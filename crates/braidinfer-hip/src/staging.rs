//! `CrossGpuStaging<T>` — single audit surface for the §11.4 host-mapped UC
//! pattern used by cross-GPU buffers in moe_p2p / multi_gpu.
//!
//! Today the same 4-step sequence (alloc portable+coherent host-mapped, then
//! `hipHostGetDevicePointer` once per GPU under a `DeviceGuard`) is open-coded
//! at every cross-GPU staging site. This type owns the buffer and the per-GPU
//! device pointer vector together so the call site can't forget the device
//! switch, append pointers in the wrong order, or use the wrong allocator.
//!
//! ## Why MTYPE_UC (portable_coherent) is required for cross-GPU buffers
//!
//! gfx11 (RDNA3) has no `buffer_wbl2` instruction and no ISA-level L2
//! invalidation for the GART aperture. A writer GPU's cached store to a
//! host-mapped page can stay in that GPU's L2 indefinitely from a peer reader's
//! perspective; `alloc_portable` may resolve to MTYPE_NC (cached) and is
//! therefore unsafe whenever the buffer is read by a different GPU after the
//! writer's ack. Only `alloc_portable_coherent` forces MTYPE_UC on every GPU's
//! mapping, which is what this type uses unconditionally.
//!
//! ## Indexing convention
//!
//! Device pointers are stored in the order the `devices` slice is passed to
//! [`CrossGpuStaging::alloc`]. The caller chooses the mapping from semantic
//! GPU id to slice position:
//!
//! - `moe_p2p` passes `[gpu0, worker0, worker1, ...]` so that `gpu_idx ==
//!   gpu_id`.
//! - `multi_gpu::attn_out` passes `[worker.device, DeviceId(0)]` per-iteration
//!   so that `dev_ptr(0)` is the worker self-view and `dev_ptr(1)` is the
//!   GPU 0 view.
//!
//! ## Future refactors (intentionally not done here)
//!
//! - `multi_gpu::init_attn_buffers` builds one `CrossGpuStaging<f32>` per
//!   worker in a loop, which means N independent host-mapped allocations of
//!   `local_nqh * head_dim` floats each. Consolidating into a single big slab
//!   (one alloc, sliced by worker) would reduce GART page-table pressure but
//!   requires reshuffling the `GpuWorker` field layout and is out of scope.

use crate::dev_ptr::{DevPtr, tags};
use crate::device::DeviceGuard;
use crate::memory::MappedHostBuffer;
use crate::{HipResult, error, ffi};
use braidinfer_core::types::DeviceId;
use std::ptr;

/// Cross-GPU host-mapped UC staging buffer: a `MappedHostBuffer<T>` plus a
/// vector of per-GPU device pointers, one per device in the `devices` slice
/// passed at construction. See module docs for the writer-side contract and
/// indexing convention.
pub struct CrossGpuStaging<T> {
    host: MappedHostBuffer<T>,
    /// `dev_ptrs[i]` is a `*mut T` valid in the context of `devices[i]` (the
    /// `devices` slice passed to [`Self::alloc`]). The semantic mapping from
    /// GPU id to slice index is the caller's choice — see module docs.
    dev_ptrs: Vec<*mut T>,
}

impl<T> CrossGpuStaging<T> {
    /// Allocate `len` elements of portable+coherent host-mapped memory and
    /// resolve a device pointer in each device's context, in the order given.
    ///
    /// Uses [`MappedHostBuffer::alloc_portable_coherent`] (MTYPE_UC on every
    /// GPU mapping; see module docs for why this allocator is mandatory).
    /// Each per-GPU pointer resolution is wrapped in a [`DeviceGuard`] which
    /// restores the caller's current device on Drop, so nested guards in the
    /// caller (e.g. an outer `DeviceGuard::switch_to(gpu0)`) are preserved.
    pub fn alloc(len: usize, devices: &[DeviceId]) -> HipResult<Self> {
        let host = MappedHostBuffer::<T>::alloc_portable_coherent(len)?;
        let mut dev_ptrs: Vec<*mut T> = Vec::with_capacity(devices.len());
        for &dev in devices {
            let _guard = DeviceGuard::switch_to(dev)?;
            let mut p: *mut std::ffi::c_void = ptr::null_mut();
            error::check(unsafe {
                ffi::hipHostGetDevicePointer(
                    &mut p,
                    host.host_ptr() as *mut std::ffi::c_void,
                    0,
                )
            })?;
            dev_ptrs.push(p as *mut T);
        }
        Ok(CrossGpuStaging { host, dev_ptrs })
    }

    /// Per-GPU device pointer. `gpu_idx` is the position in the `devices`
    /// slice passed at construction (NOT necessarily the raw `DeviceId`).
    #[track_caller]
    pub fn dev_ptr(&self, gpu_idx: usize) -> *mut T {
        self.dev_ptrs[gpu_idx]
    }

    /// CPU-side pointer: writable directly via the GART mapping.
    pub fn host_ptr(&self) -> *mut T {
        self.host.host_ptr()
    }

    /// Number of `T` elements.
    pub fn len(&self) -> usize {
        self.host.len()
    }

    pub fn is_empty(&self) -> bool {
        self.host.len() == 0
    }

    /// Zero the host-mapped backing storage via CPU `write_bytes`. Safe to
    /// call any time no GPU is actively reading the buffer (init or between
    /// dispatch batches).
    pub fn zero(&mut self) {
        unsafe { ptr::write_bytes(self.host.host_ptr(), 0, self.host.len()) };
    }

    /// Per-GPU typed device pointer tagged `HostMapped`. The backing
    /// allocation is `alloc_portable_coherent` (MTYPE_UC host-mapped
    /// fine-grained), so the correct tag is `HostMapped`.  Pass
    /// `.as_raw()` at the FFI/instruction-packing boundary; the type
    /// enforces that callers expecting cross-GPU signaling/mailbox memory
    /// cannot accidentally receive a VRAM pointer.
    ///
    /// `gpu_idx` is the position in the `devices` slice passed at construction.
    #[track_caller]
    pub fn typed_dev_ptr(&self, gpu_idx: usize) -> DevPtr<T, tags::HostMapped> {
        let ptr = self.dev_ptrs[gpu_idx];
        // SAFETY: `ptr` was resolved via hipHostGetDevicePointer from an
        // alloc_portable_coherent host-mapped allocation, which maps to
        // MTYPE_UC on every GPU — matching the `HostMapped` tag contract.
        // The pointer outlives this `CrossGpuStaging` (the host buffer owns the
        // underlying allocation and both have the same lifetime).
        unsafe { DevPtr::from_raw(ptr, self.host.len()) }
    }

    /// Borrow the underlying host buffer. Use for the rare case where a
    /// caller needs the typed `MappedHostBuffer` accessor (e.g. to obtain
    /// a `DevPtr` tagged view); prefer `dev_ptr` / `host_ptr` otherwise.
    pub fn host(&self) -> &MappedHostBuffer<T> {
        &self.host
    }
}

// CrossGpuStaging holds raw `*mut T` device pointers; they are pure
// per-context virtual addresses (not refcounts) so dropping them is a no-op,
// and the underlying MappedHostBuffer's Drop handles the host allocation.
// Not Send/Sync — same constraint as MappedHostBuffer (host memory is shared
// but device pointers are context-local).
