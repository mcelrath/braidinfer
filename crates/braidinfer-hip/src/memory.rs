use crate::dev_ptr::{DevPtr, tags};
use crate::{HipResult, error, ffi};
use braidinfer_core::types::DeviceId;
use std::marker::PhantomData;
use std::ptr;

/// Allocation class for a [`DeviceBuffer`]. Recorded at construction so that
/// typed-pointer accessors (`as_typed_local` / `as_typed_uncached` /
/// `as_typed_peer`) can assert the buffer was allocated with the right
/// constructor. The fast path is monomorphized — there is no runtime check
/// in the generic accessor; the tag is only used by the unsafe `from_raw`
/// inside each typed accessor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllocClass {
    /// `hipMalloc` — coarse-grained, device-local.
    CoarseGrainedLocal,
    /// `hipExtMallocWithFlags(HIP_DEVICE_MALLOC_UNCACHED)` — MTYPE=UC.
    UncachedDeviceLocal,
    /// portable / peer-visible coarse-grained.
    CoarseGrainedPeer,
    /// bd 4ayf B1: a NON-OWNING view into another (coarse-grained-local) buffer — e.g. a
    /// weight pointing into the bulk-loaded VRAM arena. `Drop` does NOT free; the backing
    /// buffer (the arena) owns the memory and must outlive every view into it.
    View,
}

/// GPU device memory buffer. Encodes device ID to prevent cross-device misuse.
/// Deliberately NOT Send/Sync — GPU pointers are device-local.
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
    device: DeviceId,
    alloc_class: AllocClass,
    _marker: PhantomData<*mut T>, // !Send, !Sync
}

impl<T> DeviceBuffer<T> {
    pub fn alloc(device: DeviceId, len: usize) -> HipResult<Self> {
        crate::device::Device::set_current(device)?;
        let size = len * std::mem::size_of::<T>();
        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        error::check(unsafe { ffi::hipMalloc(&mut ptr, size) })?;
        Ok(DeviceBuffer {
            ptr: ptr.cast(),
            len,
            device,
            alloc_class: AllocClass::CoarseGrainedLocal,
            _marker: PhantomData,
        })
    }

    /// Allocate uncached device memory (MTYPE=UC). Use for cross-GPU buffers
    /// read after a CPU spin-wait (no kernel-launch boundary): UC bypasses L2
    /// on **all** GPUs, so peer reads see fresh VRAM without needing
    /// hipEventWaitEvent's KMD-driven L2 invalidation. Per
    /// composable_kernel/GFX1100_ARCH.md §5.1: gfx1100 has no ISA-level L2
    /// invalidation, so UC is the only non-launch path to coherent peer reads.
    pub fn alloc_uncached(device: DeviceId, len: usize) -> HipResult<Self> {
        crate::device::Device::set_current(device)?;
        let size = len * std::mem::size_of::<T>();
        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        // hipDeviceMallocUncached = 0x3 (per ROCm hip_runtime_api.h).
        const HIP_DEVICE_MALLOC_UNCACHED: std::ffi::c_uint = 0x3;
        error::check(unsafe {
            ffi::hipExtMallocWithFlags(&mut ptr, size, HIP_DEVICE_MALLOC_UNCACHED)
        })?;
        Ok(DeviceBuffer {
            ptr: ptr.cast(),
            len,
            device,
            alloc_class: AllocClass::UncachedDeviceLocal,
            _marker: PhantomData,
        })
    }

    /// bd 4ayf B1: a NON-OWNING view into an existing coarse-grained-local buffer (an arena)
    /// at `ptr` for `len` elements. `Drop` does NOT free. SAFETY: `ptr` must point into a
    /// live coarse-grained-local `DeviceBuffer` (the arena) on `device` that outlives this
    /// view, with at least `len` valid elements. `as_ptr()`/`as_typed_local()` are valid
    /// (the arena memory is coarse-grained-local).
    pub unsafe fn view(device: DeviceId, ptr: *const T, len: usize) -> Self {
        DeviceBuffer {
            ptr: ptr as *mut T,
            len,
            device,
            alloc_class: AllocClass::View,
            _marker: PhantomData,
        }
    }

    /// Typed-pointer accessor for coarse-grained device-local buffers
    /// (i.e. allocated by [`DeviceBuffer::alloc`]). Hardware atomics
    /// (`unsafeAtomicAdd`) are safe through this pointer.
    ///
    /// # Panics
    /// Panics if this buffer was not allocated by `alloc` (debug + release).
    /// This is a programming error, not a runtime condition.
    #[track_caller]
    pub fn as_typed_local(&self) -> DevPtr<T, tags::CoarseGrainedLocal> {
        assert_eq!(
            self.alloc_class,
            AllocClass::CoarseGrainedLocal,
            "DeviceBuffer::as_typed_local() called on a buffer allocated with {:?} \
             — use the matching as_typed_* accessor",
            self.alloc_class
        );
        unsafe { DevPtr::from_raw(self.ptr, self.len) }
    }

    /// Typed-pointer accessor for MTYPE=UC buffers (allocated by
    /// [`DeviceBuffer::alloc_uncached`]). Hardware atomics are **undefined**
    /// through this pointer; the type system prevents passing it to a
    /// function that requires `CoarseGrainedLocal`.
    ///
    /// # Panics
    /// Panics if this buffer was not allocated by `alloc_uncached`.
    #[track_caller]
    pub fn as_typed_uncached(&self) -> DevPtr<T, tags::UncachedDeviceLocal> {
        assert_eq!(
            self.alloc_class,
            AllocClass::UncachedDeviceLocal,
            "DeviceBuffer::as_typed_uncached() called on a buffer allocated with {:?} \
             — use the matching as_typed_* accessor",
            self.alloc_class
        );
        unsafe { DevPtr::from_raw(self.ptr, self.len) }
    }

    /// Typed-pointer accessor for portable / peer-visible coarse-grained
    /// buffers. Hardware atomics through this pointer complete in the
    /// owning GPU's local cache and may NOT be visible to peers.
    ///
    /// # Panics
    /// Panics if this buffer was not allocated as portable/peer-visible.
    #[track_caller]
    pub fn as_typed_peer(&self) -> DevPtr<T, tags::CoarseGrainedPeer> {
        assert_eq!(
            self.alloc_class,
            AllocClass::CoarseGrainedPeer,
            "DeviceBuffer::as_typed_peer() called on a buffer allocated with {:?} \
             — use the matching as_typed_* accessor",
            self.alloc_class
        );
        unsafe { DevPtr::from_raw(self.ptr, self.len) }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns *mut T from &self for GPU instruction packing: the GPU writes
    /// through this pointer at execution time, but compilation only has &self.
    pub fn as_write_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Diagnostic: query allocation type + flags via hipPointerGetAttributes.
    /// Returns (memory_type, allocation_flags). memory_type: 1=Host, 2=Device,
    /// 3=Managed. allocation_flags: e.g. 0x3 = hipDeviceMallocUncached,
    /// 0x0 = default hipMalloc.
    pub fn pointer_attributes(&self) -> HipResult<(u32, u32)> {
        let mut attr = ffi::HipPointerAttribute::default();
        error::check(unsafe {
            ffi::hipPointerGetAttributes(&mut attr, self.ptr as *const std::ffi::c_void)
        })?;
        Ok((attr.mem_type, attr.allocation_flags))
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    pub fn copy_from_host(&mut self, data: &[T]) -> HipResult<()> {
        crate::assert_no_persistent_worker_on("DeviceBuffer::copy_from_host", self.device);
        assert!(data.len() <= self.len, "source larger than buffer");
        let size = data.len() * std::mem::size_of::<T>();
        error::check(unsafe {
            ffi::hipMemcpy(
                self.ptr.cast(),
                data.as_ptr().cast(),
                size,
                ffi::hipMemcpyHostToDevice,
            )
        })
    }

    pub fn copy_to_host(&self, data: &mut [T]) -> HipResult<()> {
        crate::assert_no_persistent_worker_on("DeviceBuffer::copy_to_host", self.device);
        assert!(data.len() >= self.len, "destination smaller than buffer");
        let size = self.len * std::mem::size_of::<T>();
        error::check(unsafe {
            ffi::hipMemcpy(
                data.as_mut_ptr().cast(),
                self.ptr.cast(),
                size,
                ffi::hipMemcpyDeviceToHost,
            )
        })
    }
}

/// Copy `len` bytes from a device pointer to a host slice.
pub fn memcpy_d2h(dst: &mut [u8], src: *const u8, len: usize) -> HipResult<()> {
    crate::assert_no_persistent_worker_current("memcpy_d2h");
    assert!(dst.len() >= len, "destination buffer too small");
    error::check(unsafe {
        ffi::hipMemcpy(
            dst.as_mut_ptr().cast(),
            src.cast(),
            len,
            ffi::hipMemcpyDeviceToHost,
        )
    })
}

/// Copy `len` bytes from a host slice to a device pointer.
pub fn memcpy_h2d(dst: *mut u8, src: &[u8], len: usize) -> HipResult<()> {
    crate::assert_no_persistent_worker_current("memcpy_h2d");
    assert!(src.len() >= len, "source buffer too small");
    error::check(unsafe {
        ffi::hipMemcpy(
            dst.cast(),
            src.as_ptr().cast(),
            len,
            ffi::hipMemcpyHostToDevice,
        )
    })
}

/// Copy `len` bytes between two device pointers (same or different GPU).
pub fn memcpy_d2d(dst: *mut u8, src: *const u8, len: usize) -> HipResult<()> {
    error::check(unsafe {
        ffi::hipMemcpy(dst.cast(), src.cast(), len, ffi::hipMemcpyDeviceToDevice)
    })
}


impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        // bd 4ayf B1: a View is a non-owning borrow into an arena — the arena owns the
        // memory and frees it; never free through a view.
        if self.alloc_class == AllocClass::View {
            return;
        }
        if !self.ptr.is_null() {
            // braidinfer-4fg: if THIS GPU still has a (possibly leaked)
            // persistent worker, hipFree blocks on SyncAllStreams which
            // waits for the worker to release CUs. Skip and accept leak.
            if crate::is_persistent_worker_active(self.device) {
                eprintln!(
                    "braidinfer: leaking {}B on {:?} (persistent worker active)",
                    self.size_bytes(),
                    self.device
                );
                return;
            }
            if crate::device::Device::set_current(self.device).is_err() {
                // Accept leak rather than free on wrong device
                eprintln!(
                    "braidinfer: leaked {}B on {:?} (set_current failed)",
                    self.size_bytes(),
                    self.device
                );
                return;
            }
            let err = unsafe { ffi::hipFree(self.ptr.cast()) };
            if err != 0 {
                eprintln!(
                    "braidinfer: hipFree failed (error {}) for {}B on {:?}",
                    err,
                    self.size_bytes(),
                    self.device
                );
            }
        }
    }
}

/// Pinned host memory buffer. Required for hipMemcpyAsync.
/// Safe to access from any thread (IS Send+Sync).
pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

// Pinned memory is host memory, safe to share across threads
unsafe impl<T: Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedBuffer<T> {}

impl<T> PinnedBuffer<T> {
    pub fn alloc(len: usize) -> HipResult<Self> {
        let size = len * std::mem::size_of::<T>();
        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        error::check(unsafe { ffi::hipHostMalloc(&mut ptr, size, ffi::hipHostMallocDefault) })?;
        Ok(PinnedBuffer {
            ptr: ptr.cast(),
            len,
            _marker: PhantomData,
        })
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// # Safety
    /// Caller must ensure no other references to this buffer exist.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Mapped host buffer: accessible from both CPU (host_ptr) and GPU (device_ptr).
/// Uses hipHostMallocMapped which maps the allocation into the GPU address space.
/// Reads/writes are coherent: GPU accesses go through GART (MTYPE_UC, no L2 caching).
/// Use for barrier flags and small shared state between CPU and a running GPU kernel.
///
/// # Choosing an allocator (snl audit 2026-05-17)
///
/// | Writer | Reader(s)           | Use                     |
/// |--------|---------------------|-------------------------|
/// | CPU    | single GPU          | `alloc`                 |
/// | CPU    | multiple GPUs       | `alloc_portable`        |
/// | GPU    | single GPU (self)   | `alloc` or `alloc_coherent` |
/// | GPU    | peer GPU(s)         | `alloc_portable_coherent` (MANDATORY) |
///
/// When a GPU writes and another GPU reads, `alloc_portable` is insufficient
/// because it may use MTYPE_NC (L2-cached) on the writer GPU — peer reads
/// can observe stale data past the ack/fence boundary. Only
/// `alloc_portable_coherent` forces MTYPE_UC. Empirical: this caused the
/// snl 4-GPU decode NaN bug fixed 2026-05-17 (normed_stage was
/// alloc_portable; switched to alloc_portable_coherent restored correctness).
pub struct MappedHostBuffer<T> {
    host_ptr: *mut T,
    device_ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

// Mapped memory is host memory, safe to share across threads
unsafe impl<T: Send> Send for MappedHostBuffer<T> {}
unsafe impl<T: Sync> Sync for MappedHostBuffer<T> {}

impl<T> MappedHostBuffer<T> {
    /// Allocate host-mapped memory for GPU→CPU signaling (ack, seq, flags).
    /// Does NOT use hipHostMallocPortable — this preserves MTYPE_UC (uncached)
    /// on the allocating GPU so that volatile writes are immediately CPU-visible.
    /// The device_ptr is valid only on the GPU that was current at allocation time.
    pub fn alloc(len: usize) -> HipResult<Self> {
        Self::alloc_impl(len, ffi::hipHostMallocMapped)
    }

    /// Allocate portable host-mapped memory: device_ptr valid from ALL GPUs.
    /// Use ONLY for buffers read by GPU kernels on DIFFERENT GPUs than the
    /// allocating GPU (e.g., normed_stage broadcast input).
    /// WARNING: may use MTYPE_NC (L2-cached) on some GPUs → GPU writes may be
    /// delayed reaching CPU. Do NOT use for GPU→CPU signaling (ack/seq fields).
    pub fn alloc_portable(len: usize) -> HipResult<Self> {
        Self::alloc_impl(len, ffi::hipHostMallocMapped | ffi::hipHostMallocPortable)
    }

    /// Force fine-grained coherent host-mapped memory. CPU writes are
    /// immediately visible to the GPU (and vice versa) — no L2 caching on
    /// either side. Use for tight CPU↔GPU polling protocols where the
    /// default `alloc` doc-claim of "MTYPE_UC on allocating GPU" may not
    /// hold under multi-GPU ROCm firmware configurations.
    ///
    /// L2-staleness test (braidinfer-pky.2 / phase-5-prime-zuk-q9z-2026-05-12):
    /// the persistent worker's volatile poll of `queue->seq_num` may wedge
    /// because the worker L2 caches a stale line under multi-GPU PCIe
    /// pressure. `alloc_coherent` forces immediate visibility.
    pub fn alloc_coherent(len: usize) -> HipResult<Self> {
        Self::alloc_impl(
            len,
            ffi::hipHostMallocMapped | ffi::hipHostMallocCoherent,
        )
    }

    /// Portable + fine-grained coherent: device_ptr valid via
    /// hipHostGetDevicePointer from every GPU's context AND CPU writes
    /// are immediately visible to GPUs on all of them (and vice versa).
    /// Used for cross-GPU shared buffers where multiple GPUs both write
    /// and read (e.g. moe output_slots, activation handoffs).
    ///
    /// Combines all three flags: Mapped (device-accessible) + Portable
    /// (per-context dev_ptr usable from every GPU) + Coherent (no L2
    /// caching on writer side → no MTYPE_NC fallback).
    pub fn alloc_portable_coherent(len: usize) -> HipResult<Self> {
        Self::alloc_impl(
            len,
            ffi::hipHostMallocMapped
                | ffi::hipHostMallocPortable
                | ffi::hipHostMallocCoherent,
        )
    }

    fn alloc_impl(len: usize, flags: u32) -> HipResult<Self> {
        let size = len * std::mem::size_of::<T>();
        let mut host_ptr: *mut std::ffi::c_void = ptr::null_mut();
        error::check(unsafe { ffi::hipHostMalloc(&mut host_ptr, size, flags) })?;
        let mut device_ptr: *mut std::ffi::c_void = ptr::null_mut();
        error::check(unsafe { ffi::hipHostGetDevicePointer(&mut device_ptr, host_ptr, 0) })?;
        // Zero-initialize: hipHostMalloc does not guarantee zeroed memory.
        let typed_ptr = host_ptr.cast::<T>();
        unsafe {
            ptr::write_bytes(typed_ptr, 0, len);
        }
        Ok(MappedHostBuffer {
            host_ptr: typed_ptr,
            device_ptr: device_ptr.cast(),
            len,
            _marker: PhantomData,
        })
    }

    /// CPU-side pointer. Use for direct CPU reads/writes.
    pub fn host_ptr(&self) -> *mut T {
        self.host_ptr
    }

    /// GPU-side pointer. Pass this to kernels via instruction slots.
    pub fn device_ptr(&self) -> *const T {
        self.device_ptr as *const T
    }

    /// GPU-side pointer alias (matches DeviceBuffer::as_ptr for uniform kernel code).
    pub fn as_ptr(&self) -> *const T {
        self.device_ptr as *const T
    }

    /// Mutable GPU-side pointer alias (matches DeviceBuffer::as_mut_ptr).
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.device_ptr
    }

    /// Returns *mut T from &self for GPU instruction packing (see DeviceBuffer::as_write_ptr).
    pub fn as_write_ptr(&self) -> *mut T {
        self.device_ptr
    }

    /// Diagnostic: query allocation type + flags via hipPointerGetAttributes
    /// on the GPU-side pointer. Returns (memory_type, allocation_flags).
    pub fn pointer_attributes(&self) -> HipResult<(u32, u32)> {
        let mut attr = ffi::HipPointerAttribute::default();
        error::check(unsafe {
            ffi::hipPointerGetAttributes(&mut attr, self.device_ptr as *const std::ffi::c_void)
        })?;
        Ok((attr.mem_type, attr.allocation_flags))
    }

    /// Typed-pointer view of the GPU-side address. Tagged
    /// [`tags::HostMapped`]: hardware atomics through this pointer are
    /// **undefined** (host fine-grained memory does not implement device
    /// atomics).
    pub fn as_typed_host_mapped(&self) -> DevPtr<T, tags::HostMapped> {
        unsafe { DevPtr::from_raw(self.device_ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for MappedHostBuffer<T> {
    fn drop(&mut self) {
        if self.host_ptr.is_null() {
            return;
        }
        // braidinfer-4fg: if any persistent worker is still active (including
        // a leaked one whose cooperative-shutdown timed out), hipHostFree
        // deadlocks because the worker still holds the device pointer to this
        // page. Skip the free and let the OS reclaim on process exit.
        if crate::any_persistent_worker_active() {
            // Persistent worker still holds the device pointer to this page.
            // hipHostFree would deadlock; skip and let OS reclaim on process exit.
            return;
        }
        let err = unsafe { ffi::hipHostFree(self.host_ptr.cast()) };
        if err != 0 {
            eprintln!(
                "braidinfer: hipHostFree failed (error {}) for mapped buffer",
                err,
            );
        }
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        // braidinfer-4fg: same hipHostFree deadlock as MappedHostBuffer.
        if crate::any_persistent_worker_active() {
            // Persistent worker holds reference; skip hipHostFree, OS reclaims on exit.
            return;
        }
        {
            let err = unsafe { ffi::hipHostFree(self.ptr.cast()) };
            if err != 0 {
                eprintln!(
                    "braidinfer: hipHostFree failed (error {}) for {}B pinned buffer",
                    err,
                    self.len * std::mem::size_of::<T>()
                );
            }
        }
    }
}
