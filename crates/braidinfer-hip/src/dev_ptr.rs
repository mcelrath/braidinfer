//! Typed device pointers — compile-time-enforced memory-type preconditions.
//!
//! PRIMARY USE — cross-GPU coherence. On gfx11 multi-GPU, whether a
//! producer→consumer handoff is CORRECT depends on the memory class of the
//! buffer, and getting it wrong is a silent coherence bug. Every major
//! multi-GPU bug in this engine was a memory-class mismatch:
//!   - **el1f**: a sentinel in coarse-grained-local VRAM is L2-cached and is
//!     NOT seen by peers' MTYPE-UC PCIe reads → a cross-GPU spin deadlocks.
//!   - **§11.20**: a host-mapped-UC `signal_ptr` to `op_d2d_copy` wedges 5/5
//!     (the working medium there is VRAM).
//!   - **9gmh**: a worker compute-PCIe-write to a peer's VRAM corrupts its L2.
//! A function that takes `DevPtr<T, HostMapped>` (the cross-GPU signaling /
//! mailbox / sentinel medium) cannot be handed a VRAM pointer, and an
//! `op_d2d_copy` signal taking `DevPtr<T, CoarseGrainedLocal>` cannot be handed
//! a host-UC one — so these mismatches become compile errors instead of
//! multi-hour wedges.
//!
//! `DevPtr<T, Tag>` is a `repr(transparent)` newtype around `*mut T` carrying
//! a phantom tag describing the memory class. Functions that require a
//! particular class take `DevPtr<T, ConcreteTag>`, so passing the wrong kind
//! of memory is a compile-time error.
//!
//! SECONDARY USE — hardware atomics. `unsafeAtomicAdd` (`global_atomic_add_f32`)
//! is 46–955× faster than the CAS fallback but has defined behavior ONLY on
//! coarse-grained-local memory (undefined on UC / host-mapped / peer). DevPtr
//! can guard this too — but note it is currently **policy-rejected** for LLM
//! shapes (measured negative vs block-cooperative reductions; see
//! `kernels/rdna3_compute.h` and bd 77r.2.4), so it is a latent guard, not an
//! active call path. The cross-GPU coherence use above is the load-bearing one.
//!
//! # Tags (memory class → cross-GPU role; HW-atomic safety in parens)
//!
//! | Tag | Allocator | Cross-GPU role (atomics) |
//! |---|---|---|
//! | [`tags::CoarseGrainedLocal`] | `DeviceBuffer::alloc` (`hipMalloc`) | device-local VRAM; L2-cached, NOT peer-visible w/o fence (atomics: **safe**) |
//! | [`tags::CoarseGrainedPeer`] | (portable VRAM allocator) | peer-readable for non-atomic loads (atomics: owning-GPU only) |
//! | [`tags::UncachedDeviceLocal`] | `DeviceBuffer::alloc_uncached` (MTYPE=UC) | cross-GPU-coherent VRAM, no launch fence (atomics: **undefined**) |
//! | [`tags::HostMapped`] | `MappedHostBuffer::alloc` (host fine-grained) | CPU↔GPU + cross-GPU signaling / mailbox / sentinels (atomics: **undefined**) |
//! | [`tags::WorkgroupShared`] | `__shared__` / LDS (kernel-side) | per-CU only (atomics: per-CU) |
//!
//! # Misuse caught at compile time
//!
//! A cross-GPU sentinel-write function takes `DevPtr<u32, HostMapped>` (the
//! host-UC signaling medium). Handing it a VRAM (`CoarseGrainedLocal`) pointer
//! — the el1f bug — is a type error, not a silent cross-GPU deadlock:
//!
//! ```compile_fail
//! use braidinfer_hip::dev_ptr::{DevPtr, tags::{HostMapped, CoarseGrainedLocal}};
//!
//! // The cross-GPU sentinel MUST live in host-mapped UC (el1f: a VRAM sentinel
//! // is L2-cached and missed by peers' MTYPE-UC reads). This signature enforces it.
//! fn cross_gpu_sentinel_write(_p: DevPtr<u32, HostMapped>, _seq: u32) {}
//!
//! // SAFETY: doctest only — never construct DevPtr from a dangling pointer in real code.
//! let vram: DevPtr<u32, CoarseGrainedLocal> =
//!     unsafe { DevPtr::from_raw(std::ptr::null_mut(), 0) };
//! cross_gpu_sentinel_write(vram, 1); // <-- type error: VRAM sentinel IS the el1f bug
//! ```
//!
//! And the correct host-mapped call compiles:
//!
//! ```
//! use braidinfer_hip::dev_ptr::{DevPtr, tags::HostMapped};
//!
//! fn cross_gpu_sentinel_write(_p: DevPtr<u32, HostMapped>, _seq: u32) {}
//!
//! let host_uc: DevPtr<u32, HostMapped> =
//!     unsafe { DevPtr::from_raw(std::ptr::null_mut(), 0) };
//! cross_gpu_sentinel_write(host_uc, 1);
//! ```

use std::marker::PhantomData;

/// Tag types describing the memory class of a [`DevPtr`].
pub mod tags {
    /// Coarse-grained, device-local. Hardware atomics (`unsafeAtomicAdd`,
    /// `global_atomic_add_f32`) are **safe** here. Default `hipMalloc`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct CoarseGrainedLocal;

    /// Coarse-grained, peer-visible (allocated via `hipExtMallocWithFlags`
    /// portable / `hipHostMallocPortable`-style). Hardware atomics are safe
    /// **only within the owning GPU's local cache domain** — the result will
    /// **not** propagate to peers without an explicit fence/sync. Prefer
    /// `CoarseGrainedLocal` for atomics; use this tag only when the buffer
    /// must be peer-readable for non-atomic loads.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct CoarseGrainedPeer;

    /// MTYPE=UC (uncached) device memory. Hardware atomics are
    /// **undefined behavior** here — the L1/L2 bypass conflicts with
    /// `unsafeAtomicAdd`'s assumptions. Used for cross-GPU coherence
    /// without kernel-launch fences (see GFX1100_ARCH.md §5.1).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct UncachedDeviceLocal;

    /// Host-mapped fine-grained memory (`hipHostMalloc` /
    /// `MappedHostBuffer`). Hardware atomics are **undefined** — fine-grained
    /// PCIe-mapped buffers do not implement device atomics. Use this tag for
    /// CPU↔GPU signaling buffers (ack flags, sequence counters, mailbox).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct HostMapped;

    /// Workgroup shared memory (LDS / `__shared__`). Per-CU; pointers do not
    /// escape a kernel launch. Included for completeness; rarely seen on the
    /// Rust side.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct WorkgroupShared;
}

/// A device pointer tagged with its memory class.
///
/// FFI: pass `as_raw()` to kernel-launch code; the raw pointer is identical
/// to what `DeviceBuffer::as_mut_ptr()` would have returned.
///
/// Deliberately NOT `Send`/`Sync`: device pointers are bound to the GPU
/// context that allocated them and the calling thread's `hipSetDevice`
/// state is part of the precondition. (Matches `DeviceBuffer<T>`'s own
/// non-Send/Sync stance.)
pub struct DevPtr<T, Tag> {
    ptr: *mut T,
    // `len` is informational only (not enforced at compile time) — it lets
    // consumers bounds-check at runtime when needed without a separate field.
    len: usize,
    _tag: PhantomData<fn() -> Tag>,
    _not_send_sync: PhantomData<*mut T>,
}

impl<T, Tag> DevPtr<T, Tag> {
    /// Construct a typed device pointer from a raw pointer + element length.
    ///
    /// # Safety
    /// The caller asserts that `ptr` was allocated with the memory class
    /// described by `Tag`, that `len` elements of `T` are valid at `ptr`,
    /// and that the pointer outlives the `DevPtr`.
    pub unsafe fn from_raw(ptr: *mut T, len: usize) -> Self {
        DevPtr {
            ptr,
            len,
            _tag: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Raw `*mut T` for FFI / kernel-launch packing.
    #[inline]
    pub fn as_raw(&self) -> *mut T {
        self.ptr
    }

    /// Raw `*const T` view.
    #[inline]
    pub fn as_raw_const(&self) -> *const T {
        self.ptr as *const T
    }

    /// Element count.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T, Tag> Clone for DevPtr<T, Tag> {
    fn clone(&self) -> Self {
        DevPtr {
            ptr: self.ptr,
            len: self.len,
            _tag: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

impl<T, Tag> Copy for DevPtr<T, Tag> {}

impl<T, Tag> std::fmt::Debug for DevPtr<T, Tag> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DevPtr")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("tag", &std::any::type_name::<Tag>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::tags::*;
    use super::*;

    // Compile-time check: `DevPtr<T, Tag>` is `repr(transparent)` — same size
    // and alignment as `*mut T`. (We can't assert layout with const_assert
    // alone, but at minimum the size matches a `*mut T` plus `usize`.)
    #[test]
    fn dev_ptr_size_matches_ptr_plus_len() {
        assert_eq!(
            std::mem::size_of::<DevPtr<f32, CoarseGrainedLocal>>(),
            std::mem::size_of::<*mut f32>() + std::mem::size_of::<usize>(),
        );
    }

    // Tags are zero-sized, so the type system pays no runtime cost.
    #[test]
    fn tags_are_zero_sized() {
        assert_eq!(std::mem::size_of::<CoarseGrainedLocal>(), 0);
        assert_eq!(std::mem::size_of::<CoarseGrainedPeer>(), 0);
        assert_eq!(std::mem::size_of::<UncachedDeviceLocal>(), 0);
        assert_eq!(std::mem::size_of::<HostMapped>(), 0);
        assert_eq!(std::mem::size_of::<WorkgroupShared>(), 0);
    }

    // Demonstrates the type system rejects tag mismatches. This compiles
    // only the OK branch; the bad branch lives in the module-level
    // `compile_fail` doctest above.
    #[test]
    fn coarse_grained_accepts_local_tag() {
        fn requires_local(_p: DevPtr<f32, CoarseGrainedLocal>) {}
        let p: DevPtr<f32, CoarseGrainedLocal> = unsafe { DevPtr::from_raw(std::ptr::null_mut(), 0) };
        requires_local(p);
    }
}
