pub mod device;
pub mod error;
pub mod ffi;
pub mod memory;
pub mod module;
pub mod stream;

pub use device::Device;
pub use error::{HipError, HipResult};
pub use memory::{
    DeviceBuffer, MappedHostBuffer, PinnedBuffer, memcpy_d2d, memcpy_d2h, memcpy_h2d,
};
pub use stream::Stream;

use braidinfer_core::types::DeviceId;
use std::sync::atomic::{AtomicU32, Ordering};

/// Bitmask of GPUs that currently have a persistent cooperative worker holding
/// all CUs. Bit `i` set => GPU `i` is busy. Any hipMemcpy / hipLaunchKernel on
/// the same GPU as a set bit would deadlock; cross-GPU operations are fine.
/// Supports up to 32 GPUs.
static PERSISTENT_WORKER_ACTIVE_MASK: AtomicU32 = AtomicU32::new(0);

pub fn set_persistent_worker_active(device: DeviceId, active: bool) {
    let bit = 1u32 << device.0;
    if active {
        PERSISTENT_WORKER_ACTIVE_MASK.fetch_or(bit, Ordering::SeqCst);
    } else {
        PERSISTENT_WORKER_ACTIVE_MASK.fetch_and(!bit, Ordering::SeqCst);
    }
}

fn is_persistent_worker_active(device: DeviceId) -> bool {
    let bit = 1u32 << device.0;
    PERSISTENT_WORKER_ACTIVE_MASK.load(Ordering::SeqCst) & bit != 0
}

/// Panics if a persistent worker is active on the given GPU.
/// Use this guard in any HIP function that would deadlock (memcpy, kernel launch, etc.).
#[track_caller]
pub fn assert_no_persistent_worker_on(op: &str, device: DeviceId) {
    if is_persistent_worker_active(device) {
        panic!(
            "HIP operation '{}' called on GPU {} while persistent worker holds all its CUs — \
             this would deadlock. Use GPU-side printf() for debugging during inference. \
             See CLAUDE.md 'What Causes Hangs in Persistent Mode'.",
            op, device.0
        );
    }
}

/// Panics if a persistent worker is active on the *current* GPU. Used for free
/// functions like `memcpy_d2h`/`memcpy_h2d` that don't carry a `DeviceId`.
#[track_caller]
pub fn assert_no_persistent_worker_current(op: &str) {
    let mut current: i32 = 0;
    unsafe { ffi::hipGetDevice(&mut current) };
    if current >= 0 {
        assert_no_persistent_worker_on(op, DeviceId(current as u32));
    }
}
