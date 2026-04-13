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

use std::sync::atomic::{AtomicBool, Ordering};

/// Set to true while the persistent cooperative kernel holds all GPU 0 CUs.
/// Any hipMemcpy / hipLaunchKernel on GPU 0 while this is set will deadlock.
static PERSISTENT_WORKER_ACTIVE: AtomicBool = AtomicBool::new(false);

pub fn set_persistent_worker_active(active: bool) {
    PERSISTENT_WORKER_ACTIVE.store(active, Ordering::SeqCst);
}

/// Panics if called while the persistent worker is active.
/// Use this guard in any HIP function that would deadlock (memcpy, kernel launch, etc.).
#[track_caller]
pub fn assert_no_persistent_worker(op: &str) {
    if PERSISTENT_WORKER_ACTIVE.load(Ordering::SeqCst) {
        panic!(
            "HIP operation '{}' called while persistent worker holds all GPU 0 CUs — \
             this would deadlock. Use GPU-side printf() for debugging during inference. \
             See CLAUDE.md 'What Causes Hangs in Persistent Mode'.",
            op
        );
    }
}
