//! CPU-scheduled persistent worker dispatch (braidinfer-czl).
//! Each GPU runs a persistent cooperative kernel polling a host-mapped work queue.
//! CPU sequences operations via memory writes — no HIP API calls in the hot path.

/// Persistent dispatch context: manages worker kernels and work queues on all GPUs.
pub struct PersistentDispatch {
    // TODO(czl Phase 2): CPU scheduler + per-GPU work queues
}
