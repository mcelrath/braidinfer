//! Per-op cycle profiling for the persistent megakernel.
//!
//! Plan: ~/.claude/plans/PLAN-op-profile.md (epic braidinfer-xiu).
//!
//! When the runtime is built with `BRAIDINFER_OP_PROFILE=1` (env var picked
//! up by build.rs and propagated to hipcc as `-DBRAIDINFER_OP_PROFILE`),
//! every dispatched opcode in `persistent_worker.hip` accumulates ticks
//! and call count into a per-opcode slot of a GPU-resident `u64` buffer.
//!
//! Slot layout: `counters[2 * opcode + 0] = cycles_total`,
//!              `counters[2 * opcode + 1] = call_count`.
//! Sized to [`NUM_SLOTS`].
//!
//! Lifecycle: allocate **before** [`crate::persistent_dispatch::PersistentDispatch::init_with_total`].
//! The device pointer is written into each `WorkerQueue`'s `op_profile`
//! field at worker launch time. `dump_after_shutdown` is unsafe — caller
//! must ensure the persistent worker has been dropped before reading
//! (a `hipMemcpy` D2H during cooperative-kernel life deadlocks per
//! kb `77r-2-1-dma-under-persistent-deadlocks-all-paths-2026-05-07`).

use std::sync::atomic::{AtomicPtr, Ordering};

use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::ffi;

/// Process-global op_profile counter pointer. PersistentDispatch reads this
/// at worker launch time. Set via `install_global` BEFORE first decode call
/// (which lazy-initializes the persistent worker). Null = profiling disabled.
static OP_PROFILE_GLOBAL: AtomicPtr<u64> = AtomicPtr::new(std::ptr::null_mut());

pub fn install_global(profile: &OpProfile) {
    OP_PROFILE_GLOBAL.store(profile.device_ptr(), Ordering::SeqCst);
}

pub fn uninstall_global() {
    OP_PROFILE_GLOBAL.store(std::ptr::null_mut(), Ordering::SeqCst);
}

pub fn get_global() -> *mut u64 {
    OP_PROFILE_GLOBAL.load(Ordering::SeqCst)
}

/// Matches `BRAIDINFER_OP_PROFILE_NUM_SLOTS` in `kernels/op_profile.h`.
/// Sized large enough to cover the OP_* enum with headroom.
pub const NUM_SLOTS: usize = 64;

/// Per-GPU profiling counters. Allocated before `PersistentDispatch::init_with_total`.
pub struct OpProfile {
    counters: DeviceBuffer<u64>,
}

impl OpProfile {
    /// Allocate a zeroed counter buffer of size `2 * NUM_SLOTS` u64 on `device`.
    pub fn alloc(device: DeviceId) -> HipResult<Self> {
        let mut counters = DeviceBuffer::<u64>::alloc(device, 2 * NUM_SLOTS)?;
        let zeros = vec![0u64; 2 * NUM_SLOTS];
        counters.copy_from_host(&zeros)?;
        Ok(Self { counters })
    }

    /// Device pointer for the persistent worker's WorkerQueue::op_profile field.
    pub fn device_ptr(&self) -> *mut u64 {
        self.counters.as_ptr() as *mut u64
    }

    /// Reset all counters to zero. UNSAFE: caller must ensure the persistent
    /// worker has been shut down (no atomic ops in flight).
    pub unsafe fn reset_after_shutdown(&mut self) -> HipResult<()> {
        let zeros = vec![0u64; 2 * NUM_SLOTS];
        self.counters.copy_from_host(&zeros)
    }

    /// Read counters out and return per-opcode statistics.
    ///
    /// SAFETY: caller must ensure the persistent worker has been shut down
    /// (PersistentDispatch dropped). Calling while the worker holds CUs
    /// deadlocks via DMA-under-cooperative-kernel.
    pub unsafe fn dump_after_shutdown(&self) -> HipResult<Vec<OpStats>> {
        let mut raw = vec![0u64; 2 * NUM_SLOTS];
        // SAFETY: counters is a DeviceBuffer<u64> of length 2 * NUM_SLOTS; raw
        // is sized to match. copy_to_host has its own assert_no_persistent_worker
        // guard which is the same lifecycle constraint we document.
        self.counters.copy_to_host(&mut raw)?;

        let mut out = Vec::with_capacity(NUM_SLOTS);
        for op in 0..NUM_SLOTS {
            // Note: cycles_total can be u64-wrap-negative briefly mid-flight,
            // but after shutdown the SUM of all (-t0) and (+t1) settles to
            // the true sum of deltas. If we observe a "small wrap" pattern
            // (e.g., 18 quintillion ticks for a few calls), it's because the
            // dispatcher was not actually shut down — return as-is and let
            // the caller flag.
            let cycles_total = raw[2 * op];
            let calls = raw[2 * op + 1];
            if calls == 0 {
                continue;
            }
            out.push(OpStats {
                opcode: op as u32,
                name: opcode_name(op as u32),
                calls,
                ticks_total: cycles_total,
                ticks_per_call: cycles_total as f64 / calls as f64,
            });
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct OpStats {
    pub opcode: u32,
    pub name: &'static str,
    pub calls: u64,
    pub ticks_total: u64,
    pub ticks_per_call: f64,
}

/// Map opcode id → human name. Mirrors kernels/opcodes.h. Unknown ids
/// return "OP_UNKNOWN".
pub fn opcode_name(op: u32) -> &'static str {
    // Order must match kernels/opcodes.h. Unrecognized ops fall through.
    match op {
        0 => "OP_HALT",
        1 => "OP_BARRIER",
        2 => "OP_LINEAR_PROJ",
        3 => "OP_RMSNORM",
        4 => "OP_CONV1D",
        5 => "OP_GDN_GATE",
        6 => "OP_GDN_RECUR",
        7 => "OP_RMSNORM_GATE",
        8 => "OP_RESIDUAL_ADD",
        9 => "OP_QK_NORM",
        10 => "OP_MROPE",
        11 => "OP_GQA_ATTN",
        12 => "OP_OUTPUT_GATE",
        13 => "OP_FFN_GATE_UP",
        14 => "OP_FFN_DOWN_RES",
        15 => "OP_EMBEDDING",
        16 => "OP_LM_HEAD",
        17 => "OP_D2D_COPY",
        18 => "OP_ATTN_PAGED",
        19 => "OP_ATTN_PREFILL",
        20 => "OP_DEINTERLEAVE",
        21 => "OP_KV_QUANTIZE",
        22 => "OP_ATTN_PAGED_Q",
        23 => "OP_MOE_GATE",
        24 => "OP_MOE_FFN",
        25 => "OP_LINEAR_PROJ_RNF4",
        26 => "OP_LINEAR_PROJ_PCG32",
        27 => "OP_RMSNORM_WX",
        28 => "OP_SILU_MUL",
        29 => "OP_FFN_GATE_UP_RNF4",
        30 => "OP_FFN_DOWN_RES_RNF4",
        31 => "OP_SIGMOID_WEIGHTED_ADD",
        32 => "OP_SCALE_ADD",
        33 => "OP_RELU_SQ",
        34 => "OP_MAMBA2_CONV1D",
        35 => "OP_SSM_UPDATE",
        36 => "OP_MAMBA2_NORM_GATED",
        37 => "OP_CONV1D_3X",
        38 => "OP_FFN_GATE_UP_WX",
        39 => "OP_FFN_GATE_UP_RNF4_WX",
        40 => "OP_LINEAR_PROJ_2X",
        41 => "OP_MOE_DISPATCH",
        42 => "OP_MOE_DISPATCH_POST",
        43 => "OP_MOE_FFN_REMOTE",
        _ => "OP_UNKNOWN",
    }
}

/// Format an OpStats list as a sortable table, sorted by ticks_total descending.
pub fn format_table(stats: &[OpStats]) -> String {
    let total: u64 = stats.iter().map(|s| s.ticks_total).sum();
    let mut sorted = stats.to_vec();
    sorted.sort_by(|a, b| b.ticks_total.cmp(&a.ticks_total));

    let mut out = String::new();
    out.push_str(&format!(
        "{:<26} {:>10} {:>16} {:>14} {:>6}\n",
        "opcode", "calls", "ticks_total", "ticks/call", "pct"
    ));
    for s in &sorted {
        // Sanity check: ticks_total / calls must round-trip to ticks_per_call.
        let computed = s.ticks_total as f64 / s.calls as f64;
        assert!(
            (computed - s.ticks_per_call).abs() < 1.0,
            "ticks/call drift in OpStats — atomic dropout?"
        );
        let pct = if total > 0 {
            100.0 * s.ticks_total as f64 / total as f64
        } else {
            0.0
        };
        out.push_str(&format!(
            "{:<26} {:>10} {:>16} {:>14.0} {:>5.1}%\n",
            s.name, s.calls, s.ticks_total, s.ticks_per_call, pct
        ));
    }
    out
}

#[allow(dead_code)]
fn _unused() {
    // suppress unused-import warning when nothing else references ffi
    let _ = ffi::hipMemcpy;
}
