//! bd w1li.1: per-op test for the SHIPPING megakernel `op_rmsnorm`.
//!
//! The legacy `rmsnorm_test.rs` validates `RmsNormKernel` (rmsnorm.hsaco) — the
//! DEPRECATED standalone kbk kernel, NOT the `op_rmsnorm` that actually ships in
//! the persistent megakernel. This test dispatches a single rmsnorm instruction
//! through the real persistent worker (`PersistentDispatch::test_dispatch_batch_slice`
//! — the same path production decode/prefill use) and asserts vs a CPU reference.
//! It is the interim, ABI-independent harness of the w1li per-op test framework
//! (P1); P2 factors `op_core` and P3 adds the am-rs-loadable `__global__` entries.
//!
//! Both shipping rmsnorm opcodes are covered (see compile_common.rs rmsnorm_opcode):
//!   OP_RMSNORM    = y = x·rms·(1+w)  (one_plus_w=true — the Qwen3.x variant)
//!   OP_RMSNORM_WX = y = x·rms·w      (one_plus_w=false)

use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::MappedHostBuffer;
use braidinfer_runtime::megakernel::srg6_4_test_api::{
    OP_RMSNORM, OP_RMSNORM_WX, RmsNormInst, SHARED_LPROJ_TOTAL,
};
use braidinfer_runtime::persistent_dispatch::PersistentDispatch;
use braidinfer_runtime::watchdog::WatchdogThread;
use std::sync::Arc;

fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// CPU reference. `one_plus_w` selects the (1+w) vs plain-w weight scaling,
/// mirroring the two megakernel opcodes (compile_common.rs rmsnorm_opcode).
fn rmsnorm_reference(input: &[f32], weight: &[u16], eps: f32, one_plus_w: bool) -> Vec<f32> {
    let n = weight.len();
    let num_rows = input.len() / n;
    let mut output = vec![0.0f32; input.len()];
    for row in 0..num_rows {
        let x = &input[row * n..(row + 1) * n];
        let y = &mut output[row * n..(row + 1) * n];
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt().recip();
        for i in 0..n {
            let w = bf16_to_f32(weight[i]);
            let scale = if one_plus_w { 1.0 + w } else { w };
            y[i] = x[i] * rms * scale;
        }
    }
    output
}

fn run_case(dispatch: &mut PersistentDispatch, opcode: u32, one_plus_w: bool, label: &str) {
    let hidden_size = 1024usize;
    let num_rows = 4usize;
    let eps = 1e-6f32;
    let n = num_rows * hidden_size;

    // Host-mapped coherent IO: GPU writes are immediately visible to host_ptr
    // reads without a stream synchronize (CPU MMIO, safe under the running worker).
    let input = MappedHostBuffer::<f32>::alloc_coherent(n).expect("alloc input");
    let weight = MappedHostBuffer::<u16>::alloc_coherent(hidden_size).expect("alloc weight");
    let mut output = MappedHostBuffer::<f32>::alloc_coherent(n).expect("alloc output");

    let input_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let weight_data: Vec<u16> = (0..hidden_size)
        .map(|i| f32_to_bf16(1.0 + (i as f32 * 0.001)))
        .collect();
    unsafe {
        let ip = input.host_ptr();
        for (i, &v) in input_data.iter().enumerate() {
            ip.add(i).write_volatile(v);
        }
        let wp = weight.host_ptr();
        for (i, &v) in weight_data.iter().enumerate() {
            wp.add(i).write_volatile(v);
        }
        let op = output.host_ptr();
        for i in 0..n {
            op.add(i).write_volatile(f32::from_bits(0xDEAD_BEEF));
        }
    }

    // grid_x = num_rows (production: 1 decode / n prefill).
    let inst = RmsNormInst::test_new(
        opcode,
        num_rows as u32,
        output.as_mut_ptr(),
        input.as_ptr(),
        weight.as_ptr(),
        hidden_size as i32,
        eps,
    )
    .into_inst();
    dispatch.test_dispatch_batch_slice(0, &[inst]);

    let mut result = vec![0.0f32; n];
    unsafe {
        let op = output.host_ptr();
        for (i, r) in result.iter_mut().enumerate() {
            *r = op.add(i).read_volatile();
        }
    }

    let expected = rmsnorm_reference(&input_data, &weight_data, eps, one_plus_w);
    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-4,
        "[{label}] op_rmsnorm max error {max_err} exceeds tolerance (shipping op diverges from CPU ref)"
    );
    println!("[{label}] op_rmsnorm test passed: max error = {max_err:.2e}");
}

#[test]
fn test_op_rmsnorm_matches_reference() {
    let device = DeviceId(0);

    // Single-GPU persistent worker on slot 0 — same dispatch path production
    // uses, no model/prefill (op_rmsnorm is a self-contained vblock op).
    let watchdog = Arc::new(WatchdogThread::spawn());
    let mut dispatch =
        PersistentDispatch::init_with_total(1, &[], SHARED_LPROJ_TOTAL, 0, watchdog)
            .expect("init dispatch");
    dispatch
        .add_device(device, SHARED_LPROJ_TOTAL)
        .expect("add GPU 0 persistent worker");

    run_case(&mut dispatch, OP_RMSNORM, true, "OP_RMSNORM (1+w)");
    run_case(&mut dispatch, OP_RMSNORM_WX, false, "OP_RMSNORM_WX (w)");

    dispatch.shutdown();
}
