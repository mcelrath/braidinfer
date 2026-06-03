//! bd w1li.1: per-op test for the SHIPPING megakernel `op_silu_mul`.
//!
//! Validates the production `op_silu_mul` (out[i] = silu(gate[i]) * up[i], silu(x)=x/(1+e^-x))
//! via the real persistent-worker dispatch path vs a CPU reference — the braidinfer-internal
//! half of the per-op test framework. The matching op_silu_mul_test_entry is am-rs-loadable.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::MappedHostBuffer;
use braidinfer_runtime::megakernel::srg6_4_test_api::{SiluMulInst, SHARED_LPROJ_TOTAL};
use braidinfer_runtime::persistent_dispatch::PersistentDispatch;
use braidinfer_runtime::watchdog::WatchdogThread;
use std::sync::Arc;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn test_op_silu_mul_matches_reference() {
    let device = DeviceId(0);
    let n = 1024usize;
    // grid_x = div_ceil(n, 256) — production convention (compile_layers.rs:415): 256 elems/vblock.
    let grid_x = ((n + 255) / 256) as u32;

    let watchdog = Arc::new(WatchdogThread::spawn());
    let mut dispatch =
        PersistentDispatch::init_with_total(1, &[], SHARED_LPROJ_TOTAL, 0, watchdog)
            .expect("init dispatch");
    dispatch
        .add_device(device, SHARED_LPROJ_TOTAL)
        .expect("add GPU 0 persistent worker");

    let gate = MappedHostBuffer::<f32>::alloc_coherent(n).expect("alloc gate");
    let up = MappedHostBuffer::<f32>::alloc_coherent(n).expect("alloc up");
    let mut output = MappedHostBuffer::<f32>::alloc_coherent(n).expect("alloc output");

    let gate_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011).sin() * 3.0).collect();
    let up_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.009).cos()).collect();
    unsafe {
        let gp = gate.host_ptr();
        for (i, &v) in gate_data.iter().enumerate() {
            gp.add(i).write_volatile(v);
        }
        let upp = up.host_ptr();
        for (i, &v) in up_data.iter().enumerate() {
            upp.add(i).write_volatile(v);
        }
        let op = output.host_ptr();
        for i in 0..n {
            op.add(i).write_volatile(f32::from_bits(0xDEAD_BEEF));
        }
    }

    let inst = SiluMulInst::test_new(
        grid_x,
        output.as_mut_ptr(),
        gate.as_ptr(),
        up.as_ptr(),
        n as i32,
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
    dispatch.shutdown();

    let max_err: f32 = result
        .iter()
        .zip(gate_data.iter().zip(up_data.iter()))
        .map(|(a, (g, u))| (a - silu(*g) * u).abs())
        .fold(0.0f32, f32::max);
    // Tolerance accommodates GPU __expf vs CPU exp (silu has an exp).
    assert!(
        max_err < 1e-3,
        "op_silu_mul max error {max_err} exceeds tolerance (shipping op diverges from CPU ref)"
    );
    println!("op_silu_mul test passed: max error = {max_err:.2e}");
}
