//! bd w1li.1: per-op test for the SHIPPING megakernel `op_residual_add`.
//!
//! Validates the production `op_residual_add` (out[i] = src[i] + residual[i]) via the real
//! persistent-worker dispatch path vs a CPU reference — the braidinfer-internal half of the
//! per-op test framework. (am-rs validated the matching op_residual_add_test_entry byte-exact
//! on VFIO; this is the regression harness on the braidinfer side.)

use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::MappedHostBuffer;
use braidinfer_runtime::megakernel::srg6_4_test_api::{ResidualAddInst, SHARED_LPROJ_TOTAL};
use braidinfer_runtime::persistent_dispatch::PersistentDispatch;
use braidinfer_runtime::watchdog::WatchdogThread;
use std::sync::Arc;

#[test]
fn test_op_residual_add_matches_reference() {
    let device = DeviceId(0);
    let n = 1024usize;
    // grid_x = div_ceil(n, 256) — production convention (compile_attention.rs:346): each vblock
    // handles 256 elements.
    let grid_x = ((n + 255) / 256) as u32;

    let watchdog = Arc::new(WatchdogThread::spawn());
    let mut dispatch =
        PersistentDispatch::init_with_total(1, &[], SHARED_LPROJ_TOTAL, 0, watchdog)
            .expect("init dispatch");
    dispatch
        .add_device(device, SHARED_LPROJ_TOTAL)
        .expect("add GPU 0 persistent worker");

    let src = MappedHostBuffer::<f32>::alloc_coherent(n).expect("alloc src");
    let residual = MappedHostBuffer::<f32>::alloc_coherent(n).expect("alloc residual");
    let mut output = MappedHostBuffer::<f32>::alloc_coherent(n).expect("alloc output");

    let src_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
    let residual_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.007).cos()).collect();
    unsafe {
        let sp = src.host_ptr();
        for (i, &v) in src_data.iter().enumerate() {
            sp.add(i).write_volatile(v);
        }
        let rp = residual.host_ptr();
        for (i, &v) in residual_data.iter().enumerate() {
            rp.add(i).write_volatile(v);
        }
        let op = output.host_ptr();
        for i in 0..n {
            op.add(i).write_volatile(f32::from_bits(0xDEAD_BEEF));
        }
    }

    let inst = ResidualAddInst::test_new(
        grid_x,
        output.as_mut_ptr(),
        src.as_ptr(),
        residual.as_ptr(),
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
        .zip(src_data.iter().zip(residual_data.iter()))
        .map(|(a, (s, r))| (a - (s + r)).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-6,
        "op_residual_add max error {max_err} exceeds tolerance (shipping op diverges from CPU ref)"
    );
    println!("op_residual_add test passed: max error = {max_err:.2e}");
}
