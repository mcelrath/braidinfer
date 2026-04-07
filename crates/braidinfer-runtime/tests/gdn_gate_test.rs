use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::GdnGateKernel;

fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0f32 + x.exp()).ln() }
}

#[test]
fn test_gdn_gate_decay_in_unit_interval() {
    let device = DeviceId(0);
    let num_heads: usize = 32;

    let a_log: Vec<f32> = (0..num_heads).map(|i| (i as f32 * 0.1) - 1.0).collect();
    let a: Vec<f32> = (0..num_heads).map(|i| (i as f32 * 0.05) - 0.8).collect();
    let dt_bias_f32: Vec<f32> = (0..num_heads).map(|i| (i as f32 * 0.03) - 0.5).collect();
    let dt_bias: Vec<u16> = dt_bias_f32.iter().copied().map(f32_to_bf16).collect();

    let expected: Vec<f32> = (0..num_heads)
        .map(|h| {
            let sp = softplus(a[h] + bf16_to_f32(dt_bias[h]));
            (-a_log[h].exp() * sp).exp()
        })
        .collect();

    // All expected values should be in (0, 1)
    for (h, &v) in expected.iter().enumerate() {
        assert!(v > 0.0 && v < 1.0, "head {h}: gate = {v} not in (0,1)");
    }

    let stream = Stream::new(device).expect("stream");
    let kernel = GdnGateKernel::load(device).expect("load kernel");

    let mut d_gate = DeviceBuffer::<f32>::alloc(device, num_heads).expect("alloc gate");
    let mut d_alog = DeviceBuffer::<f32>::alloc(device, num_heads).expect("alloc a_log");
    let mut d_a = DeviceBuffer::<f32>::alloc(device, num_heads).expect("alloc a");
    let mut d_dt = DeviceBuffer::<u16>::alloc(device, num_heads).expect("alloc dt_bias");

    d_alog.copy_from_host(&a_log).expect("copy a_log");
    d_a.copy_from_host(&a).expect("copy a");
    d_dt.copy_from_host(&dt_bias).expect("copy dt_bias");

    kernel
        .forward(&mut d_gate, &d_alog, &d_a, &d_dt, num_heads as u32, &stream)
        .expect("kernel launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; num_heads];
    d_gate.copy_to_host(&mut result).expect("copy gate");

    // Verify all decay values in (0, 1)
    for (h, &v) in result.iter().enumerate() {
        assert!(v > 0.0 && v < 1.0, "GPU head {h}: gate = {v} not in (0,1)");
    }

    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-5,
        "gdn_gate max error {max_err} exceeds tolerance"
    );
    println!("gdn_gate test passed: max error = {max_err:.2e}");
}
