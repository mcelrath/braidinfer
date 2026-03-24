use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::SiluMulKernel;

fn silu_mul_reference(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let silu_g = g / (1.0f32 + (-g).exp());
            silu_g * u
        })
        .collect()
}

#[test]
fn test_silu_mul_matches_reference() {
    let device = DeviceId(0);
    let size = 2816usize;

    let gate_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin() * 2.0).collect();
    let up_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.007).cos()).collect();

    let expected = silu_mul_reference(&gate_data, &up_data);

    let stream = Stream::new(device).expect("stream");
    let kernel = SiluMulKernel::load(device).expect("load kernel");

    let mut d_gate = DeviceBuffer::<f32>::alloc(device, size).expect("alloc gate");
    let mut d_up = DeviceBuffer::<f32>::alloc(device, size).expect("alloc up");
    let mut d_output = DeviceBuffer::<f32>::alloc(device, size).expect("alloc output");

    d_gate.copy_from_host(&gate_data).expect("copy gate");
    d_up.copy_from_host(&up_data).expect("copy up");

    kernel
        .forward(&mut d_output, &d_gate, &d_up, size as u32, &stream)
        .expect("kernel launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; size];
    d_output.copy_to_host(&mut result).expect("copy output");

    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-5,
        "SiluMul max error {max_err} exceeds tolerance"
    );
    println!("SiluMul test passed: max error = {max_err:.2e}");
}
