use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::OutputGateKernel;

#[test]
fn test_output_gate_matches_reference() {
    let device = DeviceId(0);
    let size: usize = 1024;

    let attn_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
    let gate_data: Vec<f32> = (0..size)
        .map(|i| (i as f32 * 0.007 - 0.5).cos() * 2.0)
        .collect();

    let expected: Vec<f32> = attn_data
        .iter()
        .zip(gate_data.iter())
        .map(|(a, g)| a * (1.0 / (1.0 + (-g).exp())))
        .collect();

    let stream = Stream::new(device).expect("stream");
    let kernel = OutputGateKernel::load(device).expect("load kernel");

    let mut d_output = DeviceBuffer::<f32>::alloc(device, size).expect("alloc output");
    let mut d_attn = DeviceBuffer::<f32>::alloc(device, size).expect("alloc attn");
    let mut d_gate = DeviceBuffer::<f32>::alloc(device, size).expect("alloc gate");

    d_attn.copy_from_host(&attn_data).expect("copy attn");
    d_gate.copy_from_host(&gate_data).expect("copy gate");

    kernel
        .forward(&mut d_output, &d_attn, &d_gate, size as u32, &stream)
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
        "output_gate max error {max_err} exceeds tolerance"
    );
    println!("output_gate test passed: max error = {max_err:.2e}");
}
