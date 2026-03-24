use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::RmsNormKernel;

fn rmsnorm_reference(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = weight.len();
    let num_rows = input.len() / n;
    let mut output = vec![0.0f32; input.len()];
    for row in 0..num_rows {
        let x = &input[row * n..(row + 1) * n];
        let y = &mut output[row * n..(row + 1) * n];
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt().recip();
        for i in 0..n {
            y[i] = x[i] * rms * weight[i];
        }
    }
    output
}

#[test]
fn test_rmsnorm_matches_reference() {
    let device = DeviceId(0);
    let hidden_size = 1024u32;
    let num_rows = 4u32;
    let eps = 1e-6f32;
    let n = (num_rows * hidden_size) as usize;

    // Generate test data
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let weight_data: Vec<f32> = (0..hidden_size as usize)
        .map(|i| 1.0 + (i as f32 * 0.001))
        .collect();

    // CPU reference
    let expected = rmsnorm_reference(&input_data, &weight_data, eps);

    // GPU computation
    let stream = Stream::new(device).expect("stream");
    let kernel = RmsNormKernel::load(device).expect("load kernel");

    let mut d_input = DeviceBuffer::<f32>::alloc(device, n).expect("alloc input");
    let mut d_weight =
        DeviceBuffer::<f32>::alloc(device, hidden_size as usize).expect("alloc weight");
    let mut d_output = DeviceBuffer::<f32>::alloc(device, n).expect("alloc output");

    d_input.copy_from_host(&input_data).expect("copy input");
    d_weight.copy_from_host(&weight_data).expect("copy weight");

    kernel
        .forward(&mut d_output, &d_input, &d_weight, num_rows, hidden_size, eps, &stream)
        .expect("kernel launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; n];
    d_output.copy_to_host(&mut result).expect("copy output");

    // Compare with tolerance
    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-4,
        "RMSNorm max error {max_err} exceeds tolerance"
    );
    println!("RMSNorm test passed: max error = {max_err:.2e}");
}
