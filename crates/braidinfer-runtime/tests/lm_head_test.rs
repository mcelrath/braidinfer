use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::LmHeadKernel;

fn matmul_reference(weight: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut acc = 0.0f32;
        for j in 0..in_dim {
            acc += weight[i * in_dim + j] * input[j];
        }
        output[i] = acc;
    }
    output
}

#[test]
fn test_lm_head_matches_reference() {
    let device = DeviceId(0);
    let hidden_size = 1024usize;
    let vocab_size = 512usize;

    let input_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let weight_data: Vec<f32> = (0..vocab_size * hidden_size)
        .map(|i| (i as f32 * 0.0001).cos() * 0.1)
        .collect();

    let expected = matmul_reference(&weight_data, &input_data, vocab_size, hidden_size);

    let stream = Stream::new(device).expect("stream");
    let kernel = LmHeadKernel::load(device).expect("load kernel");

    let mut d_input = DeviceBuffer::<f32>::alloc(device, hidden_size).expect("alloc input");
    let mut d_weight = DeviceBuffer::<f32>::alloc(device, vocab_size * hidden_size).expect("alloc weight");
    let mut d_output = DeviceBuffer::<f32>::alloc(device, vocab_size).expect("alloc output");

    d_input.copy_from_host(&input_data).expect("copy input");
    d_weight.copy_from_host(&weight_data).expect("copy weight");

    kernel
        .forward(&mut d_output, &d_weight, &d_input, vocab_size as u32, hidden_size as u32, &stream)
        .expect("kernel launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; vocab_size];
    d_output.copy_to_host(&mut result).expect("copy output");

    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "LmHead max error {max_err} exceeds tolerance"
    );
    println!("LmHead test passed: max error = {max_err:.2e}");
}
