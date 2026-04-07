use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::LinearProjKernel;

fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn linear_proj_reference(weight: &[u16], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut acc = 0.0f32;
        for j in 0..in_dim {
            acc += bf16_to_f32(weight[i * in_dim + j]) * input[j];
        }
        output[i] = acc;
    }
    output
}

#[test]
fn test_linear_proj_matches_reference() {
    let device = DeviceId(0);
    let in_dim = 1024usize;
    let out_dim = 3584usize; // intermediate_size from Qwen3.5-0.8B config

    let input_data: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let weight_data_f32: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32 * 0.0001).cos() * 0.1)
        .collect();
    let weight_data: Vec<u16> = weight_data_f32.iter().copied().map(f32_to_bf16).collect();

    let expected = linear_proj_reference(&weight_data, &input_data, out_dim, in_dim);

    let stream = Stream::new(device).expect("stream");
    let kernel = LinearProjKernel::load(device).expect("load kernel");

    let mut d_input = DeviceBuffer::<f32>::alloc(device, in_dim).expect("alloc input");
    let mut d_weight = DeviceBuffer::<u16>::alloc(device, out_dim * in_dim).expect("alloc weight");
    let mut d_output = DeviceBuffer::<f32>::alloc(device, out_dim).expect("alloc output");

    d_input.copy_from_host(&input_data).expect("copy input");
    d_weight.copy_from_host(&weight_data).expect("copy weight");

    kernel
        .forward(
            &mut d_output,
            &d_weight,
            &d_input,
            out_dim as u32,
            in_dim as u32,
            &stream,
        )
        .expect("kernel launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; out_dim];
    d_output.copy_to_host(&mut result).expect("copy output");

    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "LinearProj max error {max_err} exceeds tolerance"
    );
    println!("LinearProj test passed: max error = {max_err:.2e}");
}
