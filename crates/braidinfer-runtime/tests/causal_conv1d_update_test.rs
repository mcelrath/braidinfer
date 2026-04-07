use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::CausalConv1dUpdateKernel;

fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn causal_conv1d_reference(
    state: &mut Vec<f32>,
    input: &[f32],
    weight: &[u16],
    conv_dim: usize,
    kernel_size: usize,
) -> Vec<f32> {
    let hist = kernel_size - 1;
    let mut output = vec![0.0f32; conv_dim];
    for ch in 0..conv_dim {
        let s = &mut state[ch * hist..(ch + 1) * hist];
        let w = &weight[ch * kernel_size..(ch + 1) * kernel_size];
        let mut acc = 0.0f32;
        for k in 0..hist {
            acc += s[k] * bf16_to_f32(w[k]);
        }
        acc += input[ch] * bf16_to_f32(w[hist]);
        output[ch] = silu(acc);
        for i in 0..hist - 1 {
            s[i] = s[i + 1];
        }
        s[hist - 1] = input[ch];
    }
    output
}

#[test]
fn test_causal_conv1d_update_matches_reference() {
    let device = DeviceId(0);
    let conv_dim: usize = 64;
    let kernel_size: usize = 4;
    let hist = kernel_size - 1;

    let state_data: Vec<f32> = (0..conv_dim * hist)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();
    let input_data: Vec<f32> = (0..conv_dim)
        .map(|i| (i as f32 * 0.03 + 1.0).cos())
        .collect();
    let weight_data_f32: Vec<f32> = (0..conv_dim * kernel_size)
        .map(|i| (i as f32 * 0.007).sin() * 0.3 + 0.1)
        .collect();
    let weight_data: Vec<u16> = weight_data_f32.iter().copied().map(f32_to_bf16).collect();

    let mut cpu_state = state_data.clone();
    let expected = causal_conv1d_reference(
        &mut cpu_state,
        &input_data,
        &weight_data,
        conv_dim,
        kernel_size,
    );

    let stream = Stream::new(device).expect("stream");
    let kernel = CausalConv1dUpdateKernel::load(device).expect("load kernel");

    let mut d_state = DeviceBuffer::<f32>::alloc(device, conv_dim * hist).expect("alloc state");
    let mut d_input = DeviceBuffer::<f32>::alloc(device, conv_dim).expect("alloc input");
    let mut d_weight =
        DeviceBuffer::<u16>::alloc(device, conv_dim * kernel_size).expect("alloc weight");
    let mut d_output = DeviceBuffer::<f32>::alloc(device, conv_dim).expect("alloc output");

    d_state.copy_from_host(&state_data).expect("copy state");
    d_input.copy_from_host(&input_data).expect("copy input");
    d_weight.copy_from_host(&weight_data).expect("copy weight");

    kernel
        .forward(
            &mut d_state,
            &d_input,
            &d_weight,
            &mut d_output,
            conv_dim as u32,
            kernel_size as u32,
            &stream,
        )
        .expect("kernel launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; conv_dim];
    d_output.copy_to_host(&mut result).expect("copy output");

    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-5,
        "causal_conv1d_update max error {max_err} exceeds tolerance"
    );
    println!("causal_conv1d_update test passed: max error = {max_err:.2e}");
}
