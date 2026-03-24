use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::RmsNormGatedKernel;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn rmsnorm_gated_reference(
    x: &[f32],
    z: &[f32],
    weight: &[f32],
    num_heads: usize,
    value_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; num_heads * value_dim];
    for h in 0..num_heads {
        let xh = &x[h * value_dim..(h + 1) * value_dim];
        let zh = &z[h * value_dim..(h + 1) * value_dim];
        let oh = &mut output[h * value_dim..(h + 1) * value_dim];
        let sum_sq: f32 = xh.iter().map(|v| v * v).sum();
        let rms = (sum_sq / value_dim as f32 + eps).sqrt().recip();
        for i in 0..value_dim {
            oh[i] = xh[i] * rms * weight[i] * silu(zh[i]);
        }
    }
    output
}

#[test]
fn test_rmsnorm_gated_matches_reference() {
    let device = DeviceId(0);
    let num_heads: usize = 16;
    let value_dim: usize = 128;
    let eps = 1e-6f32;

    let x_data: Vec<f32> = (0..num_heads * value_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let z_data: Vec<f32> = (0..num_heads * value_dim)
        .map(|i| (i as f32 * 0.007 + 0.3).cos())
        .collect();
    let weight_data: Vec<f32> = (0..value_dim).map(|i| 1.0 + i as f32 * 0.002).collect();

    let expected = rmsnorm_gated_reference(&x_data, &z_data, &weight_data, num_heads, value_dim, eps);

    let stream = Stream::new(device).expect("stream");
    let kernel = RmsNormGatedKernel::load(device).expect("load kernel");

    let mut d_output = DeviceBuffer::<f32>::alloc(device, num_heads * value_dim).expect("alloc output");
    let mut d_x = DeviceBuffer::<f32>::alloc(device, num_heads * value_dim).expect("alloc x");
    let mut d_z = DeviceBuffer::<f32>::alloc(device, num_heads * value_dim).expect("alloc z");
    let mut d_w = DeviceBuffer::<f32>::alloc(device, value_dim).expect("alloc w");

    d_x.copy_from_host(&x_data).expect("copy x");
    d_z.copy_from_host(&z_data).expect("copy z");
    d_w.copy_from_host(&weight_data).expect("copy w");

    kernel
        .forward(
            &mut d_output,
            &d_x,
            &d_z,
            &d_w,
            num_heads as u32,
            value_dim as u32,
            eps,
            &stream,
        )
        .expect("kernel launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; num_heads * value_dim];
    d_output.copy_to_host(&mut result).expect("copy output");

    let max_err: f32 = result.iter().zip(expected.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    assert!(max_err < 1e-4, "rmsnorm_gated max error {max_err} exceeds tolerance");
    println!("rmsnorm_gated test passed: max error = {max_err:.2e}");
}
