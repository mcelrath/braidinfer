use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::QkNormKernel;

fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn rmsnorm_per_head(
    heads: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) {
    for h in 0..num_heads {
        let x = &mut heads[h * head_dim..(h + 1) * head_dim];
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / head_dim as f32 + eps).sqrt().recip();
        for i in 0..head_dim {
            x[i] = x[i] * rms * (1.0 + weight[i]);
        }
    }
}

#[test]
fn test_qk_norm_matches_reference() {
    let device = DeviceId(0);
    let num_q_heads: usize = 8;
    let num_kv_heads: usize = 4;
    let head_dim: usize = 64;
    let eps = 1e-6f32;

    let q_data: Vec<f32> = (0..num_q_heads * head_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let k_data: Vec<f32> = (0..num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.013 + 0.5).cos())
        .collect();
    let q_weight_f32: Vec<f32> = (0..head_dim).map(|i| 1.0 + i as f32 * 0.001).collect();
    let k_weight_f32: Vec<f32> = (0..head_dim).map(|i| 0.9 + i as f32 * 0.002).collect();
    let q_weight: Vec<u16> = q_weight_f32.iter().copied().map(f32_to_bf16).collect();
    let k_weight: Vec<u16> = k_weight_f32.iter().copied().map(f32_to_bf16).collect();

    let mut expected_q = q_data.clone();
    let mut expected_k = k_data.clone();
    let q_weight_ref: Vec<f32> = q_weight.iter().copied().map(bf16_to_f32).collect();
    let k_weight_ref: Vec<f32> = k_weight.iter().copied().map(bf16_to_f32).collect();
    rmsnorm_per_head(&mut expected_q, num_q_heads, head_dim, &q_weight_ref, eps);
    rmsnorm_per_head(&mut expected_k, num_kv_heads, head_dim, &k_weight_ref, eps);

    let stream = Stream::new(device).expect("stream");
    let kernel = QkNormKernel::load(device).expect("load kernel");

    let mut d_q = DeviceBuffer::<f32>::alloc(device, num_q_heads * head_dim).expect("alloc q");
    let mut d_k = DeviceBuffer::<f32>::alloc(device, num_kv_heads * head_dim).expect("alloc k");
    let mut d_qw = DeviceBuffer::<u16>::alloc(device, head_dim).expect("alloc qw");
    let mut d_kw = DeviceBuffer::<u16>::alloc(device, head_dim).expect("alloc kw");

    d_q.copy_from_host(&q_data).expect("copy q");
    d_k.copy_from_host(&k_data).expect("copy k");
    d_qw.copy_from_host(&q_weight).expect("copy qw");
    d_kw.copy_from_host(&k_weight).expect("copy kw");

    kernel
        .forward(
            &mut d_q,
            &mut d_k,
            &d_qw,
            &d_kw,
            num_q_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            eps,
            &stream,
        )
        .expect("kernel launch");

    stream.synchronize().expect("sync");

    let mut result_q = vec![0.0f32; num_q_heads * head_dim];
    let mut result_k = vec![0.0f32; num_kv_heads * head_dim];
    d_q.copy_to_host(&mut result_q).expect("copy q");
    d_k.copy_to_host(&mut result_k).expect("copy k");

    let max_err_q: f32 = result_q
        .iter()
        .zip(expected_q.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_err_k: f32 = result_k
        .iter()
        .zip(expected_k.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err_q < 1e-4,
        "qk_norm Q max error {max_err_q} exceeds tolerance"
    );
    assert!(
        max_err_k < 1e-4,
        "qk_norm K max error {max_err_k} exceeds tolerance"
    );
    println!("qk_norm test passed: Q max err = {max_err_q:.2e}, K max err = {max_err_k:.2e}");
}
