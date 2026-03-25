use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::GqaAttentionKernel;

// Qwen3.5-0.8B GQA parameters (from config.json text_config)
const NUM_Q_HEADS: u32 = 8;
const NUM_KV_HEADS: u32 = 2;
const HEAD_DIM: u32 = 256;

fn gqa_attention_reference(
    q: &[f32],          // [num_q_heads, head_dim]
    k_cache: &[f32],   // [num_kv_heads, max_seq_len, head_dim]
    v_cache: &[f32],   // [num_kv_heads, max_seq_len, head_dim]
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    max_seq_len: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let gqa_group = num_q_heads / num_kv_heads;
    let mut output = vec![0.0f32; num_q_heads * head_dim];

    for h in 0..num_q_heads {
        let kv_h = h / gqa_group;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];

        let scores: Vec<f32> = (0..seq_len)
            .map(|t| {
                let k_ptr = &k_cache[(kv_h * max_seq_len + t) * head_dim
                    ..(kv_h * max_seq_len + t + 1) * head_dim];
                let dot: f32 = q_head.iter().zip(k_ptr.iter()).map(|(a, b)| a * b).sum();
                dot * scale
            })
            .collect();

        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_s: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f32 = exp_s.iter().sum();
        let attn: Vec<f32> = exp_s.iter().map(|e| e / sum_exp).collect();

        let out = &mut output[h * head_dim..(h + 1) * head_dim];
        for t in 0..seq_len {
            let v_ptr = &v_cache[(kv_h * max_seq_len + t) * head_dim
                ..(kv_h * max_seq_len + t + 1) * head_dim];
            for i in 0..head_dim {
                out[i] += attn[t] * v_ptr[i];
            }
        }
    }
    output
}

#[test]
fn test_gqa_attention_matches_reference() {
    let device = DeviceId(0);
    let nqh = NUM_Q_HEADS as usize;
    let nkh = NUM_KV_HEADS as usize;
    let hd = HEAD_DIM as usize;
    let seq_len: usize = 32;
    let max_seq_len: usize = 128;

    // Generate test data
    let q_data: Vec<f32> = (0..nqh * hd)
        .map(|i| (i as f32 * 0.011).sin() * 0.5)
        .collect();
    let k_cache_data: Vec<f32> = (0..max_seq_len * nkh * hd)
        .map(|i| (i as f32 * 0.007).cos() * 0.3)
        .collect();
    let v_cache_data: Vec<f32> = (0..max_seq_len * nkh * hd)
        .map(|i| (i as f32 * 0.009).sin() * 0.4)
        .collect();

    // CPU reference ([H,T,D] layout — full buffer, reference reads only first seq_len per head)
    let expected = gqa_attention_reference(
        &q_data,
        &k_cache_data,
        &v_cache_data,
        nqh,
        nkh,
        hd,
        seq_len,
        max_seq_len,
    );

    // GPU computation
    let stream = Stream::new(device).expect("stream");
    let kernel = GqaAttentionKernel::load(device).expect("load gqa_attention kernel");

    let mut d_output = DeviceBuffer::<f32>::alloc(device, nqh * hd).expect("alloc output");
    let mut d_q = DeviceBuffer::<f32>::alloc(device, nqh * hd).expect("alloc q");
    let mut d_k = DeviceBuffer::<f32>::alloc(device, max_seq_len * nkh * hd).expect("alloc k");
    let mut d_v = DeviceBuffer::<f32>::alloc(device, max_seq_len * nkh * hd).expect("alloc v");

    d_q.copy_from_host(&q_data).expect("copy q");
    d_k.copy_from_host(&k_cache_data).expect("copy k");
    d_v.copy_from_host(&v_cache_data).expect("copy v");

    kernel
        .forward(
            &mut d_output,
            &d_q,
            &d_k,
            &d_v,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            seq_len as u32,
            max_seq_len as u32,
            &stream,
        )
        .expect("gqa_attention launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; nqh * hd];
    d_output.copy_to_host(&mut result).expect("copy output");

    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "GQA attention max error {max_err} exceeds tolerance"
    );
    println!("GQA attention test passed: max error = {max_err:.2e}");
}
