use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::AttnLayerFusedKernel;

// Qwen3.5-0.8B attention config
const HIDDEN_SIZE: usize = 1024;
const NUM_Q_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const ROPE_DIM: usize = 64;
const SECTION0_PAIRS: usize = 11;
const SECTION1_PAIRS: usize = 11;
const SECTION2_PAIRS: usize = 10;
const ROPE_THETA: f32 = 10_000_000.0;
const EPS: f32 = 1e-6;

const Q_OUT_DIM: usize = NUM_Q_HEADS * HEAD_DIM; // 2048
const KV_OUT_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 512
const SCRATCH_SIZE: usize = Q_OUT_DIM + KV_OUT_DIM * 2 + Q_OUT_DIM; // 5120

fn compute_inv_freq() -> Vec<f32> {
    let num_pairs = ROPE_DIM / 2;
    (0..num_pairs)
        .map(|i| {
            let exp = 2.0 * i as f32 / ROPE_DIM as f32;
            1.0 / ROPE_THETA.powf(exp)
        })
        .collect()
}

fn rmsnorm_reference(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let sum_sq: f32 = input.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt().recip();
    input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| x * rms * (1.0 + w))
        .collect()
}

fn linear_proj_reference(weight: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
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

fn mrope_reference(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    section0_pairs: usize,
    section1_pairs: usize,
    inv_freq: &[f32],
    position_ids: &[i32; 3],
) {
    let total_pairs = rope_dim / 2;
    for h in 0..num_heads {
        let head = &mut data[h * head_dim..(h + 1) * head_dim];
        for pair in 0..total_pairs {
            let section = if pair < section0_pairs {
                0
            } else if pair < section0_pairs + section1_pairs {
                1
            } else {
                2
            };
            let pos = position_ids[section] as f32;
            let theta = pos * inv_freq[pair];
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let i0 = pair * 2;
            let i1 = pair * 2 + 1;
            let x0 = head[i0];
            let x1 = head[i1];
            head[i0] = x0 * cos_t - x1 * sin_t;
            head[i1] = x0 * sin_t + x1 * cos_t;
        }
    }
}

fn gqa_attention_reference(
    q: &[f32],
    k_cache: &[f32], // [num_kv_heads, max_seq_len, head_dim]
    v_cache: &[f32],
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
                let k_ptr = &k_cache
                    [(kv_h * max_seq_len + t) * head_dim..(kv_h * max_seq_len + t + 1) * head_dim];
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
            let v_ptr = &v_cache
                [(kv_h * max_seq_len + t) * head_dim..(kv_h * max_seq_len + t + 1) * head_dim];
            for i in 0..head_dim {
                out[i] += attn[t] * v_ptr[i];
            }
        }
    }
    output
}

/// Full unfused reference: RMSNorm → QKV proj → mRoPE → KV cache write → GQA attn → output proj → residual
fn attn_layer_reference(
    input: &[f32],
    rms_weight: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    inv_freq: &[f32],
    position_ids: &[i32; 3],
    k_cache: &mut Vec<f32>,
    v_cache: &mut Vec<f32>,
    seq_pos: usize,
    seq_len: usize,
) -> Vec<f32> {
    let normed = rmsnorm_reference(input, rms_weight, EPS);
    let mut q = linear_proj_reference(w_q, &normed, Q_OUT_DIM, HIDDEN_SIZE);
    let mut k = linear_proj_reference(w_k, &normed, KV_OUT_DIM, HIDDEN_SIZE);
    let v = linear_proj_reference(w_v, &normed, KV_OUT_DIM, HIDDEN_SIZE);

    mrope_reference(
        &mut q,
        NUM_Q_HEADS,
        HEAD_DIM,
        ROPE_DIM,
        SECTION0_PAIRS,
        SECTION1_PAIRS,
        inv_freq,
        position_ids,
    );
    mrope_reference(
        &mut k,
        NUM_KV_HEADS,
        HEAD_DIM,
        ROPE_DIM,
        SECTION0_PAIRS,
        SECTION1_PAIRS,
        inv_freq,
        position_ids,
    );

    // Write K,V to cache at seq_pos ([H,T,D] layout)
    let max_seq_len = k_cache.len() / (NUM_KV_HEADS * HEAD_DIM);
    for kv_head in 0..NUM_KV_HEADS {
        let dst = (kv_head * max_seq_len + seq_pos) * HEAD_DIM;
        let src = kv_head * HEAD_DIM;
        k_cache[dst..dst + HEAD_DIM].copy_from_slice(&k[src..src + HEAD_DIM]);
        v_cache[dst..dst + HEAD_DIM].copy_from_slice(&v[src..src + HEAD_DIM]);
    }

    let attn_out = gqa_attention_reference(
        &q,
        &k_cache,
        &v_cache,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        seq_len,
        max_seq_len,
    );

    let out_proj = linear_proj_reference(w_o, &attn_out, HIDDEN_SIZE, Q_OUT_DIM);
    out_proj
        .iter()
        .zip(input.iter())
        .map(|(o, r)| o + r)
        .collect()
}

#[test]
fn test_attn_layer_fused_matches_unfused() {
    let device = DeviceId(0);
    let max_seq_len: usize = 64;

    let inv_freq = compute_inv_freq();
    let position_ids: [i32; 3] = [5, 5, 5];

    // Initial KV cache populated with tokens 0..4 (seq_pos=4, then we write pos=4, seq_len=5)
    let seq_pos: usize = 4;
    let seq_len: usize = seq_pos + 1; // after writing new token

    // Deterministic test data
    let input: Vec<f32> = (0..HIDDEN_SIZE)
        .map(|i| ((i as f32 * 0.017).sin()) * 0.1)
        .collect();
    let rms_weight: Vec<f32> = (0..HIDDEN_SIZE)
        .map(|i| 1.0 + (i as f32 * 0.001).cos() * 0.1)
        .collect();
    let w_q: Vec<f32> = (0..Q_OUT_DIM * HIDDEN_SIZE)
        .map(|i| ((i as f32 * 0.00013).sin()) * 0.02)
        .collect();
    let w_k: Vec<f32> = (0..KV_OUT_DIM * HIDDEN_SIZE)
        .map(|i| ((i as f32 * 0.00017).cos()) * 0.02)
        .collect();
    let w_v: Vec<f32> = (0..KV_OUT_DIM * HIDDEN_SIZE)
        .map(|i| ((i as f32 * 0.00019).sin()) * 0.02)
        .collect();
    let w_o: Vec<f32> = (0..HIDDEN_SIZE * Q_OUT_DIM)
        .map(|i| ((i as f32 * 0.000011).cos()) * 0.02)
        .collect();

    // Pre-filled cache for positions 0..seq_pos
    let mut k_cache_ref: Vec<f32> = (0..max_seq_len * NUM_KV_HEADS * HEAD_DIM)
        .map(|i| ((i as f32 * 0.007).cos()) * 0.3)
        .collect();
    let mut v_cache_ref: Vec<f32> = (0..max_seq_len * NUM_KV_HEADS * HEAD_DIM)
        .map(|i| ((i as f32 * 0.009).sin()) * 0.4)
        .collect();

    // CPU reference
    let expected = attn_layer_reference(
        &input,
        &rms_weight,
        &w_q,
        &w_k,
        &w_v,
        &w_o,
        &inv_freq,
        &position_ids,
        &mut k_cache_ref,
        &mut v_cache_ref,
        seq_pos,
        seq_len,
    );

    // GPU fused
    let stream = Stream::new(device).expect("stream");
    let kernel = AttnLayerFusedKernel::load(device).expect("load attn_layer_fused kernel");

    // Re-initialize GPU caches from the same initial data (before reference wrote to pos seq_pos)
    let k_cache_init: Vec<f32> = (0..max_seq_len * NUM_KV_HEADS * HEAD_DIM)
        .map(|i| ((i as f32 * 0.007).cos()) * 0.3)
        .collect();
    let v_cache_init: Vec<f32> = (0..max_seq_len * NUM_KV_HEADS * HEAD_DIM)
        .map(|i| ((i as f32 * 0.009).sin()) * 0.4)
        .collect();

    let mut d_output = DeviceBuffer::<f32>::alloc(device, HIDDEN_SIZE).expect("alloc output");
    let mut d_scratch = DeviceBuffer::<f32>::alloc(device, SCRATCH_SIZE).expect("alloc scratch");
    let mut d_input = DeviceBuffer::<f32>::alloc(device, HIDDEN_SIZE).expect("alloc input");
    let mut d_rms_w = DeviceBuffer::<f32>::alloc(device, HIDDEN_SIZE).expect("alloc rms_weight");
    let mut d_wq = DeviceBuffer::<f32>::alloc(device, Q_OUT_DIM * HIDDEN_SIZE).expect("alloc w_q");
    let mut d_wk = DeviceBuffer::<f32>::alloc(device, KV_OUT_DIM * HIDDEN_SIZE).expect("alloc w_k");
    let mut d_wv = DeviceBuffer::<f32>::alloc(device, KV_OUT_DIM * HIDDEN_SIZE).expect("alloc w_v");
    let mut d_wo = DeviceBuffer::<f32>::alloc(device, HIDDEN_SIZE * Q_OUT_DIM).expect("alloc w_o");
    let mut d_inv_freq = DeviceBuffer::<f32>::alloc(device, ROPE_DIM / 2).expect("alloc inv_freq");
    let mut d_pos = DeviceBuffer::<i32>::alloc(device, 3).expect("alloc pos");
    let mut d_k_cache = DeviceBuffer::<f32>::alloc(device, max_seq_len * NUM_KV_HEADS * HEAD_DIM)
        .expect("alloc k_cache");
    let mut d_v_cache = DeviceBuffer::<f32>::alloc(device, max_seq_len * NUM_KV_HEADS * HEAD_DIM)
        .expect("alloc v_cache");

    d_input.copy_from_host(&input).expect("copy input");
    d_rms_w
        .copy_from_host(&rms_weight)
        .expect("copy rms_weight");
    d_wq.copy_from_host(&w_q).expect("copy w_q");
    d_wk.copy_from_host(&w_k).expect("copy w_k");
    d_wv.copy_from_host(&w_v).expect("copy w_v");
    d_wo.copy_from_host(&w_o).expect("copy w_o");
    d_inv_freq.copy_from_host(&inv_freq).expect("copy inv_freq");
    d_pos.copy_from_host(&position_ids).expect("copy pos");
    d_k_cache
        .copy_from_host(&k_cache_init)
        .expect("copy k_cache");
    d_v_cache
        .copy_from_host(&v_cache_init)
        .expect("copy v_cache");

    kernel
        .forward(
            &mut d_output,
            &mut d_scratch,
            &d_input,
            &d_rms_w,
            &d_wq,
            &d_wk,
            &d_wv,
            &d_wo,
            &d_inv_freq,
            d_pos.as_ptr(),
            &mut d_k_cache,
            &mut d_v_cache,
            HIDDEN_SIZE as u32,
            NUM_Q_HEADS as u32,
            NUM_KV_HEADS as u32,
            HEAD_DIM as u32,
            ROPE_DIM as u32,
            SECTION0_PAIRS as u32,
            SECTION1_PAIRS as u32,
            SECTION2_PAIRS as u32,
            seq_pos as u32,
            seq_len as u32,
            max_seq_len as u32,
            EPS,
            &stream,
        )
        .expect("attn_layer_fused launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; HIDDEN_SIZE];
    d_output.copy_to_host(&mut result).expect("copy output");

    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-2,
        "attn_layer_fused max error {max_err} exceeds tolerance"
    );
    println!("attn_layer_fused test passed: max error = {max_err:.2e}");
}
