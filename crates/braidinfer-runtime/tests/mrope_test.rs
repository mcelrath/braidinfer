use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::MRoPEKernel;

// Qwen3.5-0.8B mRoPE parameters (from config.json text_config)
const NUM_Q_HEADS: u32 = 8;
const NUM_KV_HEADS: u32 = 2;
const HEAD_DIM: u32 = 256;    // config.json text_config.head_dim = 256
const ROPE_DIM: u32 = 64;     // head_dim(256) * partial_rotary_factor(0.25) = 64
const SECTION0_PAIRS: u32 = 11;  // temporal
const SECTION1_PAIRS: u32 = 11;  // height
const SECTION2_PAIRS: u32 = 10;  // width — total 32 pairs = rope_dim/2
const ROPE_THETA: f32 = 10_000_000.0;  // rope_theta from config

fn compute_inv_freq() -> Vec<f32> {
    let num_pairs = (ROPE_DIM / 2) as usize;
    (0..num_pairs)
        .map(|i| {
            let exp = 2.0 * i as f32 / ROPE_DIM as f32;
            1.0 / ROPE_THETA.powf(exp)
        })
        .collect()
}

fn mrope_reference(
    data: &mut [f32],  // [num_heads, head_dim]
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

#[test]
fn test_mrope_matches_reference() {
    let device = DeviceId(0);
    let nqh = NUM_Q_HEADS as usize;
    let nkh = NUM_KV_HEADS as usize;
    let hd = HEAD_DIM as usize;

    let inv_freq = compute_inv_freq();
    let position_ids: [i32; 3] = [42, 7, 13]; // use different pos_ids to test all 3 sections

    // Random-ish Q and K data
    let q_data: Vec<f32> = (0..nqh * hd).map(|i| ((i as f32 * 0.017).sin())).collect();
    let k_data: Vec<f32> = (0..nkh * hd).map(|i| ((i as f32 * 0.013).cos())).collect();

    // CPU reference
    let mut q_ref = q_data.clone();
    let mut k_ref = k_data.clone();
    mrope_reference(
        &mut q_ref,
        nqh,
        hd,
        ROPE_DIM as usize,
        SECTION0_PAIRS as usize,
        SECTION1_PAIRS as usize,
        &inv_freq,
        &position_ids,
    );
    mrope_reference(
        &mut k_ref,
        nkh,
        hd,
        ROPE_DIM as usize,
        SECTION0_PAIRS as usize,
        SECTION1_PAIRS as usize,
        &inv_freq,
        &position_ids,
    );

    // GPU computation
    let stream = Stream::new(device).expect("stream");
    let kernel = MRoPEKernel::load(device).expect("load mrope kernel");

    let mut d_q = DeviceBuffer::<f32>::alloc(device, nqh * hd).expect("alloc q");
    let mut d_k = DeviceBuffer::<f32>::alloc(device, nkh * hd).expect("alloc k");
    let mut d_inv_freq = DeviceBuffer::<f32>::alloc(device, inv_freq.len()).expect("alloc inv_freq");
    let mut d_pos = DeviceBuffer::<i32>::alloc(device, 3).expect("alloc pos");

    d_q.copy_from_host(&q_data).expect("copy q");
    d_k.copy_from_host(&k_data).expect("copy k");
    d_inv_freq.copy_from_host(&inv_freq).expect("copy inv_freq");
    d_pos.copy_from_host(&position_ids).expect("copy pos");

    kernel
        .forward(
            &mut d_q,
            &mut d_k,
            &d_inv_freq,
            &d_pos,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            ROPE_DIM,
            SECTION0_PAIRS,
            SECTION1_PAIRS,
            SECTION2_PAIRS,
            &stream,
        )
        .expect("mrope launch");

    stream.synchronize().expect("sync");

    let mut q_result = vec![0.0f32; nqh * hd];
    let mut k_result = vec![0.0f32; nkh * hd];
    d_q.copy_to_host(&mut q_result).expect("copy q back");
    d_k.copy_to_host(&mut k_result).expect("copy k back");

    let max_err_q: f32 = q_result
        .iter()
        .zip(q_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let max_err_k: f32 = k_result
        .iter()
        .zip(k_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err_q < 1e-4,
        "mRoPE Q max error {max_err_q} exceeds tolerance"
    );
    assert!(
        max_err_k < 1e-4,
        "mRoPE K max error {max_err_k} exceeds tolerance"
    );
    println!("mRoPE test passed: Q max error = {max_err_q:.2e}, K max error = {max_err_k:.2e}");
}
