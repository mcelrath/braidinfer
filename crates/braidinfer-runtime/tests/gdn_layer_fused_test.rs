use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::GdnLayerFusedKernel;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn rmsnorm_ref(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = weight.len();
    let sum_sq: f32 = input.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt().recip();
    input.iter().zip(weight).map(|(x, w)| x * rms * w).collect()
}

fn linear_ref(weight: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|i| {
            weight[i * in_dim..(i + 1) * in_dim]
                .iter()
                .zip(input)
                .map(|(w, x)| w * x)
                .sum()
        })
        .collect()
}

fn gdn_step_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    b: &[f32],
    state: &mut [f32],
    num_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; num_heads * value_dim];
    for h in 0..num_heads {
        let gate = sigmoid(g[h]);
        let beta = sigmoid(b[h]);
        let s = &mut state[h * key_dim * value_dim..(h + 1) * key_dim * value_dim];
        for i in 0..key_dim {
            for j in 0..value_dim {
                s[i * value_dim + j] =
                    gate * s[i * value_dim + j] + beta * v[h * value_dim + j] * k[h * key_dim + i];
            }
        }
        for j in 0..value_dim {
            let mut sum = 0.0f32;
            for i in 0..key_dim {
                sum += s[i * value_dim + j] * q[h * key_dim + i];
            }
            output[h * value_dim + j] = sum;
        }
    }
    output
}

#[test]
fn test_gdn_layer_fused_matches_unfused() {
    let device = DeviceId(0);
    let hidden_size: usize = 1024;
    let num_heads: usize = 16;
    let key_dim: usize = 128;
    let value_dim: usize = 128;
    let eps = 1e-6f32;

    let nk = num_heads * key_dim;
    let nv = num_heads * value_dim;
    let nh = num_heads;

    // --- Generate random-ish weights and inputs ---
    let input_data: Vec<f32> = (0..hidden_size)
        .map(|i| (i as f32 * 0.0031).sin() * 0.5)
        .collect();
    let rms_weight: Vec<f32> = (0..hidden_size).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let w_q: Vec<f32> = (0..nk * hidden_size)
        .map(|i| (i as f32 * 0.000013).sin() * 0.02)
        .collect();
    let w_k: Vec<f32> = (0..nk * hidden_size)
        .map(|i| (i as f32 * 0.000017 + 1.0).cos() * 0.02)
        .collect();
    let w_v: Vec<f32> = (0..nv * hidden_size)
        .map(|i| (i as f32 * 0.000011 + 2.0).sin() * 0.02)
        .collect();
    let w_g: Vec<f32> = (0..nh * hidden_size)
        .map(|i| (i as f32 * 0.000019 + 3.0).cos() * 0.02)
        .collect();
    let w_b: Vec<f32> = (0..nh * hidden_size)
        .map(|i| (i as f32 * 0.000023 + 4.0).sin() * 0.02)
        .collect();
    let w_o: Vec<f32> = (0..hidden_size * nv)
        .map(|i| (i as f32 * 0.000007 + 5.0).cos() * 0.02)
        .collect();
    let state_init: Vec<f32> = (0..num_heads * key_dim * value_dim)
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();

    // --- CPU reference: unfused sequence ---
    let normed = rmsnorm_ref(&input_data, &rms_weight, eps);
    let q_ref = linear_ref(&w_q, &normed, nk, hidden_size);
    let k_ref = linear_ref(&w_k, &normed, nk, hidden_size);
    let v_ref = linear_ref(&w_v, &normed, nv, hidden_size);
    let g_ref = linear_ref(&w_g, &normed, nh, hidden_size);
    let b_ref = linear_ref(&w_b, &normed, nh, hidden_size);
    let mut cpu_state = state_init.clone();
    let rec_out = gdn_step_ref(
        &q_ref,
        &k_ref,
        &v_ref,
        &g_ref,
        &b_ref,
        &mut cpu_state,
        num_heads,
        key_dim,
        value_dim,
    );
    let proj_out = linear_ref(&w_o, &rec_out, hidden_size, nv);
    let expected: Vec<f32> = proj_out
        .iter()
        .zip(&input_data)
        .map(|(p, r)| p + r)
        .collect();

    // --- GPU: fused kernel ---
    let stream = Stream::new(device).expect("stream");
    let kernel = GdnLayerFusedKernel::load(device).expect("load fused kernel");

    let scratch_size = 2 * nk + nv + 2 * nh + nv;

    let mut d_output = DeviceBuffer::<f32>::alloc(device, hidden_size).expect("alloc output");
    let mut d_scratch = DeviceBuffer::<f32>::alloc(device, scratch_size).expect("alloc scratch");
    let mut d_input = DeviceBuffer::<f32>::alloc(device, hidden_size).expect("alloc input");
    let mut d_rms_w = DeviceBuffer::<f32>::alloc(device, hidden_size).expect("alloc rms_w");
    let mut d_wq = DeviceBuffer::<f32>::alloc(device, nk * hidden_size).expect("alloc w_q");
    let mut d_wk = DeviceBuffer::<f32>::alloc(device, nk * hidden_size).expect("alloc w_k");
    let mut d_wv = DeviceBuffer::<f32>::alloc(device, nv * hidden_size).expect("alloc w_v");
    let mut d_wg = DeviceBuffer::<f32>::alloc(device, nh * hidden_size).expect("alloc w_g");
    let mut d_wb = DeviceBuffer::<f32>::alloc(device, nh * hidden_size).expect("alloc w_b");
    let mut d_wo = DeviceBuffer::<f32>::alloc(device, hidden_size * nv).expect("alloc w_o");
    let mut d_state =
        DeviceBuffer::<f32>::alloc(device, num_heads * key_dim * value_dim).expect("alloc state");

    d_input.copy_from_host(&input_data).expect("copy input");
    d_rms_w
        .copy_from_host(&rms_weight)
        .expect("copy rms_weight");
    d_wq.copy_from_host(&w_q).expect("copy w_q");
    d_wk.copy_from_host(&w_k).expect("copy w_k");
    d_wv.copy_from_host(&w_v).expect("copy w_v");
    d_wg.copy_from_host(&w_g).expect("copy w_g");
    d_wb.copy_from_host(&w_b).expect("copy w_b");
    d_wo.copy_from_host(&w_o).expect("copy w_o");
    d_state.copy_from_host(&state_init).expect("copy state");

    kernel
        .forward(
            &mut d_output,
            &mut d_scratch,
            &d_input,
            &d_rms_w,
            &d_wq,
            &d_wk,
            &d_wv,
            &d_wg,
            &d_wb,
            &d_wo,
            &mut d_state,
            hidden_size as u32,
            num_heads as u32,
            key_dim as u32,
            value_dim as u32,
            eps,
            &stream,
        )
        .expect("fused kernel launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; hidden_size];
    d_output.copy_to_host(&mut result).expect("copy output");

    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "GDN fused layer max error {max_err} exceeds tolerance"
    );
    println!("GDN fused layer test passed: max error = {max_err:.2e}");
}
