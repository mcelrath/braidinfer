use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::GdnRecurrentStepV2Kernel;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let inv = 1.0 / (norm + 1e-8f32.sqrt());
    v.iter().map(|x| x * inv).collect()
}

fn gdn_step_v2_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    b: &[f32],
    state: &mut [f32],
    num_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; num_heads * value_dim];
    for h in 0..num_heads {
        let decay = gate[h];
        let beta = sigmoid(b[h]);
        let qh = l2_normalize(&q[h * key_dim..(h + 1) * key_dim]);
        let kh = l2_normalize(&k[h * key_dim..(h + 1) * key_dim]);
        let q_scale = 1.0 / (key_dim as f32).sqrt();
        let vh = &v[h * value_dim..(h + 1) * value_dim];
        let s = &mut state[h * key_dim * value_dim..(h + 1) * key_dim * value_dim];

        for idx in 0..key_dim * value_dim {
            s[idx] *= decay;
        }

        for j in 0..value_dim {
            let mut kv_mem = 0.0f32;
            for i in 0..key_dim {
                kv_mem += s[j * key_dim + i] * kh[i];
            }
            let delta = (vh[j] - kv_mem) * beta;
            for i in 0..key_dim {
                s[j * key_dim + i] += kh[i] * delta;
            }
        }

        for j in 0..value_dim {
            let mut sum = 0.0f32;
            for i in 0..key_dim {
                sum += s[j * key_dim + i] * qh[i] * q_scale;
            }
            output[h * value_dim + j] = sum;
        }
    }
    output
}

#[test]
fn test_gdn_recurrent_step_v2_matches_reference() {
    let device = DeviceId(0);
    let num_heads: usize = 8;
    let key_dim: usize = 64;
    let value_dim: usize = 64;

    let q_data: Vec<f32> = (0..num_heads * key_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let k_data: Vec<f32> = (0..num_heads * key_dim)
        .map(|i| (i as f32 * 0.007 + 0.5).cos())
        .collect();
    let v_data: Vec<f32> = (0..num_heads * value_dim)
        .map(|i| (i as f32 * 0.013).sin())
        .collect();
    // gate values pre-computed, all in (0, 1)
    let gate_data: Vec<f32> = (0..num_heads).map(|i| 0.1 + (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..num_heads).map(|i| (i as f32 * 0.05) - 0.4).collect();
    let state_data: Vec<f32> = (0..num_heads * key_dim * value_dim)
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();

    let mut cpu_state = state_data.clone();
    let expected = gdn_step_v2_reference(
        &q_data,
        &k_data,
        &v_data,
        &gate_data,
        &b_data,
        &mut cpu_state,
        num_heads,
        key_dim,
        value_dim,
    );

    let stream = Stream::new(device).expect("stream");
    let kernel = GdnRecurrentStepV2Kernel::load(device).expect("load kernel");

    let mut d_q = DeviceBuffer::<f32>::alloc(device, num_heads * key_dim).expect("alloc q");
    let mut d_k = DeviceBuffer::<f32>::alloc(device, num_heads * key_dim).expect("alloc k");
    let mut d_v = DeviceBuffer::<f32>::alloc(device, num_heads * value_dim).expect("alloc v");
    let mut d_gate = DeviceBuffer::<f32>::alloc(device, num_heads).expect("alloc gate");
    let mut d_b = DeviceBuffer::<f32>::alloc(device, num_heads).expect("alloc b");
    let mut d_state =
        DeviceBuffer::<f32>::alloc(device, num_heads * key_dim * value_dim).expect("alloc state");
    let mut d_output =
        DeviceBuffer::<f32>::alloc(device, num_heads * value_dim).expect("alloc output");

    d_q.copy_from_host(&q_data).expect("copy q");
    d_k.copy_from_host(&k_data).expect("copy k");
    d_v.copy_from_host(&v_data).expect("copy v");
    d_gate.copy_from_host(&gate_data).expect("copy gate");
    d_b.copy_from_host(&b_data).expect("copy b");
    d_state.copy_from_host(&state_data).expect("copy state");

    kernel
        .forward(
            &d_q,
            &d_k,
            &d_v,
            &d_gate,
            &d_b,
            &mut d_state,
            &mut d_output,
            num_heads as u32,
            key_dim as u32,
            value_dim as u32,
            1,
            &stream,
        )
        .expect("kernel launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; num_heads * value_dim];
    d_output.copy_to_host(&mut result).expect("copy output");

    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "gdn_recurrent_step_v2 max error {max_err} exceeds tolerance"
    );
    println!("gdn_recurrent_step_v2 test passed: max error = {max_err:.2e}");
}
