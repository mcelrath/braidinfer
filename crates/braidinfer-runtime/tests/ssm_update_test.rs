use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;
use braidinfer_runtime::kernel::SelectiveStateUpdateKernel;

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

fn reference_ssm_update(
    ssm_state: &mut Vec<f32>,
    x: &[f32],
    dt: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    output: &mut Vec<f32>,
    num_heads: usize,
    head_dim: usize,
    state_size: usize,
    n_groups: usize,
) {
    let heads_per_group = num_heads / n_groups;
    for h in 0..num_heads {
        let g = h / heads_per_group;
        let dt_val = softplus(dt[h] + dt_bias[h]);
        let a = -a_log[h].exp();
        let da = (dt_val * a).exp();

        for dd in 0..head_dim {
            for s in 0..state_size {
                let idx = h * head_dim * state_size + dd * state_size + s;
                ssm_state[idx] =
                    da * ssm_state[idx] + dt_val * x[h * head_dim + dd] * b[g * state_size + s];
            }
        }

        for dd in 0..head_dim {
            let mut sum = 0.0f32;
            for s in 0..state_size {
                let idx = h * head_dim * state_size + dd * state_size + s;
                sum += ssm_state[idx] * c[g * state_size + s];
            }
            output[h * head_dim + dd] = sum + d[h] * x[h * head_dim + dd];
        }
    }
}

#[test]
fn test_selective_state_update_small() {
    let device = DeviceId(0);
    let stream = Stream::new(device).unwrap();
    let kernel = SelectiveStateUpdateKernel::load(device).unwrap();

    let num_heads: u32 = 4;
    let head_dim: u32 = 8;
    let state_size: u32 = 16;
    let n_groups: u32 = 2;

    let nh = num_heads as usize;
    let hd = head_dim as usize;
    let ss = state_size as usize;
    let ng = n_groups as usize;

    let x: Vec<f32> = (0..nh * hd)
        .map(|i| ((i as f32) * 0.1 - 1.6).sin())
        .collect();
    let dt: Vec<f32> = (0..nh).map(|i| 0.5 + 0.1 * i as f32).collect();
    let dt_bias: Vec<f32> = vec![0.1; nh];
    let a_log: Vec<f32> = (0..nh).map(|i| (1.0 + i as f32).ln()).collect();
    let b_data: Vec<f32> = (0..ng * ss).map(|i| ((i as f32) * 0.2).cos()).collect();
    let c_data: Vec<f32> = (0..ng * ss).map(|i| ((i as f32) * 0.3).sin()).collect();
    let d_data: Vec<f32> = vec![1.0; nh];
    let mut state_cpu: Vec<f32> = vec![0.0; nh * hd * ss];
    let mut output_cpu: Vec<f32> = vec![0.0; nh * hd];

    reference_ssm_update(
        &mut state_cpu,
        &x,
        &dt,
        &dt_bias,
        &a_log,
        &b_data,
        &c_data,
        &d_data,
        &mut output_cpu,
        nh,
        hd,
        ss,
        ng,
    );

    let mut state_dev = DeviceBuffer::<f32>::alloc(device, nh * hd * ss).unwrap();
    state_dev
        .copy_from_host(&vec![0.0f32; nh * hd * ss])
        .unwrap();
    let mut x_dev = DeviceBuffer::<f32>::alloc(device, nh * hd).unwrap();
    x_dev.copy_from_host(&x).unwrap();
    let mut dt_dev = DeviceBuffer::<f32>::alloc(device, nh).unwrap();
    dt_dev.copy_from_host(&dt).unwrap();
    let mut dt_bias_dev = DeviceBuffer::<f32>::alloc(device, nh).unwrap();
    dt_bias_dev.copy_from_host(&dt_bias).unwrap();
    let mut a_log_dev = DeviceBuffer::<f32>::alloc(device, nh).unwrap();
    a_log_dev.copy_from_host(&a_log).unwrap();
    let mut b_dev = DeviceBuffer::<f32>::alloc(device, ng * ss).unwrap();
    b_dev.copy_from_host(&b_data).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::alloc(device, ng * ss).unwrap();
    c_dev.copy_from_host(&c_data).unwrap();
    let mut d_dev = DeviceBuffer::<f32>::alloc(device, nh).unwrap();
    d_dev.copy_from_host(&d_data).unwrap();
    let mut output_dev = DeviceBuffer::<f32>::alloc(device, nh * hd).unwrap();

    kernel
        .forward(
            &mut state_dev,
            &x_dev,
            &dt_dev,
            &dt_bias_dev,
            &a_log_dev,
            &b_dev,
            &c_dev,
            &d_dev,
            &mut output_dev,
            num_heads,
            head_dim,
            state_size,
            n_groups,
            &stream,
        )
        .unwrap();
    stream.synchronize().unwrap();

    let mut output_gpu = vec![0.0f32; nh * hd];
    output_dev.copy_to_host(&mut output_gpu).unwrap();

    let max_abs_diff = output_cpu
        .iter()
        .zip(&output_gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("CPU first 8: {:?}", &output_cpu[..8]);
    println!("GPU first 8: {:?}", &output_gpu[..8]);
    println!("Max abs diff: {max_abs_diff}");
    assert!(
        max_abs_diff < 1e-4,
        "output mismatch: max_abs_diff={max_abs_diff}"
    );

    let mut state_gpu = vec![0.0f32; nh * hd * ss];
    state_dev.copy_to_host(&mut state_gpu).unwrap();
    let state_diff = state_cpu
        .iter()
        .zip(&state_gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("State max abs diff: {state_diff}");
    assert!(state_diff < 1e-4, "state mismatch: max_diff={state_diff}");
}

#[test]
fn test_selective_state_update_nemotron_dims() {
    let device = DeviceId(0);
    let stream = Stream::new(device).unwrap();
    let kernel = SelectiveStateUpdateKernel::load(device).unwrap();

    let num_heads: u32 = 64;
    let head_dim: u32 = 64;
    let state_size: u32 = 128;
    let n_groups: u32 = 8;

    let nh = num_heads as usize;
    let hd = head_dim as usize;
    let ss = state_size as usize;
    let ng = n_groups as usize;

    let x: Vec<f32> = (0..nh * hd).map(|i| ((i as f32) * 0.01).sin()).collect();
    let dt: Vec<f32> = (0..nh).map(|i| 0.5 + 0.01 * i as f32).collect();
    let dt_bias: Vec<f32> = vec![0.1; nh];
    let a_log: Vec<f32> = (0..nh).map(|i| (1.0 + i as f32).ln()).collect();
    let b_data: Vec<f32> = (0..ng * ss).map(|i| ((i as f32) * 0.01).cos()).collect();
    let c_data: Vec<f32> = (0..ng * ss).map(|i| ((i as f32) * 0.01).sin()).collect();
    let d_data: Vec<f32> = vec![1.0; nh];
    let mut state_cpu: Vec<f32> = vec![0.0; nh * hd * ss];
    let mut output_cpu: Vec<f32> = vec![0.0; nh * hd];

    reference_ssm_update(
        &mut state_cpu,
        &x,
        &dt,
        &dt_bias,
        &a_log,
        &b_data,
        &c_data,
        &d_data,
        &mut output_cpu,
        nh,
        hd,
        ss,
        ng,
    );

    let mut state_dev = DeviceBuffer::<f32>::alloc(device, nh * hd * ss).unwrap();
    state_dev
        .copy_from_host(&vec![0.0f32; nh * hd * ss])
        .unwrap();
    let mut x_dev = DeviceBuffer::<f32>::alloc(device, nh * hd).unwrap();
    x_dev.copy_from_host(&x).unwrap();
    let mut dt_dev = DeviceBuffer::<f32>::alloc(device, nh).unwrap();
    dt_dev.copy_from_host(&dt).unwrap();
    let mut dt_bias_dev = DeviceBuffer::<f32>::alloc(device, nh).unwrap();
    dt_bias_dev.copy_from_host(&dt_bias).unwrap();
    let mut a_log_dev = DeviceBuffer::<f32>::alloc(device, nh).unwrap();
    a_log_dev.copy_from_host(&a_log).unwrap();
    let mut b_dev = DeviceBuffer::<f32>::alloc(device, ng * ss).unwrap();
    b_dev.copy_from_host(&b_data).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::alloc(device, ng * ss).unwrap();
    c_dev.copy_from_host(&c_data).unwrap();
    let mut d_dev = DeviceBuffer::<f32>::alloc(device, nh).unwrap();
    d_dev.copy_from_host(&d_data).unwrap();
    let mut output_dev = DeviceBuffer::<f32>::alloc(device, nh * hd).unwrap();

    kernel
        .forward(
            &mut state_dev,
            &x_dev,
            &dt_dev,
            &dt_bias_dev,
            &a_log_dev,
            &b_dev,
            &c_dev,
            &d_dev,
            &mut output_dev,
            num_heads,
            head_dim,
            state_size,
            n_groups,
            &stream,
        )
        .unwrap();
    stream.synchronize().unwrap();

    let mut output_gpu = vec![0.0f32; nh * hd];
    output_dev.copy_to_host(&mut output_gpu).unwrap();

    let max_abs_diff = output_cpu
        .iter()
        .zip(&output_gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Nemotron dims ({nh} heads, {hd} head_dim, {ss} state_size, {ng} groups)");
    println!("Max abs diff: {max_abs_diff}");
    assert!(
        max_abs_diff < 1e-3,
        "output mismatch at Nemotron dims: max_abs_diff={max_abs_diff}"
    );
}
