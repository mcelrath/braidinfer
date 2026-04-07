use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::{
    FfnFusedKernel, LinearProjKernel, ResidualAddKernel, RmsNormKernel, SiluMulKernel,
};

const HIDDEN_SIZE: usize = 1024;
const INTERMEDIATE_SIZE: usize = 3584;
const EPS: f32 = 1e-6;

fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn rmsnorm_reference(input: &[f32], weight: &[u16], eps: f32) -> Vec<f32> {
    let n = weight.len();
    let sum_sq: f32 = input.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt().recip();
    input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| x * rms * (1.0 + bf16_to_f32(*w)))
        .collect()
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

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn ffn_reference(
    input: &[f32],
    rms_weight: &[u16],
    w_gate: &[u16],
    w_up: &[u16],
    w_down: &[u16],
) -> Vec<f32> {
    let normed = rmsnorm_reference(input, rms_weight, EPS);
    let gate = linear_proj_reference(w_gate, &normed, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    let up = linear_proj_reference(w_up, &normed, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    let act: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| silu(*g) * u)
        .collect();
    let down = linear_proj_reference(w_down, &act, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    down.iter().zip(input.iter()).map(|(d, r)| d + r).collect()
}

#[test]
fn test_ffn_fused_matches_unfused() {
    let device = DeviceId(0);

    let input_data: Vec<f32> = (0..HIDDEN_SIZE).map(|i| (i as f32 * 0.01).sin()).collect();
    let rms_weight_f32: Vec<f32> = (0..HIDDEN_SIZE).map(|i| i as f32 * 0.001).collect();
    let w_gate_f32: Vec<f32> = (0..INTERMEDIATE_SIZE * HIDDEN_SIZE)
        .map(|i| (i as f32 * 0.0001).cos() * 0.01)
        .collect();
    let w_up_f32: Vec<f32> = (0..INTERMEDIATE_SIZE * HIDDEN_SIZE)
        .map(|i| (i as f32 * 0.00013 + 1.0).sin() * 0.01)
        .collect();
    let w_down_f32: Vec<f32> = (0..HIDDEN_SIZE * INTERMEDIATE_SIZE)
        .map(|i| (i as f32 * 0.00007 + 0.5).cos() * 0.01)
        .collect();
    let rms_weight: Vec<u16> = rms_weight_f32.iter().copied().map(f32_to_bf16).collect();
    let w_gate: Vec<u16> = w_gate_f32.iter().copied().map(f32_to_bf16).collect();
    let w_up: Vec<u16> = w_up_f32.iter().copied().map(f32_to_bf16).collect();
    let w_down: Vec<u16> = w_down_f32.iter().copied().map(f32_to_bf16).collect();

    // CPU reference
    let expected = ffn_reference(&input_data, &rms_weight, &w_gate, &w_up, &w_down);

    let stream = Stream::new(device).expect("stream");

    // Allocate device buffers
    let mut d_input = DeviceBuffer::<f32>::alloc(device, HIDDEN_SIZE).expect("alloc input");
    let mut d_rms_w = DeviceBuffer::<u16>::alloc(device, HIDDEN_SIZE).expect("alloc rms_weight");
    let mut d_wgate =
        DeviceBuffer::<u16>::alloc(device, INTERMEDIATE_SIZE * HIDDEN_SIZE).expect("alloc w_gate");
    let mut d_wup =
        DeviceBuffer::<u16>::alloc(device, INTERMEDIATE_SIZE * HIDDEN_SIZE).expect("alloc w_up");
    let mut d_wdown =
        DeviceBuffer::<u16>::alloc(device, HIDDEN_SIZE * INTERMEDIATE_SIZE).expect("alloc w_down");
    let mut d_scratch =
        DeviceBuffer::<f32>::alloc(device, INTERMEDIATE_SIZE).expect("alloc scratch");
    let mut d_output = DeviceBuffer::<f32>::alloc(device, HIDDEN_SIZE).expect("alloc output");

    d_input.copy_from_host(&input_data).expect("copy input");
    d_rms_w
        .copy_from_host(&rms_weight)
        .expect("copy rms_weight");
    d_wgate.copy_from_host(&w_gate).expect("copy w_gate");
    d_wup.copy_from_host(&w_up).expect("copy w_up");
    d_wdown.copy_from_host(&w_down).expect("copy w_down");

    let kernel = FfnFusedKernel::load(device).expect("load ffn_fused kernel");

    kernel
        .forward_gate_up(
            &mut d_scratch,
            &d_input,
            &d_rms_w,
            &d_wgate,
            &d_wup,
            HIDDEN_SIZE as u32,
            INTERMEDIATE_SIZE as u32,
            EPS,
            &stream,
        )
        .expect("gate_up launch");

    kernel
        .forward_down_residual(
            &mut d_output,
            &d_input,
            &d_wdown,
            &d_scratch,
            HIDDEN_SIZE as u32,
            INTERMEDIATE_SIZE as u32,
            &stream,
        )
        .expect("down_residual launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; HIDDEN_SIZE];
    d_output.copy_to_host(&mut result).expect("copy output");

    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "FfnFused max error {max_err} exceeds tolerance"
    );
    println!("FfnFused test passed: max error = {max_err:.2e}");
}

#[test]
fn test_ffn_fused_matches_unfused_kernels() {
    let device = DeviceId(0);

    let input_data: Vec<f32> = (0..HIDDEN_SIZE).map(|i| (i as f32 * 0.01).sin()).collect();
    let rms_weight_f32: Vec<f32> = (0..HIDDEN_SIZE).map(|i| i as f32 * 0.001).collect();
    let w_gate_f32: Vec<f32> = (0..INTERMEDIATE_SIZE * HIDDEN_SIZE)
        .map(|i| (i as f32 * 0.0001).cos() * 0.01)
        .collect();
    let w_up_f32: Vec<f32> = (0..INTERMEDIATE_SIZE * HIDDEN_SIZE)
        .map(|i| (i as f32 * 0.00013 + 1.0).sin() * 0.01)
        .collect();
    let w_down_f32: Vec<f32> = (0..HIDDEN_SIZE * INTERMEDIATE_SIZE)
        .map(|i| (i as f32 * 0.00007 + 0.5).cos() * 0.01)
        .collect();
    let rms_weight: Vec<u16> = rms_weight_f32.iter().copied().map(f32_to_bf16).collect();
    let w_gate: Vec<u16> = w_gate_f32.iter().copied().map(f32_to_bf16).collect();
    let w_up: Vec<u16> = w_up_f32.iter().copied().map(f32_to_bf16).collect();
    let w_down: Vec<u16> = w_down_f32.iter().copied().map(f32_to_bf16).collect();

    let stream = Stream::new(device).expect("stream");

    // Shared weight buffers
    let mut d_input = DeviceBuffer::<f32>::alloc(device, HIDDEN_SIZE).expect("alloc input");
    let mut d_rms_w = DeviceBuffer::<u16>::alloc(device, HIDDEN_SIZE).expect("alloc rms_weight");
    let mut d_wgate =
        DeviceBuffer::<u16>::alloc(device, INTERMEDIATE_SIZE * HIDDEN_SIZE).expect("alloc w_gate");
    let mut d_wup =
        DeviceBuffer::<u16>::alloc(device, INTERMEDIATE_SIZE * HIDDEN_SIZE).expect("alloc w_up");
    let mut d_wdown =
        DeviceBuffer::<u16>::alloc(device, HIDDEN_SIZE * INTERMEDIATE_SIZE).expect("alloc w_down");

    d_input.copy_from_host(&input_data).expect("copy input");
    d_rms_w
        .copy_from_host(&rms_weight)
        .expect("copy rms_weight");
    d_wgate.copy_from_host(&w_gate).expect("copy w_gate");
    d_wup.copy_from_host(&w_up).expect("copy w_up");
    d_wdown.copy_from_host(&w_down).expect("copy w_down");

    // --- Unfused path ---
    let rms_kernel = RmsNormKernel::load(device).expect("load rmsnorm");
    let lp_kernel = LinearProjKernel::load(device).expect("load linear_proj");
    let silu_kernel = SiluMulKernel::load(device).expect("load silu_mul");
    let res_kernel = ResidualAddKernel::load(device).expect("load residual_add");

    let mut d_normed = DeviceBuffer::<f32>::alloc(device, HIDDEN_SIZE).expect("alloc normed");
    let mut d_gate_out =
        DeviceBuffer::<f32>::alloc(device, INTERMEDIATE_SIZE).expect("alloc gate_out");
    let mut d_up_out = DeviceBuffer::<f32>::alloc(device, INTERMEDIATE_SIZE).expect("alloc up_out");
    let mut d_act = DeviceBuffer::<f32>::alloc(device, INTERMEDIATE_SIZE).expect("alloc act");
    let mut d_down_out = DeviceBuffer::<f32>::alloc(device, HIDDEN_SIZE).expect("alloc down_out");
    let mut d_ref_output =
        DeviceBuffer::<f32>::alloc(device, HIDDEN_SIZE).expect("alloc ref_output");

    rms_kernel
        .forward(
            &mut d_normed,
            &d_input,
            &d_rms_w,
            1,
            HIDDEN_SIZE as u32,
            EPS,
            true,
            &stream,
        )
        .expect("rmsnorm");
    lp_kernel
        .forward(
            &mut d_gate_out,
            &d_wgate,
            &d_normed,
            INTERMEDIATE_SIZE as u32,
            HIDDEN_SIZE as u32,
            &stream,
        )
        .expect("gate proj");
    lp_kernel
        .forward(
            &mut d_up_out,
            &d_wup,
            &d_normed,
            INTERMEDIATE_SIZE as u32,
            HIDDEN_SIZE as u32,
            &stream,
        )
        .expect("up proj");
    silu_kernel
        .forward(
            &mut d_act,
            &d_gate_out,
            &d_up_out,
            INTERMEDIATE_SIZE as u32,
            &stream,
        )
        .expect("silu_mul");
    lp_kernel
        .forward(
            &mut d_down_out,
            &d_wdown,
            &d_act,
            HIDDEN_SIZE as u32,
            INTERMEDIATE_SIZE as u32,
            &stream,
        )
        .expect("down proj");
    res_kernel
        .forward(
            &mut d_ref_output,
            &d_down_out,
            &d_input,
            HIDDEN_SIZE as u32,
            &stream,
        )
        .expect("residual_add");

    stream.synchronize().expect("sync unfused");

    let mut unfused_result = vec![0.0f32; HIDDEN_SIZE];
    d_ref_output
        .copy_to_host(&mut unfused_result)
        .expect("copy unfused output");

    // --- Fused path ---
    let mut d_scratch =
        DeviceBuffer::<f32>::alloc(device, INTERMEDIATE_SIZE).expect("alloc scratch");
    let mut d_fused_output =
        DeviceBuffer::<f32>::alloc(device, HIDDEN_SIZE).expect("alloc fused output");

    let fused_kernel = FfnFusedKernel::load(device).expect("load ffn_fused");

    fused_kernel
        .forward_gate_up(
            &mut d_scratch,
            &d_input,
            &d_rms_w,
            &d_wgate,
            &d_wup,
            HIDDEN_SIZE as u32,
            INTERMEDIATE_SIZE as u32,
            EPS,
            &stream,
        )
        .expect("gate_up launch");

    fused_kernel
        .forward_down_residual(
            &mut d_fused_output,
            &d_input,
            &d_wdown,
            &d_scratch,
            HIDDEN_SIZE as u32,
            INTERMEDIATE_SIZE as u32,
            &stream,
        )
        .expect("down_residual launch");

    stream.synchronize().expect("sync fused");

    let mut fused_result = vec![0.0f32; HIDDEN_SIZE];
    d_fused_output
        .copy_to_host(&mut fused_result)
        .expect("copy fused output");

    let max_err: f32 = fused_result
        .iter()
        .zip(unfused_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "Fused vs unfused GPU kernels max error {max_err} exceeds tolerance"
    );
    println!("FfnFused vs unfused GPU kernels: max error = {max_err:.2e}");
}
