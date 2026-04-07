use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::LinearProjKernel;
use braidinfer_runtime::quant::{
    PackedWeights, WeightFormat, quantize_pc_g32_q4, quantize_rnf4_g128,
};

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounding = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + rounding) >> 16) as u16
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn linear_proj_reference(
    weight_bf16: &[u16],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut acc = 0.0f64;
        for j in 0..in_dim {
            acc += bf16_to_f32(weight_bf16[i * in_dim + j]) as f64 * input[j] as f64;
        }
        output[i] = acc as f32;
    }
    output
}

#[test]
fn test_rnf4_g128_matches_bf16() {
    let device = DeviceId(0);
    let in_dim = 256usize; // must be multiple of 128
    let out_dim = 64usize;

    let input_data: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.03).sin()).collect();
    let weight_f32: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32 * 0.0001).cos() * 0.1)
        .collect();
    let weight_bf16: Vec<u16> = weight_f32.iter().map(|&x| f32_to_bf16(x)).collect();

    let ref_output = linear_proj_reference(&weight_bf16, &input_data, out_dim, in_dim);

    let packed = quantize_rnf4_g128(&weight_bf16, out_dim, in_dim);

    let stream = Stream::new(device).expect("stream");
    let kernel = LinearProjKernel::load(device).expect("load kernel");

    let mut d_input = DeviceBuffer::<f32>::alloc(device, in_dim).expect("alloc");
    let mut d_output = DeviceBuffer::<f32>::alloc(device, out_dim).expect("alloc");
    d_input.copy_from_host(&input_data).expect("copy");

    let mut d_packed = DeviceBuffer::<u8>::alloc(device, packed.len()).expect("alloc");
    d_packed.copy_from_host(&packed).expect("copy");

    let pw = PackedWeights {
        data: d_packed,
        format: WeightFormat::Rnf4G128,
        out_dim,
        in_dim,
    };

    kernel
        .forward_packed(&mut d_output, &pw, &d_input, &stream)
        .expect("kernel");
    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; out_dim];
    d_output.copy_to_host(&mut result).expect("copy");

    let max_err: f32 = result
        .iter()
        .zip(ref_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_val: f32 = ref_output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let rel_err = max_err / max_val.max(1e-10);

    println!("rnf4_g128 test: max_err={max_err:.2e}, rel_err={rel_err:.2e}");
    println!("  bf16 ref sample: {:?}", &ref_output[..4]);
    println!("  rnf4 result:     {:?}", &result[..4]);

    assert!(
        rel_err < 0.02,
        "rnf4_g128 relative error {rel_err:.4} exceeds 2% tolerance"
    );
}

#[test]
fn test_pcg32_q4_runs() {
    let device = DeviceId(0);
    let in_dim = 256usize;
    let out_dim = 64usize;

    let input_data: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.03).sin()).collect();
    let weight_f32: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32 * 0.0001).cos() * 0.1)
        .collect();
    let weight_bf16: Vec<u16> = weight_f32.iter().map(|&x| f32_to_bf16(x)).collect();

    let ref_output = linear_proj_reference(&weight_bf16, &input_data, out_dim, in_dim);

    let packed = quantize_pc_g32_q4(&weight_bf16, out_dim, in_dim);

    let stream = Stream::new(device).expect("stream");
    let kernel = LinearProjKernel::load(device).expect("load kernel");

    let mut d_input = DeviceBuffer::<f32>::alloc(device, in_dim).expect("alloc");
    let mut d_output = DeviceBuffer::<f32>::alloc(device, out_dim).expect("alloc");
    d_input.copy_from_host(&input_data).expect("copy");

    let mut d_packed = DeviceBuffer::<u8>::alloc(device, packed.len()).expect("alloc");
    d_packed.copy_from_host(&packed).expect("copy");

    let pw = PackedWeights {
        data: d_packed,
        format: WeightFormat::PcG32Q4,
        out_dim,
        in_dim,
    };

    kernel
        .forward_packed(&mut d_output, &pw, &d_input, &stream)
        .expect("kernel");
    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; out_dim];
    d_output.copy_to_host(&mut result).expect("copy");

    let max_err: f32 = result
        .iter()
        .zip(ref_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_val: f32 = ref_output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let rel_err = max_err / max_val.max(1e-10);

    println!("pcg32_q4 test: max_err={max_err:.2e}, rel_err={rel_err:.4}");
    println!("  bf16 ref sample: {:?}", &ref_output[..4]);
    println!("  pcg32 result:    {:?}", &result[..4]);

    // Q4 has ~12.5% PPL degradation, so allow larger error
    assert!(
        rel_err < 0.2,
        "pcg32_q4 relative error {rel_err:.4} exceeds 20% tolerance"
    );
}
