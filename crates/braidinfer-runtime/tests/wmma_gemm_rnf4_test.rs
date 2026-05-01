use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::WmmaGemmKernel;

const NF4_TABLE: [f32; 16] = [
    -1.0f32, -0.6961928, -0.5250731, -0.3949175,
    -0.2844414, -0.1847734, -0.0910500, 0.0,
     0.0795803,  0.1609302,  0.2461123,  0.3379152,
     0.4407098,  0.5626170,  0.7229568,  1.0,
];

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// Generate RNF4G128 weight buffer for shape [out_dim, in_dim].
/// Returns (weight_bytes, reference_f32).
/// reference_f32[n*in_dim + k] = dequantized weight value.
fn generate_rnf4g128_weights(out_dim: usize, in_dim: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let group_size = 128usize;
    let group_bytes = 132usize;
    let num_groups = in_dim / group_size;
    assert_eq!(in_dim % group_size, 0, "in_dim must be multiple of 128");

    let total_bytes = out_dim * num_groups * group_bytes;
    let mut weight_buf = vec![0u8; total_bytes];
    let mut ref_f32 = vec![0.0f32; out_dim * in_dim];

    let mut rng = seed as u64;
    let mut next = || -> u32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng & 0xFFFFFFFF) as u32
    };

    for n in 0..out_dim {
        let row_offset = n * num_groups * group_bytes;
        for g in 0..num_groups {
            let gp = row_offset + g * group_bytes;
            let absmax1_f32 = 0.5 + (next() as f32 / u32::MAX as f32) * 0.5;
            let absmax2_f32 = 0.5 + (next() as f32 / u32::MAX as f32) * 0.5;
            let absmax1_bf16 = f32_to_bf16_bits(absmax1_f32);
            let absmax2_bf16 = f32_to_bf16_bits(absmax2_f32);
            let absmax1 = bf16_to_f32(absmax1_bf16);
            let absmax2 = bf16_to_f32(absmax2_bf16);

            weight_buf[gp + 64] = (absmax1_bf16 & 0xFF) as u8;
            weight_buf[gp + 65] = (absmax1_bf16 >> 8) as u8;
            weight_buf[gp + 130] = (absmax2_bf16 & 0xFF) as u8;
            weight_buf[gp + 131] = (absmax2_bf16 >> 8) as u8;

            for bi in 0..64 {
                let r = next();
                let i1_lo = (r & 0xF) as usize;
                let i1_hi = ((r >> 4) & 0xF) as usize;
                let i2_lo = ((r >> 8) & 0xF) as usize;
                let i2_hi = ((r >> 12) & 0xF) as usize;

                weight_buf[gp + bi] = (i1_lo | (i1_hi << 4)) as u8;
                weight_buf[gp + 66 + bi] = (i2_lo | (i2_hi << 4)) as u8;

                let k_base = g * group_size + bi * 2;
                ref_f32[n * in_dim + k_base + 0] =
                    NF4_TABLE[i1_lo] * absmax1 + NF4_TABLE[i2_lo] * absmax2;
                ref_f32[n * in_dim + k_base + 1] =
                    NF4_TABLE[i1_hi] * absmax1 + NF4_TABLE[i2_hi] * absmax2;
            }
        }
    }

    (weight_buf, ref_f32)
}

/// Reference GEMM that performs the SAME bf16 conversions as the WMMA kernel
/// (f32 activation -> bf16, dequantized weight f32 -> bf16, then accumulate in f32).
/// This isolates WMMA correctness from the bf16 precision question.
fn scalar_gemm_rnf4_bf16_equiv(a: &[f32], w_ref: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                let a_bf16 = bf16_to_f32(f32_to_bf16_bits(a[i * k + l]));
                let w_bf16 = bf16_to_f32(f32_to_bf16_bits(w_ref[j * k + l]));
                acc += a_bf16 * w_bf16;
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[test]
fn test_wmma_gemm_rnf4g128_correctness() {
    let device = DeviceId(0);
    let m = 32usize;
    let n = 32usize;
    let k = 128usize;

    let a_f32: Vec<f32> = (0..m * k).map(|i| ((i * 3 + 1) as f32 * 0.01).sin() * 0.4).collect();

    let (weight_bytes, w_ref) = generate_rnf4g128_weights(n, k, 42);

    let expected = scalar_gemm_rnf4_bf16_equiv(&a_f32, &w_ref, m, n, k);

    let stream = Stream::new(device).expect("stream");
    let kernel = WmmaGemmKernel::load(device).expect("load wmma kernel");

    let mut d_a = DeviceBuffer::<f32>::alloc(device, m * k).expect("alloc a");
    let mut d_b = DeviceBuffer::<u8>::alloc(device, weight_bytes.len()).expect("alloc b");
    let mut d_c = DeviceBuffer::<f32>::alloc(device, m * n).expect("alloc c");

    d_a.copy_from_host(&a_f32).expect("copy a");
    d_b.copy_from_host(&weight_bytes).expect("copy b");

    kernel
        .gemm_rnf4g128(&mut d_c, &d_a, &d_b, m as u32, n as u32, k as u32, &stream)
        .expect("gemm_rnf4 launch");
    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; m * n];
    d_c.copy_to_host(&mut result).expect("copy result");

    let max_err = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "WMMA RNF4G128 GEMM max error {max_err:.4e} exceeds 1e-3 tolerance"
    );
}

#[test]
fn test_wmma_gemm_rnf4g128_larger() {
    let device = DeviceId(0);
    let m = 64usize;
    let n = 64usize;
    let k = 256usize;

    let a_f32: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) as f32 * 0.005).cos() * 0.5).collect();

    let (weight_bytes, w_ref) = generate_rnf4g128_weights(n, k, 137);

    let expected = scalar_gemm_rnf4_bf16_equiv(&a_f32, &w_ref, m, n, k);

    let stream = Stream::new(device).expect("stream");
    let kernel = WmmaGemmKernel::load(device).expect("load wmma kernel");

    let mut d_a = DeviceBuffer::<f32>::alloc(device, m * k).expect("alloc a");
    let mut d_b = DeviceBuffer::<u8>::alloc(device, weight_bytes.len()).expect("alloc b");
    let mut d_c = DeviceBuffer::<f32>::alloc(device, m * n).expect("alloc c");

    d_a.copy_from_host(&a_f32).expect("copy a");
    d_b.copy_from_host(&weight_bytes).expect("copy b");

    kernel
        .gemm_rnf4g128(&mut d_c, &d_a, &d_b, m as u32, n as u32, k as u32, &stream)
        .expect("gemm_rnf4 launch");
    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; m * n];
    d_c.copy_to_host(&mut result).expect("copy result");

    let max_err = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "WMMA RNF4G128 GEMM (larger) max error {max_err:.4e} exceeds 1e-3"
    );
}
