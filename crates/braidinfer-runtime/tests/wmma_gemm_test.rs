use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::WmmaGemmKernel;

fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn scalar_gemm_bf16(a: &[u16], b: &[u16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += bf16_to_f32(a[i * k + l]) * bf16_to_f32(b[j * k + l]);
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[test]
fn test_wmma_gemm_bf16_correctness() {
    let device = DeviceId(0);
    let m = 32usize;
    let n = 32usize;
    let k = 64usize;

    let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.007).cos() * 0.5).collect();

    let a_bf16: Vec<u16> = a_f32.iter().copied().map(f32_to_bf16).collect();
    let b_bf16: Vec<u16> = b_f32.iter().copied().map(f32_to_bf16).collect();

    let expected = scalar_gemm_bf16(&a_bf16, &b_bf16, m, n, k);

    let stream = Stream::new(device).expect("stream");
    let kernel = WmmaGemmKernel::load(device).expect("load wmma kernel");

    let mut d_a = DeviceBuffer::<u16>::alloc(device, m * k).expect("alloc a");
    let mut d_b = DeviceBuffer::<u16>::alloc(device, n * k).expect("alloc b");
    let mut d_c = DeviceBuffer::<f32>::alloc(device, m * n).expect("alloc c");

    d_a.copy_from_host(&a_bf16).expect("copy a");
    d_b.copy_from_host(&b_bf16).expect("copy b");

    kernel
        .gemm_bf16(&mut d_c, &d_a, &d_b, m as u32, n as u32, k as u32, &stream)
        .expect("gemm_bf16 launch");
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
        "WMMA bf16 GEMM max error {max_err:.4e} exceeds 1e-3 tolerance"
    );
}

#[test]
fn test_wmma_gemm_bf16_larger() {
    let device = DeviceId(0);
    let m = 256usize;
    let n = 256usize;
    let k = 1024usize;

    let a_f32: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 13) as f32 * 0.001).sin() * 0.3).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|i| ((i * 5 + 7) as f32 * 0.001).cos() * 0.3).collect();

    let a_bf16: Vec<u16> = a_f32.iter().copied().map(f32_to_bf16).collect();
    let b_bf16: Vec<u16> = b_f32.iter().copied().map(f32_to_bf16).collect();

    let expected = scalar_gemm_bf16(&a_bf16, &b_bf16, m, n, k);

    let stream = Stream::new(device).expect("stream");
    let kernel = WmmaGemmKernel::load(device).expect("load wmma kernel");

    let mut d_a = DeviceBuffer::<u16>::alloc(device, m * k).expect("alloc a");
    let mut d_b = DeviceBuffer::<u16>::alloc(device, n * k).expect("alloc b");
    let mut d_c = DeviceBuffer::<f32>::alloc(device, m * n).expect("alloc c");

    d_a.copy_from_host(&a_bf16).expect("copy a");
    d_b.copy_from_host(&b_bf16).expect("copy b");

    kernel
        .gemm_bf16(&mut d_c, &d_a, &d_b, m as u32, n as u32, k as u32, &stream)
        .expect("gemm_bf16 launch");
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
        "WMMA bf16 GEMM (larger) max error {max_err:.4e} exceeds 1e-3"
    );
}
