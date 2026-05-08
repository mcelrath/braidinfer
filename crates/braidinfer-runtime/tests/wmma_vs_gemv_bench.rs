use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::{LinearProjKernel, WmmaGemmKernel};
use std::time::Instant;

fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

#[test]
#[ignore]
fn bench_wmma_vs_gemv() {
    let device = DeviceId(0);
    let stream = Stream::new(device).expect("stream");
    let wmma = WmmaGemmKernel::load(device).expect("load wmma");
    let gemv = LinearProjKernel::load(device).expect("load gemv");

    let shapes: &[(usize, usize)] = &[
        (4096, 4096),
        (4096, 1536),
        (1536, 4096),
        (5120, 5120),
        (5120, 1024),
    ];
    let m_values: &[usize] = &[1, 4, 16, 64, 256];
    let warmup = 5;
    let iters = 50;

    println!("\nshape (out_dim x in_dim) | M | wmma ms | gemv x M ms | speedup");
    println!("---");
    for &(out_dim, in_dim) in shapes {
        for &m in m_values {
            let k = in_dim;
            let n = out_dim;
            assert_eq!(k % 16, 0);
            assert_eq!(n % 16, 0);

            let a_f32: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 13) as f32 * 0.001).sin() * 0.3).collect();
            let b_f32: Vec<f32> = (0..n * k).map(|i| ((i * 5 + 7) as f32 * 0.001).cos() * 0.3).collect();
            let a_bf16: Vec<u16> = a_f32.iter().copied().map(f32_to_bf16).collect();
            let b_bf16: Vec<u16> = b_f32.iter().copied().map(f32_to_bf16).collect();

            let mut d_a_bf = DeviceBuffer::<u16>::alloc(device, m * k).unwrap();
            let mut d_b_bf = DeviceBuffer::<u16>::alloc(device, n * k).unwrap();
            let mut d_c = DeviceBuffer::<f32>::alloc(device, m * n).unwrap();
            d_a_bf.copy_from_host(&a_bf16).unwrap();
            d_b_bf.copy_from_host(&b_bf16).unwrap();

            // GEMV path expects f32 input, bf16 weights, f32 output (per-token GEMV).
            // Reuse the same a_f32 (treat M tokens as M sequential GEMV calls).
            let mut d_a_f32 = DeviceBuffer::<f32>::alloc(device, m * k).unwrap();
            d_a_f32.copy_from_host(&a_f32).unwrap();
            let mut d_y = DeviceBuffer::<f32>::alloc(device, n).unwrap();

            // Warmup WMMA
            for _ in 0..warmup {
                wmma.gemm_bf16(&mut d_c, &d_a_bf, &d_b_bf, m as u32, n as u32, k as u32, &stream).unwrap();
            }
            stream.synchronize().unwrap();

            let t0 = Instant::now();
            for _ in 0..iters {
                wmma.gemm_bf16(&mut d_c, &d_a_bf, &d_b_bf, m as u32, n as u32, k as u32, &stream).unwrap();
            }
            stream.synchronize().unwrap();
            let wmma_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // Warmup GEMV
            for _ in 0..warmup {
                for _t in 0..m {
                    gemv.forward(&mut d_y, &d_b_bf, &d_a_f32, n as u32, k as u32, &stream).unwrap();
                }
            }
            stream.synchronize().unwrap();

            let t0 = Instant::now();
            for _ in 0..iters {
                for _t in 0..m {
                    gemv.forward(&mut d_y, &d_b_bf, &d_a_f32, n as u32, k as u32, &stream).unwrap();
                }
            }
            stream.synchronize().unwrap();
            let gemv_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let speedup = gemv_ms / wmma_ms;
            println!(
                "{}x{} | M={:3} | wmma={:7.3} | gemv={:7.3} | {:5.2}x",
                out_dim, in_dim, m, wmma_ms, gemv_ms, speedup
            );
        }
    }
}
