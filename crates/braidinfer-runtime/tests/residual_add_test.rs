use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::ResidualAddKernel;

#[test]
fn test_residual_add_matches_reference() {
    let device = DeviceId(0);
    let size = 1024usize;

    let x_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
    let res_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.007).cos() * 0.5).collect();

    let expected: Vec<f32> = x_data.iter().zip(res_data.iter()).map(|(a, b)| a + b).collect();

    let stream = Stream::new(device).expect("stream");
    let kernel = ResidualAddKernel::load(device).expect("load kernel");

    let mut d_x = DeviceBuffer::<f32>::alloc(device, size).expect("alloc x");
    let mut d_res = DeviceBuffer::<f32>::alloc(device, size).expect("alloc residual");
    let mut d_output = DeviceBuffer::<f32>::alloc(device, size).expect("alloc output");

    d_x.copy_from_host(&x_data).expect("copy x");
    d_res.copy_from_host(&res_data).expect("copy residual");

    kernel
        .forward(&mut d_output, &d_x, &d_res, size as u32, &stream)
        .expect("kernel launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; size];
    d_output.copy_to_host(&mut result).expect("copy output");

    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-6,
        "ResidualAdd max error {max_err} exceeds tolerance"
    );
    println!("ResidualAdd test passed: max error = {max_err:.2e}");
}
