use braidinfer_core::types::DeviceId;
use braidinfer_hip::{DeviceBuffer, Stream};
use braidinfer_runtime::kernel::EmbeddingKernel;

#[test]
fn test_embedding_lookup() {
    let device = DeviceId(0);
    let vocab_size = 256usize;
    let hidden_size = 1024usize;
    let token_id = 42i32;

    let table_data: Vec<f32> = (0..vocab_size * hidden_size)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    let expected: Vec<f32> = table_data
        [token_id as usize * hidden_size..(token_id as usize + 1) * hidden_size]
        .to_vec();

    let stream = Stream::new(device).expect("stream");
    let kernel = EmbeddingKernel::load(device).expect("load kernel");

    let mut d_table = DeviceBuffer::<f32>::alloc(device, vocab_size * hidden_size).expect("alloc table");
    let mut d_output = DeviceBuffer::<f32>::alloc(device, hidden_size).expect("alloc output");

    d_table.copy_from_host(&table_data).expect("copy table");

    kernel
        .forward(&mut d_output, &d_table, token_id, hidden_size as u32, &stream)
        .expect("kernel launch");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; hidden_size];
    d_output.copy_to_host(&mut result).expect("copy output");

    let max_err: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-7,
        "Embedding max error {max_err} exceeds tolerance"
    );
    println!("Embedding test passed: max error = {max_err:.2e}");
}
