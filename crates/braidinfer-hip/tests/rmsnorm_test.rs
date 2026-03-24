use braidinfer_core::types::DeviceId;
use braidinfer_hip::{Device, DeviceBuffer, Stream};

#[test]
fn test_device_count() {
    let count = Device::count().expect("failed to get device count");
    assert!(count > 0, "no GPU devices found");
    println!("Found {count} GPU devices");
}

#[test]
fn test_alloc_and_copy() {
    let device = DeviceId(0);
    let n = 1024;

    let mut buf = DeviceBuffer::<f32>::alloc(device, n).expect("alloc failed");

    // Upload
    let host_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    buf.copy_from_host(&host_data).expect("H2D copy failed");

    // Download
    let mut result = vec![0.0f32; n];
    buf.copy_to_host(&mut result).expect("D2H copy failed");

    assert_eq!(host_data, result);
}

#[test]
fn test_stream_create() {
    let device = DeviceId(0);
    let stream = Stream::new(device).expect("stream creation failed");
    stream.synchronize().expect("stream sync failed");
}
