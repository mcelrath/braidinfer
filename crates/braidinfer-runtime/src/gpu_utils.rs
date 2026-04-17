use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{HipResult, ffi};

/// Copy `count` f32 elements from src at offset src_off to dst at offset dst_off (device-to-device).
pub unsafe fn d2d_copy_f32(
    dst: &mut DeviceBuffer<f32>,
    dst_off: usize,
    src: &DeviceBuffer<f32>,
    src_off: usize,
    count: usize,
    stream: &Stream,
) -> HipResult<()> {
    unsafe {
        let dst_ptr = dst.as_mut_ptr().add(dst_off) as *mut std::ffi::c_void;
        let src_ptr = src.as_ptr().add(src_off) as *const std::ffi::c_void;
        braidinfer_hip::error::check(ffi::hipMemcpyAsync(
            dst_ptr,
            src_ptr,
            count * 4,
            ffi::hipMemcpyDeviceToDevice,
            stream.raw(),
        ))
    }
}
