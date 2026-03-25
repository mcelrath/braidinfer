use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;

const OP_KV_QUANTIZE: u32 = 21;
const OP_HALT: u32 = 16;

fn build_instruction(opcode: u32, grid_x: u32, args: &[(usize, u64)]) -> [u64; 16] {
    let mut inst = [0u64; 16];
    inst[0] = opcode as u64 | ((grid_x as u64) << 32);
    for &(idx, val) in args {
        inst[idx] = val;
    }
    inst
}

fn dequant_int4(packed: u8) -> (f32, f32) {
    let even = (packed & 0xF) as i8;
    let even = if even > 7 { even - 16 } else { even };
    let odd = ((packed >> 4) & 0xF) as i8;
    let odd = if odd > 7 { odd - 16 } else { odd };
    (even as f32, odd as f32)
}

fn kernel_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("BRAIDINFER_KERNEL_DIR"))
}

#[test]
fn test_kv_quantize_residual_pc() {
    let device = DeviceId(0);
    let stream = Stream::new(device).expect("stream");

    let nkh = 2usize;
    let hd = 4usize;
    let ct = 64usize;
    let total_channels = nkh * hd;

    let mut src_data = vec![0.0f32; nkh * ct * hd];
    for h in 0..nkh {
        for t in 0..ct {
            for d in 0..hd {
                let idx = h * ct * hd + t * hd + d;
                src_data[idx] = ((h * 100 + t * 10 + d) as f32) * 0.01 - 3.0;
            }
        }
    }

    let mut src_buf = DeviceBuffer::<f32>::alloc(device, src_data.len()).expect("alloc src");
    src_buf.copy_from_host(&src_data).expect("copy src");

    let data_bytes = nkh * (ct / 2) * hd;
    let scale_elems = total_channels;

    let mut q1_data_buf = DeviceBuffer::<u8>::alloc(device, data_bytes).expect("alloc q1_data");
    let mut q1_scale_buf = DeviceBuffer::<f32>::alloc(device, scale_elems).expect("alloc q1_scale");
    let mut r_data_buf = DeviceBuffer::<u8>::alloc(device, data_bytes).expect("alloc r_data");
    let mut r_scale_buf = DeviceBuffer::<f32>::alloc(device, scale_elems).expect("alloc r_scale");

    let quant_inst = build_instruction(OP_KV_QUANTIZE, total_channels as u32, &[
        (1, src_buf.as_ptr() as u64),
        (2, q1_data_buf.as_mut_ptr() as u64),
        (3, q1_scale_buf.as_mut_ptr() as u64),
        (4, r_data_buf.as_mut_ptr() as u64),
        (5, r_scale_buf.as_mut_ptr() as u64),
        (6, nkh as u64),
        (7, hd as u64),
        (8, ct as u64),
    ]);
    let halt_inst = build_instruction(OP_HALT, 0, &[]);

    let mut program_data: Vec<u64> = Vec::new();
    program_data.extend_from_slice(&quant_inst);
    program_data.extend_from_slice(&halt_inst);

    let mut program_buf = DeviceBuffer::<u64>::alloc(device, program_data.len()).expect("alloc program");
    program_buf.copy_from_host(&program_data).expect("copy program");

    let module = Module::load(device, &kernel_dir().join("megakernel.hsaco")).expect("load module");
    let func = module.get_function("megakernel_f32").expect("get function");

    let num_instructions: i32 = 2;
    let program_ptr = program_buf.as_ptr();
    let mut args: Vec<*mut std::ffi::c_void> = vec![
        &program_ptr as *const _ as *mut std::ffi::c_void,
        &num_instructions as *const _ as *mut std::ffi::c_void,
    ];

    let shared_mem = 256u32 * 4 * 2;
    func.launch_cooperative(
        (384, 1, 1),
        (256, 1, 1),
        shared_mem,
        &stream,
        &mut args,
    ).expect("launch");
    stream.synchronize().expect("sync");

    let mut q1_data = vec![0u8; data_bytes];
    let mut q1_scales = vec![0.0f32; scale_elems];
    let mut r_data_out = vec![0u8; data_bytes];
    let mut r_scales = vec![0.0f32; scale_elems];
    q1_data_buf.copy_to_host(&mut q1_data).expect("read q1_data");
    q1_scale_buf.copy_to_host(&mut q1_scales).expect("read q1_scale");
    r_data_buf.copy_to_host(&mut r_data_out).expect("read r_data");
    r_scale_buf.copy_to_host(&mut r_scales).expect("read r_scale");

    let mut max_error = 0.0f32;
    let mut sum_sq_error = 0.0f64;
    let mut count = 0usize;

    for h in 0..nkh {
        for t in 0..ct {
            for d in 0..hd {
                let channel = h * hd + d;
                let pair = t / 2;
                let data_idx = h * (ct / 2) * hd + pair * hd + d;
                let s1 = q1_scales[channel];
                let s2 = r_scales[channel];

                let (even, odd) = dequant_int4(q1_data[data_idx]);
                let (r_even, r_odd) = dequant_int4(r_data_out[data_idx]);

                let q_val = if t % 2 == 0 { even } else { odd };
                let r_val = if t % 2 == 0 { r_even } else { r_odd };
                let reconstructed = q_val * s1 + r_val * s2;

                let original = src_data[h * ct * hd + t * hd + d];
                let error = (original - reconstructed).abs();
                max_error = max_error.max(error);
                sum_sq_error += (error as f64) * (error as f64);
                count += 1;
            }
        }
    }

    let mse = sum_sq_error / count as f64;
    let rmse = mse.sqrt();
    println!("Reconstruction: max_error={max_error:.6}, RMSE={rmse:.6}");
    println!("Scales sample: q1={:.4}, r={:.4}", q1_scales[0], r_scales[0]);
    println!("Reduction: {data_bytes} quant bytes vs {} f32 bytes = {:.2}x",
        nkh * ct * hd * 4,
        (nkh * ct * hd * 4) as f64 / (data_bytes * 2 + scale_elems * 4 * 2) as f64);

    assert!(max_error < 1.0, "max reconstruction error {max_error} too large");
    assert!(rmse < 0.1, "RMSE {rmse} too large");
}
