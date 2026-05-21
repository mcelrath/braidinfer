use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;

const OP_KV_QUANTIZE: u32 = 21;
const OP_ATTN_PAGED: u32 = 18;
const OP_ATTN_PAGED_Q: u32 = 22;
const OP_HALT: u32 = 16;
const INST_SIZE: usize = 18; // must match megakernel::mod.rs INST_SIZE
const SHARED_MEM_BYTES: u32 = 31_776;
const BLOCK_SIZE: u32 = 256;
const NUM_CUS: u32 = 48;

fn build_instruction(opcode: u32, grid_x: u32, args: &[(usize, u64)]) -> [u64; INST_SIZE] {
    let mut inst = [0u64; INST_SIZE];
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

fn launch_megakernel(
    func: &braidinfer_hip::module::Function<'_>,
    stream: &Stream,
    args: &mut [*mut std::ffi::c_void],
) {
    let blocks_per_sm = func
        .max_active_blocks_per_sm(BLOCK_SIZE, SHARED_MEM_BYTES as usize)
        .expect("occupancy");
    let num_blocks = (blocks_per_sm.max(1) as u32) * NUM_CUS;
    func.launch_cooperative(
        (num_blocks, 1, 1),
        (BLOCK_SIZE, 1, 1),
        SHARED_MEM_BYTES,
        stream,
        args,
    )
    .expect("launch");
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

    let quant_inst = build_instruction(
        OP_KV_QUANTIZE,
        total_channels as u32,
        &[
            (1, src_buf.as_ptr() as u64),
            (2, q1_data_buf.as_mut_ptr() as u64),
            (3, q1_scale_buf.as_mut_ptr() as u64),
            (4, r_data_buf.as_mut_ptr() as u64),
            (5, r_scale_buf.as_mut_ptr() as u64),
            (6, nkh as u64),
            (7, hd as u64),
            (8, ct as u64),
        ],
    );
    let halt_inst = build_instruction(OP_HALT, 0, &[]);

    let mut program_data: Vec<u64> = Vec::new();
    program_data.extend_from_slice(&quant_inst);
    program_data.extend_from_slice(&halt_inst);

    let mut program_buf =
        DeviceBuffer::<u64>::alloc(device, program_data.len()).expect("alloc program");
    program_buf
        .copy_from_host(&program_data)
        .expect("copy program");

    let module = Module::load(device, &kernel_dir().join("megakernel.hsaco")).expect("load module");
    let func = module.get_function("megakernel_f32").expect("get function");

    let num_instructions: i32 = 2;
    let program_ptr = program_buf.as_ptr();
    let mut args: Vec<*mut std::ffi::c_void> = vec![
        &program_ptr as *const _ as *mut std::ffi::c_void,
        &num_instructions as *const _ as *mut std::ffi::c_void,
    ];

    launch_megakernel(&func, &stream, &mut args);
    stream.synchronize().expect("sync");

    let mut q1_data = vec![0u8; data_bytes];
    let mut q1_scales = vec![0.0f32; scale_elems];
    let mut r_data_out = vec![0u8; data_bytes];
    let mut r_scales = vec![0.0f32; scale_elems];
    q1_data_buf
        .copy_to_host(&mut q1_data)
        .expect("read q1_data");
    q1_scale_buf
        .copy_to_host(&mut q1_scales)
        .expect("read q1_scale");
    r_data_buf
        .copy_to_host(&mut r_data_out)
        .expect("read r_data");
    r_scale_buf
        .copy_to_host(&mut r_scales)
        .expect("read r_scale");

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
    println!(
        "Scales sample: q1={:.4}, r={:.4}",
        q1_scales[0], r_scales[0]
    );
    println!(
        "Reduction: {data_bytes} quant bytes vs {} f32 bytes = {:.2}x",
        nkh * ct * hd * 4,
        (nkh * ct * hd * 4) as f64 / (data_bytes * 2 + scale_elems * 4 * 2) as f64
    );

    assert!(
        max_error < 1.0,
        "max reconstruction error {max_error} too large"
    );
    assert!(rmse < 0.1, "RMSE {rmse} too large");
}

#[test]
fn test_attn_paged_quant_vs_f32() {
    let device = DeviceId(0);
    let stream = Stream::new(device).expect("stream");

    let nqh = 4usize; // 4 Q heads
    let nkh = 2usize; // 2 KV heads (GQA ratio 2)
    let hd = 256usize;
    let ct = 64usize;
    let seq_len = ct; // exactly one full chunk
    let rope_dim = 64usize;

    // Generate random-ish Q, K, V data
    let mut q_data = vec![0.0f32; nqh * hd];
    let mut kv_f32 = vec![0.0f32; 2 * nkh * ct * hd]; // [K region, V region]
    for i in 0..q_data.len() {
        q_data[i] = ((i * 7 + 3) % 1000) as f32 * 0.001 - 0.5;
    }
    for i in 0..kv_f32.len() {
        kv_f32[i] = ((i * 13 + 7) % 1000) as f32 * 0.001 - 0.5;
    }

    // inv_freq for RoPE
    let rope_pairs = rope_dim / 2;
    let inv_freq: Vec<f32> = (0..rope_pairs)
        .map(|i| 1.0 / 10000.0f32.powf(2.0 * i as f32 / rope_dim as f32))
        .collect();

    // Position table: positions 0..63
    let positions: Vec<i32> = (0..ct as i32).collect();

    // Upload Q, inv_freq, positions
    let mut q_buf = DeviceBuffer::<f32>::alloc(device, q_data.len()).expect("alloc q");
    q_buf.copy_from_host(&q_data).expect("copy q");
    let mut inv_freq_buf =
        DeviceBuffer::<f32>::alloc(device, inv_freq.len()).expect("alloc inv_freq");
    inv_freq_buf
        .copy_from_host(&inv_freq)
        .expect("copy inv_freq");
    let mut pos_buf = DeviceBuffer::<i32>::alloc(device, positions.len()).expect("alloc pos");
    pos_buf.copy_from_host(&positions).expect("copy pos");

    // --- F32 path: upload KV as f32, run OP_ATTN_PAGED ---
    let mut kv_f32_buf = DeviceBuffer::<f32>::alloc(device, kv_f32.len()).expect("alloc kv_f32");
    kv_f32_buf.copy_from_host(&kv_f32).expect("copy kv_f32");

    let chunk_ptr = kv_f32_buf.as_ptr() as u64;
    let mut page_table_data = vec![chunk_ptr];
    let mut page_table_buf = DeviceBuffer::<u64>::alloc(device, 1).expect("alloc pt");
    page_table_buf
        .copy_from_host(&page_table_data)
        .expect("copy pt");

    let k_offset_bytes = 0u64;
    let v_offset_bytes = (nkh * ct * hd * 4) as u64;

    let mut output_f32_buf = DeviceBuffer::<f32>::alloc(device, nqh * hd).expect("alloc out_f32");
    let f32_inst = build_instruction(
        OP_ATTN_PAGED,
        nqh as u32,
        &[
            (1, output_f32_buf.as_mut_ptr() as u64),
            (2, q_buf.as_ptr() as u64),
            (3, page_table_buf.as_ptr() as u64),
            (4, pos_buf.as_ptr() as u64),
            (5, inv_freq_buf.as_ptr() as u64),
            (6, nqh as u64),
            (7, nkh as u64),
            (8, hd as u64),
            (9, seq_len as u64),
            (10, ct as u64),
            (11, rope_dim as u64),
            (12, k_offset_bytes),
            (13, v_offset_bytes),
        ],
    );

    let halt = build_instruction(OP_HALT, 0, &[]);
    let mut prog: Vec<u64> = Vec::new();
    prog.extend_from_slice(&f32_inst);
    prog.extend_from_slice(&halt);
    let mut prog_buf = DeviceBuffer::<u64>::alloc(device, prog.len()).expect("alloc prog");
    prog_buf.copy_from_host(&prog).expect("copy prog");

    let module = Module::load(device, &kernel_dir().join("megakernel.hsaco")).expect("load module");
    let func = module.get_function("megakernel_f32").expect("get function");
    let num_inst: i32 = 2;
    let prog_ptr = prog_buf.as_ptr();
    let mut args: Vec<*mut std::ffi::c_void> = vec![
        &prog_ptr as *const _ as *mut std::ffi::c_void,
        &num_inst as *const _ as *mut std::ffi::c_void,
    ];
    launch_megakernel(&func, &stream, &mut args);
    stream.synchronize().expect("sync f32");

    let mut output_f32 = vec![0.0f32; nqh * hd];
    output_f32_buf
        .copy_to_host(&mut output_f32)
        .expect("read f32 output");

    // --- Quantized path: quantize K and V, run OP_ATTN_PAGED_Q ---
    // Split kv_f32 into K and V staging buffers
    let k_f32 = &kv_f32[..nkh * ct * hd];
    let v_f32 = &kv_f32[nkh * ct * hd..];

    let data_bytes = nkh * (ct / 2) * hd;
    let scale_elems = nkh * hd;
    let scale_bytes = scale_elems * 4;
    let per_kv = 2 * (data_bytes + scale_bytes);

    // Allocate quantized chunk: [K_q1data, K_q1scale, K_rdata, K_rscale, V_q1data, V_q1scale, V_rdata, V_rscale]
    let quant_chunk_bytes = 2 * per_kv; // K + V
    let mut quant_chunk_buf =
        DeviceBuffer::<u8>::alloc(device, quant_chunk_bytes).expect("alloc qchunk");

    // Upload K f32, quantize K
    let mut k_staging = DeviceBuffer::<f32>::alloc(device, k_f32.len()).expect("alloc k");
    k_staging.copy_from_host(k_f32).expect("copy k");

    let k_q1data_ptr = unsafe { quant_chunk_buf.as_mut_ptr().add(0) };
    let k_q1scale_ptr = unsafe { quant_chunk_buf.as_mut_ptr().add(data_bytes) };
    let k_rdata_ptr = unsafe { quant_chunk_buf.as_mut_ptr().add(data_bytes + scale_bytes) };
    let k_rscale_ptr = unsafe {
        quant_chunk_buf
            .as_mut_ptr()
            .add(2 * data_bytes + scale_bytes)
    };

    let quant_k_inst = build_instruction(
        OP_KV_QUANTIZE,
        (nkh * hd) as u32,
        &[
            (1, k_staging.as_ptr() as u64),
            (2, k_q1data_ptr as u64),
            (3, k_q1scale_ptr as u64),
            (4, k_rdata_ptr as u64),
            (5, k_rscale_ptr as u64),
            (6, nkh as u64),
            (7, hd as u64),
            (8, ct as u64),
        ],
    );

    // Upload V f32, quantize V
    let mut v_staging = DeviceBuffer::<f32>::alloc(device, v_f32.len()).expect("alloc v");
    v_staging.copy_from_host(v_f32).expect("copy v");

    let v_base = per_kv;
    let v_q1data_ptr = unsafe { quant_chunk_buf.as_mut_ptr().add(v_base) };
    let v_q1scale_ptr = unsafe { quant_chunk_buf.as_mut_ptr().add(v_base + data_bytes) };
    let v_rdata_ptr = unsafe {
        quant_chunk_buf
            .as_mut_ptr()
            .add(v_base + data_bytes + scale_bytes)
    };
    let v_rscale_ptr = unsafe {
        quant_chunk_buf
            .as_mut_ptr()
            .add(v_base + 2 * data_bytes + scale_bytes)
    };

    let quant_v_inst = build_instruction(
        OP_KV_QUANTIZE,
        (nkh * hd) as u32,
        &[
            (1, v_staging.as_ptr() as u64),
            (2, v_q1data_ptr as u64),
            (3, v_q1scale_ptr as u64),
            (4, v_rdata_ptr as u64),
            (5, v_rscale_ptr as u64),
            (6, nkh as u64),
            (7, hd as u64),
            (8, ct as u64),
        ],
    );

    // --- Two-phase quantized attention ---
    // Phase 1: OP_ATTN_PAGED_Q over quantized chunk → scratch
    // Phase 2: OP_ATTN_PAGED with partial_state=scratch, seq_len=0 → normalize only

    let quant_chunk_ptr = quant_chunk_buf.as_ptr() as u64;
    page_table_data[0] = quant_chunk_ptr;
    page_table_buf
        .copy_from_host(&page_table_data)
        .expect("update pt");

    // Scratch buffer: [nqh × (2 + head_dim)] floats
    let scratch_elems = nqh * (2 + hd);
    let mut scratch_buf = DeviceBuffer::<f32>::alloc(device, scratch_elems).expect("alloc scratch");

    let attn_q_inst = build_instruction(
        OP_ATTN_PAGED_Q,
        nqh as u32,
        &[
            (1, scratch_buf.as_mut_ptr() as u64), // scratch (not final output)
            (2, q_buf.as_ptr() as u64),
            (3, page_table_buf.as_ptr() as u64),
            (4, pos_buf.as_ptr() as u64),
            (5, inv_freq_buf.as_ptr() as u64),
            (6, nqh as u64),
            (7, nkh as u64),
            (8, hd as u64),
            (9, seq_len as u64),
            (10, ct as u64),
            (11, rope_dim as u64),
            (12, 0u64),
            (13, data_bytes as u64),
            (14, (data_bytes + scale_bytes) as u64),
            (15, (2 * data_bytes + scale_bytes) as u64),
        ],
    );

    // Phase 2: OP_ATTN_PAGED with partial_state, seq_len=0 (no f32 chunks to process)
    // This just normalizes v_acc/d from the scratch buffer
    let mut output_q_buf = DeviceBuffer::<f32>::alloc(device, nqh * hd).expect("alloc out_q");
    let mut empty_pt_buf = DeviceBuffer::<u64>::alloc(device, 1).expect("alloc empty pt");
    empty_pt_buf.copy_from_host(&[0u64]).expect("copy empty pt");
    let merge_inst = build_instruction(
        OP_ATTN_PAGED,
        nqh as u32,
        &[
            (1, output_q_buf.as_mut_ptr() as u64),
            (2, q_buf.as_ptr() as u64),
            (3, empty_pt_buf.as_ptr() as u64),
            (4, pos_buf.as_ptr() as u64),
            (5, inv_freq_buf.as_ptr() as u64),
            (6, nqh as u64),
            (7, nkh as u64),
            (8, hd as u64),
            (9, 0u64), // seq_len=0: no f32 chunks
            (10, ct as u64),
            (11, rope_dim as u64),
            (12, 0u64),
            (13, 0u64),
            (14, scratch_buf.as_ptr() as u64), // partial_state from quantized pass
        ],
    );

    // Build program: quantize K, quantize V, quant attention, merge, halt
    let mut prog2: Vec<u64> = Vec::new();
    prog2.extend_from_slice(&quant_k_inst);
    prog2.extend_from_slice(&quant_v_inst);
    prog2.extend_from_slice(&attn_q_inst);
    prog2.extend_from_slice(&merge_inst);
    prog2.extend_from_slice(&halt);

    let mut prog2_buf = DeviceBuffer::<u64>::alloc(device, prog2.len()).expect("alloc prog2");
    prog2_buf.copy_from_host(&prog2).expect("copy prog2");

    let num_inst2: i32 = 5;
    let prog2_ptr = prog2_buf.as_ptr();
    let mut args2: Vec<*mut std::ffi::c_void> = vec![
        &prog2_ptr as *const _ as *mut std::ffi::c_void,
        &num_inst2 as *const _ as *mut std::ffi::c_void,
    ];
    launch_megakernel(&func, &stream, &mut args2);
    stream.synchronize().expect("sync quant");

    let mut output_q = vec![0.0f32; nqh * hd];
    output_q_buf
        .copy_to_host(&mut output_q)
        .expect("read quant output");

    // Compare
    let max_diff: f32 = output_f32
        .iter()
        .zip(output_q.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_diff: f32 = output_f32
        .iter()
        .zip(output_q.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / output_f32.len() as f32;

    println!("F32 vs Quantized attention: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}");
    println!("F32 output[0..4]: {:?}", &output_f32[..4]);
    println!("Quant output[0..4]: {:?}", &output_q[..4]);

    assert!(
        max_diff < 0.05,
        "attention output diverged: max_diff={max_diff}"
    );
}
