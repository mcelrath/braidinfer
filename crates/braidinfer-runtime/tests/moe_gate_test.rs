use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;

const OP_LINEAR_PROJ: u32 = 2;
const OP_MOE_GATE: u32 = 23;
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
fn test_moe_gate_softmax() {
    let device = DeviceId(0);
    let stream = Stream::new(device).expect("stream");

    let hidden_size = 16usize;
    let num_experts = 8usize;
    let k = 3usize;

    // Create a hidden state and gate weights
    let mut hidden = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        hidden[i] = (i as f32) * 0.1;
    }

    // Gate weights: [num_experts, hidden_size] as bf16
    let mut gate_data = vec![0u16; num_experts * hidden_size];
    for e in 0..num_experts {
        for h in 0..hidden_size {
            let val = if e == 2 {
                0.5
            } else if e == 5 {
                0.3
            } else if e == 7 {
                0.2
            } else {
                0.0
            };
            let val = val + (h as f32) * 0.01;
            gate_data[e * hidden_size + h] = (val.to_bits() >> 16) as u16; // f32→bf16 truncation
        }
    }

    let mut hidden_buf = DeviceBuffer::<f32>::alloc(device, hidden_size).expect("alloc");
    hidden_buf.copy_from_host(&hidden).expect("copy");
    let mut gate_buf =
        DeviceBuffer::<u16>::alloc(device, num_experts * hidden_size).expect("alloc");
    gate_buf.copy_from_host(&gate_data).expect("copy");
    let mut scores_buf = DeviceBuffer::<f32>::alloc(device, num_experts).expect("alloc");
    let mut expert_ids_buf = DeviceBuffer::<i32>::alloc(device, k).expect("alloc");
    let mut expert_weights_buf = DeviceBuffer::<f32>::alloc(device, k).expect("alloc");

    // Program: LINEAR_PROJ (hidden @ gate^T → scores), MOE_GATE (top-k + softmax), HALT
    let proj_inst = build_instruction(
        OP_LINEAR_PROJ,
        num_experts as u32,
        &[
            (1, scores_buf.as_mut_ptr() as u64),
            (2, gate_buf.as_ptr() as u64),
            (3, hidden_buf.as_ptr() as u64),
            (4, num_experts as u64),
            (5, hidden_size as u64),
        ],
    );
    let gate_inst = build_instruction(
        OP_MOE_GATE,
        1,
        &[
            (1, scores_buf.as_ptr() as u64),
            (2, expert_ids_buf.as_mut_ptr() as u64),
            (3, expert_weights_buf.as_mut_ptr() as u64),
            (4, num_experts as u64),
            (5, k as u64),
            (6, 0u64), // softmax mode
            (7, 0u64), // no scaling
        ],
    );
    let halt = build_instruction(OP_HALT, 0, &[]);

    let mut prog: Vec<u64> = Vec::new();
    prog.extend_from_slice(&proj_inst);
    prog.extend_from_slice(&gate_inst);
    prog.extend_from_slice(&halt);
    let mut prog_buf = DeviceBuffer::<u64>::alloc(device, prog.len()).expect("alloc");
    prog_buf.copy_from_host(&prog).expect("copy");

    let module = Module::load(device, &kernel_dir().join("megakernel.hsaco")).expect("load");
    let func = module.get_function("megakernel_f32").expect("func");
    let num_inst: i32 = 3;
    let prog_ptr = prog_buf.as_ptr();
    let mut args: Vec<*mut std::ffi::c_void> = vec![
        &prog_ptr as *const _ as *mut std::ffi::c_void,
        &num_inst as *const _ as *mut std::ffi::c_void,
    ];
    launch_megakernel(&func, &stream, &mut args);
    stream.synchronize().expect("sync");

    let mut ids = vec![0i32; k];
    let mut weights = vec![0.0f32; k];
    expert_ids_buf.copy_to_host(&mut ids).expect("read ids");
    expert_weights_buf
        .copy_to_host(&mut weights)
        .expect("read weights");

    println!("Expert IDs: {:?}", ids);
    println!("Expert weights: {:?}", weights);

    // Expert 2 should have highest score (highest gate values)
    assert_eq!(ids[0], 2, "expert 2 should be top-1");
    assert_eq!(ids[1], 5, "expert 5 should be top-2");
    assert_eq!(ids[2], 7, "expert 7 should be top-3");

    // Weights should sum to ~1.0 (softmax)
    let sum: f32 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "weights should sum to 1.0, got {sum}"
    );
    assert!(
        weights[0] > weights[1],
        "top expert should have highest weight"
    );
    println!("Weights sum: {sum:.4}");
}
