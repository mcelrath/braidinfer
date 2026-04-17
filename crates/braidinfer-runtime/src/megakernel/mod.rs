use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use std::ffi::c_void;

use crate::model::ModelConfig;
use crate::trace::TraceWriter;

/// Tokens per paged KV chunk — must match compile_attention_layer_paged.
pub const CHUNK_TOKENS: usize = 64;

// Opcode constants — auto-generated from kernels/opcodes.h (single source of truth)
include!(concat!(env!("BRAIDINFER_KERNEL_DIR"), "/opcodes.rs"));

/// Shared memory bytes per block for tiled-LDS linear_proj ops: (8 + 7680 + 256) * 4.
/// Must match SHARED_LPROJ_TOTAL in kernels/megakernel_ops.hip.
pub(crate) const SHARED_LPROJ_TOTAL: u32 = 31776;

pub(crate) const INST_SIZE: usize = 18; // 18 u64s per instruction = 144 bytes
// RDNA3 7900XTX: 96 CUs grouped into 48 WGPs. hipDeviceAttributeMultiprocessorCount=48 (WGPs).
// Cooperative kernel max blocks = blocks_per_sm * WGPs = blocks_per_sm * 48.
pub(crate) const NUM_CUS: u32 = 48;

/// A single instruction for the megakernel program.
#[derive(Clone)]
pub(crate) struct Instruction {
    pub(crate) words: [u64; INST_SIZE],
}

impl Instruction {
    pub(crate) fn new(opcode: u32, grid_x: u32) -> Self {
        let mut words = [0u64; INST_SIZE];
        words[0] = (opcode as u64) | ((grid_x as u64) << 32);
        Instruction { words }
    }

}

/// Set opcode + weight pointer for a linear projection instruction.
/// For bf16: keeps OP_LINEAR_PROJ, sets bf16 pointer.
/// For packed: switches opcode to OP_LINEAR_PROJ_RNF4/PCG32, sets u8 pointer.
pub struct MegakernelProgram {
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) device_program: DeviceBuffer<u64>,
    pub(crate) module: Module,
    pub(crate) num_blocks: u32,
    pub(crate) shared_mem: u32,
    pub(crate) device: DeviceId,
    // Indices of instructions that need per-step updates
    pub(crate) embedding_inst_idx: usize,
    pub(crate) gqa_attn_inst_indices: Vec<usize>, // seq_len changes each step
    pub(crate) kv_write_indices: Vec<Vec<(usize, usize)>>, // per attn layer, per kv_head: (k_copy_idx, v_copy_idx)
    // Base KV cache pointers (position=0) for computing per-step write offsets
    pub(crate) kv_base_ptrs: Vec<(u64, u64)>, // (k_base, v_base) per attention layer
    // KV head geometry for [H,T,D] layout pointer computation
    pub(crate) num_kv_heads_attn: usize, // nkh for attention layers
    pub(crate) head_dim_attn: usize,     // hd for attention layers
    // mRoPE position_ids device pointer (3 i32s: temporal, height, width)
    pub(crate) position_ids_dev_ptr: u64,
    // Bounds check
    pub(crate) max_seq_len: u32,
    // Paged KV cache support
    pub(crate) paged: bool,
    pub(crate) page_table: Option<DeviceBuffer<u64>>, // array of chunk base pointers, uploaded per step
    pub(crate) position_table: Option<DeviceBuffer<i32>>, // position per token, uploaded per step
    pub(crate) attn_paged_inst_indices: Vec<usize>,   // indices of OP_ATTN_PAGED instructions
    pub(crate) attn_quant_inst_indices: Vec<usize>, // indices of OP_ATTN_PAGED_Q instructions (quantized KV)
    pub(crate) last_page_table_len: usize,          // track when a new chunk was added
    // kv_stride for paged KV write offset computation (nkh * hd)
    pub(crate) kv_stride_paged: usize,
    // Quantized KV support
    pub(crate) quant_scratch: Option<DeviceBuffer<f32>>, // partial state: [nqh × (2+hd)] per attn layer
    pub(crate) quant_page_table: Option<DeviceBuffer<u64>>, // page table for sealed quantized chunks
    pub(crate) last_quant_page_table_len: usize,
    pub quantized_kv: bool, // whether this program uses quantized KV
    // Dump mode: per-instruction activation capture
    pub(crate) dump_buffer: Option<DeviceBuffer<u8>>, // slot data
    pub(crate) dump_counter: Option<DeviceBuffer<i32>>, // atomic slot counter
    pub(crate) dump_capacity: i32,
    /// (instruction_idx, layer_idx) for each OP_BARRIER in the program.
    /// CPU dispatch loop uses layer_idx to look up DistributedMoeWeights.
    pub(crate) barrier_layer_map: Vec<(usize, usize)>,
    pub(crate) _mrope_inst_indices: Vec<usize>,
    // Multi-GPU distributed QKV projection boundaries (only used when multi_gpu=true).
    // For each attention layer: (rmsnorm_idx, output_gate_idx) where:
    //   rmsnorm_idx  = index of the RMSNorm instruction (flush + dispatch QKV/GQA after this)
    //   output_gate_idx = index of OP_OUTPUT_GATE or O-proj (resume megakernel here after dispatch)
    pub(crate) multi_gpu_attn_boundaries: Vec<(usize, usize)>,
    // Prefill-specific caching: indices for per-chunk patching
    pub(crate) prefill_embedding_start: usize,   // first of N consecutive EmbeddingInst
    pub(crate) prefill_kv_entries: Vec<PrefillKvEntry>, // KV D2dCopy instructions to patch per chunk
    pub(crate) prefill_attn_inst_indices: Vec<usize>,   // AttnPrefillInst indices, one per attn layer
    pub(crate) prefill_n: usize,                 // chunk size template was compiled for
    // Prevent Send — contains raw GPU device pointers as u64
    pub(crate) _not_send: std::marker::PhantomData<*mut ()>,
}

/// One KV-write D2dCopy pair (K and V) for prefill chunk patching.
pub(crate) struct PrefillKvEntry {
    pub k_inst_idx: usize,
    pub v_inst_idx: usize,
    pub t: usize,           // token offset within chunk (0..n)
    pub h: usize,           // KV head index
    pub layer_kv_idx: usize, // index into kv_base_ptrs
}

/// Activation buffers sized for N-token prefill chunks.
/// All buffers are [batch × dim] where batch = chunk_tokens.
pub struct PrefillBuffers {
    pub hidden: DeviceBuffer<f32>,  // [N × hidden_size] — main hidden state
    pub normed: DeviceBuffer<f32>,  // [N × hidden_size]
    pub qkv: DeviceBuffer<f32>,     // [N × conv_dim] (6144 for Qwen3.5)
    pub a_proj: DeviceBuffer<f32>,  // [N × num_heads]
    pub b_proj: DeviceBuffer<f32>,  // [N × num_heads]
    pub z_proj: DeviceBuffer<f32>,  // [N × num_heads * value_dim]
    pub ffn_act: DeviceBuffer<f32>, // [N × intermediate_size]
    pub residual: DeviceBuffer<f32>, // [N × hidden_size]
    pub position_ids: DeviceBuffer<i32>, // [N × 3] — mRoPE positions per token
    // Attention layer intermediates
    pub q_gate_attn: DeviceBuffer<f32>, // [N × nqh × hd × 2]
    pub q_attn: DeviceBuffer<f32>,      // [N × nqh × hd]
    pub k_attn: DeviceBuffer<f32>,      // [N × nkh × hd]
    pub v_attn: DeviceBuffer<f32>,      // [N × nkh × hd]
    pub gate_attn: DeviceBuffer<f32>,   // [N × nqh × hd]
    pub attn_out: DeviceBuffer<f32>,    // [N × nqh × hd]
    pub gated_out: DeviceBuffer<f32>,   // [N × nqh × hd]
    pub out_proj: DeviceBuffer<f32>,    // [N × hidden_size]
    // Single-token scratch for quantized FFN unfused path (Q4 prefill)
    pub ffn_gate_scratch: DeviceBuffer<f32>, // [intermediate_size]
    pub ffn_up_scratch: DeviceBuffer<f32>,   // [intermediate_size]
    pub ffn_down_scratch: DeviceBuffer<f32>, // [hidden_size]
}

impl PrefillBuffers {
    pub fn alloc(device: DeviceId, cfg: &ModelConfig, chunk_tokens: usize) -> HipResult<Self> {
        let n = chunk_tokens;
        let hs = cfg.hidden_size;
        let nh = cfg.linear_num_heads;
        let nvh = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let conv_dim = nh * kd * 2 + nvh * vd;
        let is = cfg.intermediate_size;
        let nqh = cfg.num_q_heads;
        let nkh = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        Ok(PrefillBuffers {
            hidden: DeviceBuffer::alloc(device, n * hs)?,
            normed: DeviceBuffer::alloc(device, n * hs)?,
            qkv: DeviceBuffer::alloc(device, n * conv_dim)?,
            a_proj: DeviceBuffer::alloc(device, n * nh)?,
            b_proj: DeviceBuffer::alloc(device, n * nh)?,
            z_proj: DeviceBuffer::alloc(device, n * nvh * vd)?,
            ffn_act: DeviceBuffer::alloc(device, n * is)?,
            residual: DeviceBuffer::alloc(device, n * hs)?,
            position_ids: DeviceBuffer::alloc(device, n * 3)?,
            q_gate_attn: DeviceBuffer::alloc(
                device,
                n * nqh * hd * if cfg.has_output_gate { 2 } else { 1 },
            )?,
            q_attn: DeviceBuffer::alloc(device, n * nqh * hd)?,
            k_attn: DeviceBuffer::alloc(device, n * nkh * hd)?,
            v_attn: DeviceBuffer::alloc(device, n * nkh * hd)?,
            gate_attn: DeviceBuffer::alloc(device, n * nqh * hd)?,
            attn_out: DeviceBuffer::alloc(device, n * nqh * hd)?,
            gated_out: DeviceBuffer::alloc(device, n * nqh * hd)?,
            out_proj: DeviceBuffer::alloc(device, n * hs)?,
            ffn_gate_scratch: DeviceBuffer::alloc(device, is)?,
            ffn_up_scratch: DeviceBuffer::alloc(device, is)?,
            ffn_down_scratch: DeviceBuffer::alloc(device, hs)?,
        })
    }
}

pub(crate) mod instructions;
pub(crate) use instructions::*;

mod compile_common;
mod compile_attention;
mod compile_layers;
mod compile_moe;
mod megakernel_compile;
mod megakernel_run;

impl MegakernelProgram {
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }
    pub fn block_count(&self) -> u32 {
        self.num_blocks
    }

    /// Enable per-instruction dump mode. Allocates GPU buffer and prepends
    /// an OP_NOP header instruction encoding dump pointers (words[1-3]).
    /// This avoids adding kernel args which would increase register pressure
    /// and reduce the cooperative launch block limit.
    pub fn enable_dump(&mut self, max_slots: i32) -> HipResult<()> {
        const DUMP_SLOT_BYTES: usize = 16 + 8192 * 4;
        let buf_size = max_slots as usize * DUMP_SLOT_BYTES;
        self.dump_buffer = Some(DeviceBuffer::<u8>::alloc(self.device, buf_size)?);
        let mut counter = DeviceBuffer::<i32>::alloc(self.device, 1)?;
        counter.copy_from_host(&[0i32])?;
        self.dump_counter = Some(counter);
        self.dump_capacity = max_slots;

        // Prepend OP_NOP header with dump pointers, re-upload program
        let header = NopInst {
            opcode_gridx: make_opcode_gridx(OP_NOP, 0),
            dump_buf: self.dump_buffer.as_ref().unwrap().as_ptr(),
            max_slots: max_slots as u64,
            dump_counter: self.dump_counter.as_ref().unwrap().as_ptr(),
            _pad: [0; 14],
        }.into_inst();

        let total_words = (1 + self.instructions.len()) * INST_SIZE;
        let mut flat: Vec<u64> = Vec::with_capacity(total_words);
        flat.extend_from_slice(&header.words);
        for inst in &self.instructions {
            flat.extend_from_slice(&inst.words);
        }
        let mut new_prog = DeviceBuffer::alloc(self.device, total_words)?;
        new_prog.copy_from_host(&flat)?;
        self.device_program = new_prog;

        Ok(())
    }

    /// Read dump results after execute. Returns Vec of (opcode, inst_idx, data).
    pub fn read_dump(&self, stream: &Stream) -> HipResult<Vec<(u32, u32, Vec<f32>)>> {
        const DUMP_SLOT_BYTES: usize = 16 + 8192 * 4;
        let counter = self.dump_counter.as_ref().unwrap();
        stream.synchronize()?;
        let mut count = [0i32];
        counter.copy_to_host(&mut count)?;
        let num_slots = count[0].min(self.dump_capacity) as usize;

        let buf = self.dump_buffer.as_ref().unwrap();
        let total_bytes = num_slots * DUMP_SLOT_BYTES;
        let mut host = vec![0u8; total_bytes];
        if num_slots > 0 {
            unsafe {
                braidinfer_hip::ffi::hipMemcpyAsync(
                    host.as_mut_ptr() as *mut std::ffi::c_void,
                    buf.as_ptr() as *const std::ffi::c_void,
                    total_bytes,
                    braidinfer_hip::ffi::hipMemcpyDeviceToHost,
                    stream.raw(),
                );
            }
            stream.synchronize()?;
        }

        let mut results = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            let slot = &host[i * DUMP_SLOT_BYTES..];
            let opcode = u32::from_le_bytes(slot[0..4].try_into().unwrap());
            let inst_idx = u32::from_le_bytes(slot[4..8].try_into().unwrap());
            let size = u32::from_le_bytes(slot[8..12].try_into().unwrap()) as usize;
            let data: Vec<f32> = (0..size)
                .map(|j| {
                    let off = 16 + j * 4;
                    f32::from_le_bytes(slot[off..off + 4].try_into().unwrap())
                })
                .collect();
            results.push((opcode, inst_idx, data));
        }
        Ok(results)
    }

    pub fn dump_active(&self) -> bool {
        self.dump_buffer.is_some()
    }

    pub fn disable_dump(&mut self) -> HipResult<()> {
        self.dump_buffer = None;
        self.dump_counter = None;
        self.dump_capacity = 0;
        // Rebuild device program without the NOP header
        let total_words = self.instructions.len() * INST_SIZE;
        let mut flat: Vec<u64> = Vec::with_capacity(total_words);
        for inst in &self.instructions {
            flat.extend_from_slice(&inst.words);
        }
        let mut new_prog = DeviceBuffer::alloc(self.device, total_words)?;
        new_prog.copy_from_host(&flat)?;
        self.device_program = new_prog;
        Ok(())
    }

    /// Write dump results to a BTRC trace file compatible with compare_traces.py.
    /// Names are derived from opcode + sequential index (e.g., "inst003_LINEAR_PROJ").
    pub fn write_dump_btrc(&self, stream: &Stream, path: &str) -> HipResult<()> {
        let slots = self.read_dump(stream)?;
        let mut tw = TraceWriter::open(path).expect("failed to open dump trace file");
        for (opcode, inst_idx, data) in &slots {
            let name = format!("inst{:03}_{}", inst_idx, opcode_name(*opcode));
            tw.write_checkpoint(&name, data);
        }
        tw.close().expect("failed to close dump trace file");
        eprintln!(
            "Megakernel dump: {} instructions written to {}",
            slots.len(),
            path
        );
        Ok(())
    }

    /// Execute the megakernel program.
    pub fn execute(&self, stream: &Stream) -> HipResult<()> {
        let func = self.module.get_function("megakernel_f32")?;
        let mut prog_ptr: *const c_void = self.device_program.as_ptr().cast();
        // When dump is active, device_program has an extra OP_NOP header instruction
        let extra = if self.dump_buffer.is_some() { 1 } else { 0 };
        let mut num_inst = (self.instructions.len() + extra) as i32;

        let mut args: [*mut c_void; 2] = [
            std::ptr::addr_of_mut!(prog_ptr).cast(),
            std::ptr::addr_of_mut!(num_inst).cast(),
        ];

        func.launch_cooperative(
            (self.num_blocks, 1, 1),
            (256, 1, 1),
            self.shared_mem,
            stream,
            &mut args,
        )
    }
}

fn opcode_name(op: u32) -> &'static str {
    match op {
        OP_NOP => "NOP",
        OP_RMSNORM => "RMSNORM",
        OP_LINEAR_PROJ => "LINEAR_PROJ",
        OP_CONV1D => "CONV1D",
        OP_GDN_GATE => "GDN_GATE",
        OP_GDN_RECUR => "GDN_RECUR",
        OP_RMSNORM_GATE => "RMSNORM_GATE",
        OP_RESIDUAL_ADD => "RESIDUAL_ADD",
        OP_QK_NORM => "QK_NORM",
        OP_MROPE => "MROPE",
        OP_GQA_ATTN => "GQA_ATTN",
        OP_OUTPUT_GATE => "OUTPUT_GATE",
        OP_FFN_GATE_UP => "FFN_GATE_UP",
        OP_FFN_DOWN_RES => "FFN_DOWN_RES",
        OP_EMBEDDING => "EMBEDDING",
        OP_LM_HEAD => "LM_HEAD",
        OP_HALT => "HALT",
        OP_D2D_COPY => "D2D_COPY",
        OP_ATTN_PAGED => "ATTN_PAGED",
        OP_ATTN_PREFILL => "ATTN_PREFILL",
        OP_DEINTERLEAVE => "DEINTERLEAVE",
        OP_KV_QUANTIZE => "KV_QUANTIZE",
        OP_ATTN_PAGED_Q => "ATTN_PAGED_Q",
        OP_MOE_GATE => "MOE_GATE",
        OP_MOE_FFN => "MOE_FFN",
        OP_LINEAR_PROJ_RNF4 => "LINEAR_PROJ_RNF4",
        OP_LINEAR_PROJ_PCG32 => "LINEAR_PROJ_PCG32",
        OP_RMSNORM_WX => "RMSNORM_WX",
        OP_SILU_MUL => "SILU_MUL",
        OP_SSM_UPDATE => "SSM_UPDATE",
        OP_FFN_GATE_UP_RNF4 => "FFN_GATE_UP_RNF4",
        OP_FFN_DOWN_RES_RNF4 => "FFN_DOWN_RES_RNF4",
        OP_SIGMOID_WEIGHTED_ADD => "SIGMOID_WEIGHTED_ADD",
        _ => "UNKNOWN",
    }
}
