use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::HipResult;
use std::ffi::c_void;

use crate::model::{
    ActivationBuffers, AttentionLayerWeights, GdnLayerWeights, KvCache, GdnState,
    LayerWeights, ModelConfig, Qwen35Model,
};
use crate::paged_kv::{PageAllocator, SequenceState};

/// Tokens per paged KV chunk — must match compile_attention_layer_paged.
pub const CHUNK_TOKENS: usize = 64;

// Opcode constants — must match megakernel.hip
const OP_NOP: u32 = 0;
const OP_RMSNORM: u32 = 1;
const OP_LINEAR_PROJ: u32 = 2;
const OP_CONV1D: u32 = 3;
const OP_GDN_GATE: u32 = 4;
const OP_GDN_RECUR: u32 = 5;
const OP_RMSNORM_GATE: u32 = 6;
const OP_RESIDUAL_ADD: u32 = 7;
const OP_QK_NORM: u32 = 8;
const OP_MROPE: u32 = 9;
const OP_GQA_ATTN: u32 = 10;
const OP_OUTPUT_GATE: u32 = 11;
const OP_FFN_GATE_UP: u32 = 12;
const OP_FFN_DOWN_RES: u32 = 13;
const OP_EMBEDDING: u32 = 14;
const OP_LM_HEAD: u32 = 15;
const OP_HALT: u32 = 16;
const OP_D2D_COPY: u32 = 17;
const OP_ATTN_PAGED: u32 = 18;
const OP_ATTN_PREFILL: u32 = 19;
const OP_DEINTERLEAVE: u32 = 20;
const OP_KV_QUANTIZE: u32 = 21;
const OP_ATTN_PAGED_Q: u32 = 22;

const FLAG_NO_SYNC: u32 = 0x80000000; // bit 31: skip grid.sync() after this instruction

const INST_SIZE: usize = 16; // 16 u64s per instruction = 128 bytes
const NUM_CUS: u32 = 96;

/// A single instruction for the megakernel program.
#[derive(Clone)]
struct Instruction {
    words: [u64; INST_SIZE],
}

impl Instruction {
    fn new(opcode: u32, grid_x: u32) -> Self {
        let mut words = [0u64; INST_SIZE];
        words[0] = (opcode as u64) | ((grid_x as u64) << 32);
        Instruction { words }
    }

    fn set_ptr<T>(&mut self, idx: usize, ptr: *const T) {
        self.words[idx] = ptr as u64;
    }

    /// Set a slot to a GPU output pointer. Uses *const T because DeviceBuffer
    /// pointers are stable addresses — the GPU writes through them regardless
    /// of Rust borrow state. Named distinctly from set_ptr for documentation.
    fn set_output_ptr<T>(&mut self, idx: usize, ptr: *const T) {
        self.words[idx] = ptr as u64;
    }

    fn set_int(&mut self, idx: usize, val: i32) {
        self.words[idx] = val as u64;
    }

    fn set_float(&mut self, idx: usize, val: f32) {
        self.words[idx] = val.to_bits() as u64;
    }

    fn set_no_sync(&mut self) {
        self.words[0] |= FLAG_NO_SYNC as u64;
    }
}

/// Pre-compiled program for the megakernel.
pub struct MegakernelProgram {
    instructions: Vec<Instruction>,
    device_program: DeviceBuffer<u64>,
    module: Module,
    num_blocks: u32,
    device: DeviceId,
    // Indices of instructions that need per-step updates
    embedding_inst_idx: usize,
    mrope_inst_indices: Vec<usize>,    // one per attention layer
    gqa_attn_inst_indices: Vec<usize>, // seq_len changes each step
    kv_write_indices: Vec<Vec<(usize, usize)>>, // per attn layer, per kv_head: (k_copy_idx, v_copy_idx)
    // Base KV cache pointers (position=0) for computing per-step write offsets
    kv_base_ptrs: Vec<(u64, u64)>, // (k_base, v_base) per attention layer
    // KV head geometry for [H,T,D] layout pointer computation
    num_kv_heads_attn: usize, // nkh for attention layers
    head_dim_attn: usize,     // hd for attention layers
    // mRoPE position_ids device pointer (3 i32s: temporal, height, width)
    position_ids_dev_ptr: u64,
    // Bounds check
    max_seq_len: u32,
    // Paged KV cache support
    paged: bool,
    page_table: Option<DeviceBuffer<u64>>,     // array of chunk base pointers, uploaded per step
    position_table: Option<DeviceBuffer<i32>>, // position per token, uploaded per step
    attn_paged_inst_indices: Vec<usize>,        // indices of OP_ATTN_PAGED instructions
    last_page_table_len: usize,                 // track when a new chunk was added
    // kv_stride for paged KV write offset computation (nkh * hd)
    kv_stride_paged: usize,
    // Prevent Send — contains raw GPU device pointers as u64
    _not_send: std::marker::PhantomData<*mut ()>,
}

/// Activation buffers sized for N-token prefill chunks.
/// All buffers are [batch × dim] where batch = chunk_tokens.
pub struct PrefillBuffers {
    pub hidden: DeviceBuffer<f32>,       // [N × hidden_size] — main hidden state
    pub normed: DeviceBuffer<f32>,       // [N × hidden_size]
    pub qkv: DeviceBuffer<f32>,          // [N × conv_dim] (6144 for Qwen3.5)
    pub a_proj: DeviceBuffer<f32>,       // [N × num_heads]
    pub b_proj: DeviceBuffer<f32>,       // [N × num_heads]
    pub z_proj: DeviceBuffer<f32>,       // [N × num_heads * value_dim]
    pub ffn_act: DeviceBuffer<f32>,      // [N × intermediate_size]
    pub residual: DeviceBuffer<f32>,     // [N × hidden_size]
    pub position_ids: DeviceBuffer<i32>, // [N × 3] — mRoPE positions per token
    // Attention layer intermediates
    pub q_gate_attn: DeviceBuffer<f32>,  // [N × nqh × hd × 2]
    pub q_attn: DeviceBuffer<f32>,       // [N × nqh × hd]
    pub k_attn: DeviceBuffer<f32>,       // [N × nkh × hd]
    pub v_attn: DeviceBuffer<f32>,       // [N × nkh × hd]
    pub gate_attn: DeviceBuffer<f32>,    // [N × nqh × hd]
    pub attn_out: DeviceBuffer<f32>,     // [N × nqh × hd]
    pub gated_out: DeviceBuffer<f32>,    // [N × nqh × hd]
    pub out_proj: DeviceBuffer<f32>,     // [N × hidden_size]
}

impl PrefillBuffers {
    pub fn alloc(device: DeviceId, cfg: &ModelConfig, chunk_tokens: usize) -> HipResult<Self> {
        let n = chunk_tokens;
        let hs = cfg.hidden_size;
        let nh = cfg.linear_num_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let conv_dim = nh * kd * 2 + nh * vd;
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
            z_proj: DeviceBuffer::alloc(device, n * nh * vd)?,
            ffn_act: DeviceBuffer::alloc(device, n * is)?,
            residual: DeviceBuffer::alloc(device, n * hs)?,
            position_ids: DeviceBuffer::alloc(device, n * 3)?,
            q_gate_attn: DeviceBuffer::alloc(device, n * nqh * hd * 2)?,
            q_attn: DeviceBuffer::alloc(device, n * nqh * hd)?,
            k_attn: DeviceBuffer::alloc(device, n * nkh * hd)?,
            v_attn: DeviceBuffer::alloc(device, n * nkh * hd)?,
            gate_attn: DeviceBuffer::alloc(device, n * nqh * hd)?,
            attn_out: DeviceBuffer::alloc(device, n * nqh * hd)?,
            gated_out: DeviceBuffer::alloc(device, n * nqh * hd)?,
            out_proj: DeviceBuffer::alloc(device, n * hs)?,
        })
    }
}

fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

/// Discriminates the variant parts (KV write + attention op) of an attention layer.
enum AttentionVariant<'a> {
    /// Flat (non-paged) decode: GQA attention, KV written after mRoPE.
    FlatKv { kv_cache: &'a KvCache },
    /// Paged decode: OP_ATTN_PAGED, KV written BEFORE mRoPE.
    PagedKv { kv_cache: &'a KvCache, attn_layer_index: usize },
    /// Prefill (N tokens): OP_ATTN_PREFILL, bulk KV write after mRoPE.
    Prefill { kv_cache: &'a KvCache, start_pos: u32 },
}

impl MegakernelProgram {
    pub fn instruction_count(&self) -> usize { self.instructions.len() }
    pub fn block_count(&self) -> u32 { self.num_blocks }

    pub fn compile(model: &Qwen35Model) -> HipResult<Self> {
        Self::compile_inner(model, false)
    }

    pub fn compile_paged(model: &Qwen35Model) -> HipResult<Self> {
        Self::compile_inner(model, true)
    }

    fn compile_inner(model: &Qwen35Model, paged: bool) -> HipResult<Self> {
        let cfg = &model.config;
        let device = model.device;
        let act = &model.activations;

        // Guard: only GDN recurrent layers are supported by the megakernel
        if matches!(cfg.recurrent_kind, crate::model::RecurrentLayerKind::Mamba2 { .. }) {
            panic!("Mamba2 recurrent layers not yet supported by megakernel");
        }

        let module = Module::load(device, &crate::kernel::kernel_dir().join("megakernel.hsaco"))?;

        // Note: hipDeviceAttributeCooperativeLaunch (95) returns 0 on ROCm/RDNA3 even though
        // cooperative launch works. Skipping capability check — hipModuleLaunchCooperativeKernel
        // will return an error if unsupported.

        // Query max blocks for cooperative launch
        let func = module.get_function("megakernel_f32")?;
        let blocks_per_sm = func.max_active_blocks_per_sm(256, 256 * 4 * 2)?; // 2KB shared for gdn_recurrent
        let num_blocks = (blocks_per_sm as u32 * NUM_CUS).min(384); // conservative

        let mut instructions: Vec<Instruction> = Vec::new();
        let mut mrope_inst_indices = Vec::new();
        let mut gqa_attn_inst_indices = Vec::new();
        let mut kv_write_indices = Vec::new();
        let mut kv_base_ptrs = Vec::new();

        let hs = cfg.hidden_size;
        let nh_gdn = cfg.linear_num_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let conv_dim = nh_gdn * kd * 2 + nh_gdn * vd; // 6144
        let ck = cfg.linear_conv_kernel_dim;
        let nqh = cfg.num_q_heads;
        let nkh = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let is = cfg.intermediate_size;
        let vs = cfg.vocab_size;
        let eps = cfg.rms_norm_eps;

        // Embedding (token_id placeholder = 0, updated per step)
        let embedding_inst_idx = instructions.len();
        {
            let mut inst = Instruction::new(OP_EMBEDDING, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, act.hidden.as_ptr());
            inst.set_ptr(2, model.embed_weight.as_ptr());
            inst.set_int(3, 0); // token_id — updated per step
            inst.set_int(4, hs as i32);
            instructions.push(inst);
        }

        // Layers
        let mut attn_paged_inst_indices = Vec::new();
        let mut attn_layer_count = 0usize;

        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        for layer_i in 0..cfg.num_layers {
            if cfg.layer_is_attention[layer_i] {
                if paged {
                    Self::compile_attention_layer_paged(
                        cfg, &model.layers[layer_i], act,
                        &model.kv_caches[kv_idx],
                        attn_layer_count,
                        &mut instructions, &mut mrope_inst_indices,
                        &mut kv_write_indices, &mut kv_base_ptrs,
                        &mut attn_paged_inst_indices,
                    );
                } else {
                    Self::compile_attention_layer(
                        cfg, &model.layers[layer_i], act,
                        &model.kv_caches[kv_idx],
                        &mut instructions, &mut mrope_inst_indices,
                        &mut gqa_attn_inst_indices, &mut kv_write_indices,
                        &mut kv_base_ptrs,
                    );
                }
                attn_layer_count += 1;
                kv_idx += 1;
            } else {
                Self::compile_gdn_layer(
                    cfg, &model.layers[layer_i], act,
                    &model.gdn_conv_states[gdn_idx],
                    &model.gdn_states[gdn_idx],
                    &mut instructions,
                );
                gdn_idx += 1;
            }

            // FFN (same for both layer types)
            Self::compile_ffn(cfg, &model.layers[layer_i], act, &mut instructions);
        }

        // Final RMSNorm: copy hidden→normed, then norm normed→hidden
        {
            let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, act.normed.as_ptr());
            inst.set_ptr(2, act.hidden.as_ptr());
            inst.set_int(3, hs as i32);
            instructions.push(inst);
        }
        {
            let mut inst = Instruction::new(OP_RMSNORM, 1);
            inst.set_output_ptr(1, act.hidden.as_ptr());
            inst.set_ptr(2, act.normed.as_ptr());
            inst.set_ptr(3, model.final_norm_weight.as_ptr());
            inst.set_int(4, hs as i32);
            inst.set_float(5, eps);
            instructions.push(inst);
        }

        // LM head (= linear_proj with vocab_size output rows)
        {
            let mut inst = Instruction::new(OP_LINEAR_PROJ, vs as u32);
            inst.set_output_ptr(1, act.logits.as_ptr());
            inst.set_ptr(2, model.embed_weight.as_ptr()); // weight-tied
            inst.set_ptr(3, act.hidden.as_ptr());
            inst.set_int(4, vs as i32);
            inst.set_int(5, hs as i32);
            instructions.push(inst);
        }

        // HALT
        instructions.push(Instruction::new(OP_HALT, 0));

        // Upload program to device
        let total_words = instructions.len() * INST_SIZE;
        let mut flat: Vec<u64> = Vec::with_capacity(total_words);
        for inst in &instructions {
            flat.extend_from_slice(&inst.words);
        }
        let mut device_program = DeviceBuffer::alloc(device, total_words)?;
        device_program.copy_from_host(&flat)?;

        Ok(MegakernelProgram {
            instructions,
            device_program,
            module,
            num_blocks,
            device,
            embedding_inst_idx,
            mrope_inst_indices,
            gqa_attn_inst_indices,
            kv_write_indices,
            kv_base_ptrs,
            position_ids_dev_ptr: act.position_ids.as_ptr() as u64,
            max_seq_len: cfg.max_seq_len as u32,
            paged,
            page_table: None,
            position_table: None,
            attn_paged_inst_indices,
            last_page_table_len: 0,
            kv_stride_paged: cfg.num_kv_heads * cfg.head_dim,
            num_kv_heads_attn: cfg.num_kv_heads,
            head_dim_attn: cfg.head_dim,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Compile a one-shot prefill program for `tokens` starting at `start_pos`.
    /// GDN layers use batched projections (weight reuse across all N tokens).
    /// Attention layers run sequentially (one token at a time).
    /// Returns a program + list of per-token attention instruction indices for paged KV updates.
    pub fn compile_prefill(
        model: &Qwen35Model,
        tokens: &[u32],
        start_pos: u32,
        prefill_bufs: &mut PrefillBuffers,
    ) -> HipResult<Self> {
        let n = tokens.len();
        assert!(n > 0 && n <= CHUNK_TOKENS);
        let cfg = &model.config;
        let device = model.device;
        let act = &model.activations;

        let module = Module::load(device, &crate::kernel::kernel_dir().join("megakernel.hsaco"))?;
        let func = module.get_function("megakernel_f32")?;
        let blocks_per_sm = func.max_active_blocks_per_sm(256, 256 * 4 * 2)?;
        let num_blocks = (blocks_per_sm as u32 * NUM_CUS).min(384);

        let mut instructions: Vec<Instruction> = Vec::new();

        let hs = cfg.hidden_size;
        let nh_gdn = cfg.linear_num_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let conv_dim = nh_gdn * kd * 2 + nh_gdn * vd;
        let ck = cfg.linear_conv_kernel_dim;
        let nqh = cfg.num_q_heads;
        let nkh = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let is = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;

        // === Embedding: N lookups into prefill_bufs.hidden ===
        let embedding_inst_idx = instructions.len();
        for t in 0..n {
            let mut inst = Instruction::new(OP_EMBEDDING, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, unsafe { prefill_bufs.hidden.as_ptr().add(t * hs) });
            inst.set_ptr(2, model.embed_weight.as_ptr());
            inst.set_int(3, tokens[t] as i32);
            inst.set_int(4, hs as i32);
            if t + 1 < n { inst.set_no_sync(); }
            instructions.push(inst);
        }

        // Upload positions into prefill_bufs.position_ids: [N × 3] i32
        {
            let mut pos_data = vec![0i32; n * 3];
            for t in 0..n {
                let pos = (start_pos + t as u32) as i32;
                pos_data[t * 3] = pos;
                pos_data[t * 3 + 1] = pos;
                pos_data[t * 3 + 2] = pos;
            }
            prefill_bufs.position_ids.copy_from_host(&pos_data)?;
        }

        // === Layers ===
        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        let mut attn_layer_count = 0usize;

        for layer_i in 0..cfg.num_layers {
            if cfg.layer_is_attention[layer_i] {
                let w = match &model.layers[layer_i] {
                    LayerWeights::Attention(w) => w,
                    _ => panic!("expected attention layer"),
                };
                let kv_cache = &model.kv_caches[kv_idx];

                Self::emit_attention_layer(
                    cfg, w, act,
                    Some((prefill_bufs, n)),
                    &AttentionVariant::Prefill { kv_cache, start_pos },
                    &mut instructions,
                    &mut Vec::new(), &mut Vec::new(), &mut Vec::new(), &mut Vec::new(),
                    &mut Vec::new(),
                );

                // Batched FFN
                Self::compile_ffn_batched(cfg, &model.layers[layer_i], prefill_bufs, n, &mut instructions);
                attn_layer_count += 1;
                kv_idx += 1;
            } else {
                // GDN layers: batched projections + sequential recurrence
                let w = match &model.layers[layer_i] {
                    LayerWeights::Gdn(w) => w,
                    _ => panic!("expected GDN layer"),
                };
                let conv_state = &model.gdn_conv_states[gdn_idx];
                let gdn_state = &model.gdn_states[gdn_idx];

                // --- Batched projections ---
                // RMSNorm (grid_x=N, one block per token)
                {
                    let mut inst = Instruction::new(OP_RMSNORM, n as u32);
                    inst.set_output_ptr(1, prefill_bufs.normed.as_ptr());
                    inst.set_ptr(2, prefill_bufs.hidden.as_ptr());
                    inst.set_ptr(3, w.input_norm.as_ptr());
                    inst.set_int(4, hs as i32);
                    inst.set_float(5, eps);
                    instructions.push(inst);
                }

                // QKV projection (batch=N)
                {
                    let mut inst = Instruction::new(OP_LINEAR_PROJ, conv_dim as u32);
                    inst.set_output_ptr(1, prefill_bufs.qkv.as_ptr());
                    inst.set_ptr(2, w.w_qkv.as_ptr());
                    inst.set_ptr(3, prefill_bufs.normed.as_ptr());
                    inst.set_int(4, conv_dim as i32);
                    inst.set_int(5, hs as i32);
                    inst.set_int(6, n as i32);
                    inst.set_no_sync();
                    instructions.push(inst);
                }

                // a projection (batch=N)
                {
                    let mut inst = Instruction::new(OP_LINEAR_PROJ, nh_gdn as u32);
                    inst.set_output_ptr(1, prefill_bufs.a_proj.as_ptr());
                    inst.set_ptr(2, w.w_a.as_ptr());
                    inst.set_ptr(3, prefill_bufs.normed.as_ptr());
                    inst.set_int(4, nh_gdn as i32);
                    inst.set_int(5, hs as i32);
                    inst.set_int(6, n as i32);
                    inst.set_no_sync();
                    instructions.push(inst);
                }

                // b projection (batch=N)
                {
                    let mut inst = Instruction::new(OP_LINEAR_PROJ, nh_gdn as u32);
                    inst.set_output_ptr(1, prefill_bufs.b_proj.as_ptr());
                    inst.set_ptr(2, w.w_b.as_ptr());
                    inst.set_ptr(3, prefill_bufs.normed.as_ptr());
                    inst.set_int(4, nh_gdn as i32);
                    inst.set_int(5, hs as i32);
                    inst.set_int(6, n as i32);
                    inst.set_no_sync();
                    instructions.push(inst);
                }

                // z projection (batch=N) — SYNC before sequential part
                {
                    let mut inst = Instruction::new(OP_LINEAR_PROJ, (nh_gdn * vd) as u32);
                    inst.set_output_ptr(1, prefill_bufs.z_proj.as_ptr());
                    inst.set_ptr(2, w.w_z.as_ptr());
                    inst.set_ptr(3, prefill_bufs.normed.as_ptr());
                    inst.set_int(4, (nh_gdn * vd) as i32);
                    inst.set_int(5, hs as i32);
                    inst.set_int(6, n as i32);
                    instructions.push(inst);
                }

                // --- Sequential per-token: conv1d, gate, recurrence, norm, output, residual ---
                let q_dim = nh_gdn * kd;
                let k_dim = nh_gdn * kd;
                let v_dim = nh_gdn * vd;

                for t in 0..n {
                    // Conv1d on Q (from batched qkv[t])
                    {
                        let mut inst = Instruction::new(OP_CONV1D, div_ceil(q_dim as u32, 256));
                        inst.set_output_ptr(1, conv_state.as_ptr());
                        inst.set_ptr(2, unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim) });
                        inst.set_ptr(3, w.conv1d_weight_q.as_ptr());
                        inst.set_output_ptr(4, act.q_gdn.as_ptr());
                        inst.set_int(5, q_dim as i32);
                        inst.set_int(6, ck as i32);
                        inst.set_no_sync();
                        instructions.push(inst);
                    }
                    // Conv1d on K
                    {
                        let mut inst = Instruction::new(OP_CONV1D, div_ceil(k_dim as u32, 256));
                        inst.set_output_ptr(1, unsafe { conv_state.as_ptr().add(q_dim * (ck - 1)) });
                        inst.set_ptr(2, unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim + q_dim) });
                        inst.set_ptr(3, w.conv1d_weight_k.as_ptr());
                        inst.set_output_ptr(4, act.k_gdn.as_ptr());
                        inst.set_int(5, k_dim as i32);
                        inst.set_int(6, ck as i32);
                        inst.set_no_sync();
                        instructions.push(inst);
                    }
                    // Conv1d on V
                    {
                        let mut inst = Instruction::new(OP_CONV1D, div_ceil(v_dim as u32, 256));
                        inst.set_output_ptr(1, unsafe { conv_state.as_ptr().add((q_dim + k_dim) * (ck - 1)) });
                        inst.set_ptr(2, unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim + q_dim + k_dim) });
                        inst.set_ptr(3, w.conv1d_weight_v.as_ptr());
                        inst.set_output_ptr(4, act.v_gdn.as_ptr());
                        inst.set_int(5, v_dim as i32);
                        inst.set_int(6, ck as i32);
                        instructions.push(inst);
                    }

                    // GDN gate (from batched a_proj[t])
                    {
                        let mut inst = Instruction::new(OP_GDN_GATE, div_ceil(nh_gdn as u32, 256));
                        inst.set_output_ptr(1, act.gate_gdn.as_ptr());
                        inst.set_ptr(2, unsafe { prefill_bufs.a_proj.as_ptr().add(t * nh_gdn) });
                        inst.set_ptr(3, w.a_log.as_ptr());
                        inst.set_ptr(4, w.dt_bias.as_ptr());
                        inst.set_int(5, nh_gdn as i32);
                        instructions.push(inst);
                    }

                    // GDN recurrence
                    {
                        let mut inst = Instruction::new(OP_GDN_RECUR, nh_gdn as u32);
                        inst.set_ptr(1, act.q_gdn.as_ptr());
                        inst.set_ptr(2, act.k_gdn.as_ptr());
                        inst.set_ptr(3, act.v_gdn.as_ptr());
                        inst.set_ptr(4, act.gate_gdn.as_ptr());
                        inst.set_ptr(5, unsafe { prefill_bufs.b_proj.as_ptr().add(t * nh_gdn) });
                        inst.set_output_ptr(6, gdn_state.recurrent.as_ptr());
                        inst.set_output_ptr(7, act.recurrent_out.as_ptr());
                        inst.set_int(8, kd as i32);
                        inst.set_int(9, vd as i32);
                        instructions.push(inst);
                    }

                    // RMSNorm gated (z from batched z_proj[t])
                    {
                        let mut inst = Instruction::new(OP_RMSNORM_GATE, nh_gdn as u32);
                        inst.set_output_ptr(1, act.normed_gated.as_ptr());
                        inst.set_ptr(2, act.recurrent_out.as_ptr());
                        inst.set_ptr(3, unsafe { prefill_bufs.z_proj.as_ptr().add(t * nh_gdn * vd) });
                        inst.set_ptr(4, w.output_norm.as_ptr());
                        inst.set_int(5, nh_gdn as i32);
                        inst.set_int(6, vd as i32);
                        inst.set_float(7, eps);
                        instructions.push(inst);
                    }

                    // Output projection
                    {
                        let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
                        inst.set_output_ptr(1, act.out_proj.as_ptr());
                        inst.set_ptr(2, w.w_out.as_ptr());
                        inst.set_ptr(3, act.normed_gated.as_ptr());
                        inst.set_int(4, hs as i32);
                        inst.set_int(5, (nh_gdn * vd) as i32);
                        instructions.push(inst);
                    }

                    // Residual: hidden[t] = out_proj + hidden[t]
                    {
                        let hidden_t = unsafe { prefill_bufs.hidden.as_ptr().add(t * hs) };
                        let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
                        inst.set_output_ptr(1, hidden_t);
                        inst.set_ptr(2, act.out_proj.as_ptr());
                        inst.set_ptr(3, hidden_t);
                        inst.set_int(4, hs as i32);
                        instructions.push(inst);
                    }
                }

                // --- Batched FFN ---
                let (post_norm, w_gate, w_up, w_down) = (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down);

                // FFN gate+up (batch=N)
                {
                    let mut inst = Instruction::new(OP_FFN_GATE_UP, (is * n) as u32);
                    inst.set_output_ptr(1, prefill_bufs.ffn_act.as_ptr());
                    inst.set_ptr(2, prefill_bufs.hidden.as_ptr());
                    inst.set_ptr(3, post_norm.as_ptr());
                    inst.set_ptr(4, w_gate.as_ptr());
                    inst.set_ptr(5, w_up.as_ptr());
                    inst.set_int(6, hs as i32);
                    inst.set_int(7, is as i32);
                    inst.set_float(8, eps);
                    inst.set_int(9, n as i32);
                    instructions.push(inst);
                }

                // Save residual (N hidden states)
                {
                    let mut inst = Instruction::new(OP_D2D_COPY, div_ceil((n * hs) as u32, 256));
                    inst.set_output_ptr(1, prefill_bufs.residual.as_ptr());
                    inst.set_ptr(2, prefill_bufs.hidden.as_ptr());
                    inst.set_int(3, (n * hs) as i32);
                    instructions.push(inst);
                }

                // FFN down + residual (batch=N)
                {
                    let mut inst = Instruction::new(OP_FFN_DOWN_RES, (hs * n) as u32);
                    inst.set_output_ptr(1, prefill_bufs.hidden.as_ptr());
                    inst.set_ptr(2, prefill_bufs.residual.as_ptr());
                    inst.set_ptr(3, w_down.as_ptr());
                    inst.set_ptr(4, prefill_bufs.ffn_act.as_ptr());
                    inst.set_int(5, hs as i32);
                    inst.set_int(6, is as i32);
                    inst.set_int(7, n as i32);
                    instructions.push(inst);
                }

                gdn_idx += 1;
            }
        }

        // === Final norm + LM head (last token only) ===
        // Copy last token's hidden to act.hidden
        {
            let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, act.hidden.as_ptr());
            inst.set_ptr(2, unsafe { prefill_bufs.hidden.as_ptr().add((n - 1) * hs) });
            inst.set_int(3, hs as i32);
            instructions.push(inst);
        }
        // RMSNorm
        {
            let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, act.normed.as_ptr());
            inst.set_ptr(2, act.hidden.as_ptr());
            inst.set_int(3, hs as i32);
            instructions.push(inst);
        }
        {
            let mut inst = Instruction::new(OP_RMSNORM, 1);
            inst.set_output_ptr(1, act.hidden.as_ptr());
            inst.set_ptr(2, act.normed.as_ptr());
            inst.set_ptr(3, model.final_norm_weight.as_ptr());
            inst.set_int(4, hs as i32);
            inst.set_float(5, eps);
            instructions.push(inst);
        }
        // LM head
        {
            let mut inst = Instruction::new(OP_LINEAR_PROJ, cfg.vocab_size as u32);
            inst.set_output_ptr(1, act.logits.as_ptr());
            inst.set_ptr(2, model.embed_weight.as_ptr());
            inst.set_ptr(3, act.hidden.as_ptr());
            inst.set_int(4, cfg.vocab_size as i32);
            inst.set_int(5, hs as i32);
            instructions.push(inst);
        }
        instructions.push(Instruction::new(OP_HALT, 0));

        // Upload
        let total_words = instructions.len() * INST_SIZE;
        let mut flat: Vec<u64> = Vec::with_capacity(total_words);
        for inst in &instructions {
            flat.extend_from_slice(&inst.words);
        }
        let mut device_program = DeviceBuffer::alloc(device, total_words)?;
        device_program.copy_from_host(&flat)?;

        Ok(MegakernelProgram {
            instructions,
            device_program,
            module,
            num_blocks,
            device,
            embedding_inst_idx,
            mrope_inst_indices: Vec::new(),
            gqa_attn_inst_indices: Vec::new(),
            kv_write_indices: Vec::new(),
            kv_base_ptrs: Vec::new(),
            position_ids_dev_ptr: act.position_ids.as_ptr() as u64,
            max_seq_len: cfg.max_seq_len as u32,
            paged: false,
            page_table: None,
            position_table: None,
            attn_paged_inst_indices: Vec::new(),
            last_page_table_len: 0,
            kv_stride_paged: cfg.num_kv_heads * cfg.head_dim,
            num_kv_heads_attn: cfg.num_kv_heads,
            head_dim_attn: cfg.head_dim,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Shared attention-layer emit helper.
    ///
    /// Emits steps 1–7 (RMSNorm → QKV proj → deinterleave → QK-norm → [KV-write] → mRoPE)
    /// and steps 10–12 (output-gate → out-proj+residual).
    /// Steps 8–9 (KV-write placement and attention op) vary by variant and are inserted
    /// at the appropriate point inside this function.
    ///
    /// Returns the index of the emitted mRoPE instruction.
    #[allow(clippy::too_many_arguments)]
    fn emit_attention_layer(
        cfg: &ModelConfig,
        w: &AttentionLayerWeights,
        act: &ActivationBuffers,
        // Prefill overrides: if Some, use prefill pointers and batch=n; else batch=1 and act.*
        prefill: Option<(&PrefillBuffers, usize)>,
        variant: &AttentionVariant,
        instructions: &mut Vec<Instruction>,
        mrope_indices: &mut Vec<usize>,
        gqa_attn_inst_indices: &mut Vec<usize>,
        kv_write_indices: &mut Vec<Vec<(usize, usize)>>,
        kv_base_ptrs: &mut Vec<(u64, u64)>,
        attn_paged_indices: &mut Vec<usize>,
    ) {
        let hs = cfg.hidden_size;
        let nqh = cfg.num_q_heads;
        let nkh = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let eps = cfg.rms_norm_eps;
        let rd = cfg.rope_dim;
        let n = prefill.as_ref().map_or(1, |&(_, n)| n);

        // Buffer pointers — either prefill or single-token activation buffers
        let (normed_ptr, hidden_ptr, q_gate_attn_ptr, k_attn_ptr, v_attn_ptr,
             q_attn_ptr, gate_attn_ptr, attn_out_ptr, gated_out_ptr,
             out_proj_ptr, position_ids_ptr, ffn_hidden_ptr) =
        if let Some((pb, _)) = &prefill {
            (pb.normed.as_ptr(), pb.hidden.as_ptr(),
             pb.q_gate_attn.as_ptr(), pb.k_attn.as_ptr(), pb.v_attn.as_ptr(),
             pb.q_attn.as_ptr(), pb.gate_attn.as_ptr(),
             pb.attn_out.as_ptr(), pb.gated_out.as_ptr(),
             pb.out_proj.as_ptr(), pb.position_ids.as_ptr(), pb.hidden.as_ptr())
        } else {
            (act.normed.as_ptr(), act.hidden.as_ptr(),
             act.q_gate_attn.as_ptr(), act.k_attn.as_ptr(), act.v_attn.as_ptr(),
             act.q_attn.as_ptr(), act.gate_attn.as_ptr(),
             act.attn_out.as_ptr(), act.gated_out.as_ptr(),
             act.out_proj.as_ptr(), act.position_ids.as_ptr(), act.hidden.as_ptr())
        };

        // 1. RMSNorm
        {
            let mut inst = Instruction::new(OP_RMSNORM, n as u32);
            inst.set_output_ptr(1, normed_ptr);
            inst.set_ptr(2, hidden_ptr);
            inst.set_ptr(3, w.input_norm.as_ptr());
            inst.set_int(4, hs as i32);
            inst.set_float(5, eps);
            instructions.push(inst);
        }

        // 2. Q+gate, K, V projections
        {
            let mut inst = Instruction::new(OP_LINEAR_PROJ, (nqh * hd * 2) as u32);
            inst.set_output_ptr(1, q_gate_attn_ptr);
            inst.set_ptr(2, w.w_q_gate.as_ptr());
            inst.set_ptr(3, normed_ptr);
            inst.set_int(4, (nqh * hd * 2) as i32);
            inst.set_int(5, hs as i32);
            if n > 1 { inst.set_int(6, n as i32); }
            inst.set_no_sync();
            instructions.push(inst);
        }
        {
            let mut inst = Instruction::new(OP_LINEAR_PROJ, (nkh * hd) as u32);
            inst.set_output_ptr(1, k_attn_ptr);
            inst.set_ptr(2, w.w_k.as_ptr());
            inst.set_ptr(3, normed_ptr);
            inst.set_int(4, (nkh * hd) as i32);
            inst.set_int(5, hs as i32);
            if n > 1 { inst.set_int(6, n as i32); }
            inst.set_no_sync();
            instructions.push(inst);
        }
        {
            let mut inst = Instruction::new(OP_LINEAR_PROJ, (nkh * hd) as u32);
            inst.set_output_ptr(1, v_attn_ptr);
            inst.set_ptr(2, w.w_v.as_ptr());
            inst.set_ptr(3, normed_ptr);
            inst.set_int(4, (nkh * hd) as i32);
            inst.set_int(5, hs as i32);
            if n > 1 { inst.set_int(6, n as i32); }
            instructions.push(inst);
        }

        // 3. Deinterleave Q+gate → Q, gate
        if n > 1 {
            // Batched: single OP_DEINTERLEAVE
            let total_elems = n * nqh * hd;
            let mut inst = Instruction::new(OP_DEINTERLEAVE, div_ceil(total_elems as u32, 256));
            inst.set_output_ptr(1, q_attn_ptr);
            inst.set_output_ptr(2, gate_attn_ptr);
            inst.set_ptr(3, q_gate_attn_ptr);
            inst.set_int(4, nqh as i32);
            inst.set_int(5, hd as i32);
            inst.set_int(6, n as i32);
            instructions.push(inst);
        } else {
            // Single-token: per-head D2D_COPY loop
            for h in 0..nqh {
                let src_q_offset = h * hd * 2;
                let src_g_offset = h * hd * 2 + hd;
                let dst_offset = h * hd;
                let is_last = h == nqh - 1;
                let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
                inst.set_output_ptr(1, unsafe { q_attn_ptr.add(dst_offset) });
                inst.set_ptr(2, unsafe { q_gate_attn_ptr.add(src_q_offset) });
                inst.set_int(3, hd as i32);
                inst.set_no_sync();
                instructions.push(inst);
                let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
                inst.set_output_ptr(1, unsafe { gate_attn_ptr.add(dst_offset) });
                inst.set_ptr(2, unsafe { q_gate_attn_ptr.add(src_g_offset) });
                inst.set_int(3, hd as i32);
                if !is_last { inst.set_no_sync(); }
                instructions.push(inst);
            }
        }

        // 4. QK norm
        {
            let mut inst = Instruction::new(OP_QK_NORM, (n * (nqh + nkh)) as u32);
            inst.set_output_ptr(1, q_attn_ptr);
            inst.set_output_ptr(2, k_attn_ptr);
            inst.set_ptr(3, w.q_norm.as_ptr());
            inst.set_ptr(4, w.k_norm.as_ptr());
            inst.set_int(5, nqh as i32);
            inst.set_int(6, nkh as i32);
            inst.set_int(7, hd as i32);
            inst.set_float(8, eps);
            if n > 1 { inst.set_int(9, n as i32); }
            instructions.push(inst);
        }

        // Steps 5 (KV write paged — before mRoPE) or 6 (KV write flat/prefill — after mRoPE)
        // is emitted by the variant block below. We emit mRoPE between them when needed.

        // Variant-specific: KV write placement and attention op
        match variant {
            AttentionVariant::FlatKv { kv_cache } => {
                // mRoPE first (step 5), then KV write (step 6), then GQA (step 7)
                let mrope_idx = instructions.len();
                mrope_indices.push(mrope_idx);
                {
                    let mut inst = Instruction::new(OP_MROPE, (n * (nqh + nkh)) as u32);
                    inst.set_output_ptr(1, q_attn_ptr);
                    inst.set_output_ptr(2, k_attn_ptr);
                    inst.set_ptr(3, act.inv_freq.as_ptr());
                    inst.set_ptr(4, position_ids_ptr);
                    inst.set_int(5, nqh as i32);
                    inst.set_int(6, nkh as i32);
                    inst.set_int(7, hd as i32);
                    inst.set_int(8, rd as i32);
                    inst.set_int(9, cfg.mrope_section[0] as i32);
                    inst.set_int(10, cfg.mrope_section[1] as i32);
                    inst.set_int(11, cfg.mrope_section[2] as i32);
                    inst.set_int(12, n as i32);
                    instructions.push(inst);
                }

                // Per-head D2D_COPY for [H,T,D] layout: each head's cache slot
                // is at base + h * max_seq_len * hd, updated at runtime with position offset.
                let max_sl = cfg.max_seq_len;
                let head_stride = max_sl * hd; // elements between consecutive heads
                {
                    let mut head_indices = Vec::new();
                    for h in 0..nkh {
                        let k_copy_idx = instructions.len();
                        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
                        inst.set_output_ptr(1, unsafe { kv_cache.k.as_ptr().add(h * head_stride) });
                        inst.set_ptr(2, unsafe { k_attn_ptr.add(h * hd) });
                        inst.set_int(3, hd as i32);
                        inst.set_no_sync();
                        instructions.push(inst);
                        let v_copy_idx = instructions.len();
                        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
                        inst.set_output_ptr(1, unsafe { kv_cache.v.as_ptr().add(h * head_stride) });
                        inst.set_ptr(2, unsafe { v_attn_ptr.add(h * hd) });
                        inst.set_int(3, hd as i32);
                        if h < nkh - 1 { inst.set_no_sync(); }
                        instructions.push(inst);
                        head_indices.push((k_copy_idx, v_copy_idx));
                    }
                    kv_write_indices.push(head_indices);
                    kv_base_ptrs.push((kv_cache.k.as_ptr() as u64, kv_cache.v.as_ptr() as u64));
                }

                // GQA attention
                let gqa_idx = instructions.len();
                gqa_attn_inst_indices.push(gqa_idx);
                let mut inst = Instruction::new(OP_GQA_ATTN, nqh as u32);
                inst.set_output_ptr(1, attn_out_ptr);
                inst.set_ptr(2, q_attn_ptr);
                inst.set_ptr(3, kv_cache.k.as_ptr());
                inst.set_ptr(4, kv_cache.v.as_ptr());
                inst.set_int(5, nqh as i32);
                inst.set_int(6, nkh as i32);
                inst.set_int(7, hd as i32);
                inst.set_int(8, 1); // seq_len — updated per step
                inst.set_int(9, cfg.max_seq_len as i32);
                instructions.push(inst);
            }

            AttentionVariant::PagedKv { kv_cache, attn_layer_index } => {
                // KV write BEFORE mRoPE (paged stores pre-RoPE K)
                // Per-head D2D_COPY for [H,T,D] chunk layout
                let kv_stride = nkh * hd;
                let chunk_tokens: usize = 64;
                let layer_k_offset_bytes =
                    (*attn_layer_index * 2 * chunk_tokens * kv_stride * std::mem::size_of::<f32>()) as u64;
                let layer_v_offset_bytes =
                    layer_k_offset_bytes + (chunk_tokens * kv_stride * std::mem::size_of::<f32>()) as u64;

                let chunk_head_stride = chunk_tokens * hd; // elements between heads within chunk
                let mut head_indices = Vec::new();
                for h in 0..nkh {
                    let k_copy_idx = instructions.len();
                    {
                        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
                        inst.set_output_ptr(1, unsafe { kv_cache.k.as_ptr().add(h * chunk_head_stride) });
                        inst.set_ptr(2, unsafe { k_attn_ptr.add(h * hd) });
                        inst.set_int(3, hd as i32);
                        inst.set_no_sync();
                        instructions.push(inst);
                    }
                    let v_copy_idx = instructions.len();
                    {
                        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
                        inst.set_output_ptr(1, unsafe { kv_cache.v.as_ptr().add(h * chunk_head_stride) });
                        inst.set_ptr(2, unsafe { v_attn_ptr.add(h * hd) });
                        inst.set_int(3, hd as i32);
                        if h < nkh - 1 { inst.set_no_sync(); }
                        instructions.push(inst);
                    }
                    head_indices.push((k_copy_idx, v_copy_idx));
                }
                kv_write_indices.push(head_indices);
                kv_base_ptrs.push((kv_cache.k.as_ptr() as u64, kv_cache.v.as_ptr() as u64));

                // mRoPE after KV write
                let mrope_idx = instructions.len();
                mrope_indices.push(mrope_idx);
                {
                    let mut inst = Instruction::new(OP_MROPE, (nqh + nkh) as u32);
                    inst.set_output_ptr(1, q_attn_ptr);
                    inst.set_output_ptr(2, k_attn_ptr);
                    inst.set_ptr(3, act.inv_freq.as_ptr());
                    inst.set_ptr(4, position_ids_ptr);
                    inst.set_int(5, nqh as i32);
                    inst.set_int(6, nkh as i32);
                    inst.set_int(7, hd as i32);
                    inst.set_int(8, rd as i32);
                    inst.set_int(9, cfg.mrope_section[0] as i32);
                    inst.set_int(10, cfg.mrope_section[1] as i32);
                    inst.set_int(11, cfg.mrope_section[2] as i32);
                    inst.set_int(12, 1); // batch=1 for decode
                    instructions.push(inst);
                }

                // OP_ATTN_PAGED
                let paged_idx = instructions.len();
                attn_paged_indices.push(paged_idx);
                {
                    let mut inst = Instruction::new(OP_ATTN_PAGED, nqh as u32);
                    inst.set_output_ptr(1, attn_out_ptr);
                    inst.set_ptr(2, q_attn_ptr);
                    inst.set_int(3, 0); // page_table_ptr — patched per step
                    inst.set_int(4, 0); // position_table_ptr — patched per step
                    inst.set_ptr(5, act.inv_freq.as_ptr());
                    inst.set_int(6, nqh as i32);
                    inst.set_int(7, nkh as i32);
                    inst.set_int(8, hd as i32);
                    inst.set_int(9, 1); // seq_len — patched per step
                    inst.set_int(10, chunk_tokens as i32);
                    inst.set_int(11, rd as i32);
                    inst.words[12] = layer_k_offset_bytes;
                    inst.words[13] = layer_v_offset_bytes;
                    instructions.push(inst);
                }
            }

            AttentionVariant::Prefill { kv_cache, start_pos } => {
                // mRoPE first (batched)
                let mrope_idx = instructions.len();
                mrope_indices.push(mrope_idx);
                {
                    let mut inst = Instruction::new(OP_MROPE, (n * (nqh + nkh)) as u32);
                    inst.set_output_ptr(1, q_attn_ptr);
                    inst.set_output_ptr(2, k_attn_ptr);
                    inst.set_ptr(3, act.inv_freq.as_ptr());
                    inst.set_ptr(4, position_ids_ptr);
                    inst.set_int(5, nqh as i32);
                    inst.set_int(6, nkh as i32);
                    inst.set_int(7, hd as i32);
                    inst.set_int(8, rd as i32);
                    inst.set_int(9, cfg.mrope_section[0] as i32);
                    inst.set_int(10, cfg.mrope_section[1] as i32);
                    inst.set_int(11, cfg.mrope_section[2] as i32);
                    inst.set_int(12, n as i32);
                    instructions.push(inst);
                }

                // Per-head KV write for N tokens ([H,T,D] layout)
                // Source k_attn is [N, nkh, hd], dest is [nkh, max_seq_len, hd]
                // For each head h, copy N*hd elements from src offset [0,h,0] stride nkh*hd
                // to dst offset [h, start_pos, 0].
                // Since source is token-major [N, nkh, hd], we need per-token-per-head copies
                // OR a new scatter op. For prefill N is small (≤64), so N*nkh copies of hd floats.
                let max_sl = cfg.max_seq_len;
                for t in 0..n {
                    for h in 0..nkh {
                        let src_off = (t * nkh + h) * hd;
                        let dst_off = h * max_sl * hd + (*start_pos as usize + t) * hd;
                        let k_dst = unsafe { kv_cache.k.as_ptr().add(dst_off) };
                        let k_src = unsafe { k_attn_ptr.add(src_off) };
                        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
                        inst.set_output_ptr(1, k_dst);
                        inst.set_ptr(2, k_src);
                        inst.set_int(3, hd as i32);
                        inst.set_no_sync();
                        instructions.push(inst);

                        let v_dst = unsafe { kv_cache.v.as_ptr().add(dst_off) };
                        let v_src = unsafe { v_attn_ptr.add(src_off) };
                        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
                        inst.set_output_ptr(1, v_dst);
                        inst.set_ptr(2, v_src);
                        inst.set_int(3, hd as i32);
                        if t == n - 1 && h == nkh - 1 {
                            // last copy syncs
                        } else {
                            inst.set_no_sync();
                        }
                        instructions.push(inst);
                    }
                }

                // OP_ATTN_PREFILL
                {
                    let mut inst = Instruction::new(OP_ATTN_PREFILL, (n * nqh) as u32);
                    inst.set_output_ptr(1, attn_out_ptr);
                    inst.set_ptr(2, q_attn_ptr);
                    inst.set_ptr(3, kv_cache.k.as_ptr());
                    inst.set_ptr(4, kv_cache.v.as_ptr());
                    inst.set_int(5, nqh as i32);
                    inst.set_int(6, nkh as i32);
                    inst.set_int(7, hd as i32);
                    inst.set_int(8, *start_pos as i32);
                    inst.set_int(9, n as i32);
                    inst.set_int(10, cfg.max_seq_len as i32);
                    instructions.push(inst);
                }
            }
        }

        // 10. Output gate
        {
            let gate_size = n * nqh * hd;
            let mut inst = Instruction::new(OP_OUTPUT_GATE, div_ceil(gate_size as u32, 256));
            inst.set_output_ptr(1, gated_out_ptr);
            inst.set_ptr(2, attn_out_ptr);
            inst.set_ptr(3, gate_attn_ptr);
            inst.set_int(4, gate_size as i32);
            instructions.push(inst);
        }

        // 11. Output projection + residual
        {
            let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
            inst.set_output_ptr(1, out_proj_ptr);
            inst.set_ptr(2, w.w_o.as_ptr());
            inst.set_ptr(3, gated_out_ptr);
            inst.set_int(4, hs as i32);
            inst.set_int(5, (nqh * hd) as i32);
            if n > 1 { inst.set_int(6, n as i32); }
            instructions.push(inst);
        }
        if n > 1 {
            // Batched residual: hidden = hidden + out_proj (N tokens)
            let total = n * hs;
            let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(total as u32, 256));
            inst.set_output_ptr(1, ffn_hidden_ptr);
            inst.set_ptr(2, out_proj_ptr);
            inst.set_ptr(3, ffn_hidden_ptr);
            inst.set_int(4, total as i32);
            instructions.push(inst);
        } else if prefill.is_some() {
            // Single-token prefill: residual uses prefill buffer (hidden_ptr = pb.hidden)
            let total = hs;
            let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(total as u32, 256));
            inst.set_output_ptr(1, hidden_ptr);
            inst.set_ptr(2, out_proj_ptr);
            inst.set_ptr(3, hidden_ptr);
            inst.set_int(4, total as i32);
            instructions.push(inst);
        } else {
            // Single-token decode: two-step residual via act.residual scratch
            {
                let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
                inst.set_output_ptr(1, act.residual.as_ptr());
                inst.set_ptr(2, hidden_ptr);
                inst.set_int(3, hs as i32);
                instructions.push(inst);
            }
            {
                let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
                inst.set_output_ptr(1, act.hidden.as_ptr());
                inst.set_ptr(2, out_proj_ptr);
                inst.set_ptr(3, act.residual.as_ptr());
                inst.set_int(4, hs as i32);
                instructions.push(inst);
            }
        }
    }

    fn compile_gdn_layer(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        act: &ActivationBuffers,
        conv_state: &DeviceBuffer<f32>,
        gdn_state: &GdnState,
        instructions: &mut Vec<Instruction>,
    ) {
        let w = match layer {
            LayerWeights::Gdn(w) => w,
            _ => panic!("expected GDN layer"),
        };
        let hs = cfg.hidden_size;
        let nh = cfg.linear_num_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let ck = cfg.linear_conv_kernel_dim;
        let qkv_dim = nh * kd * 2 + nh * vd; // 6144
        let eps = cfg.rms_norm_eps;

        // 1. RMSNorm
        let mut inst = Instruction::new(OP_RMSNORM, 1);
        inst.set_output_ptr(1, act.normed.as_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_ptr(3, w.input_norm.as_ptr());
        inst.set_int(4, hs as i32);
        inst.set_float(5, eps);
        instructions.push(inst);

        // 2. QKV projection [6144, 1024] @ [1024] → [6144]
        // NO_SYNC: next 3 instructions (a/b/z proj) read normed, not qkv
        let mut inst = Instruction::new(OP_LINEAR_PROJ, qkv_dim as u32);
        inst.set_output_ptr(1, act.qkv.as_ptr());
        inst.set_ptr(2, w.w_qkv.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, qkv_dim as i32);
        inst.set_int(5, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // 3. Project a [nh], b [nh], z [nh*vd] — all read normed, write disjoint buffers
        let mut inst = Instruction::new(OP_LINEAR_PROJ, nh as u32);
        inst.set_output_ptr(1, act.a_proj.as_ptr());
        inst.set_ptr(2, w.w_a.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, nh as i32);
        inst.set_int(5, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        let mut inst = Instruction::new(OP_LINEAR_PROJ, nh as u32);
        inst.set_output_ptr(1, act.b_proj.as_ptr());
        inst.set_ptr(2, w.w_b.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, nh as i32);
        inst.set_int(5, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // z proj: SYNC here ensures QKV+a+b+z all complete before conv1d reads qkv
        let mut inst = Instruction::new(OP_LINEAR_PROJ, (nh * vd) as u32);
        inst.set_output_ptr(1, act.z_proj.as_ptr());
        inst.set_ptr(2, w.w_z.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, (nh * vd) as i32);
        inst.set_int(5, hs as i32);
        instructions.push(inst);

        // 4. Causal conv1d on QKV (3 separate calls for q, k, v slices)
        let q_dim = nh * kd; // 2048
        let k_dim = nh * kd; // 2048
        let v_dim = nh * vd; // 2048

        // Conv on Q portion — NO_SYNC: conv_k reads different qkv slice, writes different state/output
        let mut inst = Instruction::new(OP_CONV1D, div_ceil(q_dim as u32, 256));
        inst.set_output_ptr(1, conv_state.as_ptr());
        inst.set_ptr(2, act.qkv.as_ptr());
        inst.set_ptr(3, w.conv1d_weight_q.as_ptr());
        inst.set_output_ptr(4, act.q_gdn.as_ptr());
        inst.set_int(5, q_dim as i32);
        inst.set_int(6, ck as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // Conv on K portion — NO_SYNC: conv_v reads different slice
        let mut inst = Instruction::new(OP_CONV1D, div_ceil(k_dim as u32, 256));
        inst.set_output_ptr(1, unsafe { conv_state.as_ptr().add(q_dim * (ck - 1)) });
        inst.set_ptr(2, unsafe { act.qkv.as_ptr().add(q_dim) });
        inst.set_ptr(3, w.conv1d_weight_k.as_ptr());
        inst.set_output_ptr(4, act.k_gdn.as_ptr());
        inst.set_int(5, k_dim as i32);
        inst.set_int(6, ck as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // Conv on V portion
        let mut inst = Instruction::new(OP_CONV1D, div_ceil(v_dim as u32, 256));
        inst.set_output_ptr(1, unsafe { conv_state.as_ptr().add((q_dim + k_dim) * (ck - 1)) });
        inst.set_ptr(2, unsafe { act.qkv.as_ptr().add(q_dim + k_dim) });
        inst.set_ptr(3, w.conv1d_weight_v.as_ptr());
        inst.set_output_ptr(4, act.v_gdn.as_ptr());
        inst.set_int(5, v_dim as i32);
        inst.set_int(6, ck as i32);
        instructions.push(inst);

        // 5. GDN gate
        let mut inst = Instruction::new(OP_GDN_GATE, div_ceil(nh as u32, 256));
        inst.set_output_ptr(1, act.gate_gdn.as_ptr());
        inst.set_ptr(2, act.a_proj.as_ptr());
        inst.set_ptr(3, w.a_log.as_ptr());
        inst.set_ptr(4, w.dt_bias.as_ptr());
        inst.set_int(5, nh as i32);
        instructions.push(inst);

        // 6. GDN recurrent (one block per head)
        let mut inst = Instruction::new(OP_GDN_RECUR, nh as u32);
        inst.set_ptr(1, act.q_gdn.as_ptr());
        inst.set_ptr(2, act.k_gdn.as_ptr());
        inst.set_ptr(3, act.v_gdn.as_ptr());
        inst.set_ptr(4, act.gate_gdn.as_ptr());
        inst.set_ptr(5, act.b_proj.as_ptr());
        inst.set_output_ptr(6, gdn_state.recurrent.as_ptr());
        inst.set_output_ptr(7, act.recurrent_out.as_ptr());
        inst.set_int(8, kd as i32);
        inst.set_int(9, vd as i32);
        instructions.push(inst);

        // 7. RMSNorm gated
        let mut inst = Instruction::new(OP_RMSNORM_GATE, nh as u32);
        inst.set_output_ptr(1, act.normed_gated.as_ptr());
        inst.set_ptr(2, act.recurrent_out.as_ptr());
        inst.set_ptr(3, act.z_proj.as_ptr());
        inst.set_ptr(4, w.output_norm.as_ptr());
        inst.set_int(5, nh as i32);
        inst.set_int(6, vd as i32);
        inst.set_float(7, eps);
        instructions.push(inst);

        // 8. Output projection [1024, 2048]
        let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
        inst.set_output_ptr(1, act.out_proj.as_ptr());
        inst.set_ptr(2, w.w_out.as_ptr());
        inst.set_ptr(3, act.normed_gated.as_ptr());
        inst.set_int(4, hs as i32);
        inst.set_int(5, (nh * vd) as i32);
        instructions.push(inst);

        // 9. Residual: copy hidden→residual, then add
        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
        inst.set_output_ptr(1, act.residual.as_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_int(3, hs as i32);
        instructions.push(inst);

        let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
        inst.set_output_ptr(1, act.hidden.as_ptr());
        inst.set_ptr(2, act.out_proj.as_ptr());
        inst.set_ptr(3, act.residual.as_ptr());
        inst.set_int(4, hs as i32);
        instructions.push(inst);
    }

    fn compile_attention_layer(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        act: &ActivationBuffers,
        kv_cache: &KvCache,
        instructions: &mut Vec<Instruction>,
        mrope_indices: &mut Vec<usize>,
        gqa_indices: &mut Vec<usize>,
        kv_write_indices: &mut Vec<Vec<(usize, usize)>>,
        kv_base_ptrs: &mut Vec<(u64, u64)>,
    ) {
        let w = match layer {
            LayerWeights::Attention(w) => w,
            _ => panic!("expected attention layer"),
        };
        Self::emit_attention_layer(
            cfg, w, act, None,
            &AttentionVariant::FlatKv { kv_cache },
            instructions, mrope_indices, gqa_indices, kv_write_indices, kv_base_ptrs,
            &mut Vec::new(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_attention_layer_paged(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        act: &ActivationBuffers,
        kv_cache: &KvCache,
        attn_layer_index: usize,
        instructions: &mut Vec<Instruction>,
        mrope_indices: &mut Vec<usize>,
        kv_write_indices: &mut Vec<Vec<(usize, usize)>>,
        kv_base_ptrs: &mut Vec<(u64, u64)>,
        attn_paged_indices: &mut Vec<usize>,
    ) {
        let w = match layer {
            LayerWeights::Attention(w) => w,
            _ => panic!("expected attention layer"),
        };
        Self::emit_attention_layer(
            cfg, w, act, None,
            &AttentionVariant::PagedKv { kv_cache, attn_layer_index },
            instructions, mrope_indices, &mut Vec::new(), kv_write_indices, kv_base_ptrs,
            attn_paged_indices,
        );
    }

    fn compile_ffn_batched(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        bufs: &PrefillBuffers,
        n: usize,
        instructions: &mut Vec<Instruction>,
    ) {
        let hs = cfg.hidden_size;
        let is = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;

        let (post_norm, w_gate, w_up, w_down) = match layer {
            LayerWeights::Gdn(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
            LayerWeights::Attention(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
        };

        // FFN gate+up (batch=N)
        let mut inst = Instruction::new(OP_FFN_GATE_UP, (is * n) as u32);
        inst.set_output_ptr(1, bufs.ffn_act.as_ptr());
        inst.set_ptr(2, bufs.hidden.as_ptr());
        inst.set_ptr(3, post_norm.as_ptr());
        inst.set_ptr(4, w_gate.as_ptr());
        inst.set_ptr(5, w_up.as_ptr());
        inst.set_int(6, hs as i32);
        inst.set_int(7, is as i32);
        inst.set_float(8, eps);
        inst.set_int(9, n as i32);
        instructions.push(inst);

        // Save residual
        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil((n * hs) as u32, 256));
        inst.set_output_ptr(1, bufs.residual.as_ptr());
        inst.set_ptr(2, bufs.hidden.as_ptr());
        inst.set_int(3, (n * hs) as i32);
        instructions.push(inst);

        // FFN down + residual (batch=N)
        let mut inst = Instruction::new(OP_FFN_DOWN_RES, (hs * n) as u32);
        inst.set_output_ptr(1, bufs.hidden.as_ptr());
        inst.set_ptr(2, bufs.residual.as_ptr());
        inst.set_ptr(3, w_down.as_ptr());
        inst.set_ptr(4, bufs.ffn_act.as_ptr());
        inst.set_int(5, hs as i32);
        inst.set_int(6, is as i32);
        inst.set_int(7, n as i32);
        instructions.push(inst);
    }

    fn compile_ffn(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        act: &ActivationBuffers,
        instructions: &mut Vec<Instruction>,
    ) {
        let hs = cfg.hidden_size;
        let is = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;

        let (post_norm, w_gate, w_up, w_down) = match layer {
            LayerWeights::Gdn(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
            LayerWeights::Attention(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
        };

        // FFN gate+up (fused with RMSNorm)
        let mut inst = Instruction::new(OP_FFN_GATE_UP, is as u32);
        inst.set_output_ptr(1, act.ffn_act.as_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_ptr(3, post_norm.as_ptr());
        inst.set_ptr(4, w_gate.as_ptr());
        inst.set_ptr(5, w_up.as_ptr());
        inst.set_int(6, hs as i32);
        inst.set_int(7, is as i32);
        inst.set_float(8, eps);
        instructions.push(inst);

        // Save residual
        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
        inst.set_output_ptr(1, act.residual.as_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_int(3, hs as i32);
        instructions.push(inst);

        // FFN down + residual
        let mut inst = Instruction::new(OP_FFN_DOWN_RES, hs as u32);
        inst.set_output_ptr(1, act.hidden.as_ptr());
        inst.set_ptr(2, act.residual.as_ptr());
        inst.set_ptr(3, w_down.as_ptr());
        inst.set_ptr(4, act.ffn_act.as_ptr());
        inst.set_int(5, hs as i32);
        inst.set_int(6, is as i32);
        instructions.push(inst);
    }

    /// Update per-step fields (token_id, position) and upload only changed instructions.
    pub fn update_step(&mut self, token_id: u32, position: u32, stream: &Stream) -> HipResult<()> {
        assert!(position < self.max_seq_len, "position {position} >= max_seq_len {}", self.max_seq_len);


        // Update embedding token_id
        self.instructions[self.embedding_inst_idx].set_int(3, token_id as i32);

        // Update position_ids on device ([temporal, height, width] = all equal position for text)
        // Use synchronous hipMemcpy because pos_data is a stack local
        let pos_data = [position as i32, position as i32, position as i32];
        braidinfer_hip::error::check(unsafe {
            braidinfer_hip::ffi::hipMemcpy(
                self.position_ids_dev_ptr as *mut std::ffi::c_void,
                pos_data.as_ptr().cast(),
                3 * std::mem::size_of::<i32>(),
                braidinfer_hip::ffi::hipMemcpyHostToDevice,
            )
        })?;

        // Update KV cache write offsets (position-dependent, [H,T,D] layout)
        let nkh = self.num_kv_heads_attn;
        let hd = self.head_dim_attn;
        let max_sl = self.max_seq_len as usize;
        let head_stride = max_sl * hd; // elements between consecutive heads
        for (layer_i, head_indices) in self.kv_write_indices.iter().enumerate() {
            let (k_base, v_base) = self.kv_base_ptrs[layer_i];
            for (h, &(k_idx, v_idx)) in head_indices.iter().enumerate() {
                let offset = (h * head_stride + position as usize * hd) * std::mem::size_of::<f32>();
                self.instructions[k_idx].words[1] = k_base + offset as u64;
                self.instructions[v_idx].words[1] = v_base + offset as u64;
            }
        }

        // Update GQA attention seq_len
        let seq_len = position + 1;
        for &idx in &self.gqa_attn_inst_indices {
            self.instructions[idx].set_int(8, seq_len as i32);
        }

        // Upload entire instruction buffer in one hipMemcpyAsync call.
        // ~500 instructions × 128 bytes = ~64KB; one 64KB copy is cheaper than 24× 128-byte copies.
        let flat: Vec<u64> = self.instructions.iter().flat_map(|i| i.words).collect();
        let dev_ptr = self.device_program.as_mut_ptr();
        let size = flat.len() * std::mem::size_of::<u64>();
        braidinfer_hip::error::check(unsafe {
            braidinfer_hip::ffi::hipMemcpyAsync(
                dev_ptr.cast(),
                flat.as_ptr().cast(),
                size,
                braidinfer_hip::ffi::hipMemcpyHostToDevice,
                stream.raw(),
            )
        })?;
        Ok(())
    }

    /// Update per-step fields for the paged KV path.
    /// Must be called before `execute()` each decode step.
    pub fn update_step_paged(
        &mut self,
        token_id: u32,
        position: u32,
        seq: &SequenceState,
        allocator: &PageAllocator,
        stream: &Stream,
    ) -> HipResult<()> {
        assert!(position < self.max_seq_len, "position {position} >= max_seq_len {}", self.max_seq_len);
        assert!(self.paged, "update_step_paged called on non-paged program");

        // 1. Patch embedding token_id
        self.instructions[self.embedding_inst_idx].set_int(3, token_id as i32);

        // 2. Append scalar position to position_table on device at offset [position]
        // Use synchronous hipMemcpy (not Async) because source is a stack local
        {
            let pos_scalar = position as i32;
            let pos_table_ptr = self.position_table.as_ref().expect("position_table not allocated").as_ptr();
            let dst = unsafe { (pos_table_ptr as *mut u8).add(position as usize * std::mem::size_of::<i32>()) };
            braidinfer_hip::error::check(unsafe {
                braidinfer_hip::ffi::hipMemcpy(
                    dst.cast(),
                    std::ptr::addr_of!(pos_scalar).cast(),
                    std::mem::size_of::<i32>(),
                    braidinfer_hip::ffi::hipMemcpyHostToDevice,
                )
            })?;
        }

        // Also update position_ids for mRoPE (same as flat path)
        // Use synchronous hipMemcpy because pos_data is a stack local
        let pos_data = [position as i32, position as i32, position as i32];
        braidinfer_hip::error::check(unsafe {
            braidinfer_hip::ffi::hipMemcpy(
                self.position_ids_dev_ptr as *mut std::ffi::c_void,
                pos_data.as_ptr().cast(),
                3 * std::mem::size_of::<i32>(),
                braidinfer_hip::ffi::hipMemcpyHostToDevice,
            )
        })?;

        // 3. Patch KV write D2D_COPY destinations from paged chunk layout [H,T,D]
        // current_chunk_offset() returns len (post-increment from append_token).
        // The write target is len-1 (the slot just reserved).
        let chunk_offset = (seq.current_chunk_offset() as usize).saturating_sub(1);
        let kv_stride = self.kv_stride_paged;
        let nkh = self.num_kv_heads_attn;
        let hd = self.head_dim_attn;
        let chunk_head_stride = CHUNK_TOKENS * hd; // elements between heads within chunk

        for (layer_i, head_indices) in self.kv_write_indices.iter().enumerate() {
            let chunk_slot = if seq.chunks.is_empty() { 0 } else {
                seq.chunks.last().unwrap().slot_index()
            };
            let chunk_base = allocator.slot_ptr(chunk_slot) as u64;
            // layout: [layer0_K[nkh, chunk_tokens, hd], layer0_V[...], layer1_K, ...]
            let layer_k_offset = (layer_i * 2 * CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>()) as u64;
            let layer_v_offset = layer_k_offset + (CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>()) as u64;
            for (h, &(k_idx, v_idx)) in head_indices.iter().enumerate() {
                let head_byte_off = (h * chunk_head_stride + chunk_offset * hd) * std::mem::size_of::<f32>();
                let k_ptr = chunk_base + layer_k_offset + head_byte_off as u64;
                let v_ptr = chunk_base + layer_v_offset + head_byte_off as u64;
                self.instructions[k_idx].words[1] = k_ptr;
                self.instructions[v_idx].words[1] = v_ptr;
            }
        }

        // 4. Patch OP_ATTN_PAGED seq_len and page_table/position_table ptrs
        let seq_len = (position + 1) as i32;
        let page_table_ptr = self.page_table.as_ref().expect("page_table not allocated").as_ptr() as u64;
        let pos_table_ptr = self.position_table.as_ref().expect("position_table not allocated").as_ptr() as u64;
        for &idx in &self.attn_paged_inst_indices {
            self.instructions[idx].words[3] = page_table_ptr;
            self.instructions[idx].words[4] = pos_table_ptr;
            self.instructions[idx].set_int(9, seq_len);
        }

        // 5. Upload page_table if chunk list changed
        if seq.chunks.len() != self.last_page_table_len {
            let page_table_dev = self.page_table.as_mut().expect("page_table not allocated");
            let host_ptrs: Vec<u64> = seq.chunks.iter()
                .map(|c| allocator.slot_ptr(c.slot_index()) as u64)
                .collect();
            let dst = page_table_dev.as_mut_ptr() as *mut u8;
            let bytes = host_ptrs.len() * std::mem::size_of::<u64>();
            braidinfer_hip::error::check(unsafe {
                braidinfer_hip::ffi::hipMemcpyAsync(
                    dst.cast(),
                    host_ptrs.as_ptr().cast(),
                    bytes,
                    braidinfer_hip::ffi::hipMemcpyHostToDevice,
                    stream.raw(),
                )
            })?;
            self.last_page_table_len = seq.chunks.len();
        }

        // 6. Upload entire instruction buffer in one hipMemcpyAsync call.
        let flat: Vec<u64> = self.instructions.iter().flat_map(|i| i.words).collect();
        let dev_ptr = self.device_program.as_mut_ptr();
        let size = flat.len() * std::mem::size_of::<u64>();
        braidinfer_hip::error::check(unsafe {
            braidinfer_hip::ffi::hipMemcpyAsync(
                dev_ptr.cast(),
                flat.as_ptr().cast(),
                size,
                braidinfer_hip::ffi::hipMemcpyHostToDevice,
                stream.raw(),
            )
        })?;
        Ok(())
    }

    /// Allocate the next chunk if the current one just filled up.
    /// Call after execute() + stream sync, before next update_step_paged().
    pub fn post_step_paged(
        &self,
        position: u32,
        seq: &mut SequenceState,
        allocator: &mut PageAllocator,
    ) -> HipResult<()> {
        // If we just wrote the last token slot in this chunk, pre-allocate the next chunk.
        if (position as usize + 1) % CHUNK_TOKENS == 0 {
            seq.append_token(allocator)?;
        }
        Ok(())
    }

    /// Lazily allocate the page_table and position_table device buffers.
    /// Must be called once before the first update_step_paged().
    pub fn init_paged_buffers(&mut self, max_chunks: usize) -> HipResult<()> {
        if self.page_table.is_none() {
            self.page_table = Some(DeviceBuffer::alloc(self.device, max_chunks)?);
        }
        if self.position_table.is_none() {
            self.position_table = Some(DeviceBuffer::alloc(self.device, self.max_seq_len as usize)?);
        }
        Ok(())
    }

    /// Execute the megakernel program.
    pub fn execute(&self, stream: &Stream) -> HipResult<()> {
        let func = self.module.get_function("megakernel_f32")?;
        let mut prog_ptr: *const c_void = self.device_program.as_ptr().cast();
        let mut num_inst = self.instructions.len() as i32;

        let mut args: [*mut c_void; 2] = [
            std::ptr::addr_of_mut!(prog_ptr).cast(),
            std::ptr::addr_of_mut!(num_inst).cast(),
        ];

        func.launch_cooperative(
            (self.num_blocks, 1, 1),
            (256, 1, 1),
            256 * 4 * 2, // 2KB shared memory (enough for gdn_recurrent's 2 arrays)
            stream,
            &mut args,
        )
    }
}
