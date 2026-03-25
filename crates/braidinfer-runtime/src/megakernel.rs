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

    fn set_mut_ptr<T>(&mut self, idx: usize, ptr: *const T) {
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
    kv_write_indices: Vec<(usize, usize)>, // (k_copy_idx, v_copy_idx) per attn layer
    // Base KV cache pointers (position=0) for computing per-step write offsets
    kv_base_ptrs: Vec<(u64, u64)>, // (k_base, v_base) per attention layer
    // mRoPE position_ids device pointer (3 i32s: temporal, height, width)
    position_ids_dev_ptr: u64,
    // Bounds check
    max_seq_len: u32,
}

fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

impl MegakernelProgram {
    pub fn instruction_count(&self) -> usize { self.instructions.len() }
    pub fn block_count(&self) -> u32 { self.num_blocks }

    pub fn compile(model: &Qwen35Model) -> HipResult<Self> {
        let cfg = &model.config;
        let device = model.device;
        let act = &model.activations;

        let module = Module::load(device, &crate::kernel::kernel_dir().join("megakernel.hsaco"))?;

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
            inst.set_mut_ptr(1, act.hidden.as_ptr());
            inst.set_ptr(2, model.embed_weight.as_ptr());
            inst.set_int(3, 0); // token_id — updated per step
            inst.set_int(4, hs as i32);
            instructions.push(inst);
        }

        // Layers
        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        for layer_i in 0..cfg.num_layers {
            if cfg.layer_is_attention[layer_i] {
                Self::compile_attention_layer(
                    cfg, &model.layers[layer_i], act,
                    &model.kv_caches[kv_idx],
                    &mut instructions, &mut mrope_inst_indices,
                    &mut gqa_attn_inst_indices, &mut kv_write_indices,
                    &mut kv_base_ptrs,
                );
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
            inst.set_mut_ptr(1, act.normed.as_ptr());
            inst.set_ptr(2, act.hidden.as_ptr());
            inst.set_int(3, hs as i32);
            instructions.push(inst);
        }
        {
            let mut inst = Instruction::new(OP_RMSNORM, 1);
            inst.set_mut_ptr(1, act.hidden.as_ptr());
            inst.set_ptr(2, act.normed.as_ptr());
            inst.set_ptr(3, model.final_norm_weight.as_ptr());
            inst.set_int(4, hs as i32);
            inst.set_float(5, eps);
            instructions.push(inst);
        }

        // LM head (= linear_proj with vocab_size output rows)
        {
            let mut inst = Instruction::new(OP_LINEAR_PROJ, vs as u32);
            inst.set_mut_ptr(1, act.logits.as_ptr());
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
        })
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
        inst.set_mut_ptr(1, act.normed.as_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_ptr(3, w.input_norm.as_ptr());
        inst.set_int(4, hs as i32);
        inst.set_float(5, eps);
        instructions.push(inst);

        // 2. QKV projection [6144, 1024] @ [1024] → [6144]
        // NO_SYNC: next 3 instructions (a/b/z proj) read normed, not qkv
        let mut inst = Instruction::new(OP_LINEAR_PROJ, qkv_dim as u32);
        inst.set_mut_ptr(1, act.qkv.as_ptr());
        inst.set_ptr(2, w.w_qkv.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, qkv_dim as i32);
        inst.set_int(5, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // 3. Project a [nh], b [nh], z [nh*vd] — all read normed, write disjoint buffers
        let mut inst = Instruction::new(OP_LINEAR_PROJ, nh as u32);
        inst.set_mut_ptr(1, act.a_proj.as_ptr());
        inst.set_ptr(2, w.w_a.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, nh as i32);
        inst.set_int(5, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        let mut inst = Instruction::new(OP_LINEAR_PROJ, nh as u32);
        inst.set_mut_ptr(1, act.b_proj.as_ptr());
        inst.set_ptr(2, w.w_b.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, nh as i32);
        inst.set_int(5, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // z proj: SYNC here ensures QKV+a+b+z all complete before conv1d reads qkv
        let mut inst = Instruction::new(OP_LINEAR_PROJ, (nh * vd) as u32);
        inst.set_mut_ptr(1, act.z_proj.as_ptr());
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
        inst.set_mut_ptr(1, conv_state.as_ptr());
        inst.set_ptr(2, act.qkv.as_ptr());
        inst.set_ptr(3, w.conv1d_weight_q.as_ptr());
        inst.set_mut_ptr(4, act.q_gdn.as_ptr());
        inst.set_int(5, q_dim as i32);
        inst.set_int(6, ck as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // Conv on K portion — NO_SYNC: conv_v reads different slice
        let mut inst = Instruction::new(OP_CONV1D, div_ceil(k_dim as u32, 256));
        inst.set_mut_ptr(1, unsafe { conv_state.as_ptr().add(q_dim * (ck - 1)) });
        inst.set_ptr(2, unsafe { act.qkv.as_ptr().add(q_dim) });
        inst.set_ptr(3, w.conv1d_weight_k.as_ptr());
        inst.set_mut_ptr(4, act.k_gdn.as_ptr());
        inst.set_int(5, k_dim as i32);
        inst.set_int(6, ck as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // Conv on V portion
        let mut inst = Instruction::new(OP_CONV1D, div_ceil(v_dim as u32, 256));
        inst.set_mut_ptr(1, unsafe { conv_state.as_ptr().add((q_dim + k_dim) * (ck - 1)) });
        inst.set_ptr(2, unsafe { act.qkv.as_ptr().add(q_dim + k_dim) });
        inst.set_ptr(3, w.conv1d_weight_v.as_ptr());
        inst.set_mut_ptr(4, act.v_gdn.as_ptr());
        inst.set_int(5, v_dim as i32);
        inst.set_int(6, ck as i32);
        instructions.push(inst);

        // 5. GDN gate
        let mut inst = Instruction::new(OP_GDN_GATE, div_ceil(nh as u32, 256));
        inst.set_mut_ptr(1, act.gate_gdn.as_ptr());
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
        inst.set_mut_ptr(6, gdn_state.recurrent.as_ptr());
        inst.set_mut_ptr(7, act.recurrent_out.as_ptr());
        inst.set_int(8, kd as i32);
        inst.set_int(9, vd as i32);
        instructions.push(inst);

        // 7. RMSNorm gated
        let mut inst = Instruction::new(OP_RMSNORM_GATE, nh as u32);
        inst.set_mut_ptr(1, act.normed_gated.as_ptr());
        inst.set_ptr(2, act.recurrent_out.as_ptr());
        inst.set_ptr(3, act.z_proj.as_ptr());
        inst.set_ptr(4, w.output_norm.as_ptr());
        inst.set_int(5, nh as i32);
        inst.set_int(6, vd as i32);
        inst.set_float(7, eps);
        instructions.push(inst);

        // 8. Output projection [1024, 2048]
        let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
        inst.set_mut_ptr(1, act.out_proj.as_ptr());
        inst.set_ptr(2, w.w_out.as_ptr());
        inst.set_ptr(3, act.normed_gated.as_ptr());
        inst.set_int(4, hs as i32);
        inst.set_int(5, (nh * vd) as i32);
        instructions.push(inst);

        // 9. Residual: copy hidden→residual, then add
        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
        inst.set_mut_ptr(1, act.residual.as_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_int(3, hs as i32);
        instructions.push(inst);

        let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
        inst.set_mut_ptr(1, act.hidden.as_ptr());
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
        kv_write_indices: &mut Vec<(usize, usize)>,
        kv_base_ptrs: &mut Vec<(u64, u64)>,
    ) {
        let w = match layer {
            LayerWeights::Attention(w) => w,
            _ => panic!("expected attention layer"),
        };
        let hs = cfg.hidden_size;
        let nqh = cfg.num_q_heads;
        let nkh = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let eps = cfg.rms_norm_eps;
        let rd = cfg.rope_dim;

        // 1. RMSNorm
        let mut inst = Instruction::new(OP_RMSNORM, 1);
        inst.set_mut_ptr(1, act.normed.as_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_ptr(3, w.input_norm.as_ptr());
        inst.set_int(4, hs as i32);
        inst.set_float(5, eps);
        instructions.push(inst);

        // 2. Q+gate [4096], K [512], V [512] projections — all read normed, write disjoint
        let mut inst = Instruction::new(OP_LINEAR_PROJ, (nqh * hd * 2) as u32);
        inst.set_mut_ptr(1, act.q_gate_attn.as_ptr());
        inst.set_ptr(2, w.w_q_gate.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, (nqh * hd * 2) as i32);
        inst.set_int(5, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        let mut inst = Instruction::new(OP_LINEAR_PROJ, (nkh * hd) as u32);
        inst.set_mut_ptr(1, act.k_attn.as_ptr());
        inst.set_ptr(2, w.w_k.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, (nkh * hd) as i32);
        inst.set_int(5, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // V proj: SYNC here ensures Q+gate, K, V all complete before deinterleave
        let mut inst = Instruction::new(OP_LINEAR_PROJ, (nkh * hd) as u32);
        inst.set_mut_ptr(1, act.v_attn.as_ptr());
        inst.set_ptr(2, w.w_v.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, (nkh * hd) as i32);
        inst.set_int(5, hs as i32);
        instructions.push(inst);

        // 3. Q/gate deinterleave: 8 heads × 2 copies each = 16 d2d copies
        // All write to non-overlapping regions — skip sync on all but last
        for h in 0..nqh {
            let src_q_offset = h * hd * 2;
            let src_g_offset = h * hd * 2 + hd;
            let dst_offset = h * hd;
            let is_last = h == nqh - 1;

            let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
            inst.set_mut_ptr(1, unsafe { act.q_attn.as_ptr().add(dst_offset) });
            inst.set_ptr(2, unsafe { act.q_gate_attn.as_ptr().add(src_q_offset) });
            inst.set_int(3, hd as i32);
            inst.set_no_sync();
            instructions.push(inst);

            let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
            inst.set_mut_ptr(1, unsafe { act.gate_attn.as_ptr().add(dst_offset) });
            inst.set_ptr(2, unsafe { act.q_gate_attn.as_ptr().add(src_g_offset) });
            inst.set_int(3, hd as i32);
            if !is_last { inst.set_no_sync(); }
            instructions.push(inst);
        }

        // 4. QK norm
        let mut inst = Instruction::new(OP_QK_NORM, (nqh + nkh) as u32);
        inst.set_mut_ptr(1, act.q_attn.as_ptr());
        inst.set_mut_ptr(2, act.k_attn.as_ptr());
        inst.set_ptr(3, w.q_norm.as_ptr());
        inst.set_ptr(4, w.k_norm.as_ptr());
        inst.set_int(5, nqh as i32);
        inst.set_int(6, nkh as i32);
        inst.set_int(7, hd as i32);
        inst.set_float(8, eps);
        instructions.push(inst);

        // 5. mRoPE (position updated per step)
        let mrope_idx = instructions.len();
        mrope_indices.push(mrope_idx);
        let mut inst = Instruction::new(OP_MROPE, (nqh + nkh) as u32);
        inst.set_mut_ptr(1, act.q_attn.as_ptr());
        inst.set_mut_ptr(2, act.k_attn.as_ptr());
        inst.set_ptr(3, act.inv_freq.as_ptr());
        inst.set_ptr(4, act.position_ids.as_ptr());
        inst.set_int(5, nqh as i32);
        inst.set_int(6, nkh as i32);
        inst.set_int(7, hd as i32);
        inst.set_int(8, rd as i32);
        inst.set_int(9, cfg.mrope_section[0] as i32);
        inst.set_int(10, cfg.mrope_section[1] as i32);
        inst.set_int(11, cfg.mrope_section[2] as i32);
        instructions.push(inst);

        // 6. Write K,V to cache — k_copy NO_SYNC (v_copy writes different buffer)
        let kv_stride = nkh * hd;
        let k_copy_idx = instructions.len();
        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(kv_stride as u32, 256));
        inst.set_mut_ptr(1, kv_cache.k.as_ptr());
        inst.set_ptr(2, act.k_attn.as_ptr());
        inst.set_int(3, kv_stride as i32);
        inst.set_no_sync();
        instructions.push(inst);

        let v_copy_idx = instructions.len();
        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(kv_stride as u32, 256));
        inst.set_mut_ptr(1, kv_cache.v.as_ptr());
        inst.set_ptr(2, act.v_attn.as_ptr());
        inst.set_int(3, kv_stride as i32);
        instructions.push(inst);
        kv_write_indices.push((k_copy_idx, v_copy_idx));
        kv_base_ptrs.push((kv_cache.k.as_ptr() as u64, kv_cache.v.as_ptr() as u64));

        // 7. GQA attention (seq_len updated per step)
        let gqa_idx = instructions.len();
        gqa_indices.push(gqa_idx);
        let mut inst = Instruction::new(OP_GQA_ATTN, nqh as u32);
        inst.set_mut_ptr(1, act.attn_out.as_ptr());
        inst.set_ptr(2, act.q_attn.as_ptr());
        inst.set_ptr(3, kv_cache.k.as_ptr());
        inst.set_ptr(4, kv_cache.v.as_ptr());
        inst.set_int(5, nqh as i32);
        inst.set_int(6, nkh as i32);
        inst.set_int(7, hd as i32);
        inst.set_int(8, 1); // seq_len — updated per step
        inst.set_int(9, cfg.max_seq_len as i32);
        instructions.push(inst);

        // 8. Output gate
        let gate_size = nqh * hd;
        let mut inst = Instruction::new(OP_OUTPUT_GATE, div_ceil(gate_size as u32, 256));
        inst.set_mut_ptr(1, act.gated_out.as_ptr());
        inst.set_ptr(2, act.attn_out.as_ptr());
        inst.set_ptr(3, act.gate_attn.as_ptr());
        inst.set_int(4, gate_size as i32);
        instructions.push(inst);

        // 9. Output projection [1024, 2048]
        let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
        inst.set_mut_ptr(1, act.out_proj.as_ptr());
        inst.set_ptr(2, w.w_o.as_ptr());
        inst.set_ptr(3, act.gated_out.as_ptr());
        inst.set_int(4, hs as i32);
        inst.set_int(5, (nqh * hd) as i32);
        instructions.push(inst);

        // 10. Residual
        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
        inst.set_mut_ptr(1, act.residual.as_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_int(3, hs as i32);
        instructions.push(inst);

        let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
        inst.set_mut_ptr(1, act.hidden.as_ptr());
        inst.set_ptr(2, act.out_proj.as_ptr());
        inst.set_ptr(3, act.residual.as_ptr());
        inst.set_int(4, hs as i32);
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
        inst.set_mut_ptr(1, act.ffn_act.as_ptr());
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
        inst.set_mut_ptr(1, act.residual.as_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_int(3, hs as i32);
        instructions.push(inst);

        // FFN down + residual
        let mut inst = Instruction::new(OP_FFN_DOWN_RES, hs as u32);
        inst.set_mut_ptr(1, act.hidden.as_ptr());
        inst.set_ptr(2, act.residual.as_ptr());
        inst.set_ptr(3, w_down.as_ptr());
        inst.set_ptr(4, act.ffn_act.as_ptr());
        inst.set_int(5, hs as i32);
        inst.set_int(6, is as i32);
        instructions.push(inst);
    }

    /// Update per-step fields (token_id, position) and upload only changed instructions.
    pub fn update_step(&mut self, token_id: u32, position: u32) -> HipResult<()> {
        assert!(position < self.max_seq_len, "position {position} >= max_seq_len {}", self.max_seq_len);

        let cfg_nkh_hd = 512usize; // nkh * hd = 2 * 256
        let mut changed: Vec<usize> = Vec::new();

        // Update embedding token_id
        self.instructions[self.embedding_inst_idx].set_int(3, token_id as i32);
        changed.push(self.embedding_inst_idx);

        // Update position_ids on device ([temporal, height, width] = all equal position for text)
        let pos_data = [position as i32, position as i32, position as i32];
        braidinfer_hip::error::check(unsafe {
            braidinfer_hip::ffi::hipMemcpy(
                self.position_ids_dev_ptr as *mut std::ffi::c_void,
                pos_data.as_ptr().cast(),
                3 * std::mem::size_of::<i32>(),
                braidinfer_hip::ffi::hipMemcpyHostToDevice,
            )
        })?;

        // Update KV cache write offsets (position-dependent)
        let pos_offset = position as usize * cfg_nkh_hd;
        for (layer_i, &(k_idx, v_idx)) in self.kv_write_indices.iter().enumerate() {
            let (k_base, v_base) = self.kv_base_ptrs[layer_i];
            let k_ptr = k_base + (pos_offset * std::mem::size_of::<f32>()) as u64;
            let v_ptr = v_base + (pos_offset * std::mem::size_of::<f32>()) as u64;
            self.instructions[k_idx].words[1] = k_ptr;
            self.instructions[v_idx].words[1] = v_ptr;
            changed.push(k_idx);
            changed.push(v_idx);
        }

        // Update GQA attention seq_len
        let seq_len = position + 1;
        for &idx in &self.gqa_attn_inst_indices {
            self.instructions[idx].set_int(8, seq_len as i32);
            changed.push(idx);
        }

        // Upload only the changed instructions (each is INST_SIZE u64s = 128 bytes)
        let dev_ptr = self.device_program.as_mut_ptr();
        for inst_idx in changed {
            let words = &self.instructions[inst_idx].words;
            let byte_offset = inst_idx * INST_SIZE * std::mem::size_of::<u64>();
            let size = INST_SIZE * std::mem::size_of::<u64>();
            braidinfer_hip::error::check(unsafe {
                braidinfer_hip::ffi::hipMemcpy(
                    (dev_ptr as *mut u8).add(byte_offset).cast(),
                    words.as_ptr().cast(),
                    size,
                    braidinfer_hip::ffi::hipMemcpyHostToDevice,
                )
            })?;
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
