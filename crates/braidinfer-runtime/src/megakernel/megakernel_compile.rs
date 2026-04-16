//! Megakernel program compilation: translates model config + weights into instruction streams.
//! Extracted from megakernel.rs for maintainability.

use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;

use super::instructions::*;
use super::{CHUNK_TOKENS, INST_OPCODE_MASK, INST_SIZE, Instruction, MegakernelProgram, NUM_CUS, PrefillBuffers};
#[allow(unused_imports)]
use super::{
    OP_ATTN_PAGED, OP_ATTN_PAGED_Q, OP_ATTN_PREFILL, OP_CONV1D, OP_D2D_COPY,
    OP_DEINTERLEAVE, OP_EMBEDDING, OP_FFN_DOWN_RES, OP_FFN_DOWN_RES_RNF4, OP_FFN_GATE_UP,
    OP_FFN_GATE_UP_RNF4, OP_GDN_GATE, OP_GDN_RECUR, OP_GQA_ATTN, OP_HALT, OP_KV_QUANTIZE,
    OP_LINEAR_PROJ, OP_LINEAR_PROJ_PCG32, OP_LINEAR_PROJ_RNF4, OP_LM_HEAD, OP_MAMBA2_CONV1D,
    OP_MAMBA2_NORM_GATED, OP_MOE_DISPATCH, OP_MOE_FFN, OP_MOE_GATE, OP_MROPE, OP_OUTPUT_GATE,
    OP_QK_NORM, OP_RELU_SQ, OP_RESIDUAL_ADD, OP_RMSNORM, OP_RMSNORM_GATE, OP_RMSNORM_WX,
    OP_SIGMOID_WEIGHTED_ADD, OP_SILU_MUL, OP_SSM_UPDATE,
};
use crate::model::{
    ActivationBuffers, AttentionLayerWeights, GdnState, KvCache, LayerWeights, Mamba2State, Model,
    ModelConfig, RecurrentLayerKind,
};

fn emit_linear_proj(inst: &mut Instruction, weight: &crate::model::LinearWeight, ptr_slot: usize) {
    use crate::model::{LinearWeight, WeightFormat};
    match weight {
        LinearWeight::Bf16(buf) => {
            inst.words[ptr_slot] = buf.as_ptr() as u64;
        }
        LinearWeight::Packed(pw) => {
            let op = match pw.format {
                WeightFormat::Rnf4G128 => OP_LINEAR_PROJ_RNF4,
                WeightFormat::PcG32Q4 => OP_LINEAR_PROJ_PCG32,
                WeightFormat::Bf16 => OP_LINEAR_PROJ,
            };
            // Replace opcode (low 32 bits), preserve grid_x (high 32 bits)
            inst.words[0] = (inst.words[0] & 0xFFFF_FFFF_0000_0000u64) | op as u64;
            inst.words[ptr_slot] = pw.data.as_ptr() as u64;
        }
    }
}

/// Emit a batched linear projection. For bf16, uses single batched instruction.
/// For quantized (PCG32/RNF4), emits per-token loop (kernel batching TODO: braidinfer-xxy).
fn emit_batched_linear_proj(
    weight: &crate::model::LinearWeight,
    output: *mut f32,
    input: *const f32,
    out_dim: usize,
    in_dim: usize,
    n: usize,
    no_sync: bool,
    instructions: &mut Vec<Instruction>,
) {
    let (opcode, w_ptr) = linear_proj_opcode_ptr(weight);
    let inst = LinearProjInst::new(opcode, out_dim as u32, output, w_ptr, input, out_dim as i32, in_dim as i32, n as i32);
    let inst = if no_sync { inst.no_sync() } else { inst };
    instructions.push(inst.into_inst());
}

/// Return (opcode, weight_data_ptr) for a LinearWeight.
fn linear_proj_opcode_ptr(weight: &crate::model::LinearWeight) -> (u32, *const u8) {
    use crate::model::{LinearWeight, WeightFormat};
    match weight {
        LinearWeight::Bf16(buf) => (OP_LINEAR_PROJ, buf.as_ptr() as *const u8),
        LinearWeight::Packed(pw) => {
            let op = match pw.format {
                WeightFormat::Rnf4G128 => OP_LINEAR_PROJ_RNF4,
                WeightFormat::PcG32Q4 => OP_LINEAR_PROJ_PCG32,
                WeightFormat::Bf16 => OP_LINEAR_PROJ,
            };
            (op, pw.data.as_ptr())
        }
    }
}

/// Choose RMSNorm opcode based on model config.
fn rmsnorm_opcode(one_plus_w: bool) -> u32 {
    if one_plus_w {
        OP_RMSNORM
    } else {
        OP_RMSNORM_WX
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
    PagedKv { attn_layer_index: usize },
    /// Prefill (N tokens): OP_ATTN_PREFILL, bulk KV write after mRoPE.
    Prefill {
        kv_cache: &'a KvCache,
        start_pos: u32,
    },
}

impl MegakernelProgram {
    pub fn compile(model: &Model) -> HipResult<Self> {
        Self::compile_inner(model, false, false)
    }

    pub fn compile_paged(model: &Model) -> HipResult<Self> {
        Self::compile_inner(model, true, false)
    }

    /// Compile for GPU-native P2P MoE dispatch (OP_MOE_DISPATCH).
    /// MoE layers emit OP_MOE_DISPATCH — handled entirely inside the megakernel by op_moe_dispatch.
    /// No CPU involvement in the hot path; workers on GPUs 1-3 run moe_worker_kernel.
    pub fn compile_multi_gpu_p2p(
        model: &Model,
        p2p: &crate::moe_p2p::MoeP2pContext,
    ) -> HipResult<Self> {
        Self::compile_inner_p2p(model, p2p)
    }

    fn compile_inner(model: &Model, paged: bool, multi_gpu: bool) -> HipResult<Self> {
        let cfg = &model.config;
        let device = model.device;
        let act = &model.activations;


        let module = Module::load(
            device,
            &crate::kernel::kernel_dir().join("megakernel.hsaco"),
        )?;

        // Note: hipDeviceAttributeCooperativeLaunch (95) returns 0 on ROCm/RDNA3 even though
        // cooperative launch works. Skipping capability check — hipModuleLaunchCooperativeKernel
        // will return an error if unsupported.

        let has_moe = cfg
            .layers
            .iter()
            .any(|l| matches!(l.ffn_type, crate::model::FfnType::MoE { .. }));
        // OP_MOE_GATE needs 1024 floats = 4KB. GDN recurrent needs 2KB.
        // OP_LINEAR_PROJ_PCG32/RNF4 tiled-LDS: (8+7680+256)*4 = 31776 bytes per block.
        // 2 blocks/CU: 2*31776 = 63552 < 65536 ✓ — no occupancy reduction.
        let base_shared = if has_moe { 1024u32 * 4 } else { 256u32 * 4 * 2 };
        let shared_mem = base_shared.max(31776u32);
        let func = module.get_function("megakernel_f32")?;
        let blocks_per_sm = func.max_active_blocks_per_sm(256, shared_mem as usize)?;
        eprintln!(
            "  megakernel: shared_mem={shared_mem} blocks_per_sm={blocks_per_sm} NUM_CUS={NUM_CUS}"
        );
        // NUM_CUS=48 = WGP count (MultiprocessorCount on RDNA3). Max cooperative blocks = blocks_per_sm * WGPs.
        // blocks_per_sm=0 means LDS-limited; fall back to 1/WGP.
        let blocks_per_sm_clamped = blocks_per_sm.max(1) as u32;
        let num_blocks = blocks_per_sm_clamped * NUM_CUS;

        let mut instructions: Vec<Instruction> = Vec::new();
        let mut mrope_inst_indices: Vec<usize> = Vec::new();
        let mut gqa_attn_inst_indices = Vec::new();
        let mut kv_write_indices = Vec::new();
        let mut kv_base_ptrs = Vec::new();
        let mut barrier_layer_map: Vec<(usize, usize)> = Vec::new();
        let mut multi_gpu_attn_boundaries: Vec<(usize, usize)> = Vec::new();

        let hs = cfg.hidden_size;
        let nh_gdn = cfg.linear_num_heads;
        let nvh_gdn = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let _conv_dim = nh_gdn * kd * 2 + nvh_gdn * vd;
        let _ck = cfg.linear_conv_kernel_dim;
        let _nqh = cfg.num_q_heads;
        let _nkh = cfg.num_kv_heads;
        let _hd = cfg.head_dim;
        let _is = cfg.intermediate_size;
        let vs = cfg.vocab_size;
        let eps = cfg.rms_norm_eps;

        // Embedding (token_id placeholder = 0, updated per step)
        let embedding_inst_idx = instructions.len();
        instructions.push(EmbeddingInst::new(
            div_ceil(hs as u32, 256),
            act.hidden.as_write_ptr(),
            model.embed_weight.as_ptr(),
            0, // token_id — updated per step
            hs as i32,
        ).into_inst());

        // Layers
        let mut attn_paged_inst_indices = Vec::new();
        let mut attn_quant_inst_indices = Vec::new();
        let mut attn_layer_count = 0usize;

        let mut gdn_idx = 0usize;
        let mut mamba2_idx = 0usize;
        let mut kv_idx = 0usize;
        for layer_i in 0..cfg.num_layers {
            use crate::model::LayerType;
            match cfg.layers[layer_i].layer_type {
                LayerType::Attention => {
                    if paged {
                        Self::compile_attention_layer_paged(
                            cfg,
                            &model.layers[layer_i],
                            act,
                            attn_layer_count,
                            &mut instructions,
                            &mut mrope_inst_indices,
                            &mut kv_write_indices,
                            &mut kv_base_ptrs,
                            &mut attn_paged_inst_indices,
                            &mut attn_quant_inst_indices,
                        );
                    } else if multi_gpu {
                        // Distribute QKV projection + GQA across GPUs.
                        // Only RMSNorm + output-gate/O-proj/residual in megakernel.
                        Self::compile_attention_layer_multi_gpu(
                            cfg,
                            &model.layers[layer_i],
                            act,
                            &mut instructions,
                            &mut multi_gpu_attn_boundaries,
                        );
                    } else {
                        Self::compile_attention_layer(
                            cfg,
                            &model.layers[layer_i],
                            act,
                            model
                                .legacy_kv_caches
                                .as_ref()
                                .expect("legacy KV cache not initialized for flat megakernel")
                                .get(kv_idx)
                                .expect("missing legacy KV cache"),
                            &mut instructions,
                            &mut mrope_inst_indices,
                            &mut gqa_attn_inst_indices,
                            &mut kv_write_indices,
                            &mut kv_base_ptrs,
                        );
                    }
                    attn_layer_count += 1;
                    kv_idx += 1;
                }
                LayerType::Gdn => {
                    Self::compile_gdn_layer(
                        cfg,
                        &model.layers[layer_i],
                        act,
                        &model.gdn_conv_states[gdn_idx],
                        &model.gdn_states[gdn_idx],
                        &mut instructions,
                    );
                    gdn_idx += 1;
                }
                LayerType::Mamba2 => {
                    Self::compile_mamba2_layer(
                        cfg,
                        &model.layers[layer_i],
                        act,
                        &model.mamba2_states[mamba2_idx],
                        &mut instructions,
                    );
                    mamba2_idx += 1;
                }
                LayerType::MoeFfn => {
                    // Standalone MoE FFN layer (Nemotron 'E' layers): norm + MoE dispatch + residual
                    // The mixer (Mamba2/Attention) was handled in its own layer; this is FFN-only.
                    // Skip here — handled below in the MoE FFN section.
                }
                LayerType::LfmConv => {
                    panic!("LfmConv layers not yet implemented in megakernel (braidinfer-aes.4)");
                }
            }

            // FFN dispatch
            match &cfg.layers[layer_i].ffn_type {
                crate::model::FfnType::Dense => {
                    Self::compile_ffn(cfg, &model.layers[layer_i], act, &mut instructions);
                }
                crate::model::FfnType::MoE { .. } => {
                    if multi_gpu {
                        let moe = model.moe_weights[layer_i].as_ref().unwrap();
                        let barrier_inst_idx = Self::compile_moe_ffn_multi_gpu(
                            cfg,
                            layer_i,
                            &model.layers[layer_i],
                            moe,
                            act,
                            &mut instructions,
                        );
                        barrier_layer_map.push((barrier_inst_idx, layer_i));
                    } else {
                        Self::compile_moe_ffn(
                            cfg,
                            layer_i,
                            &model.layers[layer_i],
                            model.moe_weights[layer_i].as_ref().unwrap(),
                            act,
                            &mut instructions,
                        );
                    }
                }
                crate::model::FfnType::None => {
                    // No FFN for this layer (Nemotron M/* layers)
                }
            }
        }

        // Final RMSNorm: copy hidden→normed, then norm normed→hidden
        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.normed.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
        instructions.push(RmsNormInst::new(
            rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
            act.hidden.as_write_ptr(), act.normed.as_ptr(),
            model.final_norm_weight.as_ptr(), hs as i32, eps,
        ).into_inst());

        // LM head
        {
            let lm_weight = if model.config.tie_word_embeddings {
                model.embed_weight.as_ptr() as *const u8
            } else {
                model.lm_head_weight.as_ptr() as *const u8
            };
            instructions.push(LinearProjInst::new(OP_LINEAR_PROJ, vs as u32, act.logits.as_write_ptr(), lm_weight, act.hidden.as_ptr(), vs as i32, hs as i32, 0).into_inst());
        }

        // HALT
        instructions.push(HaltInst::new().into_inst());

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
            shared_mem,
            device,
            embedding_inst_idx,
            _mrope_inst_indices: mrope_inst_indices,
            gqa_attn_inst_indices,
            kv_write_indices,
            kv_base_ptrs,
            position_ids_dev_ptr: act.position_ids.as_ptr() as u64,
            max_seq_len: cfg.max_seq_len as u32,
            paged,
            page_table: None,
            position_table: None,
            attn_paged_inst_indices,
            attn_quant_inst_indices,
            last_page_table_len: 0,
            kv_stride_paged: cfg.num_kv_heads * cfg.head_dim,
            quant_scratch: None,
            quant_page_table: None,
            last_quant_page_table_len: 0,
            quantized_kv: false,
            dump_buffer: None,
            dump_counter: None,
            dump_capacity: 0,
            num_kv_heads_attn: cfg.num_kv_heads,
            head_dim_attn: cfg.head_dim,
            barrier_layer_map,
            multi_gpu_attn_boundaries,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Compile a one-shot prefill program for `tokens` starting at `start_pos`.
    /// GDN layers use batched projections (weight reuse across all N tokens).
    /// Attention layers run sequentially (one token at a time).
    /// Returns a program + list of per-token attention instruction indices for paged KV updates.
    pub fn compile_prefill(
        model: &Model,
        tokens: &[u32],
        start_pos: u32,
        prefill_bufs: &mut PrefillBuffers,
    ) -> HipResult<Self> {
        let n = tokens.len();
        assert!(n > 0 && n <= CHUNK_TOKENS);
        let cfg = &model.config;
        let device = model.device;
        let act = &model.activations;

        let module = Module::load(
            device,
            &crate::kernel::kernel_dir().join("megakernel.hsaco"),
        )?;
        let shared_mem = (256u32 * 4 * 2)
            .max((cfg.hidden_size as u32) * 4)
            .max(31776u32);
        let func = module.get_function("megakernel_f32")?;
        let blocks_per_sm = func.max_active_blocks_per_sm(256, shared_mem as usize)?;
        eprintln!(
            "  megakernel(paged): shared_mem={shared_mem} blocks_per_sm={blocks_per_sm} NUM_CUS={NUM_CUS}"
        );
        // NUM_CUS=48 = WGP count (MultiprocessorCount on RDNA3). Max cooperative blocks = blocks_per_sm * WGPs.
        let blocks_per_sm_clamped = blocks_per_sm.max(1) as u32;
        let num_blocks = blocks_per_sm_clamped * NUM_CUS;
        let mut instructions: Vec<Instruction> = Vec::new();

        let hs = cfg.hidden_size;
        let nh_gdn = cfg.linear_num_heads;
        let nvh_gdn = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let conv_dim = nh_gdn * kd * 2 + nvh_gdn * vd;
        let ck = cfg.linear_conv_kernel_dim;
        let _nqh = cfg.num_q_heads;
        let _nkh = cfg.num_kv_heads;
        let _hd = cfg.head_dim;
        let _is = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;

        // === Embedding: N lookups into prefill_bufs.hidden ===
        let embedding_inst_idx = instructions.len();
        for t in 0..n {
            let inst = EmbeddingInst::new(
                div_ceil(hs as u32, 256),
                unsafe { prefill_bufs.hidden.as_write_ptr().add(t * hs) },
                model.embed_weight.as_ptr(),
                tokens[t] as i32,
                hs as i32,
            );
            let inst = if t + 1 < n { inst.no_sync() } else { inst };
            instructions.push(inst.into_inst());
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
        let mut _attn_layer_count = 0usize;

        for layer_i in 0..cfg.num_layers {
            use crate::model::LayerType;
            if cfg.layers[layer_i].layer_type == LayerType::Attention {
                let w = match &model.layers[layer_i] {
                    LayerWeights::Attention(w) => w,
                    _ => panic!("expected attention layer"),
                };
                let kv_cache = model
                    .legacy_kv_caches
                    .as_ref()
                    .expect("legacy KV cache not initialized for flat prefill")
                    .get(kv_idx)
                    .expect("missing legacy KV cache");

                Self::emit_attention_layer(
                    cfg,
                    w,
                    act,
                    Some((prefill_bufs, n)),
                    &AttentionVariant::Prefill {
                        kv_cache,
                        start_pos,
                    },
                    &mut instructions,
                    &mut Vec::new(),
                    &mut Vec::new(),
                    &mut Vec::new(),
                    &mut Vec::new(),
                    &mut Vec::new(),
                    &mut Vec::new(),
                );

                // Batched FFN
                Self::compile_ffn_batched(
                    cfg,
                    &model.layers[layer_i],
                    prefill_bufs,
                    n,
                    &mut instructions,
                );
                _attn_layer_count += 1;
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
                    instructions.push(RmsNormInst::new(
                        rmsnorm_opcode(cfg.rms_norm_one_plus_w),
                        n as u32,
                        prefill_bufs.normed.as_write_ptr(),
                        prefill_bufs.hidden.as_ptr(),
                        w.input_norm.as_ptr(),
                        hs as i32,
                        eps,
                    ).into_inst());
                }

                // QKV projection (batch=N)
                emit_batched_linear_proj(
                    &w.w_qkv,
                    prefill_bufs.qkv.as_write_ptr(),
                    prefill_bufs.normed.as_ptr(),
                    conv_dim,
                    hs,
                    n,
                    true,
                    &mut instructions,
                );

                // a projection (batch=N)
                emit_batched_linear_proj(
                    &w.w_a,
                    prefill_bufs.a_proj.as_write_ptr(),
                    prefill_bufs.normed.as_ptr(),
                    nvh_gdn,
                    hs,
                    n,
                    true,
                    &mut instructions,
                );

                // b projection (batch=N)
                emit_batched_linear_proj(
                    &w.w_b,
                    prefill_bufs.b_proj.as_write_ptr(),
                    prefill_bufs.normed.as_ptr(),
                    nvh_gdn,
                    hs,
                    n,
                    true,
                    &mut instructions,
                );

                // z projection (batch=N) — SYNC before sequential part
                emit_batched_linear_proj(
                    &w.w_z,
                    prefill_bufs.z_proj.as_write_ptr(),
                    prefill_bufs.normed.as_ptr(),
                    nvh_gdn * vd,
                    hs,
                    n,
                    false,
                    &mut instructions,
                );

                // --- Sequential per-token: conv1d, gate, recurrence, norm, output, residual ---
                let q_dim = nh_gdn * kd;
                let k_dim = nh_gdn * kd;
                let v_dim = nvh_gdn * vd;

                for t in 0..n {
                    // Conv1d on Q (from batched qkv[t])
                    instructions.push(Conv1dInst::new(
                        div_ceil(q_dim as u32, 256),
                        conv_state.as_write_ptr(),
                        unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim) },
                        w.conv1d_weight_q.as_ptr(),
                        act.q_gdn.as_write_ptr(),
                        q_dim as i32,
                        ck as i32,
                    ).no_sync().into_inst());

                    // Conv1d on K
                    instructions.push(Conv1dInst::new(
                        div_ceil(k_dim as u32, 256),
                        unsafe { conv_state.as_write_ptr().add(q_dim * (ck - 1)) },
                        unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim + q_dim) },
                        w.conv1d_weight_k.as_ptr(),
                        act.k_gdn.as_write_ptr(),
                        k_dim as i32,
                        ck as i32,
                    ).no_sync().into_inst());

                    // Conv1d on V
                    instructions.push(Conv1dInst::new(
                        div_ceil(v_dim as u32, 256),
                        unsafe { conv_state.as_write_ptr().add((q_dim + k_dim) * (ck - 1)) },
                        unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim + q_dim + k_dim) },
                        w.conv1d_weight_v.as_ptr(),
                        act.v_gdn.as_write_ptr(),
                        v_dim as i32,
                        ck as i32,
                    ).into_inst());

                    // GDN gate (from batched a_proj[t])
                    instructions.push(GdnGateInst::new(
                        div_ceil(nvh_gdn as u32, 256),
                        act.gate_gdn.as_write_ptr(),
                        unsafe { prefill_bufs.a_proj.as_ptr().add(t * nvh_gdn) },
                        w.a_log.as_ptr(),
                        w.dt_bias.as_ptr(),
                        nvh_gdn as i32,
                    ).into_inst());

                    // GDN recurrence (nvh heads with GQA key sharing)
                    {
                        let gqa_group = nvh_gdn / nh_gdn;
                        instructions.push(GdnRecurInst::new(
                            nvh_gdn as u32,
                            act.q_gdn.as_ptr(),
                            act.k_gdn.as_ptr(),
                            act.v_gdn.as_ptr(),
                            act.gate_gdn.as_ptr(),
                            unsafe { prefill_bufs.b_proj.as_ptr().add(t * nvh_gdn) },
                            gdn_state.recurrent.as_write_ptr(),
                            act.recurrent_out.as_write_ptr(),
                            kd as i32,
                            vd as i32,
                            gqa_group as i32,
                        ).into_inst());
                    }

                    // RMSNorm gated (z from batched z_proj[t])
                    instructions.push(RmsNormGateInst::new(
                        nvh_gdn as u32,
                        act.normed_gated.as_write_ptr(),
                        act.recurrent_out.as_ptr(),
                        unsafe { prefill_bufs.z_proj.as_ptr().add(t * nvh_gdn * vd) },
                        w.output_norm.as_ptr(),
                        nvh_gdn as i32,
                        vd as i32,
                        eps,
                    ).into_inst());

                    // Output projection
                    {
                        let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
                        inst.words[1] = act.out_proj.as_write_ptr() as u64;
                        emit_linear_proj(&mut inst, &w.w_out, 2);
                        inst.words[3] = act.normed_gated.as_ptr() as u64;
                        inst.words[4] = hs as u64;
                        inst.words[5] = (nvh_gdn * vd) as u64;
                        instructions.push(inst);
                    }

                    // Residual: hidden[t] = out_proj + hidden[t]
                    {
                        let hidden_t = unsafe { prefill_bufs.hidden.as_write_ptr().add(t * hs) };
                        instructions.push(ResidualAddInst::new(
                            div_ceil(hs as u32, 256),
                            hidden_t,
                            act.out_proj.as_ptr(),
                            hidden_t,
                            hs as i32,
                        ).into_inst());
                    }
                }

                // --- Batched FFN ---
                Self::compile_ffn_batched(
                    cfg,
                    &model.layers[layer_i],
                    prefill_bufs,
                    n,
                    &mut instructions,
                );

                gdn_idx += 1;
            }
        }

        // === Final norm + LM head (last token only) ===
        // Copy last token's hidden to act.hidden
        instructions.push(D2dCopyInst::new(
            div_ceil(hs as u32, 256),
            act.hidden.as_write_ptr(),
            unsafe { prefill_bufs.hidden.as_ptr().add((n - 1) * hs) },
            hs as i32,
        ).into_inst());
        // D2D copy normed ← hidden (buffer copy before final RMSNorm)
        instructions.push(D2dCopyInst::new(
            div_ceil(hs as u32, 256),
            act.normed.as_write_ptr(),
            act.hidden.as_ptr(),
            hs as i32,
        ).into_inst());
        // Final RMSNorm: hidden ← rmsnorm(normed, final_norm_weight)
        instructions.push(RmsNormInst::new(
            rmsnorm_opcode(cfg.rms_norm_one_plus_w),
            1,
            act.hidden.as_write_ptr(),
            act.normed.as_ptr(),
            model.final_norm_weight.as_ptr(),
            hs as i32,
            eps,
        ).into_inst());
        // LM head
        {
            let lm_w_ptr = if cfg.tie_word_embeddings {
                model.embed_weight.as_ptr()
            } else {
                model.lm_head_weight.as_ptr()
            };
            instructions.push(LinearProjInst::new(
                OP_LINEAR_PROJ,
                cfg.vocab_size as u32,
                act.logits.as_write_ptr(),
                lm_w_ptr as *const u8,
                act.hidden.as_ptr(),
                cfg.vocab_size as i32,
                hs as i32,
                0,
            ).into_inst());
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
            shared_mem,
            device,
            embedding_inst_idx,
            _mrope_inst_indices: Vec::new(),
            gqa_attn_inst_indices: Vec::new(),
            kv_write_indices: Vec::new(),
            kv_base_ptrs: Vec::new(),
            position_ids_dev_ptr: act.position_ids.as_ptr() as u64,
            max_seq_len: cfg.max_seq_len as u32,
            paged: false,
            page_table: None,
            position_table: None,
            attn_paged_inst_indices: Vec::new(),
            attn_quant_inst_indices: Vec::new(),
            last_page_table_len: 0,
            kv_stride_paged: cfg.num_kv_heads * cfg.head_dim,
            quant_scratch: None,
            quant_page_table: None,
            last_quant_page_table_len: 0,
            quantized_kv: false,
            dump_buffer: None,
            dump_counter: None,
            dump_capacity: 0,
            num_kv_heads_attn: cfg.num_kv_heads,
            head_dim_attn: cfg.head_dim,
            barrier_layer_map: Vec::new(),
            multi_gpu_attn_boundaries: Vec::new(),
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
        prefill: Option<(&PrefillBuffers, usize)>,
        variant: &AttentionVariant,
        instructions: &mut Vec<Instruction>,
        mrope_indices: &mut Vec<usize>,
        gqa_attn_inst_indices: &mut Vec<usize>,
        kv_write_indices: &mut Vec<Vec<(usize, usize)>>,
        kv_base_ptrs: &mut Vec<(u64, u64)>,
        attn_paged_indices: &mut Vec<usize>,
        attn_quant_indices: &mut Vec<usize>,
    ) {
        let hs = cfg.hidden_size;
        let nqh = cfg.num_q_heads;
        let nkh = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let eps = cfg.rms_norm_eps;
        let rd = cfg.rope_dim;
        let n = prefill.as_ref().map_or(1, |&(_, n)| n);

        // Buffer pointers — either prefill or single-token activation buffers
        let (
            normed_ptr,
            hidden_ptr,
            q_gate_attn_ptr,
            k_attn_ptr,
            v_attn_ptr,
            q_attn_ptr,
            gate_attn_ptr,
            attn_out_ptr,
            gated_out_ptr,
            out_proj_ptr,
            position_ids_ptr,
            ffn_hidden_ptr,
        ) = if let Some((pb, _)) = &prefill {
            (
                pb.normed.as_write_ptr(),
                pb.hidden.as_write_ptr(),
                pb.q_gate_attn.as_write_ptr(),
                pb.k_attn.as_write_ptr(),
                pb.v_attn.as_write_ptr(),
                pb.q_attn.as_write_ptr(),
                pb.gate_attn.as_write_ptr(),
                pb.attn_out.as_write_ptr(),
                pb.gated_out.as_write_ptr(),
                pb.out_proj.as_write_ptr(),
                pb.position_ids.as_ptr(),
                pb.hidden.as_write_ptr(),
            )
        } else {
            (
                act.normed.as_write_ptr(),
                act.hidden.as_write_ptr(),
                act.q_gate_attn.as_write_ptr(),
                act.k_attn.as_write_ptr(),
                act.v_attn.as_write_ptr(),
                act.q_attn.as_write_ptr(),
                act.gate_attn.as_write_ptr(),
                act.attn_out.as_write_ptr(),
                act.gated_out.as_write_ptr(),
                act.out_proj.as_write_ptr(),
                act.position_ids.as_ptr(),
                act.hidden.as_write_ptr(),
            )
        };

        // 1. RMSNorm
        instructions.push(RmsNormInst::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), n as u32, normed_ptr, hidden_ptr, w.input_norm.as_ptr(), hs as i32, eps).into_inst());

        // 2. Q(+gate), K, V projections
        let q_mult = if cfg.has_output_gate { 2 } else { 1 };
        emit_batched_linear_proj(&w.w_q_gate, q_gate_attn_ptr, normed_ptr, nqh * hd * q_mult, hs, n, true, instructions);
        emit_batched_linear_proj(&w.w_k, k_attn_ptr, normed_ptr, nkh * hd, hs, n, true, instructions);
        emit_batched_linear_proj(&w.w_v, v_attn_ptr, normed_ptr, nkh * hd, hs, n, false, instructions);

        // 3. Deinterleave Q+gate → Q, gate
        if !cfg.has_output_gate {
            let total = n * nqh * hd;
            instructions.push(D2dCopyInst::new(div_ceil(total as u32, 256), q_attn_ptr, q_gate_attn_ptr as *const f32, total as i32).into_inst());
        } else {
            let total_elems = n * nqh * hd;
            instructions.push(DeinterleaveInst::new(div_ceil(total_elems as u32, 256), q_attn_ptr, gate_attn_ptr, q_gate_attn_ptr as *const f32, nqh as i32, hd as i32, n as i32).into_inst());
        }

        // 4a. KV write for PagedKv — BEFORE QK-norm so cache stores pre-norm K/V.
        //     Pre-norm K has full dynamic range; quantizing post-norm K (±0.06) is catastrophic
        //     (TVD=0.97). See exterior_algebra kb-20260328-115542-e172dd.
        //     QK-norm is applied at attention time after dequant (in op_attn_paged / op_attn_paged_quant).
        let paged_kv_write_before_norm = matches!(variant, AttentionVariant::PagedKv { .. });
        let mut paged_layer_k_offset: u64 = 0;
        let mut paged_layer_v_offset: u64 = 0;
        if paged_kv_write_before_norm {
            if let AttentionVariant::PagedKv { attn_layer_index } = &variant {
                let kv_stride = nkh * hd;
                let chunk_tokens: usize = 64;
                paged_layer_k_offset =
                    (*attn_layer_index * 2 * chunk_tokens * kv_stride * std::mem::size_of::<f32>())
                        as u64;
                paged_layer_v_offset = paged_layer_k_offset
                    + (chunk_tokens * kv_stride * std::mem::size_of::<f32>()) as u64;
                let _chunk_head_stride = chunk_tokens * hd;
                let mut head_indices = Vec::new();
                for h in 0..nkh {
                    let k_copy_idx = instructions.len();
                    let ki = D2dCopyInst::new(div_ceil(hd as u32, 256), std::ptr::null_mut::<f32>(), unsafe { k_attn_ptr.add(h * hd) as *const f32 }, hd as i32).no_sync();
                    instructions.push(ki.into_inst());
                    let v_copy_idx = instructions.len();
                    let vi = D2dCopyInst::new(div_ceil(hd as u32, 256), std::ptr::null_mut::<f32>(), unsafe { v_attn_ptr.add(h * hd) as *const f32 }, hd as i32);
                    let vi = if h < nkh - 1 { vi.no_sync() } else { vi };
                    instructions.push(vi.into_inst());
                    head_indices.push((k_copy_idx, v_copy_idx));
                }
                kv_write_indices.push(head_indices);
            }
        }

        // 4b. QK norm (only for models that have qk_norm weights — e.g. Qwen3.5, not Mistral)
        if cfg.has_qk_norm {
            instructions.push(QkNormInst::new(
                (n * (nqh + nkh)) as u32,
                q_attn_ptr,
                k_attn_ptr,
                w.q_norm.as_ptr(),
                w.k_norm.as_ptr(),
                nqh as i32,
                nkh as i32,
                hd as i32,
                eps,
                if n > 1 { n as i32 } else { 0 },
            ).into_inst());
        }

        // Steps 5/6: variant-specific attention ops. PagedKv KV write already done above.

        // Variant-specific: KV write placement and attention op
        match variant {
            AttentionVariant::FlatKv { kv_cache } => {
                // mRoPE first (step 5), then KV write (step 6), then GQA (step 7)
                let mrope_idx = instructions.len();
                mrope_indices.push(mrope_idx);
                instructions.push(MropeInst::new(
                    (n * (nqh + nkh)) as u32,
                    q_attn_ptr, k_attn_ptr,
                    act.inv_freq.as_ptr(), position_ids_ptr,
                    nqh as i32, nkh as i32, hd as i32, rd as i32,
                    cfg.mrope_sections()[0] as i32,
                    cfg.mrope_sections()[1] as i32,
                    cfg.mrope_sections()[2] as i32,
                    n as i32,
                ).into_inst());

                // Per-head D2D_COPY for [H,T,D] layout: each head's cache slot
                // is at base + h * max_seq_len * hd, updated at runtime with position offset.
                let max_sl = cfg.max_seq_len;
                let head_stride = max_sl * hd; // elements between consecutive heads
                {
                    let mut head_indices = Vec::new();
                    for h in 0..nkh {
                        let k_copy_idx = instructions.len();
                        let ki = D2dCopyInst::new(
                            div_ceil(hd as u32, 256),
                            unsafe { kv_cache.k.as_write_ptr().add(h * head_stride) },
                            unsafe { k_attn_ptr.add(h * hd) as *const f32 },
                            hd as i32,
                        ).no_sync();
                        instructions.push(ki.into_inst());
                        let v_copy_idx = instructions.len();
                        let vi = D2dCopyInst::new(
                            div_ceil(hd as u32, 256),
                            unsafe { kv_cache.v.as_write_ptr().add(h * head_stride) },
                            unsafe { v_attn_ptr.add(h * hd) as *const f32 },
                            hd as i32,
                        );
                        let vi = if h < nkh - 1 { vi.no_sync() } else { vi };
                        instructions.push(vi.into_inst());
                        head_indices.push((k_copy_idx, v_copy_idx));
                    }
                    kv_write_indices.push(head_indices);
                    kv_base_ptrs.push((kv_cache.k.as_ptr() as u64, kv_cache.v.as_ptr() as u64));
                }

                // GQA attention
                let gqa_idx = instructions.len();
                gqa_attn_inst_indices.push(gqa_idx);
                instructions.push(GqaAttnInst::new(
                    nqh as u32,
                    attn_out_ptr,
                    q_attn_ptr as *const f32,
                    kv_cache.k.as_ptr(),
                    kv_cache.v.as_ptr(),
                    nqh as i32, nkh as i32, hd as i32,
                    1, // seq_len — updated per step
                    cfg.max_seq_len as i32,
                ).into_inst());
            }

            AttentionVariant::PagedKv { attn_layer_index } => {
                // KV write already emitted above (step 4a, before QK-norm).
                // Cache now stores pre-QK-norm K/V for quantization quality.

                // mRoPE after KV write (applied to working Q/K buffers, not cache)
                let mrope_idx = instructions.len();
                mrope_indices.push(mrope_idx);
                instructions.push(MropeInst::new(
                    (nqh + nkh) as u32,
                    q_attn_ptr, k_attn_ptr,
                    act.inv_freq.as_ptr(), position_ids_ptr,
                    nqh as i32, nkh as i32, hd as i32, rd as i32,
                    cfg.mrope_sections()[0] as i32,
                    cfg.mrope_sections()[1] as i32,
                    cfg.mrope_sections()[2] as i32,
                    1, // batch=1 for decode
                ).into_inst());

                // OP_ATTN_PAGED_Q: quantized attention (grid_x=0 initially, patched when chunks seal)
                let quant_idx = instructions.len();
                attn_quant_indices.push(quant_idx);
                {
                    use crate::paged_kv::quantized_kv_offsets;
                    let chunk_tokens: usize = CHUNK_TOKENS;
                    let (q1d, q1s, rd_off, rs) =
                        quantized_kv_offsets(cfg, chunk_tokens, *attn_layer_index, false);
                    let k_norm_ptr = if cfg.has_qk_norm { w.k_norm.as_ptr() } else { std::ptr::null() };
                    instructions.push(AttnPagedQInst::new(
                        q_attn_ptr as *const f32,
                        act.inv_freq.as_ptr(),
                        nqh as i32, nkh as i32, hd as i32,
                        chunk_tokens as i32,
                        rd as i32,
                        q1d as u64, q1s as u64, rd_off as u64, rs as u64,
                        k_norm_ptr,
                    ).no_sync().into_inst()); // no sync between quant and f32 attention
                }

                // OP_ATTN_PAGED: f32 attention on active chunk + merge from scratch
                let paged_idx = instructions.len();
                attn_paged_indices.push(paged_idx);
                {
                    let k_norm_ptr = if cfg.has_qk_norm { w.k_norm.as_ptr() } else { std::ptr::null() };
                    instructions.push(AttnPagedInst::new(
                        nqh as u32,
                        attn_out_ptr,
                        q_attn_ptr as *const f32,
                        act.inv_freq.as_ptr(),
                        nqh as i32, nkh as i32, hd as i32,
                        1, // seq_len — patched per step
                        CHUNK_TOKENS as i32,
                        rd as i32,
                        paged_layer_k_offset,
                        paged_layer_v_offset,
                        k_norm_ptr,
                    ).into_inst());
                }
            }

            AttentionVariant::Prefill {
                kv_cache,
                start_pos,
            } => {
                // mRoPE first (batched)
                let mrope_idx = instructions.len();
                mrope_indices.push(mrope_idx);
                instructions.push(MropeInst::new(
                    (n * (nqh + nkh)) as u32,
                    q_attn_ptr, k_attn_ptr,
                    act.inv_freq.as_ptr(), position_ids_ptr,
                    nqh as i32, nkh as i32, hd as i32, rd as i32,
                    cfg.mrope_sections()[0] as i32,
                    cfg.mrope_sections()[1] as i32,
                    cfg.mrope_sections()[2] as i32,
                    n as i32,
                ).into_inst());

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
                        let k_dst = unsafe { kv_cache.k.as_write_ptr().add(dst_off) };
                        let k_src = unsafe { k_attn_ptr.add(src_off) };
                        let ki = D2dCopyInst::new(
                            div_ceil(hd as u32, 256), k_dst, k_src as *const f32, hd as i32,
                        ).no_sync();
                        instructions.push(ki.into_inst());

                        let v_dst = unsafe { kv_cache.v.as_write_ptr().add(dst_off) };
                        let v_src = unsafe { v_attn_ptr.add(src_off) };
                        let vi = D2dCopyInst::new(
                            div_ceil(hd as u32, 256), v_dst, v_src as *const f32, hd as i32,
                        );
                        let vi = if t == n - 1 && h == nkh - 1 { vi } else { vi.no_sync() };
                        instructions.push(vi.into_inst());
                    }
                }

                // OP_ATTN_PREFILL
                instructions.push(AttnPrefillInst::new(
                    (n * nqh) as u32,
                    attn_out_ptr,
                    q_attn_ptr as *const f32,
                    kv_cache.k.as_ptr(),
                    kv_cache.v.as_ptr(),
                    nqh as i32, nkh as i32, hd as i32,
                    *start_pos as i32,
                    n as i32,
                    cfg.max_seq_len as i32,
                ).into_inst());
            }
        }

        // 10. Output gate (Qwen3.5 only) or pass-through
        let final_attn_ptr = if cfg.has_output_gate {
            let gate_size = n * nqh * hd;
            instructions.push(OutputGateInst::new(
                div_ceil(gate_size as u32, 256),
                gated_out_ptr,
                attn_out_ptr as *const f32,
                gate_attn_ptr as *const f32,
                gate_size as i32,
            ).into_inst());
            gated_out_ptr
        } else {
            attn_out_ptr // skip output gate, use attention output directly
        };

        // 11. Output projection + residual
        emit_batched_linear_proj(
            &w.w_o,
            out_proj_ptr,
            final_attn_ptr,
            hs,
            nqh * hd,
            n,
            false,
            instructions,
        );
        if n > 1 {
            // Batched residual: hidden = hidden + out_proj (N tokens)
            let total = n * hs;
            instructions.push(ResidualAddInst::new(
                div_ceil(total as u32, 256),
                ffn_hidden_ptr,
                out_proj_ptr as *const f32,
                ffn_hidden_ptr as *const f32,
                total as i32,
            ).into_inst());
        } else if prefill.is_some() {
            // Single-token prefill: residual uses prefill buffer (hidden_ptr = pb.hidden)
            instructions.push(ResidualAddInst::new(
                div_ceil(hs as u32, 256),
                hidden_ptr,
                out_proj_ptr as *const f32,
                hidden_ptr as *const f32,
                hs as i32,
            ).into_inst());
        } else {
            // Single-token decode: two-step residual via act.residual scratch
            instructions.push(D2dCopyInst::new(
                div_ceil(hs as u32, 256),
                act.residual.as_write_ptr(),
                hidden_ptr as *const f32,
                hs as i32,
            ).into_inst());
            instructions.push(ResidualAddInst::new(
                div_ceil(hs as u32, 256),
                act.hidden.as_write_ptr(),
                out_proj_ptr as *const f32,
                act.residual.as_ptr(),
                hs as i32,
            ).into_inst());
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
        let nvh = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let ck = cfg.linear_conv_kernel_dim;
        let qkv_dim = nh * kd * 2 + nvh * vd;
        let eps = cfg.rms_norm_eps;

        // 1. RMSNorm
        instructions.push(RmsNormInst::new(
            rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
            act.normed.as_write_ptr(), act.hidden.as_ptr(), w.input_norm.as_ptr(), hs as i32, eps,
        ).into_inst());

        // 2. QKV projection — NO_SYNC: next 3 instructions (a/b/z proj) read normed, not qkv
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.w_qkv);
            instructions.push(LinearProjInst::new(op, qkv_dim as u32, act.qkv.as_write_ptr(), wp, act.normed.as_ptr(), qkv_dim as i32, hs as i32, 0).no_sync().into_inst());
        }

        // 3. Project a, b, z
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.w_a);
            instructions.push(LinearProjInst::new(op, nvh as u32, act.a_proj.as_write_ptr(), wp, act.normed.as_ptr(), nvh as i32, hs as i32, 0).no_sync().into_inst());
        }
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.w_b);
            instructions.push(LinearProjInst::new(op, nvh as u32, act.b_proj.as_write_ptr(), wp, act.normed.as_ptr(), nvh as i32, hs as i32, 0).no_sync().into_inst());
        }
        // z proj: SYNC ensures QKV+a+b+z all complete before conv1d reads qkv
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.w_z);
            instructions.push(LinearProjInst::new(op, (nvh * vd) as u32, act.z_proj.as_write_ptr(), wp, act.normed.as_ptr(), (nvh * vd) as i32, hs as i32, 0).into_inst());
        }

        // 4. Causal conv1d on QKV (3 separate calls for q, k, v slices)
        let q_dim = nh * kd;
        let k_dim = nh * kd;
        let v_dim = nvh * vd;

        // Conv on Q — NO_SYNC
        instructions.push(Conv1dInst::new(
            div_ceil(q_dim as u32, 256),
            conv_state.as_write_ptr(), act.qkv.as_ptr(), w.conv1d_weight_q.as_ptr(), act.q_gdn.as_write_ptr(),
            q_dim as i32, ck as i32,
        ).no_sync().into_inst());

        // Conv on K — NO_SYNC
        instructions.push(Conv1dInst::new(
            div_ceil(k_dim as u32, 256),
            unsafe { conv_state.as_write_ptr().add(q_dim * (ck - 1)) },
            unsafe { act.qkv.as_ptr().add(q_dim) },
            w.conv1d_weight_k.as_ptr(), act.k_gdn.as_write_ptr(),
            k_dim as i32, ck as i32,
        ).no_sync().into_inst());

        // Conv on V
        instructions.push(Conv1dInst::new(
            div_ceil(v_dim as u32, 256),
            unsafe { conv_state.as_write_ptr().add((q_dim + k_dim) * (ck - 1)) },
            unsafe { act.qkv.as_ptr().add(q_dim + k_dim) },
            w.conv1d_weight_v.as_ptr(), act.v_gdn.as_write_ptr(),
            v_dim as i32, ck as i32,
        ).into_inst());

        // 5. GDN gate
        let gqa_group = nvh / nh;
        instructions.push(GdnGateInst::new(
            div_ceil(nvh as u32, 256),
            act.gate_gdn.as_write_ptr(), act.a_proj.as_ptr(), w.a_log.as_ptr(), w.dt_bias.as_ptr(), nvh as i32,
        ).into_inst());

        // 6. GDN recurrent
        instructions.push(GdnRecurInst::new(
            nvh as u32,
            act.q_gdn.as_ptr(), act.k_gdn.as_ptr(), act.v_gdn.as_ptr(), act.gate_gdn.as_ptr(), act.b_proj.as_ptr(),
            gdn_state.recurrent.as_write_ptr(), act.recurrent_out.as_write_ptr(),
            kd as i32, vd as i32, gqa_group as i32,
        ).into_inst());

        // 7. RMSNorm gated
        instructions.push(RmsNormGateInst::new(
            nvh as u32,
            act.normed_gated.as_write_ptr(), act.recurrent_out.as_ptr(), act.z_proj.as_ptr(), w.output_norm.as_ptr(),
            nvh as i32, vd as i32, eps,
        ).into_inst());

        // 8. Output projection
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.w_out);
            instructions.push(LinearProjInst::new(op, hs as u32, act.out_proj.as_write_ptr(), wp, act.normed_gated.as_ptr(), hs as i32, (nvh * vd) as i32, 0).into_inst());
        }

        // 9. Residual
        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
        instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), act.out_proj.as_ptr(), act.residual.as_ptr(), hs as i32).into_inst());
    }

    fn compile_mamba2_layer(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        act: &ActivationBuffers,
        state: &Mamba2State,
        instructions: &mut Vec<Instruction>,
    ) {
        let w = match layer {
            LayerWeights::Mamba2(w) => w,
            _ => panic!("expected Mamba2 layer"),
        };
        let (nh, hd, sd, ck, ng, cd) = match &cfg.recurrent_kind {
            RecurrentLayerKind::Mamba2 {
                num_heads,
                head_dim,
                state_dim,
                conv_kernel,
                n_groups,
                conv_dim,
                ..
            } => (*num_heads, *head_dim, *state_dim, *conv_kernel, *n_groups, *conv_dim),
            _ => panic!("compile_mamba2_layer but no Mamba2 config"),
        };
        let hs = cfg.hidden_size;
        let intermediate = nh * hd;         // gate size + ssm output size
        let in_proj_size = intermediate + cd + nh; // gate + xBC + dt
        let eps = cfg.rms_norm_eps;
        let group_size = intermediate / ng; // value_dim per norm group

        // 1. RMSNorm
        instructions.push(RmsNormInst::new(
            rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
            act.normed.as_write_ptr(), act.hidden.as_ptr(), w.input_norm.as_ptr(), hs as i32, eps,
        ).into_inst());

        // 2. in_proj
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.in_proj);
            instructions.push(LinearProjInst::new(op, in_proj_size as u32, act.mamba2_in_proj.as_write_ptr(), wp, act.normed.as_ptr(), in_proj_size as i32, hs as i32, 0).into_inst());
        }

        // 3. conv1d
        instructions.push(Mamba2Conv1dInst::new(
            div_ceil(cd as u32, 256),
            state.conv.as_write_ptr(),
            unsafe { act.mamba2_in_proj.as_ptr().add(intermediate) },
            w.conv1d_weight.as_ptr(),
            w.conv1d_bias.as_ptr(),
            act.mamba2_conv_out.as_write_ptr(),
            cd as i32, ck as i32,
        ).into_inst());

        // 4. SSM update
        instructions.push(SsmUpdateInst::new(
            nh as u32,
            state.ssm.as_write_ptr(),
            act.mamba2_conv_out.as_ptr(),
            unsafe { act.mamba2_in_proj.as_ptr().add(intermediate + cd) },
            w.dt_bias.as_ptr(),
            w.a_log.as_ptr(),
            unsafe { act.mamba2_conv_out.as_ptr().add(intermediate) },
            unsafe { act.mamba2_conv_out.as_ptr().add(intermediate + ng * sd) },
            w.d.as_ptr(),
            act.mamba2_ssm_out.as_write_ptr(),
            nh as i32, hd as i32, sd as i32, ng as i32,
        ).into_inst());

        // 5. mamba2_norm_gated
        instructions.push(Mamba2NormGatedInst::new(
            ng as u32,
            act.mamba2_conv_out.as_write_ptr(),
            act.mamba2_ssm_out.as_ptr(),
            act.mamba2_in_proj.as_ptr(),
            w.norm_weight.as_ptr(),
            ng as i32, group_size as i32, eps,
        ).into_inst());

        // 6. out_proj
        {
            let (op, wp) = linear_proj_opcode_ptr(&w.out_proj);
            instructions.push(LinearProjInst::new(op, hs as u32, act.out_proj.as_write_ptr(), wp, act.mamba2_conv_out.as_ptr(), hs as i32, intermediate as i32, 0).into_inst());
        }

        // 7. Residual
        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
        instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), act.out_proj.as_ptr(), act.residual.as_ptr(), hs as i32).into_inst());
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
            cfg,
            w,
            act,
            None,
            &AttentionVariant::FlatKv { kv_cache },
            instructions,
            mrope_indices,
            gqa_indices,
            kv_write_indices,
            kv_base_ptrs,
            &mut Vec::new(),
            &mut Vec::new(),
        );
    }

    /// Multi-GPU attention: emit only RMSNorm + output-gate + O-proj + residual.
    /// QKV projection, deinterleave, QK-norm, KV-write, mRoPE, and GQA are handled by
    /// dispatch_head_parallel_attention() (runs on all GPUs in parallel via persistent workers).
    ///
    /// Records `(rmsnorm_idx, output_gate_idx)` into `multi_gpu_boundaries`:
    ///   rmsnorm_idx     = index of RMSNorm instruction (flush and dispatch after this)
    ///   output_gate_idx = index of first post-GQA instruction (resume megakernel here)
    fn compile_attention_layer_multi_gpu(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        act: &ActivationBuffers,
        instructions: &mut Vec<Instruction>,
        multi_gpu_boundaries: &mut Vec<(usize, usize)>,
    ) {
        let w = match layer {
            LayerWeights::Attention(w) => w,
            _ => panic!("expected attention layer"),
        };
        let hs = cfg.hidden_size;
        let nqh = cfg.num_q_heads;
        let hd = cfg.head_dim;
        let eps = cfg.rms_norm_eps;

        // Activation buffer pointers (single-token decode only; prefill uses full path)
        let hidden_ptr = act.hidden.as_write_ptr();
        let normed_ptr = act.normed.as_write_ptr();
        let attn_out_ptr = act.attn_out.as_write_ptr();
        let gated_out_ptr = act.gated_out.as_write_ptr();
        let gate_attn_ptr = act.gate_attn.as_write_ptr();
        let out_proj_ptr = act.out_proj.as_write_ptr();

        // 1. RMSNorm
        instructions.push(RmsNormInst::new(
            rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
            normed_ptr, hidden_ptr, w.input_norm.as_ptr(), hs as i32, eps,
        ).into_inst());
        // 1b. Copy normed → normed_stage (GART/MappedHostBuffer, write-through to system RAM).
        // This is the flush boundary for P2P dispatch: after GPU 0 acks this instruction,
        // normed_stage is coherently visible to GPUs 1-3 without L2 cache coherence issues.
        // (Regular device VRAM is NOT coherent for PCIe P2P on RDNA3; GART is.)
        let normed_stage_copy_idx = instructions.len();
        instructions.push(D2dCopyInst::new(
            div_ceil(hs as u32, 256),
            act.normed_stage.as_write_ptr(),
            normed_ptr,
            hs as i32,
        ).into_inst());
        // Steps 2–9 (QKV proj, deinterleave, QK-norm, KV write, mRoPE, GQA) handled by dispatcher.
        // output_gate_idx points to the first instruction AFTER the dispatched block.
        let output_gate_idx = instructions.len(); // == normed_stage_copy_idx + 1

        // 10. Output gate (Qwen3.5) or pass-through
        let final_attn_ptr = if cfg.has_output_gate {
            let gate_size = nqh * hd;
            instructions.push(OutputGateInst::new(
                div_ceil(gate_size as u32, 256),
                gated_out_ptr, attn_out_ptr, gate_attn_ptr, gate_size as i32,
            ).into_inst());
            gated_out_ptr
        } else {
            attn_out_ptr
        };

        // 11. Output projection + residual (single-token decode path)
        emit_batched_linear_proj(
            &w.w_o,
            out_proj_ptr,
            final_attn_ptr,
            hs,
            nqh * hd,
            1,
            false,
            instructions,
        );
        // Two-step residual (single-token decode): copy hidden → residual, then add
        instructions.push(D2dCopyInst::new(
            div_ceil(hs as u32, 256),
            act.residual.as_write_ptr(), hidden_ptr, hs as i32,
        ).into_inst());
        instructions.push(ResidualAddInst::new(
            div_ceil(hs as u32, 256),
            act.hidden.as_write_ptr(), out_proj_ptr, act.residual.as_ptr(), hs as i32,
        ).into_inst());

        multi_gpu_boundaries.push((normed_stage_copy_idx, output_gate_idx));
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_attention_layer_paged(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        act: &ActivationBuffers,
        attn_layer_index: usize,
        instructions: &mut Vec<Instruction>,
        mrope_indices: &mut Vec<usize>,
        kv_write_indices: &mut Vec<Vec<(usize, usize)>>,
        kv_base_ptrs: &mut Vec<(u64, u64)>,
        attn_paged_indices: &mut Vec<usize>,
        attn_quant_indices: &mut Vec<usize>,
    ) {
        let w = match layer {
            LayerWeights::Attention(w) => w,
            _ => panic!("expected attention layer"),
        };
        Self::emit_attention_layer(
            cfg,
            w,
            act,
            None,
            &AttentionVariant::PagedKv { attn_layer_index },
            instructions,
            mrope_indices,
            &mut Vec::new(),
            kv_write_indices,
            kv_base_ptrs,
            attn_paged_indices,
            attn_quant_indices,
        );
    }

    fn compile_ffn_batched(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        bufs: &PrefillBuffers,
        n: usize,
        instructions: &mut Vec<Instruction>,
    ) {
        use crate::model::LinearWeight;
        let hs = cfg.hidden_size;
        let is = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;

        let (post_norm, w_gate, w_up, w_down) = match layer {
            LayerWeights::Gdn(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
            LayerWeights::Attention(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
            _ => panic!("prefill FFN only for Gdn/Attention layers"),
        };

        let all_bf16 = matches!(w_gate, LinearWeight::Bf16(_))
            && matches!(w_up, LinearWeight::Bf16(_))
            && matches!(w_down, LinearWeight::Bf16(_));

        if all_bf16 {
            // Fused path: OP_FFN_GATE_UP + OP_FFN_DOWN_RES (bf16 only, processes all N tokens)
            instructions.push(FfnGateUpInst::new(
                OP_FFN_GATE_UP, (is * n) as u32,
                bufs.ffn_act.as_write_ptr(), bufs.hidden.as_ptr(), post_norm.as_ptr(),
                w_gate.as_bf16_ptr() as *const u8, w_up.as_bf16_ptr() as *const u8,
                hs as i32, is as i32, eps, n as i32,
            ).into_inst());
            instructions.push(D2dCopyInst::new(
                div_ceil((n * hs) as u32, 256),
                bufs.residual.as_write_ptr(), bufs.hidden.as_ptr(), (n * hs) as i32,
            ).into_inst());
            instructions.push(FfnDownResInst::new(
                OP_FFN_DOWN_RES, (hs * n) as u32,
                bufs.hidden.as_write_ptr(), bufs.residual.as_ptr(),
                w_down.as_bf16_ptr() as *const u8, bufs.ffn_act.as_ptr(),
                hs as i32, is as i32, n as i32,
            ).into_inst());
        } else {
            // Unfused path for quantized weights: process one token at a time.
            // Uses ffn_gate_scratch/ffn_up_scratch/ffn_down_scratch as single-token intermediates.
            for t in 0..n {
                let hidden_t = unsafe { bufs.hidden.as_write_ptr().add(t * hs) };
                let normed_t = unsafe { bufs.normed.as_write_ptr().add(t * hs) };
                let residual_t = unsafe { bufs.residual.as_write_ptr().add(t * hs) };

                // D2D_COPY: hidden[t] → residual[t]  (no_sync: RMSNorm reads hidden, not residual)
                instructions.push(D2dCopyInst::new(
                    div_ceil(hs as u32, 256), residual_t, hidden_t, hs as i32,
                ).no_sync().into_inst());

                // RMSNorm: hidden[t] → normed[t]
                instructions.push(RmsNormInst::new(
                    rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
                    normed_t, hidden_t, post_norm.as_ptr(), hs as i32, eps,
                ).into_inst());

                // Gate: normed[t] → ffn_gate_scratch  (no_sync: up reads same normed)
                {
                    let mut inst = Instruction::new(OP_LINEAR_PROJ, is as u32);
                    emit_linear_proj(&mut inst, w_gate, 2);
                    inst.words[1] = bufs.ffn_gate_scratch.as_write_ptr() as u64;
                    inst.words[3] = normed_t as u64;
                    inst.words[4] = is as u64;
                    inst.words[5] = hs as u64;
                    inst.words[0] |= super::FLAG_NO_SYNC as u64;
                    instructions.push(inst);
                }

                // Up: normed[t] → ffn_up_scratch
                {
                    let mut inst = Instruction::new(OP_LINEAR_PROJ, is as u32);
                    emit_linear_proj(&mut inst, w_up, 2);
                    inst.words[1] = bufs.ffn_up_scratch.as_write_ptr() as u64;
                    inst.words[3] = normed_t as u64;
                    inst.words[4] = is as u64;
                    inst.words[5] = hs as u64;
                    instructions.push(inst);
                }

                // SiLU(gate) * up → ffn_act[t..t+is]
                let ffn_act_t = unsafe { bufs.ffn_act.as_write_ptr().add(t * is) };
                instructions.push(SiluMulInst::new(
                    div_ceil(is as u32, 256),
                    ffn_act_t, bufs.ffn_gate_scratch.as_ptr(), bufs.ffn_up_scratch.as_ptr(), is as i32,
                ).into_inst());

                // Down: ffn_act[t] → ffn_down_scratch
                {
                    let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
                    emit_linear_proj(&mut inst, w_down, 2);
                    inst.words[1] = bufs.ffn_down_scratch.as_write_ptr() as u64;
                    inst.words[3] = ffn_act_t as u64;
                    inst.words[4] = hs as u64;
                    inst.words[5] = is as u64;
                    instructions.push(inst);
                }

                // Residual: ffn_down_scratch + residual[t] → hidden[t]
                instructions.push(ResidualAddInst::new(
                    div_ceil(hs as u32, 256),
                    hidden_t, bufs.ffn_down_scratch.as_ptr(), residual_t, hs as i32,
                ).into_inst());
            }
        }
    }

    fn compile_ffn(
        cfg: &ModelConfig,
        layer: &LayerWeights,
        act: &ActivationBuffers,
        instructions: &mut Vec<Instruction>,
    ) {
        use crate::model::LinearWeight;
        let hs = cfg.hidden_size;
        let is = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;

        let (post_norm, w_gate, w_up, w_down) = match layer {
            LayerWeights::Gdn(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
            LayerWeights::Attention(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
            _ => panic!("prefill FFN only for Gdn/Attention layers"),
        };

        let all_bf16 = matches!(w_gate, LinearWeight::Bf16(_))
            && matches!(w_up, LinearWeight::Bf16(_))
            && matches!(w_down, LinearWeight::Bf16(_));

        let all_rnf4 = matches!(w_gate, LinearWeight::Packed(pw) if pw.format == crate::quant::WeightFormat::Rnf4G128)
            && matches!(w_up, LinearWeight::Packed(pw) if pw.format == crate::quant::WeightFormat::Rnf4G128)
            && matches!(w_down, LinearWeight::Packed(pw) if pw.format == crate::quant::WeightFormat::Rnf4G128);

        if all_bf16 {
            // Fused path: OP_FFN_GATE_UP + OP_FFN_DOWN_RES (bf16 only, batch=0=single token)
            instructions.push(FfnGateUpInst::new(
                OP_FFN_GATE_UP, is as u32,
                act.ffn_act.as_write_ptr(), act.hidden.as_ptr(), post_norm.as_ptr(),
                w_gate.as_bf16_ptr() as *const u8, w_up.as_bf16_ptr() as *const u8,
                hs as i32, is as i32, eps, 0,
            ).into_inst());
            instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
            instructions.push(FfnDownResInst::new(
                OP_FFN_DOWN_RES, hs as u32,
                act.hidden.as_write_ptr(), act.residual.as_ptr(), w_down.as_bf16_ptr() as *const u8, act.ffn_act.as_ptr(),
                hs as i32, is as i32, 0,
            ).into_inst());
        } else if all_rnf4 {
            let w_gate_ptr = match w_gate { LinearWeight::Packed(pw) => pw.data.as_ptr(), _ => unreachable!() };
            let w_up_ptr   = match w_up   { LinearWeight::Packed(pw) => pw.data.as_ptr(), _ => unreachable!() };
            let w_down_ptr = match w_down { LinearWeight::Packed(pw) => pw.data.as_ptr(), _ => unreachable!() };
            instructions.push(FfnGateUpInst::new(
                OP_FFN_GATE_UP_RNF4, is as u32,
                act.ffn_act.as_write_ptr(), act.hidden.as_ptr(), post_norm.as_ptr(),
                w_gate_ptr, w_up_ptr,
                hs as i32, is as i32, eps, 0,
            ).into_inst());
            instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
            instructions.push(FfnDownResInst::new(
                OP_FFN_DOWN_RES_RNF4, hs as u32,
                act.hidden.as_write_ptr(), act.residual.as_ptr(), w_down_ptr, act.ffn_act.as_ptr(),
                hs as i32, is as i32, 0,
            ).into_inst());
        } else {
            // Unfused path for quantized weights (decode n=1 only)
            instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).no_sync().into_inst());
            instructions.push(RmsNormInst::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1, act.normed.as_write_ptr(), act.hidden.as_ptr(), post_norm.as_ptr(), hs as i32, eps).into_inst());
            {
                let (op, wp) = linear_proj_opcode_ptr(w_gate);
                instructions.push(LinearProjInst::new(op, is as u32, act.ffn_gate.as_write_ptr(), wp, act.normed.as_ptr(), is as i32, hs as i32, 0).no_sync().into_inst());
            }
            {
                let (op, wp) = linear_proj_opcode_ptr(w_up);
                instructions.push(LinearProjInst::new(op, is as u32, act.ffn_up.as_write_ptr(), wp, act.normed.as_ptr(), is as i32, hs as i32, 0).into_inst());
            }
            instructions.push(SiluMulInst::new(div_ceil(is as u32, 256), act.ffn_act.as_write_ptr(), act.ffn_gate.as_ptr(), act.ffn_up.as_ptr(), is as i32).into_inst());
            {
                let (op, wp) = linear_proj_opcode_ptr(w_down);
                instructions.push(LinearProjInst::new(op, hs as u32, act.ffn_down.as_write_ptr(), wp, act.ffn_act.as_ptr(), hs as i32, is as i32, 0).into_inst());
            }
            instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), act.ffn_down.as_ptr(), act.residual.as_ptr(), hs as i32).into_inst());
        }
    }

    /// Emit shared expert instructions into `instructions`.
    fn emit_shared_expert(
        se: &crate::weights::DenseFfnWeights,
        moe: &crate::model::MoeWeights,
        act: &ActivationBuffers,
        hs: usize,
        se_is: usize,
        instructions: &mut Vec<Instruction>,
    ) {
        if moe.has_gate_proj {
            let (op, wp) = linear_proj_opcode_ptr(&se.gate_proj);
            instructions.push(LinearProjInst::new(op, se_is as u32, act.moe_expert_gate.as_write_ptr(), wp, act.normed.as_ptr(), se_is as i32, hs as i32, 0).no_sync().into_inst());
            let (op, wp) = linear_proj_opcode_ptr(&se.up_proj);
            instructions.push(LinearProjInst::new(op, se_is as u32, act.moe_expert_up.as_write_ptr(), wp, act.normed.as_ptr(), se_is as i32, hs as i32, 0).into_inst());
            instructions.push(SiluMulInst::new(div_ceil(se_is as u32, 256), act.moe_expert_act.as_write_ptr(), act.moe_expert_gate.as_ptr(), act.moe_expert_up.as_ptr(), se_is as i32).into_inst());
        } else {
            let (op, wp) = linear_proj_opcode_ptr(&se.up_proj);
            instructions.push(LinearProjInst::new(op, se_is as u32, act.moe_expert_up.as_write_ptr(), wp, act.normed.as_ptr(), se_is as i32, hs as i32, 0).into_inst());
            instructions.push(ReluSqInst::new(div_ceil(se_is as u32, 256), act.moe_expert_act.as_write_ptr(), act.moe_expert_up.as_ptr(), se_is as i32).into_inst());
        }
        let (op, wp) = linear_proj_opcode_ptr(&se.down_proj);
        instructions.push(LinearProjInst::new(op, hs as u32, act.moe_expert_out.as_write_ptr(), wp, act.moe_expert_act.as_ptr(), hs as i32, se_is as i32, 0).into_inst());

        if let Some(ref gate_buf) = moe.shared_expert_gate {
            instructions.push(LinearProjInst::new(OP_LINEAR_PROJ, 1, act.moe_scores.as_write_ptr(), gate_buf.as_ptr() as *const u8, act.normed.as_ptr(), 1, hs as i32, 0).into_inst());
            instructions.push(SigmoidWeightedAddInst::new(div_ceil(hs as u32, 256), act.ffn_down.as_write_ptr(), act.moe_scores.as_ptr(), act.moe_expert_out.as_ptr(), hs as i32).into_inst());
        } else {
            instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.ffn_down.as_write_ptr(), act.ffn_down.as_ptr(), act.moe_expert_out.as_ptr(), hs as i32).into_inst());
        }
    }

    // Overload for ffn_down_stage output (multi-GPU path)
    fn emit_shared_expert_stage(
        se: &crate::weights::DenseFfnWeights,
        moe: &crate::model::MoeWeights,
        act: &ActivationBuffers,
        hs: usize,
        se_is: usize,
        instructions: &mut Vec<Instruction>,
    ) {
        if moe.has_gate_proj {
            let (op, wp) = linear_proj_opcode_ptr(&se.gate_proj);
            instructions.push(LinearProjInst::new(op, se_is as u32, act.moe_expert_gate.as_write_ptr(), wp, act.normed.as_ptr(), se_is as i32, hs as i32, 0).no_sync().into_inst());
            let (op, wp) = linear_proj_opcode_ptr(&se.up_proj);
            instructions.push(LinearProjInst::new(op, se_is as u32, act.moe_expert_up.as_write_ptr(), wp, act.normed.as_ptr(), se_is as i32, hs as i32, 0).into_inst());
            instructions.push(SiluMulInst::new(div_ceil(se_is as u32, 256), act.moe_expert_act.as_write_ptr(), act.moe_expert_gate.as_ptr(), act.moe_expert_up.as_ptr(), se_is as i32).into_inst());
        } else {
            let (op, wp) = linear_proj_opcode_ptr(&se.up_proj);
            instructions.push(LinearProjInst::new(op, se_is as u32, act.moe_expert_up.as_write_ptr(), wp, act.normed.as_ptr(), se_is as i32, hs as i32, 0).into_inst());
            instructions.push(ReluSqInst::new(div_ceil(se_is as u32, 256), act.moe_expert_act.as_write_ptr(), act.moe_expert_up.as_ptr(), se_is as i32).into_inst());
        }
        let (op, wp) = linear_proj_opcode_ptr(&se.down_proj);
        instructions.push(LinearProjInst::new(op, hs as u32, act.moe_expert_out.as_write_ptr(), wp, act.moe_expert_act.as_ptr(), hs as i32, se_is as i32, 0).into_inst());

        if let Some(ref gate_buf) = moe.shared_expert_gate {
            instructions.push(LinearProjInst::new(OP_LINEAR_PROJ, 1, act.moe_scores.as_write_ptr(), gate_buf.as_ptr() as *const u8, act.normed.as_ptr(), 1, hs as i32, 0).into_inst());
            instructions.push(SigmoidWeightedAddInst::new(div_ceil(hs as u32, 256), act.ffn_down_stage.as_write_ptr(), act.moe_scores.as_ptr(), act.moe_expert_out.as_ptr(), hs as i32).into_inst());
        } else {
            instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.ffn_down_stage.as_write_ptr(), act.ffn_down_stage.as_ptr() as *const f32, act.moe_expert_out.as_ptr(), hs as i32).into_inst());
        }
    }

    /// Compile MoE FFN for one layer: norm + gate + OP_MOE_GATE + OP_MOE_FFN + shared expert + residual.
    fn compile_moe_ffn(
        cfg: &ModelConfig,
        layer_idx: usize,
        layer: &LayerWeights,
        moe: &crate::model::MoeWeights,
        act: &ActivationBuffers,
        instructions: &mut Vec<Instruction>,
    ) {
        use crate::model::{FfnType, GateType};
        let hs = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        let (k, gate_type, eis) = match &cfg.layers[layer_idx].ffn_type {
            FfnType::MoE {
                num_active,
                gate_type,
                expert_intermediate_size,
                ..
            } => (*num_active, gate_type.clone(), *expert_intermediate_size),
            _ => unreachable!(),
        };
        let ne = moe.num_experts;

        // Get norm weight pointer
        let norm_ptr = match layer {
            LayerWeights::Attention(w) => w.post_norm.as_ptr(),
            LayerWeights::Gdn(w) => w.post_norm.as_ptr(),
            LayerWeights::MoeFfn(w) => w.input_norm.as_ptr(),
            _ => panic!("no norm weight for MoE FFN layer"),
        };

        // D2D_COPY: hidden → residual (NO_SYNC: norm reads hidden)
        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).no_sync().into_inst());

        // RMSNorm: hidden → normed
        instructions.push(RmsNormInst::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1, act.normed.as_write_ptr(), act.hidden.as_ptr(), norm_ptr, hs as i32, eps).into_inst());

        // Gate projection: normed → moe_scores[num_experts]
        instructions.push(LinearProjInst::new(OP_LINEAR_PROJ, ne as u32, act.moe_scores.as_write_ptr(), moe.gate.as_ptr() as *const u8, act.normed.as_ptr(), ne as i32, hs as i32, 0).into_inst());

        // OP_MOE_GATE: top-k selection on GPU
        let (gate_mode, rsf) = match &gate_type {
            GateType::Softmax => (0u32, 1.0f32),
            GateType::NormTopK { routed_scaling_factor } => (1, *routed_scaling_factor),
            GateType::Sigmoid { routed_scaling_factor } => (2, *routed_scaling_factor),
        };
        let bias_ptr = moe.score_correction_bias_gpu.as_ref().map(|b| b.as_ptr() as *const u8).unwrap_or(std::ptr::null());
        instructions.push(MoeGateInst::new(act.moe_scores.as_ptr(), act.moe_expert_ids.as_write_ptr(), act.moe_expert_weights.as_write_ptr(), ne as i32, k as i32, gate_mode, rsf, bias_ptr).into_inst());

        // OP_MOE_FFN: fused expert loop (internal grid.sync())
        // Currently only supports PcG32Q4 weights in the GPU kernel
        assert!(
            matches!(
                moe.expert_gate_up.weight_format(),
                crate::quant::WeightFormat::PcG32Q4
            ),
            "OP_MOE_FFN only supports PcG32Q4 expert weights (got {:?})",
            moe.expert_gate_up.weight_format()
        );
        let gate_up_expert_stride = if moe.has_gate_proj {
            moe.expert_gate_up.row_byte_offset_dim(2 * eis, hs)
        } else {
            moe.expert_gate_up.row_byte_offset_dim(eis, hs)
        };
        let down_expert_stride = moe.expert_down.row_byte_offset_dim(hs, eis);
        let gate_up_row_stride = moe.expert_gate_up.row_byte_offset_dim(1, hs);

        let flags =
            (if moe.has_gate_proj { 1u32 } else { 0 }) | (if !moe.has_gate_proj { 2 } else { 0 }); // bit1 = relu²

        let grid_x = std::cmp::max(eis, hs) as u32;

        instructions.push(Instruction {
            words: {
                let mut w = [0u64; INST_SIZE];
                w[0] = make_opcode_gridx(OP_MOE_FFN, grid_x);
                w[1] = act.moe_expert_ids.as_ptr() as u64;
                w[2] = act.moe_expert_weights.as_ptr() as u64;
                w[3] = act.normed.as_ptr() as u64;
                w[4] = act.ffn_down.as_write_ptr() as u64;
                w[5] = moe.expert_gate_up.raw_data_ptr() as u64;
                w[6] = gate_up_expert_stride as u64;
                w[7] = moe.expert_down.raw_data_ptr() as u64;
                w[8] = down_expert_stride as u64;
                w[9] = k as u64;
                w[10] = (hs | (eis << 16)) as u64;
                w[11] = flags as u64;
                w[12] = act.moe_expert_gate.as_ptr() as u64;
                w[13] = act.moe_expert_up.as_ptr() as u64;
                w[14] = act.moe_expert_act.as_ptr() as u64;
                w[15] = act.moe_expert_out.as_ptr() as u64;
                w[16] = gate_up_row_stride as u64;
                w
            }
        });

        // Shared expert (if present)
        if let Some(ref se) = moe.shared_expert {
            let se_is = match &cfg.layers[layer_idx].ffn_type {
                FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } => {
                    if *shared_intermediate_size > 0 { *shared_intermediate_size } else { *expert_intermediate_size }
                }
                _ => eis,
            };

            Self::emit_shared_expert(se, moe, act, hs, se_is, instructions);
        }

        // Residual: hidden = residual + ffn_down
        instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), act.residual.as_ptr(), act.ffn_down.as_ptr(), hs as i32).into_inst());
    }

    /// Multi-GPU variant: emit norm + gate proj + OP_MOE_GATE + OP_BARRIER.
    /// CPU dispatch loop handles expert FFN; megakernel resumes for shared expert + residual.
    /// Returns the instruction index of the emitted OP_BARRIER.
    ///
    /// Note: barrier_flag_ptr and resume_flag_ptr are patched in after MoeBarrierState is
    /// allocated (in compile_multi_gpu). Initially zero; patched by execute_multi_gpu().
    fn compile_moe_ffn_multi_gpu(
        cfg: &ModelConfig,
        layer_idx: usize,
        layer: &LayerWeights,
        moe: &crate::model::MoeWeights,
        act: &ActivationBuffers,
        instructions: &mut Vec<Instruction>,
    ) -> usize {
        use crate::model::{FfnType, GateType};
        let hs = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        let (k, gate_type, ne, eis) = match &cfg.layers[layer_idx].ffn_type {
            FfnType::MoE {
                num_active,
                gate_type,
                num_experts,
                expert_intermediate_size,
                ..
            } => (
                *num_active,
                gate_type.clone(),
                *num_experts,
                *expert_intermediate_size,
            ),
            _ => unreachable!(),
        };

        let norm_ptr = match layer {
            LayerWeights::Attention(w) => w.post_norm.as_ptr(),
            LayerWeights::Gdn(w) => w.post_norm.as_ptr(),
            LayerWeights::MoeFfn(w) => w.input_norm.as_ptr(),
            _ => panic!("no norm weight for MoE FFN layer"),
        };

        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).no_sync().into_inst());
        instructions.push(RmsNormInst::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1, act.normed.as_write_ptr(), act.hidden.as_ptr(), norm_ptr, hs as i32, eps).into_inst());
        instructions.push(LinearProjInst::new(OP_LINEAR_PROJ, ne as u32, act.moe_scores.as_write_ptr(), moe.gate.as_ptr() as *const u8, act.normed.as_ptr(), ne as i32, hs as i32, 0).into_inst());
        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.normed_stage.as_write_ptr(), act.normed.as_ptr(), hs as i32).no_sync().into_inst());

        let (gate_mode, rsf) = match &gate_type {
            GateType::Softmax => (0u32, 1.0f32),
            GateType::NormTopK { routed_scaling_factor } => (1, *routed_scaling_factor),
            GateType::Sigmoid { routed_scaling_factor } => (2, *routed_scaling_factor),
        };
        let bias_ptr = moe.score_correction_bias_gpu.as_ref().map(|b| b.as_ptr() as *const u8).unwrap_or(std::ptr::null());
        instructions.push(MoeGateInst::new(act.moe_scores.as_ptr(), act.moe_expert_ids.as_write_ptr(), act.moe_expert_weights.as_write_ptr(), ne as i32, k as i32, gate_mode, rsf, bias_ptr).into_inst());

        // OP_BARRIER: grid_x=1: only block 0 runs op_barrier
        let barrier_inst_idx = instructions.len();
        instructions.push(BarrierInst::new(layer_idx as i32).into_inst());

        // After barrier: compute shared expert (if present) and add to ffn_down_stage.
        if let Some(ref se) = moe.shared_expert {
            let se_is = match &cfg.layers[layer_idx].ffn_type {
                FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } => {
                    if *shared_intermediate_size > 0 { *shared_intermediate_size } else { *expert_intermediate_size }
                }
                _ => eis,
            };
            Self::emit_shared_expert_stage(se, moe, act, hs, se_is, instructions);
        }

        // Final residual: hidden = residual + ffn_down_stage
        instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), act.residual.as_ptr(), act.ffn_down_stage.as_ptr() as *const f32, hs as i32).into_inst());

        barrier_inst_idx
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GPU-native P2P path: OP_MOE_DISPATCH
    // ─────────────────────────────────────────────────────────────────────────

    /// Compile the full instruction stream for GPU-native P2P MoE dispatch.
    /// MoE layers emit OP_MOE_DISPATCH; all other layers are identical to compile_inner.
    ///
    /// For models with moe_latent_size (e.g. Nemotron-H):
    ///   - fc1_latent_proj (normed→moe_latent) is emitted before OP_MOE_DISPATCH
    ///   - OP_MOE_DISPATCH uses moe_latent as activation and writes to moe_latent
    ///   - fc2_latent_proj (moe_latent→ffn_down_stage) + residual_add are emitted after
    fn compile_inner_p2p(model: &Model, p2p: &crate::moe_p2p::MoeP2pContext) -> HipResult<Self> {
        let mut prog = Self::compile_inner(model, false, true)?;

        let cfg = &model.config;
        let act = &model.activations;
        let hs = cfg.hidden_size;

        // Collect: (barrier_idx → layer_idx) for all MoE barriers
        let barrier_map: std::collections::HashMap<usize, usize> = prog
            .barrier_layer_map
            .iter()
            .map(|&(bi, li)| (bi, li))
            .collect();

        // Pass 1: patch OP_BARRIER → OP_MOE_DISPATCH and surrounding instructions in-place
        for &(barrier_idx, layer_idx) in &prog.barrier_layer_map {
            let moe = model.moe_weights[layer_idx].as_ref().unwrap();
            let dist = model.distributed_moe[layer_idx].as_ref();
            let (k, eis) = match &cfg.layers[layer_idx].ffn_type {
                crate::model::FfnType::MoE {
                    num_active,
                    expert_intermediate_size,
                    ..
                } => (*num_active, *expert_intermediate_size),
                _ => unreachable!(),
            };
            let has_gate = if moe.has_gate_proj { 1u64 } else { 0u64 };
            let num_workers = p2p.workers.len();
            let num_gpus = p2p.num_gpus;
            // gate_up_in_dim: expert input dimension (hs for standard MoE, moe_latent_size for Nemotron-H)
            let gupd = dist.map(|d| d.gate_up_in_dim).unwrap_or(hs);
            let has_latent = moe.fc1_latent_proj.is_some();

            // Replace D2D_COPY(normed→normed_stage) at barrier_idx-2 with:
            //   - fc1_latent_proj(normed→moe_latent) if fc1 exists, else NOP
            // compile_moe_ffn_multi_gpu order: ..., D2D_COPY(normed→normed_stage), OP_MOE_GATE, OP_BARRIER
            if barrier_idx >= 2 {
                let prev2 = &prog.instructions[barrier_idx - 2];
                let prev2_opcode = (prev2.words[0] & INST_OPCODE_MASK) as u32;
                let normed_stage_ptr = act.normed_stage.as_ptr() as u64;
                if prev2_opcode == OP_D2D_COPY && prev2.words[1] == normed_stage_ptr {
                    if let Some(ref fc1) = moe.fc1_latent_proj {
                        // Emit fc1: normed(hs) → moe_latent(gupd)
                        let mut fc1_inst = Instruction::new(OP_LINEAR_PROJ, gupd as u32);
                        fc1_inst.words[1] = act.moe_latent.as_write_ptr() as u64;
                        emit_linear_proj(&mut fc1_inst, fc1, 2);
                        fc1_inst.words[3] = act.normed.as_ptr() as u64;
                        fc1_inst.words[4] = gupd as u64;
                        fc1_inst.words[5] = hs as u64;
                        prog.instructions[barrier_idx - 2] = fc1_inst;
                    } else {
                        // NOP: no normed_stage copy needed in P2P path
                        prog.instructions[barrier_idx - 2] = Instruction::new(OP_D2D_COPY, 0);
                    }
                }
            }

            // Build OP_MOE_DISPATCH instruction
            // For Nemotron-H (has_latent): activation = moe_latent (gupd elements)
            //                              final_output = moe_latent (gupd; fc2 projects to hs after)
            // For standard MoE:           activation = normed (hs)
            //                              final_output = ffn_down_stage (hs)
            let activation_ptr = if has_latent {
                act.moe_latent.as_ptr() as u64
            } else {
                act.normed.as_ptr() as u64
            };
            let final_output_ptr = if moe.fc2_latent_proj.is_some() {
                act.moe_latent.as_ptr() as u64  // experts write latent; fc2 projects to ffn_down_stage
            } else {
                act.ffn_down_stage.as_ptr() as u64
            };

            let mut inst = Instruction::new(OP_MOE_DISPATCH, 0);
            inst.words[1] = p2p.work_queue.device_ptr() as u64;
            inst.words[2] = p2p.output_slots.as_ptr() as u64;
            inst.words[3] = final_output_ptr;
            inst.words[4] = act.moe_expert_ids.as_ptr() as u64;
            inst.words[5] = act.moe_expert_weights.as_ptr() as u64;
            inst.words[6] = p2p.seq_counter.device_ptr() as u64;
            inst.words[7] = ((num_workers as u64) << 32) | (hs as u64); // num_workers | hidden_size (slot stride)
            inst.words[8] = ((layer_idx as u64) << 32) | (k as u64);
            inst.words[9] = ((eis as u64) << 32) | has_gate;
            inst.words[10] = activation_ptr;
            inst.words[11] = p2p.gpu0_layer_config_ptrs.as_ptr() as u64;
            inst.words[12] = p2p.gpu0_scratch_gate.as_ptr() as u64;
            inst.words[13] = p2p.gpu0_scratch_up.as_ptr() as u64;
            inst.words[14] = p2p.gpu0_scratch_act.as_ptr() as u64;
            inst.words[15] = num_gpus as u64;
            inst.words[16] = gupd as u64; // gate_up_in_dim (0 → kernel defaults to hs)

            prog.instructions[barrier_idx] = inst;
        }

        // Pass 2: rebuild instruction stream to insert fc2+residual_add after OP_MOE_DISPATCH
        // (needed for Nemotron-H). Also skip stale shared_up_proj at barrier_idx+1 for those layers.
        //
        // For models with fc2_latent_proj:
        //   - compile_moe_ffn_multi_gpu returns early → emits LINEAR_PROJ(shared_up) at barrier_idx+1
        //     and NO residual_add. We skip that stale instruction and insert fc2+residual_add instead.
        // For models without fc2:
        //   - normal flow: no insertion needed (residual_add already present).
        let stale_positions: std::collections::HashSet<usize> = prog
            .barrier_layer_map
            .iter()
            .filter(|&&(bi, li)| {
                model.moe_weights[li]
                    .as_ref()
                    .map(|m| m.fc2_latent_proj.is_some())
                    .unwrap_or(false)
                    && bi + 1 < prog.instructions.len()
            })
            .map(|&(bi, _)| bi + 1)
            .collect();

        let has_fc2_layers: bool = prog
            .barrier_layer_map
            .iter()
            .any(|&(_, li)| {
                model.moe_weights[li]
                    .as_ref()
                    .map(|m| m.fc2_latent_proj.is_some())
                    .unwrap_or(false)
            });

        if has_fc2_layers {
            let mut new_instructions =
                Vec::with_capacity(prog.instructions.len() + 2 * barrier_map.len());
            // old_to_new[i] = index of old instruction i in new_instructions (-1 if skipped)
            let mut old_to_new: Vec<i64> = vec![-1i64; prog.instructions.len()];
            for (i, inst) in prog.instructions.iter().enumerate() {
                if stale_positions.contains(&i) {
                    // Skip the stale shared_up_proj left by compile_moe_ffn_multi_gpu's early return
                    continue;
                }
                old_to_new[i] = new_instructions.len() as i64;
                new_instructions.push(inst.clone());
                // After OP_MOE_DISPATCH: insert fc2_latent_proj + residual_add (when fc2 exists)
                if let Some(&layer_idx) = barrier_map.get(&i) {
                    let moe = model.moe_weights[layer_idx].as_ref().unwrap();
                    let dist = model.distributed_moe[layer_idx].as_ref();
                    let gupd = dist.map(|d| d.gate_up_in_dim).unwrap_or(hs);
                    let cfg = &model.config;
                    if let Some(ref fc2) = moe.fc2_latent_proj {
                        // fc2: moe_latent(gupd) → ffn_down_stage(hs)
                        {
                            let mut fc2_inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
                            fc2_inst.words[1] = act.ffn_down_stage.as_write_ptr() as u64;
                            emit_linear_proj(&mut fc2_inst, fc2, 2);
                            fc2_inst.words[3] = act.moe_latent.as_ptr() as u64;
                            fc2_inst.words[4] = hs as u64;
                            fc2_inst.words[5] = gupd as u64;
                            new_instructions.push(fc2_inst);
                        }

                        // Shared expert (relu² + down_proj, missing in compile_moe_ffn_multi_gpu early return)
                        if let Some(ref se) = moe.shared_expert {
                            let se_is = match &cfg.layers[layer_idx].ffn_type {
                                crate::model::FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } => {
                                    if *shared_intermediate_size > 0 { *shared_intermediate_size } else { *expert_intermediate_size }
                                }
                                _ => { let Some(ref d) = model.distributed_moe[layer_idx] else { continue }; d.expert_intermediate_size }
                            };
                            if !moe.has_gate_proj {
                                // relu² path (Nemotron-H): up_proj → relu² → down_proj
                                {
                                    let mut up_inst = Instruction::new(OP_LINEAR_PROJ, se_is as u32);
                                    up_inst.words[1] = act.moe_expert_up.as_write_ptr() as u64;
                                    emit_linear_proj(&mut up_inst, &se.up_proj, 2);
                                    up_inst.words[3] = act.normed.as_ptr() as u64;
                                    up_inst.words[4] = se_is as u64;
                                    up_inst.words[5] = hs as u64;
                                    new_instructions.push(up_inst);
                                }
                                new_instructions.push(ReluSqInst::new(
                                    div_ceil(se_is as u32, 256),
                                    act.moe_expert_act.as_write_ptr(),
                                    act.moe_expert_up.as_ptr(),
                                    se_is as i32,
                                ).into_inst());
                                {
                                    let mut dn_inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
                                    dn_inst.words[1] = act.moe_expert_out.as_write_ptr() as u64;
                                    emit_linear_proj(&mut dn_inst, &se.down_proj, 2);
                                    dn_inst.words[3] = act.moe_expert_act.as_ptr() as u64;
                                    dn_inst.words[4] = hs as u64;
                                    dn_inst.words[5] = se_is as u64;
                                    new_instructions.push(dn_inst);
                                }

                                if let Some(ref gate_buf) = moe.shared_expert_gate {
                                    // gate @ normed → moe_scores (1 scalar)
                                    new_instructions.push(LinearProjInst::new(
                                        OP_LINEAR_PROJ, 1,
                                        act.moe_scores.as_write_ptr(),
                                        gate_buf.as_ptr() as *const u8,
                                        act.normed.as_ptr(),
                                        1i32, hs as i32, 0,
                                    ).into_inst());

                                    // ffn_down_stage += sigmoid(moe_scores[0]) * moe_expert_out
                                    new_instructions.push(SigmoidWeightedAddInst::new(
                                        div_ceil(hs as u32, 256),
                                        act.ffn_down_stage.as_write_ptr(),
                                        act.moe_scores.as_ptr(),
                                        act.moe_expert_out.as_ptr(),
                                        hs as i32,
                                    ).into_inst());
                                } else {
                                    // ffn_down_stage += moe_expert_out
                                    new_instructions.push(ResidualAddInst::new(
                                        div_ceil(hs as u32, 256),
                                        act.ffn_down_stage.as_write_ptr(),
                                        act.ffn_down_stage.as_ptr() as *const f32,
                                        act.moe_expert_out.as_ptr(),
                                        hs as i32,
                                    ).into_inst());
                                }
                            }
                            // Note: has_gate_proj shared expert already handled correctly by
                            // compile_moe_ffn_multi_gpu (no early return in that branch).
                        }

                        // residual_add: hidden = residual + ffn_down_stage
                        // (missing for Nemotron-H because compile_moe_ffn_multi_gpu returns early)
                        new_instructions.push(ResidualAddInst::new(
                            div_ceil(hs as u32, 256),
                            act.hidden.as_write_ptr(),
                            act.residual.as_ptr(),
                            act.ffn_down_stage.as_ptr() as *const f32,
                            hs as i32,
                        ).into_inst());
                    }
                }
            }
            prog.instructions = new_instructions;

            // Remap multi_gpu_attn_boundaries to new instruction indices.
            // old_to_new maps old index → new index (-1 if skipped).
            prog.multi_gpu_attn_boundaries = prog.multi_gpu_attn_boundaries.iter().enumerate().map(|(_attn_i, &(flush, resume))| {
                let new_flush = old_to_new[flush];
                let new_resume = old_to_new[resume];
                assert!(new_flush >= 0, "attn flush_idx {flush} was in stale_positions — logic error");
                assert!(new_resume >= 0, "attn resume_idx {resume} was in stale_positions — logic error");
                let nf = new_flush as usize;
                let nr = new_resume as usize;
                // Verify: flush instruction should be OP_RMSNORM
                (nf, nr)
            }).collect();
        }

        // Clear barrier_layer_map so it's not misinterpreted after OP_BARRIER→OP_MOE_DISPATCH patch
        prog.barrier_layer_map.clear();

        Ok(prog)
    }
}
