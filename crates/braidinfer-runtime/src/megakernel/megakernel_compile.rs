//! Megakernel program compilation: translates model config + weights into instruction streams.
//! Extracted from megakernel.rs for maintainability.

use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::HipResult;

use crate::model::{
    ActivationBuffers, AttentionLayerWeights, KvCache, GdnState,
    LayerWeights, ModelConfig, Model,
};
use super::{Instruction, MegakernelProgram, PrefillBuffers, INST_SIZE, NUM_CUS, CHUNK_TOKENS};
#[allow(unused_imports)]
use super::{OP_RMSNORM, OP_LINEAR_PROJ, OP_CONV1D, OP_GDN_GATE, OP_GDN_RECUR,
    OP_RMSNORM_GATE, OP_RESIDUAL_ADD, OP_QK_NORM, OP_MROPE, OP_GQA_ATTN, OP_OUTPUT_GATE,
    OP_FFN_GATE_UP, OP_FFN_DOWN_RES, OP_EMBEDDING, OP_LM_HEAD, OP_HALT, OP_D2D_COPY,
    OP_ATTN_PAGED, OP_ATTN_PREFILL, OP_DEINTERLEAVE, OP_KV_QUANTIZE, OP_ATTN_PAGED_Q,
    OP_MOE_GATE, OP_MOE_FFN, OP_LINEAR_PROJ_RNF4, OP_LINEAR_PROJ_PCG32, OP_RMSNORM_WX,
    OP_SILU_MUL, OP_FFN_GATE_UP_RNF4, OP_FFN_DOWN_RES_RNF4,
    OP_SIGMOID_WEIGHTED_ADD, OP_BARRIER};

fn emit_linear_proj(inst: &mut Instruction, weight: &crate::model::LinearWeight, ptr_slot: usize) {
    use crate::model::{LinearWeight, WeightFormat};
    match weight {
        LinearWeight::Bf16(buf) => {
            inst.set_ptr(ptr_slot, buf.as_ptr());
        }
        LinearWeight::Packed(pw) => {
            let op = match pw.format {
                WeightFormat::Rnf4G128 => OP_LINEAR_PROJ_RNF4,
                WeightFormat::PcG32Q4 => OP_LINEAR_PROJ_PCG32,
                WeightFormat::Bf16 => OP_LINEAR_PROJ,
            };
            // Replace opcode (low 32 bits), preserve grid_x (high 32 bits)
            inst.words[0] = (inst.words[0] & 0xFFFF_FFFF_0000_0000u64) | op as u64;
            inst.set_ptr(ptr_slot, pw.data.as_ptr());
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
    // All weight formats support batched projection via slot 6
    let mut inst = Instruction::new(OP_LINEAR_PROJ, out_dim as u32);
    inst.set_output_ptr(1, output);
    emit_linear_proj(&mut inst, weight, 2);
    inst.set_ptr(3, input);
    inst.set_int(4, out_dim as i32);
    inst.set_int(5, in_dim as i32);
    inst.set_int(6, n as i32);
    if no_sync { inst.set_no_sync(); }
    instructions.push(inst);
}

/// Choose RMSNorm opcode based on model config.
fn rmsnorm_opcode(one_plus_w: bool) -> u32 {
    if one_plus_w { OP_RMSNORM } else { OP_RMSNORM_WX }
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
    pub fn compile(model: &Model) -> HipResult<Self> {
        Self::compile_inner(model, false, false)
    }

    pub fn compile_paged(model: &Model) -> HipResult<Self> {
        Self::compile_inner(model, true, false)
    }

    /// Compile for multi-GPU MoE models. MoE layers emit OP_BARRIER instead of OP_MOE_FFN.
    /// The CPU dispatch loop (decode_step_megakernel_moe) handles expert dispatch per barrier.
    pub fn compile_multi_gpu(model: &Model) -> HipResult<Self> {
        let mut prog = Self::compile_inner(model, false, true)?;
        let barrier_state = super::MoeBarrierState::new()?;
        // Patch barrier flag pointers into all OP_BARRIER instructions
        let bflag_dev = barrier_state.barrier.device_ptr() as u64;
        let rflag_dev = barrier_state.resume.device_ptr() as u64;
        for &(inst_idx, _layer_idx) in &prog.barrier_layer_map {
            prog.instructions[inst_idx].words[1] = bflag_dev;
            prog.instructions[inst_idx].words[2] = rflag_dev;
        }
        prog.moe_barrier = Some(barrier_state);
        Ok(prog)
    }

    fn compile_inner(model: &Model, paged: bool, multi_gpu: bool) -> HipResult<Self> {
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

        let has_moe = cfg.layers.iter().any(|l| matches!(l.ffn_type, crate::model::FfnType::MoE { .. }));
        // OP_MOE_GATE needs 1024 floats (512 selection + 512 raw) = 4KB
        // GDN recurrent + warp reduction needs 2KB
        let shared_mem = if has_moe { 1024u32 * 4 } else { 256u32 * 4 * 2 };
        let func = module.get_function("megakernel_f32")?;
        let blocks_per_sm = func.max_active_blocks_per_sm(256, shared_mem as usize)?;
        // Cap at 192 blocks (2/CU): empirically optimal for virtual block loop.
        // Higher counts increase cooperative launch overhead without improving throughput
        // since the virtual block loop already distributes work across all blocks.
        let num_blocks = (blocks_per_sm as u32 * NUM_CUS).min(192);

        let mut instructions: Vec<Instruction> = Vec::new();
        let mut mrope_inst_indices: Vec<usize> = Vec::new();
        let mut gqa_attn_inst_indices = Vec::new();
        let mut kv_write_indices = Vec::new();
        let mut kv_base_ptrs = Vec::new();
        let mut barrier_layer_map: Vec<(usize, usize)> = Vec::new();

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
        {
            let mut inst = Instruction::new(OP_EMBEDDING, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, act.hidden.as_write_ptr());
            inst.set_ptr(2, model.embed_weight.as_ptr());
            inst.set_int(3, 0); // token_id — updated per step
            inst.set_int(4, hs as i32);
            instructions.push(inst);
        }

        // Layers
        let mut attn_paged_inst_indices = Vec::new();
        let mut attn_quant_inst_indices = Vec::new();
        let mut attn_layer_count = 0usize;

        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        for layer_i in 0..cfg.num_layers {
            use crate::model::LayerType;
            match cfg.layers[layer_i].layer_type {
                LayerType::Attention => {
                    if paged {
                        Self::compile_attention_layer_paged(
                            cfg, &model.layers[layer_i], act,
                            &model.kv_caches[kv_idx],
                            attn_layer_count,
                            &mut instructions, &mut mrope_inst_indices,
                            &mut kv_write_indices, &mut kv_base_ptrs,
                            &mut attn_paged_inst_indices,
                            &mut attn_quant_inst_indices,
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
                }
                LayerType::Gdn => {
                    Self::compile_gdn_layer(
                        cfg, &model.layers[layer_i], act,
                        &model.gdn_conv_states[gdn_idx],
                        &model.gdn_states[gdn_idx],
                        &mut instructions,
                    );
                    gdn_idx += 1;
                }
                LayerType::Mamba2 => {
                    panic!("Mamba2 layers not yet implemented in megakernel (braidinfer-ce9)");
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
                            cfg, layer_i, &model.layers[layer_i], moe, act, &mut instructions,
                        );
                        barrier_layer_map.push((barrier_inst_idx, layer_i));
                    } else {
                        Self::compile_moe_ffn(
                            cfg, layer_i, &model.layers[layer_i],
                            model.moe_weights[layer_i].as_ref().unwrap(),
                            act, &mut instructions,
                        );
                    }
                }
                crate::model::FfnType::None => {
                    // No FFN for this layer (Nemotron M/* layers)
                }
            }
        }

        // Final RMSNorm: copy hidden→normed, then norm normed→hidden
        {
            let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, act.normed.as_write_ptr());
            inst.set_ptr(2, act.hidden.as_ptr());
            inst.set_int(3, hs as i32);
            instructions.push(inst);
        }
        {
            let mut inst = Instruction::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1);
            inst.set_output_ptr(1, act.hidden.as_write_ptr());
            inst.set_ptr(2, act.normed.as_ptr());
            inst.set_ptr(3, model.final_norm_weight.as_ptr());
            inst.set_int(4, hs as i32);
            inst.set_float(5, eps);
            instructions.push(inst);
        }

        // LM head (= linear_proj with vocab_size output rows)
        {
            let mut inst = Instruction::new(OP_LINEAR_PROJ, vs as u32);
            inst.set_output_ptr(1, act.logits.as_write_ptr());
            inst.set_ptr(2, if model.config.tie_word_embeddings { model.embed_weight.as_ptr() } else { model.lm_head_weight.as_ptr() });
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
            moe_barrier: None,
            barrier_layer_map,
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

        let module = Module::load(device, &crate::kernel::kernel_dir().join("megakernel.hsaco"))?;
        let shared_mem = (256u32 * 4 * 2).max((cfg.hidden_size as u32) * 4);
        let func = module.get_function("megakernel_f32")?;
        let blocks_per_sm = func.max_active_blocks_per_sm(256, shared_mem as usize)?;
        // Cap at 192 blocks (2/CU): empirically optimal for virtual block loop.
        // Higher counts increase cooperative launch overhead without improving throughput
        // since the virtual block loop already distributes work across all blocks.
        let num_blocks = (blocks_per_sm as u32 * NUM_CUS).min(192);
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
            let mut inst = Instruction::new(OP_EMBEDDING, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, unsafe { prefill_bufs.hidden.as_write_ptr().add(t * hs) });
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
        let mut _attn_layer_count = 0usize;

        for layer_i in 0..cfg.num_layers {
            use crate::model::LayerType;
            if cfg.layers[layer_i].layer_type == LayerType::Attention {
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
                    &mut Vec::new(), &mut Vec::new(),
                );

                // Batched FFN
                Self::compile_ffn_batched(cfg, &model.layers[layer_i], prefill_bufs, n, &mut instructions);
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
                    let mut inst = Instruction::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), n as u32);
                    inst.set_output_ptr(1, prefill_bufs.normed.as_write_ptr());
                    inst.set_ptr(2, prefill_bufs.hidden.as_ptr());
                    inst.set_ptr(3, w.input_norm.as_ptr());
                    inst.set_int(4, hs as i32);
                    inst.set_float(5, eps);
                    instructions.push(inst);
                }

                // QKV projection (batch=N)
                emit_batched_linear_proj(
                    &w.w_qkv, prefill_bufs.qkv.as_write_ptr(), prefill_bufs.normed.as_ptr(),
                    conv_dim, hs, n, true, &mut instructions,
                );

                // a projection (batch=N)
                emit_batched_linear_proj(
                    &w.w_a, prefill_bufs.a_proj.as_write_ptr(), prefill_bufs.normed.as_ptr(),
                    nvh_gdn, hs, n, true, &mut instructions,
                );

                // b projection (batch=N)
                emit_batched_linear_proj(
                    &w.w_b, prefill_bufs.b_proj.as_write_ptr(), prefill_bufs.normed.as_ptr(),
                    nvh_gdn, hs, n, true, &mut instructions,
                );

                // z projection (batch=N) — SYNC before sequential part
                emit_batched_linear_proj(
                    &w.w_z, prefill_bufs.z_proj.as_write_ptr(), prefill_bufs.normed.as_ptr(),
                    nvh_gdn * vd, hs, n, false, &mut instructions,
                );

                // --- Sequential per-token: conv1d, gate, recurrence, norm, output, residual ---
                let q_dim = nh_gdn * kd;
                let k_dim = nh_gdn * kd;
                let v_dim = nvh_gdn * vd;

                for t in 0..n {
                    // Conv1d on Q (from batched qkv[t])
                    {
                        let mut inst = Instruction::new(OP_CONV1D, div_ceil(q_dim as u32, 256));
                        inst.set_output_ptr(1, conv_state.as_write_ptr());
                        inst.set_ptr(2, unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim) });
                        inst.set_ptr(3, w.conv1d_weight_q.as_ptr());
                        inst.set_output_ptr(4, act.q_gdn.as_write_ptr());
                        inst.set_int(5, q_dim as i32);
                        inst.set_int(6, ck as i32);
                        inst.set_no_sync();
                        instructions.push(inst);
                    }
                    // Conv1d on K
                    {
                        let mut inst = Instruction::new(OP_CONV1D, div_ceil(k_dim as u32, 256));
                        inst.set_output_ptr(1, unsafe { conv_state.as_write_ptr().add(q_dim * (ck - 1)) });
                        inst.set_ptr(2, unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim + q_dim) });
                        inst.set_ptr(3, w.conv1d_weight_k.as_ptr());
                        inst.set_output_ptr(4, act.k_gdn.as_write_ptr());
                        inst.set_int(5, k_dim as i32);
                        inst.set_int(6, ck as i32);
                        inst.set_no_sync();
                        instructions.push(inst);
                    }
                    // Conv1d on V
                    {
                        let mut inst = Instruction::new(OP_CONV1D, div_ceil(v_dim as u32, 256));
                        inst.set_output_ptr(1, unsafe { conv_state.as_write_ptr().add((q_dim + k_dim) * (ck - 1)) });
                        inst.set_ptr(2, unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim + q_dim + k_dim) });
                        inst.set_ptr(3, w.conv1d_weight_v.as_ptr());
                        inst.set_output_ptr(4, act.v_gdn.as_write_ptr());
                        inst.set_int(5, v_dim as i32);
                        inst.set_int(6, ck as i32);
                        instructions.push(inst);
                    }

                    // GDN gate (from batched a_proj[t])
                    {
                        let mut inst = Instruction::new(OP_GDN_GATE, div_ceil(nvh_gdn as u32, 256));
                        inst.set_output_ptr(1, act.gate_gdn.as_write_ptr());
                        inst.set_ptr(2, unsafe { prefill_bufs.a_proj.as_ptr().add(t * nvh_gdn) });
                        inst.set_ptr(3, w.a_log.as_ptr());
                        inst.set_ptr(4, w.dt_bias.as_ptr());
                        inst.set_int(5, nvh_gdn as i32);
                        instructions.push(inst);
                    }

                    // GDN recurrence (nvh heads with GQA key sharing)
                    {
                        let gqa_group = nvh_gdn / nh_gdn;
                        let mut inst = Instruction::new(OP_GDN_RECUR, nvh_gdn as u32);
                        inst.set_ptr(1, act.q_gdn.as_ptr());
                        inst.set_ptr(2, act.k_gdn.as_ptr());
                        inst.set_ptr(3, act.v_gdn.as_ptr());
                        inst.set_ptr(4, act.gate_gdn.as_ptr());
                        inst.set_ptr(5, unsafe { prefill_bufs.b_proj.as_ptr().add(t * nvh_gdn) });
                        inst.set_output_ptr(6, gdn_state.recurrent.as_write_ptr());
                        inst.set_output_ptr(7, act.recurrent_out.as_write_ptr());
                        inst.set_int(8, kd as i32);
                        inst.set_int(9, vd as i32);
                        inst.set_int(10, gqa_group as i32);
                        instructions.push(inst);
                    }

                    // RMSNorm gated (z from batched z_proj[t])
                    {
                        let mut inst = Instruction::new(OP_RMSNORM_GATE, nvh_gdn as u32);
                        inst.set_output_ptr(1, act.normed_gated.as_write_ptr());
                        inst.set_ptr(2, act.recurrent_out.as_ptr());
                        inst.set_ptr(3, unsafe { prefill_bufs.z_proj.as_ptr().add(t * nvh_gdn * vd) });
                        inst.set_ptr(4, w.output_norm.as_ptr());
                        inst.set_int(5, nvh_gdn as i32);
                        inst.set_int(6, vd as i32);
                        inst.set_float(7, eps);
                        instructions.push(inst);
                    }

                    // Output projection
                    {
                        let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
                        inst.set_output_ptr(1, act.out_proj.as_write_ptr());
                        emit_linear_proj(&mut inst, &w.w_out, 2);
                        inst.set_ptr(3, act.normed_gated.as_ptr());
                        inst.set_int(4, hs as i32);
                        inst.set_int(5, (nvh_gdn * vd) as i32);
                        instructions.push(inst);
                    }

                    // Residual: hidden[t] = out_proj + hidden[t]
                    {
                        let hidden_t = unsafe { prefill_bufs.hidden.as_write_ptr().add(t * hs) };
                        let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
                        inst.set_output_ptr(1, hidden_t);
                        inst.set_ptr(2, act.out_proj.as_ptr());
                        inst.set_ptr(3, hidden_t);
                        inst.set_int(4, hs as i32);
                        instructions.push(inst);
                    }
                }

                // --- Batched FFN ---
                Self::compile_ffn_batched(cfg, &model.layers[layer_i], prefill_bufs, n, &mut instructions);

                gdn_idx += 1;
            }
        }

        // === Final norm + LM head (last token only) ===
        // Copy last token's hidden to act.hidden
        {
            let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, act.hidden.as_write_ptr());
            inst.set_ptr(2, unsafe { prefill_bufs.hidden.as_ptr().add((n - 1) * hs) });
            inst.set_int(3, hs as i32);
            instructions.push(inst);
        }
        // RMSNorm
        {
            let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, act.normed.as_write_ptr());
            inst.set_ptr(2, act.hidden.as_ptr());
            inst.set_int(3, hs as i32);
            instructions.push(inst);
        }
        {
            let mut inst = Instruction::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1);
            inst.set_output_ptr(1, act.hidden.as_write_ptr());
            inst.set_ptr(2, act.normed.as_ptr());
            inst.set_ptr(3, model.final_norm_weight.as_ptr());
            inst.set_int(4, hs as i32);
            inst.set_float(5, eps);
            instructions.push(inst);
        }
        // LM head
        {
            let mut inst = Instruction::new(OP_LINEAR_PROJ, cfg.vocab_size as u32);
            inst.set_output_ptr(1, act.logits.as_write_ptr());
            inst.set_ptr(2, if cfg.tie_word_embeddings { model.embed_weight.as_ptr() } else { model.lm_head_weight.as_ptr() });
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
            moe_barrier: None,
            barrier_layer_map: Vec::new(),
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
        let (normed_ptr, hidden_ptr, q_gate_attn_ptr, k_attn_ptr, v_attn_ptr,
             q_attn_ptr, gate_attn_ptr, attn_out_ptr, gated_out_ptr,
             out_proj_ptr, position_ids_ptr, ffn_hidden_ptr) =
        if let Some((pb, _)) = &prefill {
            (pb.normed.as_write_ptr(), pb.hidden.as_write_ptr(),
             pb.q_gate_attn.as_write_ptr(), pb.k_attn.as_write_ptr(), pb.v_attn.as_write_ptr(),
             pb.q_attn.as_write_ptr(), pb.gate_attn.as_write_ptr(),
             pb.attn_out.as_write_ptr(), pb.gated_out.as_write_ptr(),
             pb.out_proj.as_write_ptr(), pb.position_ids.as_ptr(), pb.hidden.as_write_ptr())
        } else {
            (act.normed.as_write_ptr(), act.hidden.as_write_ptr(),
             act.q_gate_attn.as_write_ptr(), act.k_attn.as_write_ptr(), act.v_attn.as_write_ptr(),
             act.q_attn.as_write_ptr(), act.gate_attn.as_write_ptr(),
             act.attn_out.as_write_ptr(), act.gated_out.as_write_ptr(),
             act.out_proj.as_write_ptr(), act.position_ids.as_ptr(), act.hidden.as_write_ptr())
        };

        // 1. RMSNorm
        {
            let mut inst = Instruction::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), n as u32);
            inst.set_output_ptr(1, normed_ptr);
            inst.set_ptr(2, hidden_ptr);
            inst.set_ptr(3, w.input_norm.as_ptr());
            inst.set_int(4, hs as i32);
            inst.set_float(5, eps);
            instructions.push(inst);
        }

        // 2. Q(+gate), K, V projections
        let q_mult = if cfg.has_output_gate { 2 } else { 1 };
        emit_batched_linear_proj(
            &w.w_q_gate, q_gate_attn_ptr, normed_ptr,
            nqh * hd * q_mult, hs, n, true, instructions,
        );
        emit_batched_linear_proj(
            &w.w_k, k_attn_ptr, normed_ptr,
            nkh * hd, hs, n, true, instructions,
        );
        emit_batched_linear_proj(
            &w.w_v, v_attn_ptr, normed_ptr,
            nkh * hd, hs, n, false, instructions,
        );

        // 3. Deinterleave Q+gate → Q, gate (only for gated Q models like Qwen3.5)
        if !cfg.has_output_gate {
            // No gate: Q projection writes directly to q_gate_attn which IS q_attn
            // Just copy q_gate_attn → q_attn (they may be different buffers)
            if n > 1 {
                let total = n * nqh * hd;
                let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(total as u32, 256));
                inst.set_output_ptr(1, q_attn_ptr);
                inst.set_ptr(2, q_gate_attn_ptr);
                inst.set_int(3, total as i32);
                instructions.push(inst);
            } else {
                let mut inst = Instruction::new(OP_D2D_COPY, div_ceil((nqh * hd) as u32, 256));
                inst.set_output_ptr(1, q_attn_ptr);
                inst.set_ptr(2, q_gate_attn_ptr);
                inst.set_int(3, (nqh * hd) as i32);
                instructions.push(inst);
            }
        } else if n > 1 {
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

        // 4a. KV write for PagedKv — BEFORE QK-norm so cache stores pre-norm K/V.
        //     Pre-norm K has full dynamic range; quantizing post-norm K (±0.06) is catastrophic
        //     (TVD=0.97). See exterior_algebra kb-20260328-115542-e172dd.
        //     QK-norm is applied at attention time after dequant (in op_attn_paged / op_attn_paged_quant).
        let paged_kv_write_before_norm = matches!(variant, AttentionVariant::PagedKv { .. });
        let mut paged_layer_k_offset: u64 = 0;
        let mut paged_layer_v_offset: u64 = 0;
        if paged_kv_write_before_norm {
            if let AttentionVariant::PagedKv { kv_cache, attn_layer_index } = &variant {
                let kv_stride = nkh * hd;
                let chunk_tokens: usize = 64;
                paged_layer_k_offset =
                    (*attn_layer_index * 2 * chunk_tokens * kv_stride * std::mem::size_of::<f32>()) as u64;
                paged_layer_v_offset =
                    paged_layer_k_offset + (chunk_tokens * kv_stride * std::mem::size_of::<f32>()) as u64;
                let chunk_head_stride = chunk_tokens * hd;
                let mut head_indices = Vec::new();
                for h in 0..nkh {
                    let k_copy_idx = instructions.len();
                    {
                        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
                        inst.set_output_ptr(1, unsafe { kv_cache.k.as_write_ptr().add(h * chunk_head_stride) });
                        inst.set_ptr(2, unsafe { k_attn_ptr.add(h * hd) });
                        inst.set_int(3, hd as i32);
                        inst.set_no_sync();
                        instructions.push(inst);
                    }
                    let v_copy_idx = instructions.len();
                    {
                        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
                        inst.set_output_ptr(1, unsafe { kv_cache.v.as_write_ptr().add(h * chunk_head_stride) });
                        inst.set_ptr(2, unsafe { v_attn_ptr.add(h * hd) });
                        inst.set_int(3, hd as i32);
                        if h < nkh - 1 { inst.set_no_sync(); }
                        instructions.push(inst);
                    }
                    head_indices.push((k_copy_idx, v_copy_idx));
                }
                kv_write_indices.push(head_indices);
                kv_base_ptrs.push((kv_cache.k.as_ptr() as u64, kv_cache.v.as_ptr() as u64));
            }
        }

        // 4b. QK norm (only for models that have qk_norm weights — e.g. Qwen3.5, not Mistral)
        if cfg.has_qk_norm {
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

        // Steps 5/6: variant-specific attention ops. PagedKv KV write already done above.

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
                        inst.set_output_ptr(1, unsafe { kv_cache.k.as_write_ptr().add(h * head_stride) });
                        inst.set_ptr(2, unsafe { k_attn_ptr.add(h * hd) });
                        inst.set_int(3, hd as i32);
                        inst.set_no_sync();
                        instructions.push(inst);
                        let v_copy_idx = instructions.len();
                        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
                        inst.set_output_ptr(1, unsafe { kv_cache.v.as_write_ptr().add(h * head_stride) });
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

            AttentionVariant::PagedKv { kv_cache: _, attn_layer_index } => {
                // KV write already emitted above (step 4a, before QK-norm).
                // Cache now stores pre-QK-norm K/V for quantization quality.

                // mRoPE after KV write (applied to working Q/K buffers, not cache)
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

                // OP_ATTN_PAGED_Q: quantized attention (grid_x=0 initially, patched when chunks seal)
                let quant_idx = instructions.len();
                attn_quant_indices.push(quant_idx);
                {
                    use crate::paged_kv::quantized_kv_offsets;
                    let chunk_tokens: usize = CHUNK_TOKENS;
                    let (q1d, q1s, rd_off, rs) = quantized_kv_offsets(cfg, chunk_tokens, *attn_layer_index, false);
                    let mut inst = Instruction::new(OP_ATTN_PAGED_Q, 0); // grid_x=0: skip until chunks are quantized
                    inst.set_int(1, 0);  // scratch ptr — patched when quantized KV is enabled
                    inst.set_ptr(2, q_attn_ptr);
                    inst.set_int(3, 0);  // quant page_table — patched per step
                    inst.set_int(4, 0);  // position_table — patched per step
                    inst.set_ptr(5, act.inv_freq.as_ptr());
                    inst.set_int(6, nqh as i32);
                    inst.set_int(7, nkh as i32);
                    inst.set_int(8, hd as i32);
                    inst.set_int(9, 0);  // quant_seq_len — patched per step
                    inst.set_int(10, chunk_tokens as i32);
                    inst.set_int(11, rd as i32);
                    inst.words[12] = q1d as u64;
                    inst.words[13] = q1s as u64;
                    inst.words[14] = rd_off as u64;
                    inst.words[15] = rs as u64;
                    // Pass k_norm weight for QK-norm after dequant (null = no QK-norm)
                    inst.set_ptr(16, if cfg.has_qk_norm { w.k_norm.as_ptr() } else { std::ptr::null() });
                    inst.set_no_sync(); // no sync between quant and f32 attention
                    instructions.push(inst);
                }

                // OP_ATTN_PAGED: f32 attention on active chunk + merge from scratch
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
                    inst.set_int(10, CHUNK_TOKENS as i32);
                    inst.set_int(11, rd as i32);
                    inst.words[12] = paged_layer_k_offset;
                    inst.words[13] = paged_layer_v_offset;
                    inst.words[14] = 0; // partial_state — patched when quantized KV enabled
                    // Pass k_norm weight for QK-norm after loading from cache (slot 16)
                    inst.set_ptr(16, if cfg.has_qk_norm { w.k_norm.as_ptr() } else { std::ptr::null() });
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
                        let k_dst = unsafe { kv_cache.k.as_write_ptr().add(dst_off) };
                        let k_src = unsafe { k_attn_ptr.add(src_off) };
                        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hd as u32, 256));
                        inst.set_output_ptr(1, k_dst);
                        inst.set_ptr(2, k_src);
                        inst.set_int(3, hd as i32);
                        inst.set_no_sync();
                        instructions.push(inst);

                        let v_dst = unsafe { kv_cache.v.as_write_ptr().add(dst_off) };
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

        // 10. Output gate (Qwen3.5 only) or pass-through
        let final_attn_ptr = if cfg.has_output_gate {
            let gate_size = n * nqh * hd;
            let mut inst = Instruction::new(OP_OUTPUT_GATE, div_ceil(gate_size as u32, 256));
            inst.set_output_ptr(1, gated_out_ptr);
            inst.set_ptr(2, attn_out_ptr);
            inst.set_ptr(3, gate_attn_ptr);
            inst.set_int(4, gate_size as i32);
            instructions.push(inst);
            gated_out_ptr
        } else {
            attn_out_ptr // skip output gate, use attention output directly
        };

        // 11. Output projection + residual
        emit_batched_linear_proj(
            &w.w_o, out_proj_ptr, final_attn_ptr,
            hs, nqh * hd, n, false, instructions,
        );
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
                inst.set_output_ptr(1, act.residual.as_write_ptr());
                inst.set_ptr(2, hidden_ptr);
                inst.set_int(3, hs as i32);
                instructions.push(inst);
            }
            {
                let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
                inst.set_output_ptr(1, act.hidden.as_write_ptr());
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
        let nvh = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let ck = cfg.linear_conv_kernel_dim;
        let qkv_dim = nh * kd * 2 + nvh * vd;
        let eps = cfg.rms_norm_eps;

        // 1. RMSNorm
        let mut inst = Instruction::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1);
        inst.set_output_ptr(1, act.normed.as_write_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_ptr(3, w.input_norm.as_ptr());
        inst.set_int(4, hs as i32);
        inst.set_float(5, eps);
        instructions.push(inst);

        // 2. QKV projection [6144, 1024] @ [1024] → [6144]
        // NO_SYNC: next 3 instructions (a/b/z proj) read normed, not qkv
        let mut inst = Instruction::new(OP_LINEAR_PROJ, qkv_dim as u32);
        inst.set_output_ptr(1, act.qkv.as_write_ptr());
        emit_linear_proj(&mut inst, &w.w_qkv, 2);
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, qkv_dim as i32);
        inst.set_int(5, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // 3. Project a [nvh], b [nvh], z [nvh*vd]
        let mut inst = Instruction::new(OP_LINEAR_PROJ, nvh as u32);
        inst.set_output_ptr(1, act.a_proj.as_write_ptr());
        emit_linear_proj(&mut inst, &w.w_a, 2);
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, nvh as i32);
        inst.set_int(5, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        let mut inst = Instruction::new(OP_LINEAR_PROJ, nvh as u32);
        inst.set_output_ptr(1, act.b_proj.as_write_ptr());
        emit_linear_proj(&mut inst, &w.w_b, 2);
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, nvh as i32);
        inst.set_int(5, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // z proj: SYNC here ensures QKV+a+b+z all complete before conv1d reads qkv
        let mut inst = Instruction::new(OP_LINEAR_PROJ, (nvh * vd) as u32);
        inst.set_output_ptr(1, act.z_proj.as_write_ptr());
        emit_linear_proj(&mut inst, &w.w_z, 2);
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, (nvh * vd) as i32);
        inst.set_int(5, hs as i32);
        instructions.push(inst);

        // 4. Causal conv1d on QKV (3 separate calls for q, k, v slices)
        let q_dim = nh * kd; // 2048
        let k_dim = nh * kd; // 2048
        let v_dim = nvh * vd; // 2048

        // Conv on Q portion — NO_SYNC: conv_k reads different qkv slice, writes different state/output
        let mut inst = Instruction::new(OP_CONV1D, div_ceil(q_dim as u32, 256));
        inst.set_output_ptr(1, conv_state.as_write_ptr());
        inst.set_ptr(2, act.qkv.as_ptr());
        inst.set_ptr(3, w.conv1d_weight_q.as_ptr());
        inst.set_output_ptr(4, act.q_gdn.as_write_ptr());
        inst.set_int(5, q_dim as i32);
        inst.set_int(6, ck as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // Conv on K portion — NO_SYNC: conv_v reads different slice
        let mut inst = Instruction::new(OP_CONV1D, div_ceil(k_dim as u32, 256));
        inst.set_output_ptr(1, unsafe { conv_state.as_write_ptr().add(q_dim * (ck - 1)) });
        inst.set_ptr(2, unsafe { act.qkv.as_ptr().add(q_dim) });
        inst.set_ptr(3, w.conv1d_weight_k.as_ptr());
        inst.set_output_ptr(4, act.k_gdn.as_write_ptr());
        inst.set_int(5, k_dim as i32);
        inst.set_int(6, ck as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // Conv on V portion
        let mut inst = Instruction::new(OP_CONV1D, div_ceil(v_dim as u32, 256));
        inst.set_output_ptr(1, unsafe { conv_state.as_write_ptr().add((q_dim + k_dim) * (ck - 1)) });
        inst.set_ptr(2, unsafe { act.qkv.as_ptr().add(q_dim + k_dim) });
        inst.set_ptr(3, w.conv1d_weight_v.as_ptr());
        inst.set_output_ptr(4, act.v_gdn.as_write_ptr());
        inst.set_int(5, v_dim as i32);
        inst.set_int(6, ck as i32);
        instructions.push(inst);

        // 5. GDN gate (nvh heads — per value head)
        let mut inst = Instruction::new(OP_GDN_GATE, div_ceil(nvh as u32, 256));
        inst.set_output_ptr(1, act.gate_gdn.as_write_ptr());
        inst.set_ptr(2, act.a_proj.as_ptr());
        inst.set_ptr(3, w.a_log.as_ptr());
        inst.set_ptr(4, w.dt_bias.as_ptr());
        inst.set_int(5, nvh as i32);
        instructions.push(inst);

        // 6. GDN recurrent (nvh heads, GQA key sharing)
        let gqa_group = nvh / nh;
        let mut inst = Instruction::new(OP_GDN_RECUR, nvh as u32);
        inst.set_ptr(1, act.q_gdn.as_ptr());
        inst.set_ptr(2, act.k_gdn.as_ptr());
        inst.set_ptr(3, act.v_gdn.as_ptr());
        inst.set_ptr(4, act.gate_gdn.as_ptr());
        inst.set_ptr(5, act.b_proj.as_ptr());
        inst.set_output_ptr(6, gdn_state.recurrent.as_write_ptr());
        inst.set_output_ptr(7, act.recurrent_out.as_write_ptr());
        inst.set_int(8, kd as i32);
        inst.set_int(9, vd as i32);
        inst.set_int(10, gqa_group as i32);
        instructions.push(inst);

        // 7. RMSNorm gated
        let mut inst = Instruction::new(OP_RMSNORM_GATE, nvh as u32);
        inst.set_output_ptr(1, act.normed_gated.as_write_ptr());
        inst.set_ptr(2, act.recurrent_out.as_ptr());
        inst.set_ptr(3, act.z_proj.as_ptr());
        inst.set_ptr(4, w.output_norm.as_ptr());
        inst.set_int(5, nvh as i32);
        inst.set_int(6, vd as i32);
        inst.set_float(7, eps);
        instructions.push(inst);

        // 8. Output projection [1024, 2048]
        let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
        inst.set_output_ptr(1, act.out_proj.as_write_ptr());
        emit_linear_proj(&mut inst, &w.w_out, 2);
        inst.set_ptr(3, act.normed_gated.as_ptr());
        inst.set_int(4, hs as i32);
        inst.set_int(5, (nvh * vd) as i32);
        instructions.push(inst);

        // 9. Residual: copy hidden→residual, then add
        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
        inst.set_output_ptr(1, act.residual.as_write_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_int(3, hs as i32);
        instructions.push(inst);

        let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
        inst.set_output_ptr(1, act.hidden.as_write_ptr());
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
            &mut Vec::new(), &mut Vec::new(),
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
        attn_quant_indices: &mut Vec<usize>,
    ) {
        let w = match layer {
            LayerWeights::Attention(w) => w,
            _ => panic!("expected attention layer"),
        };
        Self::emit_attention_layer(
            cfg, w, act, None,
            &AttentionVariant::PagedKv { kv_cache, attn_layer_index },
            instructions, mrope_indices, &mut Vec::new(), kv_write_indices, kv_base_ptrs,
            attn_paged_indices, attn_quant_indices,
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
            let mut inst = Instruction::new(OP_FFN_GATE_UP, (is * n) as u32);
            inst.set_output_ptr(1, bufs.ffn_act.as_write_ptr());
            inst.set_ptr(2, bufs.hidden.as_ptr());
            inst.set_ptr(3, post_norm.as_ptr());
            inst.set_ptr(4, w_gate.as_bf16_ptr());
            inst.set_ptr(5, w_up.as_bf16_ptr());
            inst.set_int(6, hs as i32);
            inst.set_int(7, is as i32);
            inst.set_float(8, eps);
            inst.set_int(9, n as i32);
            instructions.push(inst);

            let mut inst = Instruction::new(OP_D2D_COPY, div_ceil((n * hs) as u32, 256));
            inst.set_output_ptr(1, bufs.residual.as_write_ptr());
            inst.set_ptr(2, bufs.hidden.as_ptr());
            inst.set_int(3, (n * hs) as i32);
            instructions.push(inst);

            let mut inst = Instruction::new(OP_FFN_DOWN_RES, (hs * n) as u32);
            inst.set_output_ptr(1, bufs.hidden.as_write_ptr());
            inst.set_ptr(2, bufs.residual.as_ptr());
            inst.set_ptr(3, w_down.as_bf16_ptr());
            inst.set_ptr(4, bufs.ffn_act.as_ptr());
            inst.set_int(5, hs as i32);
            inst.set_int(6, is as i32);
            inst.set_int(7, n as i32);
            instructions.push(inst);
        } else {
            // Unfused path for quantized weights: process one token at a time.
            // Uses ffn_gate_scratch/ffn_up_scratch/ffn_down_scratch as single-token intermediates.
            for t in 0..n {
                let hidden_t = unsafe { bufs.hidden.as_write_ptr().add(t * hs) };
                let normed_t = unsafe { bufs.normed.as_write_ptr().add(t * hs) };
                let residual_t = unsafe { bufs.residual.as_write_ptr().add(t * hs) };

                // D2D_COPY: hidden[t] → residual[t]  (no_sync: RMSNorm reads hidden, not residual)
                let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
                inst.set_output_ptr(1, residual_t);
                inst.set_ptr(2, hidden_t);
                inst.set_int(3, hs as i32);
                inst.set_no_sync();
                instructions.push(inst);

                // RMSNorm: hidden[t] → normed[t]
                let mut inst = Instruction::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1);
                inst.set_output_ptr(1, normed_t);
                inst.set_ptr(2, hidden_t);
                inst.set_ptr(3, post_norm.as_ptr());
                inst.set_int(4, hs as i32);
                inst.set_float(5, eps);
                instructions.push(inst);

                // Gate: normed[t] → ffn_gate_scratch  (no_sync: up reads same normed)
                let mut inst = Instruction::new(OP_LINEAR_PROJ, is as u32);
                emit_linear_proj(&mut inst, w_gate, 2);
                inst.set_output_ptr(1, bufs.ffn_gate_scratch.as_write_ptr());
                inst.set_ptr(3, normed_t);
                inst.set_int(4, is as i32);
                inst.set_int(5, hs as i32);
                inst.set_no_sync();
                instructions.push(inst);

                // Up: normed[t] → ffn_up_scratch
                let mut inst = Instruction::new(OP_LINEAR_PROJ, is as u32);
                emit_linear_proj(&mut inst, w_up, 2);
                inst.set_output_ptr(1, bufs.ffn_up_scratch.as_write_ptr());
                inst.set_ptr(3, normed_t);
                inst.set_int(4, is as i32);
                inst.set_int(5, hs as i32);
                instructions.push(inst);

                // SiLU(gate) * up → ffn_act[t..t+is] (reuse ffn_act as scratch per token)
                let ffn_act_t = unsafe { bufs.ffn_act.as_write_ptr().add(t * is) };
                let mut inst = Instruction::new(OP_SILU_MUL, div_ceil(is as u32, 256));
                inst.set_output_ptr(1, ffn_act_t);
                inst.set_ptr(2, bufs.ffn_gate_scratch.as_ptr());
                inst.set_ptr(3, bufs.ffn_up_scratch.as_ptr());
                inst.set_int(4, is as i32);
                instructions.push(inst);

                // Down: ffn_act[t] → ffn_down_scratch
                let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
                emit_linear_proj(&mut inst, w_down, 2);
                inst.set_output_ptr(1, bufs.ffn_down_scratch.as_write_ptr());
                inst.set_ptr(3, ffn_act_t);
                inst.set_int(4, hs as i32);
                inst.set_int(5, is as i32);
                instructions.push(inst);

                // Residual: ffn_down_scratch + residual[t] → hidden[t]
                let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
                inst.set_output_ptr(1, hidden_t);
                inst.set_ptr(2, bufs.ffn_down_scratch.as_ptr());
                inst.set_ptr(3, residual_t);
                inst.set_int(4, hs as i32);
                instructions.push(inst);
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
            // Fused path: OP_FFN_GATE_UP + OP_FFN_DOWN_RES (bf16 only)
            let mut inst = Instruction::new(OP_FFN_GATE_UP, is as u32);
            inst.set_output_ptr(1, act.ffn_act.as_write_ptr());
            inst.set_ptr(2, act.hidden.as_ptr());
            inst.set_ptr(3, post_norm.as_ptr());
            inst.set_ptr(4, w_gate.as_bf16_ptr());
            inst.set_ptr(5, w_up.as_bf16_ptr());
            inst.set_int(6, hs as i32);
            inst.set_int(7, is as i32);
            inst.set_float(8, eps);
            instructions.push(inst);

            let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, act.residual.as_write_ptr());
            inst.set_ptr(2, act.hidden.as_ptr());
            inst.set_int(3, hs as i32);
            instructions.push(inst);

            let mut inst = Instruction::new(OP_FFN_DOWN_RES, hs as u32);
            inst.set_output_ptr(1, act.hidden.as_write_ptr());
            inst.set_ptr(2, act.residual.as_ptr());
            inst.set_ptr(3, w_down.as_bf16_ptr());
            inst.set_ptr(4, act.ffn_act.as_ptr());
            inst.set_int(5, hs as i32);
            inst.set_int(6, is as i32);
            instructions.push(inst);
        } else if all_rnf4 {
            // Fused path: OP_FFN_GATE_UP_RNF4 + OP_FFN_DOWN_RES_RNF4 (rnf4 decode n=1 only)
            let w_gate_ptr = match w_gate { LinearWeight::Packed(pw) => pw.data.as_ptr(), _ => unreachable!() };
            let w_up_ptr   = match w_up   { LinearWeight::Packed(pw) => pw.data.as_ptr(), _ => unreachable!() };
            let w_down_ptr = match w_down { LinearWeight::Packed(pw) => pw.data.as_ptr(), _ => unreachable!() };

            let mut inst = Instruction::new(OP_FFN_GATE_UP_RNF4, is as u32);
            inst.set_output_ptr(1, act.ffn_act.as_write_ptr());
            inst.set_ptr(2, act.hidden.as_ptr());
            inst.set_ptr(3, post_norm.as_ptr());
            inst.set_ptr(4, w_gate_ptr);
            inst.set_ptr(5, w_up_ptr);
            inst.set_int(6, hs as i32);
            inst.set_int(7, is as i32);
            inst.set_float(8, eps);
            instructions.push(inst);

            let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, act.residual.as_write_ptr());
            inst.set_ptr(2, act.hidden.as_ptr());
            inst.set_int(3, hs as i32);
            instructions.push(inst);

            let mut inst = Instruction::new(OP_FFN_DOWN_RES_RNF4, hs as u32);
            inst.set_output_ptr(1, act.hidden.as_write_ptr());
            inst.set_ptr(2, act.residual.as_ptr());
            inst.set_ptr(3, w_down_ptr);
            inst.set_ptr(4, act.ffn_act.as_ptr());
            inst.set_int(5, hs as i32);
            inst.set_int(6, is as i32);
            instructions.push(inst);
        } else {
            // Unfused path for quantized weights (decode n=1 only)
            // D2D_COPY: hidden → residual (NO_SYNC: RMSNorm reads hidden, not residual)
            let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, act.residual.as_write_ptr());
            inst.set_ptr(2, act.hidden.as_ptr());
            inst.set_int(3, hs as i32);
            inst.set_no_sync();
            instructions.push(inst);

            // RMSNorm: hidden → normed
            let mut inst = Instruction::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1);
            inst.set_output_ptr(1, act.normed.as_write_ptr());
            inst.set_ptr(2, act.hidden.as_ptr());
            inst.set_ptr(3, post_norm.as_ptr());
            inst.set_int(4, hs as i32);
            inst.set_float(5, eps);
            instructions.push(inst);

            // Gate: normed → ffn_gate (NO_SYNC: up_proj reads same normed, writes different buf)
            let mut inst = Instruction::new(OP_LINEAR_PROJ, is as u32);
            emit_linear_proj(&mut inst, w_gate, 2);
            inst.set_output_ptr(1, act.ffn_gate.as_write_ptr());
            inst.set_ptr(3, act.normed.as_ptr());
            inst.set_int(4, is as i32);
            inst.set_int(5, hs as i32);
            inst.set_no_sync();
            instructions.push(inst);

            // Up: normed → ffn_up
            let mut inst = Instruction::new(OP_LINEAR_PROJ, is as u32);
            emit_linear_proj(&mut inst, w_up, 2);
            inst.set_output_ptr(1, act.ffn_up.as_write_ptr());
            inst.set_ptr(3, act.normed.as_ptr());
            inst.set_int(4, is as i32);
            inst.set_int(5, hs as i32);
            instructions.push(inst);

            // SiLU(gate) * up → ffn_act
            let mut inst = Instruction::new(OP_SILU_MUL, div_ceil(is as u32, 256));
            inst.set_output_ptr(1, act.ffn_act.as_write_ptr());
            inst.set_ptr(2, act.ffn_gate.as_ptr());
            inst.set_ptr(3, act.ffn_up.as_ptr());
            inst.set_int(4, is as i32);
            instructions.push(inst);

            // Down: ffn_act → ffn_down
            let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
            emit_linear_proj(&mut inst, w_down, 2);
            inst.set_output_ptr(1, act.ffn_down.as_write_ptr());
            inst.set_ptr(3, act.ffn_act.as_ptr());
            inst.set_int(4, hs as i32);
            inst.set_int(5, is as i32);
            instructions.push(inst);

            // Residual: ffn_down + residual → hidden
            let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
            inst.set_output_ptr(1, act.hidden.as_write_ptr());
            inst.set_ptr(2, act.ffn_down.as_ptr());
            inst.set_ptr(3, act.residual.as_ptr());
            inst.set_int(4, hs as i32);
            instructions.push(inst);
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
            FfnType::MoE { num_active, gate_type, expert_intermediate_size, .. } =>
                (*num_active, gate_type.clone(), *expert_intermediate_size),
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
        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
        inst.set_output_ptr(1, act.residual.as_write_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_int(3, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // RMSNorm: hidden → normed
        let mut inst = Instruction::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1);
        inst.set_output_ptr(1, act.normed.as_write_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_ptr(3, norm_ptr);
        inst.set_int(4, hs as i32);
        inst.set_float(5, eps);
        instructions.push(inst);

        // Gate projection: normed → moe_scores[num_experts]
        let mut inst = Instruction::new(OP_LINEAR_PROJ, ne as u32);
        inst.set_output_ptr(1, act.moe_scores.as_write_ptr());
        inst.set_ptr(2, moe.gate.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, ne as i32);
        inst.set_int(5, hs as i32);
        instructions.push(inst);

        // OP_MOE_GATE: top-k selection on GPU
        let (gate_mode, rsf) = match &gate_type {
            GateType::Softmax => (0u32, 1.0f32),
            GateType::NormTopK { routed_scaling_factor } => (1, *routed_scaling_factor),
            GateType::Sigmoid { routed_scaling_factor } => (2, *routed_scaling_factor),
        };
        let bias_ptr = moe.score_correction_bias_gpu.as_ref()
            .map(|b| b.as_ptr() as *const u8).unwrap_or(std::ptr::null());

        let mut inst = Instruction::new(OP_MOE_GATE, 1);
        inst.set_ptr(1, act.moe_scores.as_ptr());
        inst.set_ptr(2, act.moe_expert_ids.as_ptr());
        inst.set_ptr(3, act.moe_expert_weights.as_ptr());
        inst.set_int(4, ne as i32);
        inst.set_int(5, k as i32);
        inst.set_int(6, gate_mode as i32);
        inst.set_float(7, rsf);
        inst.set_ptr(8, bias_ptr);
        instructions.push(inst);

        // OP_MOE_FFN: fused expert loop (internal grid.sync())
        // Currently only supports PcG32Q4 weights in the GPU kernel
        assert!(
            matches!(moe.expert_gate_up.weight_format(), crate::quant::WeightFormat::PcG32Q4),
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

        let flags = (if moe.has_gate_proj { 1u32 } else { 0 })
                  | (if !moe.has_gate_proj { 2 } else { 0 }); // bit1 = relu²

        let grid_x = std::cmp::max(eis, hs) as u32;

        let mut inst = Instruction::new(OP_MOE_FFN, grid_x);
        inst.set_ptr(1, act.moe_expert_ids.as_ptr());
        inst.set_ptr(2, act.moe_expert_weights.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_output_ptr(4, act.ffn_down.as_write_ptr());
        inst.set_ptr(5, moe.expert_gate_up.raw_data_ptr());
        inst.words[6] = gate_up_expert_stride as u64;
        inst.set_ptr(7, moe.expert_down.raw_data_ptr());
        inst.words[8] = down_expert_stride as u64;
        inst.set_int(9, k as i32);
        inst.set_int(10, (hs | (eis << 16)) as i32);
        inst.set_int(11, flags as i32);
        inst.set_ptr(12, act.moe_expert_gate.as_ptr());
        inst.set_ptr(13, act.moe_expert_up.as_ptr());
        inst.set_ptr(14, act.moe_expert_act.as_ptr());
        inst.set_ptr(15, act.moe_expert_out.as_ptr());
        inst.words[16] = gate_up_row_stride as u64;
        instructions.push(inst);

        // Shared expert (if present)
        if let Some(ref se) = moe.shared_expert {
            let se_is = match &cfg.layers[layer_idx].ffn_type {
                FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } =>
                    if *shared_intermediate_size > 0 { *shared_intermediate_size } else { *expert_intermediate_size },
                _ => eis,
            };

            if moe.has_gate_proj {
                // gate_proj → gate scratch
                let mut inst = Instruction::new(OP_LINEAR_PROJ, se_is as u32);
                emit_linear_proj(&mut inst, &se.gate_proj, 2);
                inst.set_output_ptr(1, act.moe_expert_gate.as_write_ptr());
                inst.set_ptr(3, act.normed.as_ptr());
                inst.set_int(4, se_is as i32);
                inst.set_int(5, hs as i32);
                inst.set_no_sync();
                instructions.push(inst);

                // up_proj → up scratch
                let mut inst = Instruction::new(OP_LINEAR_PROJ, se_is as u32);
                emit_linear_proj(&mut inst, &se.up_proj, 2);
                inst.set_output_ptr(1, act.moe_expert_up.as_write_ptr());
                inst.set_ptr(3, act.normed.as_ptr());
                inst.set_int(4, se_is as i32);
                inst.set_int(5, hs as i32);
                instructions.push(inst);

                // silu_mul
                let mut inst = Instruction::new(OP_SILU_MUL, div_ceil(se_is as u32, 256));
                inst.set_output_ptr(1, act.moe_expert_act.as_write_ptr());
                inst.set_ptr(2, act.moe_expert_gate.as_ptr());
                inst.set_ptr(3, act.moe_expert_up.as_ptr());
                inst.set_int(4, se_is as i32);
                instructions.push(inst);
            } else {
                // up_proj → up scratch
                let mut inst = Instruction::new(OP_LINEAR_PROJ, se_is as u32);
                emit_linear_proj(&mut inst, &se.up_proj, 2);
                inst.set_output_ptr(1, act.moe_expert_up.as_write_ptr());
                inst.set_ptr(3, act.normed.as_ptr());
                inst.set_int(4, se_is as i32);
                inst.set_int(5, hs as i32);
                instructions.push(inst);

                // relu² — use OP_SILU_MUL with same input for gate+up (relu² handled by kernel)
                // Actually, we need a relu² op... for now emit as two ops: silu_mul won't work.
                // TODO: add OP_RELU_SQ or handle in shared expert path
                // Workaround: emit the relu² computation via the standalone kernel path
                // For now, skip shared expert in megakernel for relu² models (Nemotron)
                // This is acceptable since Nemotron uses Mamba2 layers which aren't in megakernel yet
            }

            // down_proj → expert_out scratch
            let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
            emit_linear_proj(&mut inst, &se.down_proj, 2);
            inst.set_output_ptr(1, act.moe_expert_out.as_write_ptr());
            inst.set_ptr(3, act.moe_expert_act.as_ptr());
            inst.set_int(4, hs as i32);
            inst.set_int(5, se_is as i32);
            instructions.push(inst);

            // Shared expert gating: ffn_down += sigmoid(gate @ normed) * expert_out
            if let Some(ref gate_buf) = moe.shared_expert_gate {
                // Compute dot product: gate_weight @ normed → scalar (reuse moe_scores[0])
                let mut inst = Instruction::new(OP_LINEAR_PROJ, 1);
                inst.set_output_ptr(1, act.moe_scores.as_write_ptr());
                inst.set_ptr(2, gate_buf.as_ptr());
                inst.set_ptr(3, act.normed.as_ptr());
                inst.set_int(4, 1i32);   // out_dim = 1
                inst.set_int(5, hs as i32);
                instructions.push(inst);

                // ffn_down += sigmoid(scalar) * expert_out
                let mut inst = Instruction::new(OP_SIGMOID_WEIGHTED_ADD, div_ceil(hs as u32, 256));
                inst.set_output_ptr(1, act.ffn_down.as_write_ptr());
                inst.set_ptr(2, act.moe_scores.as_ptr());
                inst.set_ptr(3, act.moe_expert_out.as_ptr());
                inst.set_int(4, hs as i32);
                instructions.push(inst);
            } else {
                // No gate: ffn_down += expert_out
                let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
                inst.set_output_ptr(1, act.ffn_down.as_write_ptr());
                inst.set_ptr(2, act.ffn_down.as_ptr());
                inst.set_ptr(3, act.moe_expert_out.as_ptr());
                inst.set_int(4, hs as i32);
                instructions.push(inst);
            }
        }

        // Residual: hidden = residual + ffn_down
        let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
        inst.set_output_ptr(1, act.hidden.as_write_ptr());
        inst.set_ptr(2, act.residual.as_ptr());
        inst.set_ptr(3, act.ffn_down.as_ptr());
        inst.set_int(4, hs as i32);
        instructions.push(inst);
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
            FfnType::MoE { num_active, gate_type, num_experts, expert_intermediate_size, .. } =>
                (*num_active, gate_type.clone(), *num_experts, *expert_intermediate_size),
            _ => unreachable!(),
        };

        let norm_ptr = match layer {
            LayerWeights::Attention(w) => w.post_norm.as_ptr(),
            LayerWeights::Gdn(w) => w.post_norm.as_ptr(),
            LayerWeights::MoeFfn(w) => w.input_norm.as_ptr(),
            _ => panic!("no norm weight for MoE FFN layer"),
        };

        // D2D_COPY: hidden → residual
        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
        inst.set_output_ptr(1, act.residual.as_write_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_int(3, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // RMSNorm: hidden → normed
        let mut inst = Instruction::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1);
        inst.set_output_ptr(1, act.normed.as_write_ptr());
        inst.set_ptr(2, act.hidden.as_ptr());
        inst.set_ptr(3, norm_ptr);
        inst.set_int(4, hs as i32);
        inst.set_float(5, eps);
        instructions.push(inst);

        // Gate projection: normed → moe_scores[num_experts]
        let mut inst = Instruction::new(OP_LINEAR_PROJ, ne as u32);
        inst.set_output_ptr(1, act.moe_scores.as_write_ptr());
        inst.set_ptr(2, moe.gate.as_ptr());
        inst.set_ptr(3, act.normed.as_ptr());
        inst.set_int(4, ne as i32);
        inst.set_int(5, hs as i32);
        instructions.push(inst);

        // D2D_COPY: normed → normed_stage (GART/pinned memory, CPU-readable without hipMemcpy)
        // Must happen before OP_BARRIER so CPU can read activation for worker broadcast.
        let mut inst = Instruction::new(OP_D2D_COPY, div_ceil(hs as u32, 256));
        inst.set_output_ptr(1, act.normed_stage.as_write_ptr());
        inst.set_ptr(2, act.normed.as_ptr());
        inst.set_int(3, hs as i32);
        inst.set_no_sync();
        instructions.push(inst);

        // OP_MOE_GATE: top-k selection; writes to moe_expert_ids/weights (GART memory, CPU-readable)
        let (gate_mode, rsf) = match &gate_type {
            GateType::Softmax => (0u32, 1.0f32),
            GateType::NormTopK { routed_scaling_factor } => (1, *routed_scaling_factor),
            GateType::Sigmoid { routed_scaling_factor } => (2, *routed_scaling_factor),
        };
        let bias_ptr = moe.score_correction_bias_gpu.as_ref()
            .map(|b| b.as_ptr() as *const u8).unwrap_or(std::ptr::null());
        let mut inst = Instruction::new(OP_MOE_GATE, 1);
        inst.set_ptr(1, act.moe_scores.as_ptr());
        inst.set_ptr(2, act.moe_expert_ids.as_ptr());
        inst.set_ptr(3, act.moe_expert_weights.as_ptr());
        inst.set_int(4, ne as i32);
        inst.set_int(5, k as i32);
        inst.set_int(6, gate_mode as i32);
        inst.set_float(7, rsf);
        inst.set_ptr(8, bias_ptr);
        instructions.push(inst);

        // OP_BARRIER: park megakernel, CPU dispatches expert FFN into act.ffn_down_stage, then resumes.
        // barrier_flag_ptr and resume_flag_ptr are null here; patched in execute_multi_gpu().
        let barrier_inst_idx = instructions.len();
        let mut inst = Instruction::new(OP_BARRIER, 1);  // grid_x=1: only block 0 runs op_barrier
        inst.set_ptr(1, std::ptr::null::<u32>());  // barrier_flag — patched per-execute
        inst.set_ptr(2, std::ptr::null::<u32>());  // resume_flag — patched per-execute
        inst.set_int(3, layer_idx as i32);
        instructions.push(inst);

        // After barrier: compute shared expert (if present) and add to ffn_down_stage.
        // GPU 0 has all SMs available again after resume.
        if let Some(ref se) = moe.shared_expert {
            let se_is = match &cfg.layers[layer_idx].ffn_type {
                FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } =>
                    if *shared_intermediate_size > 0 { *shared_intermediate_size } else { *expert_intermediate_size },
                _ => eis,
            };

            if moe.has_gate_proj {
                let mut inst = Instruction::new(OP_LINEAR_PROJ, se_is as u32);
                emit_linear_proj(&mut inst, &se.gate_proj, 2);
                inst.set_output_ptr(1, act.moe_expert_gate.as_write_ptr());
                inst.set_ptr(3, act.normed.as_ptr());
                inst.set_int(4, se_is as i32);
                inst.set_int(5, hs as i32);
                inst.set_no_sync();
                instructions.push(inst);

                let mut inst = Instruction::new(OP_LINEAR_PROJ, se_is as u32);
                emit_linear_proj(&mut inst, &se.up_proj, 2);
                inst.set_output_ptr(1, act.moe_expert_up.as_write_ptr());
                inst.set_ptr(3, act.normed.as_ptr());
                inst.set_int(4, se_is as i32);
                inst.set_int(5, hs as i32);
                instructions.push(inst);

                let mut inst = Instruction::new(OP_SILU_MUL, div_ceil(se_is as u32, 256));
                inst.set_output_ptr(1, act.moe_expert_act.as_write_ptr());
                inst.set_ptr(2, act.moe_expert_gate.as_ptr());
                inst.set_ptr(3, act.moe_expert_up.as_ptr());
                inst.set_int(4, se_is as i32);
                instructions.push(inst);
            } else {
                // relu² shared expert — not yet supported in megakernel multi-GPU path.
                // Only affects models without gate_proj (e.g. Nemotron), which use Mamba2
                // and are handled by a different path. Skip down_proj for now.
                let mut inst = Instruction::new(OP_LINEAR_PROJ, se_is as u32);
                emit_linear_proj(&mut inst, &se.up_proj, 2);
                inst.set_output_ptr(1, act.moe_expert_up.as_write_ptr());
                inst.set_ptr(3, act.normed.as_ptr());
                inst.set_int(4, se_is as i32);
                inst.set_int(5, hs as i32);
                instructions.push(inst);
                // TODO(braidinfer-xsz): add OP_RELU_SQ opcode; for now skip shared expert down_proj on relu² models
                return barrier_inst_idx;
            }

            let mut inst = Instruction::new(OP_LINEAR_PROJ, hs as u32);
            emit_linear_proj(&mut inst, &se.down_proj, 2);
            inst.set_output_ptr(1, act.moe_expert_out.as_write_ptr());
            inst.set_ptr(3, act.moe_expert_act.as_ptr());
            inst.set_int(4, hs as i32);
            inst.set_int(5, se_is as i32);
            instructions.push(inst);

            // Add shared expert output into ffn_down_stage
            if let Some(ref gate_buf) = moe.shared_expert_gate {
                let mut inst = Instruction::new(OP_LINEAR_PROJ, 1);
                inst.set_output_ptr(1, act.moe_scores.as_write_ptr());
                inst.set_ptr(2, gate_buf.as_ptr());
                inst.set_ptr(3, act.normed.as_ptr());
                inst.set_int(4, 1i32);
                inst.set_int(5, hs as i32);
                instructions.push(inst);

                let mut inst = Instruction::new(OP_SIGMOID_WEIGHTED_ADD, div_ceil(hs as u32, 256));
                inst.set_output_ptr(1, act.ffn_down_stage.as_write_ptr());
                inst.set_ptr(2, act.moe_scores.as_ptr());
                inst.set_ptr(3, act.moe_expert_out.as_ptr());
                inst.set_int(4, hs as i32);
                instructions.push(inst);
            } else {
                // No gate: ffn_down_stage += shared_expert_out
                let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
                inst.set_output_ptr(1, act.ffn_down_stage.as_write_ptr());
                inst.set_ptr(2, act.ffn_down_stage.as_ptr() as *const f32);
                inst.set_ptr(3, act.moe_expert_out.as_ptr());
                inst.set_int(4, hs as i32);
                instructions.push(inst);
            }
        }

        // Final residual: hidden = residual + ffn_down_stage
        // ffn_down_stage contains: gathered worker expert outputs (CPU-written) + shared expert
        let mut inst = Instruction::new(OP_RESIDUAL_ADD, div_ceil(hs as u32, 256));
        inst.set_output_ptr(1, act.hidden.as_write_ptr());
        inst.set_ptr(2, act.residual.as_ptr());
        inst.set_ptr(3, act.ffn_down_stage.as_ptr() as *const f32);
        inst.set_int(4, hs as i32);
        instructions.push(inst);

        barrier_inst_idx
    }
}
