//! Megakernel program compilation: translates model config + weights into instruction streams.
//! Extracted from megakernel.rs for maintainability.

use braidinfer_hip::HipResult;
use braidinfer_hip::module::Module;
use std::sync::Arc;

use super::compile_common::{AttentionVariant, div_ceil, emit_batched_linear_proj, linear_proj_opcode_ptr, rmsnorm_opcode};

use super::upload_program;
use super::instructions::*;
use super::{CHUNK_TOKENS, Instruction, MegakernelProgram, NUM_CUS, PrefillBuffers};
#[allow(unused_imports)]
use super::{
    OP_ATTN_PAGED, OP_ATTN_PAGED_Q, OP_ATTN_PREFILL, OP_BARRIER, OP_CONV1D, OP_D2D_COPY,
    OP_DEINTERLEAVE, OP_EMBEDDING, OP_FFN_DOWN_RES, OP_FFN_DOWN_RES_RNF4, OP_FFN_GATE_UP,
    OP_FFN_GATE_UP_RNF4, OP_GDN_GATE, OP_GDN_RECUR, OP_GQA_ATTN, OP_HALT, OP_KV_QUANTIZE,
    OP_LINEAR_PROJ, OP_LINEAR_PROJ_PCG32, OP_LINEAR_PROJ_RNF4, OP_LM_HEAD, OP_MAMBA2_CONV1D,
    OP_MAMBA2_NORM_GATED, OP_MOE_DISPATCH, OP_MOE_DISPATCH_POST, OP_MOE_FFN, OP_MOE_GATE, OP_MROPE, OP_OUTPUT_GATE,
    OP_QK_NORM, OP_RELU_SQ, OP_RESIDUAL_ADD, OP_RMSNORM, OP_RMSNORM_GATE, OP_RMSNORM_WX,
    OP_SIGMOID_WEIGHTED_ADD, OP_SILU_MUL, OP_SSM_UPDATE,
};
use crate::model::{LayerWeights, Model};
use crate::watchdog::WatchdogThread;

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
                        num_blocks,
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
                        // emit_post_barrier=false for fc2_latent_proj layers (Nemotron-H): the
                        // correct post-barrier sequence (fc2 + shared_expert + residual_add) is
                        // inserted by compile_inner_p2p after patching OP_BARRIER→OP_MOE_DISPATCH.
                        let emit_post_barrier = moe.fc2_latent_proj.is_none();
                        let barrier_inst_idx = Self::compile_moe_ffn_multi_gpu(
                            cfg,
                            layer_i,
                            &model.layers[layer_i],
                            moe,
                            act,
                            &mut instructions,
                            emit_post_barrier,
                        );
                        barrier_layer_map.push((barrier_inst_idx, layer_i));
                    } else {
                        let moe = model.moe_weights[layer_i].as_ref().unwrap();
                        // Skip if weights are lite-loaded (empty expert buffers, multi-GPU model).
                        // When MULTI_GPU=1, expert_gate_up is a zero-size placeholder; using its
                        // pointer in the megakernel would fault.
                        if moe.expert_gate_up.num_elements() > 0 {
                            Self::compile_moe_ffn(
                                cfg,
                                layer_i,
                                &model.layers[layer_i],
                                moe,
                                act,
                                &mut instructions,
                            );
                        }
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
            // braidinfer-bfd: lm_head dispatched as OP_LM_HEAD (LDS-tiled
            // full-grid variant) instead of OP_LINEAR_PROJ. Numerically
            // equivalent — same accumulation order, only input source
            // changes from VRAM to LDS-staged.
            instructions.push(LinearProjInst::new(OP_LM_HEAD, vs as u32, act.logits.as_write_ptr(), lm_weight, act.hidden.as_ptr(), vs as i32, hs as i32, 0).into_inst());
        }

        // HALT
        instructions.push(HaltInst::new().into_inst());

        // Upload program to device
        let device_program = upload_program(device, &instructions)?;
        let flat_program: Vec<u64> = instructions.iter().flat_map(|i| i.words).collect();

        let watchdog = WatchdogThread::spawn();
        let wd_state_dev = watchdog.register(device)?;
        let wd_dev_ptr = wd_state_dev as *mut std::ffi::c_void;

        Ok(MegakernelProgram {
            instructions,
            device_program,
            module: Arc::new(module),
            num_blocks,
            shared_mem,
            device,
            embedding_inst_idx,
            _mrope_inst_indices: mrope_inst_indices,
            gqa_attn_inst_indices,
            position_ids_dev_ptr: act.position_ids.as_ptr() as u64,
            kv: super::KvConfig {
                max_seq_len: cfg.max_seq_len as u32,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                kv_write_indices,
                kv_base_ptrs,
            },
            paged,
            paged_kv: if paged {
                Some(super::PagedKvState {
                    page_table: None,
                    position_table: None,
                    attn_paged_inst_indices,
                    attn_quant_inst_indices,
                    last_page_table_len: 0,
                    kv_stride_paged: cfg.num_kv_heads * cfg.head_dim,
                })
            } else {
                None
            },
            quantized_kv: false,
            quant_kv: None,
            prefill_cache: None,
            dump_buffer: None,
            dump_counter: None,
            dump_capacity: 0,
            barrier_layer_map,
            multi_gpu_attn_boundaries,
            flat_program,
            _watchdog: watchdog,
            wd_dev_ptr,
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
            instructions.push(inst.into_inst());
        }

        // Upload positions into prefill_bufs.position_ids: [N × 3] i32
        prefill_bufs.write_positions(start_pos, n)?;

        // === Layers ===
        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        let mut _attn_layer_count = 0usize;
        let mut prefill_kv_entries: Vec<super::PrefillKvEntry> = Vec::new();
        let mut prefill_attn_inst_indices: Vec<usize> = Vec::new();
        let mut prefill_kv_base_ptrs: Vec<(u64, u64)> = Vec::new();

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

                // Record KV base pointers for this attention layer (at position 0)
                prefill_kv_base_ptrs.push((kv_cache.k.as_ptr() as u64, kv_cache.v.as_ptr() as u64));
                let layer_kv_idx = prefill_kv_base_ptrs.len() - 1;

                let attn_start = instructions.len();
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
                    kv_idx == 0 && super::k_trace_5ax_enabled(),
                );

                // Scan emitted instructions for KV-write D2dCopy and AttnPrefillInst.
                // KV writes are emitted by the Prefill variant AFTER the MROPE instruction
                // (compile_attention.rs:283-309). Scanning from `attn_start` would mis-identify
                // the leading q_gate→q_attn D2dCopy emitted at compile_attention.rs:87-89 when
                // !cfg.has_output_gate (Mistral/Llama) as the first KV-write — corrupting it
                // (and offsetting all subsequent KV-write entries by one) when update_prefill_chunk
                // patches the cached program for sequential N=1 prefill calls.
                let nkh = cfg.num_kv_heads;
                let kv_scan_start = (attn_start..instructions.len())
                    .find(|&idx| instructions[idx].words[0] as u32 == OP_MROPE)
                    .map(|i| i + 1)
                    .unwrap_or(attn_start);
                let mut kv_pair_count = 0usize;
                for idx in kv_scan_start..instructions.len() {
                    let opcode = instructions[idx].words[0] as u32;
                    if opcode == OP_D2D_COPY && kv_pair_count < n * nkh * 2 {
                        // KV writes: laid out as [t0h0K, t0h0V, t0h1K, t0h1V, ..., tnhkK, tnhkV]
                        let pair_flat = kv_pair_count / 2; // which (t, h) pair
                        let is_v = kv_pair_count % 2 == 1;
                        if !is_v {
                            let t = pair_flat / nkh;
                            let h = pair_flat % nkh;
                            // peek ahead for V
                            let v_idx = idx + 1;
                            prefill_kv_entries.push(super::PrefillKvEntry {
                                k_inst_idx: idx,
                                v_inst_idx: v_idx,
                                t, h, layer_kv_idx,
                            });
                        }
                        kv_pair_count += 1;
                    } else if opcode == OP_ATTN_PREFILL {
                        prefill_attn_inst_indices.push(idx);
                    }
                }

                // Batched FFN
                Self::compile_ffn_batched(
                    cfg,
                    layer_i,
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
                    ).into_inst());

                    // Conv1d on K
                    instructions.push(Conv1dInst::new(
                        div_ceil(k_dim as u32, 256),
                        unsafe { conv_state.as_write_ptr().add(q_dim * (ck - 1)) },
                        unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim + q_dim) },
                        w.conv1d_weight_k.as_ptr(),
                        act.k_gdn.as_write_ptr(),
                        k_dim as i32,
                        ck as i32,
                    ).into_inst());

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
                        let blocks_per_head = (num_blocks / nvh_gdn as u32).max(1);
                        instructions.push(GdnRecurInst::new(
                            nvh_gdn as u32 * blocks_per_head, nvh_gdn as u32,
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
                        let (lp_op, lp_w) = linear_proj_opcode_ptr(&w.w_out);
                        instructions.push(LinearProjInst::new(
                            lp_op, hs as u32,
                            act.out_proj.as_write_ptr(), lp_w, act.normed_gated.as_ptr(),
                            hs as i32, (nvh_gdn * vd) as i32, 0,
                        ).into_inst());
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
                    layer_i,
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
        let device_program = upload_program(device, &instructions)?;
        let flat_program: Vec<u64> = instructions.iter().flat_map(|i| i.words).collect();

        let watchdog = WatchdogThread::spawn();
        let wd_state_dev = watchdog.register(device)?;
        let wd_dev_ptr = wd_state_dev as *mut std::ffi::c_void;

        Ok(MegakernelProgram {
            instructions,
            device_program,
            module: Arc::new(module),
            num_blocks,
            shared_mem,
            device,
            embedding_inst_idx,
            _mrope_inst_indices: Vec::new(),
            gqa_attn_inst_indices: Vec::new(),
            position_ids_dev_ptr: prefill_bufs.position_ids.as_ptr() as u64,
            kv: super::KvConfig {
                max_seq_len: cfg.max_seq_len as u32,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                kv_write_indices: Vec::new(),
                kv_base_ptrs: prefill_kv_base_ptrs,
            },
            paged: false,
            paged_kv: None,
            quantized_kv: false,
            quant_kv: None,
            prefill_cache: Some(super::PrefillCacheState {
                embedding_start: embedding_inst_idx,
                kv_entries: prefill_kv_entries,
                attn_inst_indices: prefill_attn_inst_indices,
                n,
            }),
            dump_buffer: None,
            dump_counter: None,
            dump_capacity: 0,
            barrier_layer_map: Vec::new(),
            multi_gpu_attn_boundaries: Vec::new(),
            flat_program,
            _watchdog: watchdog,
            wd_dev_ptr,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Compile a one-shot prefill program for `tokens` starting at `start_pos` using
    /// paged KV layout. Unlike `compile_prefill` (which writes to legacy flat KV caches),
    /// this function writes KV into paged chunk slots allocated from `allocator`, populating
    /// `seq` so that subsequent paged decode steps can attend to the prefill context.
    ///
    /// Algorithm: emit N sequential single-token steps, each writing KV into the appropriate
    /// paged chunk slot and attending over tokens 0..t via OP_ATTN_PAGED. This produces
    /// correct causal attention and populates paged KV in one pass.
    ///
    /// Caller must: pre-init page_table and position_table buffers on `mk_paged`
    /// (via `init_paged_buffers`), then call this function, then execute the program.
    /// After execution, `seq` and `allocator` reflect the prefill state; decode steps
    /// continue from `seq.seq_len == tokens.len()`.
    ///
    /// The returned program is a one-shot program (not re-usable like compile_prefill);
    /// chunk addresses are baked in. For persistent prefill, compile once per prefill call.
    pub fn compile_prefill_paged(
        model: &Model,
        tokens: &[u32],
        start_pos: u32,
        seq: &mut crate::paged_kv::SequenceState,
        allocator: &mut crate::paged_kv::PageAllocator,
        page_table_buf: &braidinfer_hip::memory::MappedHostBuffer<u64>,
        position_table_buf: &braidinfer_hip::memory::MappedHostBuffer<i32>,
        prefill_bufs: &mut PrefillBuffers,
        attn_paged_inst_indices_out: &mut Vec<usize>,
        attn_quant_inst_indices_out: &mut Vec<usize>,
        kv_write_indices_out: &mut Vec<Vec<(usize, usize)>>,
    ) -> HipResult<Self> {
        let n = tokens.len();
        assert!(n > 0 && n <= CHUNK_TOKENS * 16, "prefill_paged: too many tokens");
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
        let blocks_per_sm_clamped = blocks_per_sm.max(1) as u32;
        let num_blocks = blocks_per_sm_clamped * NUM_CUS;
        let mut instructions: Vec<Instruction> = Vec::new();

        let hs = cfg.hidden_size;
        let nkh = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let eps = cfg.rms_norm_eps;
        let kv_stride = nkh * hd;

        // Allocate all paged chunk slots needed for N tokens up front.
        // Pre-allocate to bake addresses into instructions.
        for i in 0..n {
            let pos = start_pos as i32 + i as i32;
            seq.append_token(pos, allocator)?;
        }

        // Write page_table and position_table to host-mapped buffers.
        // These are needed by OP_ATTN_PAGED at execution time.
        {
            let host_pt = page_table_buf.host_ptr();
            for (i, chunk) in seq.chunks.iter().enumerate() {
                let addr = allocator.slot_ptr(chunk.slot_index()) as u64;
                unsafe { host_pt.add(i).write_volatile(addr); }
            }
            // position_table layout: 3 ints per token (mRoPE temporal/height/width).
            // For text-only models all 3 are written equal — op_attn_paged reads
            // the section via mrope_section sizes.
            let host_pos = position_table_buf.host_ptr();
            for (i, &pos) in seq.positions.iter().enumerate() {
                unsafe {
                    let base = host_pos.add(i * 3);
                    base.add(0).write_volatile(pos);
                    base.add(1).write_volatile(pos);
                    base.add(2).write_volatile(pos);
                }
            }
        }

        let page_table_ptr = page_table_buf.as_ptr() as u64;
        let pos_table_ptr = position_table_buf.as_ptr() as u64;

        // Pre-collect chunk pointers and offsets for all N tokens.
        // seq.chunks was just filled by the append_token loop above.
        // chunk slot for token t = seq.chunks[t / CHUNK_TOKENS]
        // offset within chunk = t % CHUNK_TOKENS
        let chunk_size = CHUNK_TOKENS;

        // Embedding placeholder index (first instruction) — updated per token inline.
        let embedding_inst_idx = 0; // placeholder; actual index tracked per token below

        // We track per-layer kv_write_indices across all N tokens.
        // kv_write_indices_out[layer][t * nkh + h] = (k_idx, v_idx).
        // But the existing structure is Vec<Vec<(usize, usize)>> per layer.
        // We initialise empty per-layer vecs here and fill them per token.
        let num_attn_layers = cfg
            .layers
            .iter()
            .filter(|l| l.layer_type == crate::model::LayerType::Attention)
            .count();
        // kv_write_indices_out will be filled with one entry per attention layer
        // (each entry = Vec of (k_idx, v_idx) pairs for ALL tokens × ALL heads,
        // ordered token-major: [t=0 h=0, t=0 h=1, ..., t=N-1 h=nkh-1]).
        // This is a flat vector; update_step_paged is NOT called on prefill programs,
        // so the kv_write_indices here are for documentation only (not patched at runtime).
        // Initialize empty per-layer vecs.
        for _ in 0..num_attn_layers {
            kv_write_indices_out.push(Vec::new());
        }

        // Per-token loop: emit instructions for each token sequentially.
        // Each token t: embed → QKV proj → paged KV write → mRoPE → ATTN_PAGED → ...
        let mut embedding_inst_idx_first = instructions.len();
        let mut layer_first_attn_layer_kv: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_attn_layers];

        for t in 0..n {
            let pos = start_pos + t as u32;
            let token_id = tokens[t];
            let seq_len_at_t = pos + 1; // tokens 0..t+1 visible to token t

            // Embedding for token t → act.hidden (single token)
            if t == 0 { embedding_inst_idx_first = instructions.len(); }
            instructions.push(EmbeddingInst::new(
                div_ceil(hs as u32, 256),
                act.hidden.as_write_ptr(),
                model.embed_weight.as_ptr(),
                token_id as i32,
                hs as i32,
            ).into_inst());

            // Position ID for this token (write into position_ids — host-mapped)
            // The actual host-mapped write is done above (position_table_buf).
            // For mRoPE, prefill_bufs.position_ids is used.
            // Set up position for this token in prefill_bufs.position_ids.
            // We can't do this at compile time (it's a host-mapped write).
            // Instead we write position_ids at execute time via the host-mapped position_ids.
            // For compile_prefill_paged we bake position in the MropeInst's position_ids pointer.

            let mut attn_layer_t = 0usize;
            for layer_i in 0..cfg.num_layers {
                use crate::model::LayerType;
                match cfg.layers[layer_i].layer_type {
                    LayerType::Attention => {
                        let layer_weights = match &model.layers[layer_i] {
                            crate::weights::LayerWeights::Attention(w) => w,
                            _ => panic!("expected attention layer"),
                        };

                        // Compute chunk slot and offset for token t in this layer's paged KV.
                        // layout: [layer0_K[nkh, chunk_tokens, hd], layer0_V[...], layer1_K, ...]
                        let chunk_idx_for_t = t / chunk_size;
                        let offset_in_chunk = t % chunk_size;
                        let chunk_slot = seq.chunks[chunk_idx_for_t].slot_index();
                        let chunk_base = allocator.slot_ptr(chunk_slot) as u64;
                        let layer_k_offset = (attn_layer_t * 2 * chunk_size * kv_stride * std::mem::size_of::<f32>()) as u64;
                        let layer_v_offset = layer_k_offset + (chunk_size * kv_stride * std::mem::size_of::<f32>()) as u64;

                        // Attention layer: RMSNorm + QKV proj + KV write + mRoPE + ATTN_PAGED + output
                        let nqh = cfg.num_q_heads;
                        let rd = cfg.rope_dim;

                        // RMSNorm
                        instructions.push(RmsNormInst::new(
                            rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
                            act.normed.as_write_ptr(), act.hidden.as_write_ptr(),
                            layer_weights.input_norm.as_ptr(), hs as i32, eps,
                        ).into_inst());

                        // QKV projections
                        let q_mult = if cfg.has_output_gate { 2 } else { 1 };
                        emit_batched_linear_proj(&layer_weights.w_q_gate, act.q_gate_attn.as_write_ptr(), act.normed.as_ptr(), nqh * hd * q_mult, hs, 1, &mut instructions);
                        emit_batched_linear_proj(&layer_weights.w_k, act.k_attn.as_write_ptr(), act.normed.as_ptr(), nkh * hd, hs, 1, &mut instructions);
                        emit_batched_linear_proj(&layer_weights.w_v, act.v_attn.as_write_ptr(), act.normed.as_ptr(), nkh * hd, hs, 1, &mut instructions);

                        // Deinterleave Q+gate → q_attn, gate_attn
                        if !cfg.has_output_gate {
                            let total = nqh * hd;
                            instructions.push(D2dCopyInst::new(div_ceil(total as u32, 256), act.q_attn.as_write_ptr(), act.q_gate_attn.as_ptr(), total as i32).into_inst());
                        } else {
                            instructions.push(DeinterleaveInst::new(div_ceil((nqh * hd) as u32, 256), act.q_attn.as_write_ptr(), act.gate_attn.as_write_ptr(), act.q_gate_attn.as_ptr(), nqh as i32, hd as i32, 1).into_inst());
                        }

                        // KV write into paged chunk slot (pre-mRoPE, same as decode path)
                        let chunk_head_stride = chunk_size * hd;
                        let mut head_kv_pairs: Vec<(usize, usize)> = Vec::new();
                        for h in 0..nkh {
                            let head_byte_off = (h * chunk_head_stride + offset_in_chunk * hd) * std::mem::size_of::<f32>();
                            let k_dst = (chunk_base + layer_k_offset + head_byte_off as u64) as *mut f32;
                            let v_dst = (chunk_base + layer_v_offset + head_byte_off as u64) as *mut f32;
                            let k_src = unsafe { act.k_attn.as_ptr().add(h * hd) };
                            let v_src = unsafe { act.v_attn.as_ptr().add(h * hd) };
                            let k_idx = instructions.len();
                            instructions.push(D2dCopyInst::new(div_ceil(hd as u32, 256), k_dst, k_src, hd as i32).into_inst());
                            let v_idx = instructions.len();
                            instructions.push(D2dCopyInst::new(div_ceil(hd as u32, 256), v_dst, v_src, hd as i32).into_inst());
                            head_kv_pairs.push((k_idx, v_idx));
                        }
                        layer_first_attn_layer_kv[attn_layer_t].extend(head_kv_pairs);

                        // QK-norm (post-write, pre-mRoPE)
                        if cfg.has_qk_norm {
                            instructions.push(QkNormInst::new(
                                (nqh + nkh) as u32,
                                act.q_attn.as_write_ptr(), act.k_attn.as_write_ptr(),
                                layer_weights.q_norm.as_ptr(), layer_weights.k_norm.as_ptr(),
                                nqh as i32, nkh as i32, hd as i32, eps, 0,
                            ).into_inst());
                        }

                        // mRoPE (single token, uses act.position_ids which is set per-step by set_position)
                        instructions.push(MropeInst::new(
                            (nqh + nkh) as u32,
                            act.q_attn.as_write_ptr(), act.k_attn.as_write_ptr(),
                            act.inv_freq.as_ptr(), act.position_ids.as_ptr(),
                            nqh as i32, nkh as i32, hd as i32, rd as i32,
                            cfg.mrope_sections()[0] as i32, cfg.mrope_sections()[1] as i32, cfg.mrope_sections()[2] as i32, 1,
                        ).into_inst());

                        // OP_ATTN_PAGED_Q (disabled: grid_x=0; quantized KV not used during prefill)
                        let quant_idx = instructions.len();
                        if t == 0 { attn_quant_inst_indices_out.push(quant_idx); }
                        {
                            use crate::paged_kv::quantized_kv_offsets;
                            let chunk_tokens = CHUNK_TOKENS;
                            let (q1d, q1s, rd_off, rs) = quantized_kv_offsets(cfg, chunk_tokens, attn_layer_t, false);
                            let k_norm_ptr = if cfg.has_qk_norm { layer_weights.k_norm.as_ptr() } else { std::ptr::null() };
                            instructions.push(AttnPagedQInst::new(
                                act.q_attn.as_ptr(), act.inv_freq.as_ptr(),
                                nqh as i32, nkh as i32, hd as i32, chunk_tokens as i32, rd as i32,
                                q1d as u64, q1s as u64, rd_off as u64, rs as u64, k_norm_ptr,
                            ).into_inst());
                        }

                        // OP_ATTN_PAGED: attend over all visible paged chunks for token t
                        // page_table_ptr points to the host-mapped buffer (written above).
                        // seq_len = t+1, but we use offset within active chunk.
                        let paged_idx = instructions.len();
                        if t == 0 { attn_paged_inst_indices_out.push(paged_idx); }
                        {
                            let k_norm_ptr = if cfg.has_qk_norm { layer_weights.k_norm.as_ptr() } else { std::ptr::null() };
                            // Point into paged KV offsets for this layer.
                            let layer_k_off = (attn_layer_t * 2 * chunk_size * kv_stride * std::mem::size_of::<f32>()) as u64;
                            let layer_v_off = layer_k_off + (chunk_size * kv_stride * std::mem::size_of::<f32>()) as u64;
                            let mrope = cfg.mrope_sections();
                            instructions.push(AttnPagedInst::new(
                                nqh as u32,
                                act.attn_out.as_write_ptr(),
                                act.q_attn.as_ptr(),
                                act.inv_freq.as_ptr(),
                                nqh as i32, nkh as i32, hd as i32,
                                seq_len_at_t as i32,
                                CHUNK_TOKENS as i32,
                                rd as i32,
                                layer_k_off,
                                layer_v_off,
                                k_norm_ptr,
                                eps,
                                mrope[0] as i32,
                                mrope[1] as i32,
                            ).into_inst());
                            // Patch page_table and pos_table pointers into the instruction
                            let last_idx = instructions.len() - 1;
                            unsafe {
                                let inst = instructions[last_idx].words.as_mut_ptr() as *mut AttnPagedInst;
                                (*inst).page_table = page_table_ptr;
                                (*inst).pos_table = pos_table_ptr;
                                (*inst).partial_state = 0;
                            }
                        }

                        // Output gate + O-proj + residual (single-token decode path)
                        let final_attn_ptr = if cfg.has_output_gate {
                            let gate_size = nqh * hd;
                            instructions.push(OutputGateInst::new(
                                div_ceil(gate_size as u32, 256),
                                act.gated_out.as_write_ptr(), act.attn_out.as_ptr(), act.gate_attn.as_ptr(), gate_size as i32,
                            ).into_inst());
                            act.gated_out.as_write_ptr()
                        } else {
                            act.attn_out.as_write_ptr()
                        };
                        emit_batched_linear_proj(&layer_weights.w_o, act.out_proj.as_write_ptr(), final_attn_ptr, hs, nqh * hd, 1, &mut instructions);
                        // Two-step residual: hidden → residual, then residual + out_proj → hidden
                        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.residual.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
                        instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), act.out_proj.as_ptr(), act.residual.as_ptr(), hs as i32).into_inst());

                        attn_layer_t += 1;
                    }
                    LayerType::Gdn | LayerType::Mamba2 | LayerType::MoeFfn | LayerType::LfmConv => {
                        // Non-attention layers are not supported in compile_prefill_paged
                        // (this path is for pure attention models like Qwen3.5).
                        // If a model has GDN/Mamba2/MoE layers, use compile_prefill instead.
                        // TODO(braidinfer-8gz follow-up): add GDN/Mamba2 support here.
                    }
                }

                // Dense FFN (for attention layers that have dense FFN)
                match &cfg.layers[layer_i].ffn_type {
                    crate::model::FfnType::Dense => {
                        Self::compile_ffn(cfg, &model.layers[layer_i], act, &mut instructions);
                    }
                    crate::model::FfnType::None => {}
                    crate::model::FfnType::MoE { .. } => {
                        // MoE FFN not supported in compile_prefill_paged
                    }
                }
            }
        }

        // Copy kv_write_indices_out from per-layer tracking
        for (layer_i, kv_pairs) in layer_first_attn_layer_kv.into_iter().enumerate() {
            kv_write_indices_out[layer_i] = kv_pairs;
        }

        // Final RMSNorm + LM head (on the last token's hidden state, now in act.hidden)
        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.normed.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
        instructions.push(RmsNormInst::new(
            rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
            act.hidden.as_write_ptr(), act.normed.as_ptr(),
            model.final_norm_weight.as_ptr(), hs as i32, eps,
        ).into_inst());
        {
            let lm_w_ptr = if cfg.tie_word_embeddings {
                model.embed_weight.as_ptr() as *const u8
            } else {
                model.lm_head_weight.as_ptr() as *const u8
            };
            instructions.push(LinearProjInst::new(
                OP_LM_HEAD, cfg.vocab_size as u32,
                act.logits.as_write_ptr(), lm_w_ptr, act.hidden.as_ptr(),
                cfg.vocab_size as i32, hs as i32, 0,
            ).into_inst());
        }
        instructions.push(Instruction::new(OP_HALT, 0));

        let device_program = upload_program(device, &instructions)?;
        let flat_program: Vec<u64> = instructions.iter().flat_map(|i| i.words).collect();

        // Suppress unused variable warning
        let _ = prefill_bufs;
        let _ = embedding_inst_idx_first;
        let _ = embedding_inst_idx;

        let watchdog = WatchdogThread::spawn();
        let wd_state_dev = watchdog.register(device)?;
        let wd_dev_ptr = wd_state_dev as *mut std::ffi::c_void;

        Ok(MegakernelProgram {
            instructions,
            device_program,
            module: Arc::new(module),
            num_blocks,
            shared_mem,
            device,
            embedding_inst_idx: embedding_inst_idx_first,
            _mrope_inst_indices: Vec::new(),
            gqa_attn_inst_indices: Vec::new(),
            position_ids_dev_ptr: act.position_ids.as_ptr() as u64,
            kv: super::KvConfig {
                max_seq_len: cfg.max_seq_len as u32,
                num_kv_heads: nkh,
                head_dim: hd,
                kv_write_indices: Vec::new(), // one-shot program; no per-step patching
                kv_base_ptrs: Vec::new(),
            },
            paged: true,
            paged_kv: Some(super::PagedKvState {
                page_table: None,    // caller owns these; not managed by the program
                position_table: None,
                attn_paged_inst_indices: attn_paged_inst_indices_out.clone(),
                attn_quant_inst_indices: attn_quant_inst_indices_out.clone(),
                last_page_table_len: seq.chunks.len(),
                kv_stride_paged: kv_stride,
            }),
            quantized_kv: false,
            quant_kv: None,
            prefill_cache: None,
            dump_buffer: None,
            dump_counter: None,
            dump_capacity: 0,
            barrier_layer_map: Vec::new(),
            multi_gpu_attn_boundaries: Vec::new(),
            flat_program,
            _watchdog: watchdog,
            wd_dev_ptr,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Compile a prefill segment covering layers [layer_start, layer_end).
    /// Used by prefill_mixed_chunk to batch non-MoE layer spans.
    /// Does NOT emit embedding instructions (caller sets up prefill_bufs.hidden before calling).
    /// If `is_last_segment` is true, appends final norm + LM head (writing to act.hidden/logits).
    /// Otherwise just ends with HALT leaving prefill_bufs.hidden updated.
    pub fn compile_prefill_segment(
        model: &Model,
        tokens: &[u32],
        start_pos: u32,
        layer_start: usize,
        layer_end: usize,
        is_last_segment: bool,
        prefill_bufs: &mut PrefillBuffers,
    ) -> HipResult<Self> {
        let module = Arc::new(Module::load(
            model.device,
            &crate::kernel::kernel_dir().join("megakernel.hsaco"),
        )?);
        Self::compile_prefill_segment_with_module(
            model, module, tokens, start_pos, layer_start, layer_end,
            is_last_segment, prefill_bufs,
        )
    }

    pub fn compile_prefill_segment_with_module(
        model: &Model,
        module: Arc<Module>,
        tokens: &[u32],
        start_pos: u32,
        layer_start: usize,
        layer_end: usize,
        is_last_segment: bool,
        prefill_bufs: &mut PrefillBuffers,
    ) -> HipResult<Self> {
        let n = tokens.len();
        assert!(n > 0 && n <= CHUNK_TOKENS);
        assert!(layer_start <= layer_end);
        assert!(layer_end <= model.config.num_layers);
        let cfg = &model.config;
        let device = model.device;
        let act = &model.activations;
        let shared_mem = (256u32 * 4 * 2)
            .max((cfg.hidden_size as u32) * 4)
            .max(31776u32);
        let func = module.get_function("megakernel_f32")?;
        let blocks_per_sm = func.max_active_blocks_per_sm(256, shared_mem as usize)?;
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
        let eps = cfg.rms_norm_eps;

        // Position IDs are written before every execute() in state.rs (cached
        // programs share the buffer); see PrefillBuffers::write_positions.

        // Count GDN/KV/Mamba2 indices up to layer_start
        let mut gdn_idx = 0usize;
        let mut mamba2_idx = 0usize;
        let mut kv_idx = 0usize;
        for i in 0..layer_start {
            use crate::model::LayerType;
            match cfg.layers[i].layer_type {
                LayerType::Gdn => gdn_idx += 1,
                LayerType::Mamba2 => mamba2_idx += 1,
                LayerType::Attention => kv_idx += 1,
                _ => {}
            }
        }

        let mut prefill_kv_entries: Vec<super::PrefillKvEntry> = Vec::new();
        let mut prefill_attn_inst_indices: Vec<usize> = Vec::new();
        let mut prefill_kv_base_ptrs: Vec<(u64, u64)> = Vec::new();
        let mut _attn_layer_count = 0usize;

        for layer_i in layer_start..layer_end {
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

                prefill_kv_base_ptrs.push((kv_cache.k.as_ptr() as u64, kv_cache.v.as_ptr() as u64));
                let layer_kv_idx = prefill_kv_base_ptrs.len() - 1;

                let attn_start = instructions.len();
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
                    kv_idx == 0 && super::k_trace_5ax_enabled(),
                );

                let nkh = cfg.num_kv_heads;
                // Start scan AFTER OP_MROPE: pre-MROPE D2D_COPYs (q_gate→q_attn deinterleave
                // for !has_output_gate, plus 5ax K-trace snapshots) are NOT KV writes.
                let kv_scan_start = (attn_start..instructions.len())
                    .find(|&idx| instructions[idx].words[0] as u32 == OP_MROPE)
                    .map(|i| i + 1)
                    .unwrap_or(attn_start);
                let mut kv_pair_count = 0usize;
                for idx in kv_scan_start..instructions.len() {
                    let opcode = instructions[idx].words[0] as u32;
                    if opcode == OP_D2D_COPY && kv_pair_count < n * nkh * 2 {
                        let pair_flat = kv_pair_count / 2;
                        let is_v = kv_pair_count % 2 == 1;
                        if !is_v {
                            let t = pair_flat / nkh;
                            let h = pair_flat % nkh;
                            let v_idx = idx + 1;
                            prefill_kv_entries.push(super::PrefillKvEntry {
                                k_inst_idx: idx,
                                v_inst_idx: v_idx,
                                t, h, layer_kv_idx,
                            });
                        }
                        kv_pair_count += 1;
                    } else if opcode == OP_ATTN_PREFILL {
                        prefill_attn_inst_indices.push(idx);
                    }
                }

                Self::compile_ffn_batched(cfg, layer_i, &model.layers[layer_i], prefill_bufs, n, &mut instructions);
                _attn_layer_count += 1;
                kv_idx += 1;
            } else if cfg.layers[layer_i].layer_type == LayerType::Gdn {
                let w = match &model.layers[layer_i] {
                    LayerWeights::Gdn(w) => w,
                    _ => panic!("expected GDN layer"),
                };
                let conv_state = &model.gdn_conv_states[gdn_idx];
                let gdn_state = &model.gdn_states[gdn_idx];

                // Batched projections
                instructions.push(RmsNormInst::new(
                    rmsnorm_opcode(cfg.rms_norm_one_plus_w),
                    n as u32,
                    prefill_bufs.normed.as_write_ptr(),
                    prefill_bufs.hidden.as_ptr(),
                    w.input_norm.as_ptr(),
                    hs as i32,
                    eps,
                ).into_inst());

                emit_batched_linear_proj(&w.w_qkv, prefill_bufs.qkv.as_write_ptr(), prefill_bufs.normed.as_ptr(), conv_dim, hs, n, &mut instructions);
                emit_batched_linear_proj(&w.w_a, prefill_bufs.a_proj.as_write_ptr(), prefill_bufs.normed.as_ptr(), nvh_gdn, hs, n, &mut instructions);
                emit_batched_linear_proj(&w.w_b, prefill_bufs.b_proj.as_write_ptr(), prefill_bufs.normed.as_ptr(), nvh_gdn, hs, n, &mut instructions);
                emit_batched_linear_proj(&w.w_z, prefill_bufs.z_proj.as_write_ptr(), prefill_bufs.normed.as_ptr(), nvh_gdn * vd, hs, n, &mut instructions);

                let q_dim = nh_gdn * kd;
                let k_dim = nh_gdn * kd;
                let v_dim = nvh_gdn * vd;

                for t in 0..n {
                    instructions.push(Conv1dInst::new(div_ceil(q_dim as u32, 256), conv_state.as_write_ptr(), unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim) }, w.conv1d_weight_q.as_ptr(), act.q_gdn.as_write_ptr(), q_dim as i32, ck as i32).into_inst());
                    instructions.push(Conv1dInst::new(div_ceil(k_dim as u32, 256), unsafe { conv_state.as_write_ptr().add(q_dim * (ck - 1)) }, unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim + q_dim) }, w.conv1d_weight_k.as_ptr(), act.k_gdn.as_write_ptr(), k_dim as i32, ck as i32).into_inst());
                    instructions.push(Conv1dInst::new(div_ceil(v_dim as u32, 256), unsafe { conv_state.as_write_ptr().add((q_dim + k_dim) * (ck - 1)) }, unsafe { prefill_bufs.qkv.as_ptr().add(t * conv_dim + q_dim + k_dim) }, w.conv1d_weight_v.as_ptr(), act.v_gdn.as_write_ptr(), v_dim as i32, ck as i32).into_inst());
                    {
                        let gqa_group = nvh_gdn / nh_gdn;
                        let blocks_per_head = (num_blocks / nvh_gdn as u32).max(1);
                        instructions.push(GdnGateInst::new(div_ceil(nvh_gdn as u32, 256), act.gate_gdn.as_write_ptr(), unsafe { prefill_bufs.a_proj.as_ptr().add(t * nvh_gdn) }, w.a_log.as_ptr(), w.dt_bias.as_ptr(), nvh_gdn as i32).into_inst());
                        instructions.push(GdnRecurInst::new(nvh_gdn as u32 * blocks_per_head, nvh_gdn as u32, act.q_gdn.as_ptr(), act.k_gdn.as_ptr(), act.v_gdn.as_ptr(), act.gate_gdn.as_ptr(), unsafe { prefill_bufs.b_proj.as_ptr().add(t * nvh_gdn) }, gdn_state.recurrent.as_write_ptr(), act.recurrent_out.as_write_ptr(), kd as i32, vd as i32, gqa_group as i32).into_inst());
                    }
                    instructions.push(RmsNormGateInst::new(nvh_gdn as u32, act.normed_gated.as_write_ptr(), act.recurrent_out.as_ptr(), unsafe { prefill_bufs.z_proj.as_ptr().add(t * nvh_gdn * vd) }, w.output_norm.as_ptr(), nvh_gdn as i32, vd as i32, eps).into_inst());
                    {
                        let (lp_op, lp_w) = linear_proj_opcode_ptr(&w.w_out);
                        instructions.push(LinearProjInst::new(lp_op, hs as u32, act.out_proj.as_write_ptr(), lp_w, act.normed_gated.as_ptr(), hs as i32, (nvh_gdn * vd) as i32, 0).into_inst());
                    }
                    let hidden_t = unsafe { prefill_bufs.hidden.as_write_ptr().add(t * hs) };
                    instructions.push(ResidualAddInst::new(div_ceil(hs as u32, 256), hidden_t, act.out_proj.as_ptr(), hidden_t, hs as i32).into_inst());
                }

                Self::compile_ffn_batched(cfg, layer_i, &model.layers[layer_i], prefill_bufs, n, &mut instructions);
                gdn_idx += 1;
            } else if cfg.layers[layer_i].layer_type == LayerType::Mamba2 {
                let state = &model.mamba2_states[mamba2_idx];
                // Sequential per-token Mamba2: for t=0..n, D2D(hidden[t]→act.hidden),
                // run Mamba2 step (updates ssm/conv state in place), D2D(act.hidden→hidden[t]).
                for t in 0..n {
                    let hidden_t = unsafe { prefill_bufs.hidden.as_ptr().add(t * hs) };
                    let hidden_t_w = unsafe { prefill_bufs.hidden.as_write_ptr().add(t * hs) };
                    instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), hidden_t, hs as i32).into_inst());
                    Self::compile_mamba2_layer(cfg, &model.layers[layer_i], act, state, &mut instructions);
                    instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), hidden_t_w, act.hidden.as_ptr(), hs as i32).into_inst());
                }
                mamba2_idx += 1;
                // compile_ffn_batched skips Mamba2 layers (FfnType::None), so no call needed.
            } else if cfg.layers[layer_i].layer_type == LayerType::MoeFfn {
                // MoeFfn layers are handled by the CPU path in prefill_mixed_chunk — skip here.
            }
        }

        if is_last_segment {
            // Final norm + LM head (last token only)
            instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), unsafe { prefill_bufs.hidden.as_ptr().add((n - 1) * hs) }, hs as i32).into_inst());
            instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.normed.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
            instructions.push(RmsNormInst::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1, act.hidden.as_write_ptr(), act.normed.as_ptr(), model.final_norm_weight.as_ptr(), hs as i32, eps).into_inst());
            {
                let lm_w_ptr = if cfg.tie_word_embeddings { model.embed_weight.as_ptr() } else { model.lm_head_weight.as_ptr() };
                instructions.push(LinearProjInst::new(OP_LM_HEAD, cfg.vocab_size as u32, act.logits.as_write_ptr(), lm_w_ptr as *const u8, act.hidden.as_ptr(), cfg.vocab_size as i32, hs as i32, 0).into_inst());
            }
        }
        instructions.push(Instruction::new(OP_HALT, 0));

        // Diagnostic: BRAIDINFER_DUMP_PROGRAM=1 prints opcode list per compiled segment.
        if std::env::var("BRAIDINFER_DUMP_PROGRAM").is_ok() {
            eprintln!(
                "[dump] compile_prefill_segment: layers {layer_start}..{layer_end} n={n} start_pos={start_pos} num_inst={}",
                instructions.len()
            );
            for (pc, inst) in instructions.iter().enumerate() {
                let op = inst.words[0] as u32;
                let grid_x = (inst.words[0] >> 32) as u32;
                eprintln!("  [{pc:4}] {} grid_x={grid_x}", super::opcode_name_str(op));
            }
        }

        let device_program = upload_program(device, &instructions)?;
        let flat_program: Vec<u64> = instructions.iter().flat_map(|i| i.words).collect();

        let watchdog = WatchdogThread::spawn();
        let wd_state_dev = watchdog.register(device)?;
        let wd_dev_ptr = wd_state_dev as *mut std::ffi::c_void;

        Ok(MegakernelProgram {
            instructions,
            device_program,
            module,
            num_blocks,
            shared_mem,
            device,
            embedding_inst_idx: 0,
            _mrope_inst_indices: Vec::new(),
            gqa_attn_inst_indices: Vec::new(),
            position_ids_dev_ptr: prefill_bufs.position_ids.as_ptr() as u64,
            kv: super::KvConfig {
                max_seq_len: cfg.max_seq_len as u32,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                kv_write_indices: Vec::new(),
                kv_base_ptrs: prefill_kv_base_ptrs,
            },
            paged: false,
            paged_kv: None,
            quantized_kv: false,
            quant_kv: None,
            prefill_cache: Some(super::PrefillCacheState {
                embedding_start: 0,
                kv_entries: prefill_kv_entries,
                attn_inst_indices: prefill_attn_inst_indices,
                n,
            }),
            dump_buffer: None,
            dump_counter: None,
            dump_capacity: 0,
            barrier_layer_map: Vec::new(),
            multi_gpu_attn_boundaries: Vec::new(),
            flat_program,
            _watchdog: watchdog,
            wd_dev_ptr,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Compile a tiny program that applies final RMSNorm + LM head to the last token
    /// in prefill_bufs.hidden and writes logits to act.logits.
    /// Used when the model's last layer is a standalone MoeFfn (no dense is_last span).
    pub(crate) fn compile_final_norm_lm_head(
        model: &Model,
        module: Arc<Module>,
        prefill_bufs: &PrefillBuffers,
        n: usize,
    ) -> HipResult<Self> {
        let cfg = &model.config;
        let act = &model.activations;
        let hs = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let vs = cfg.vocab_size;
        let grid_x = div_ceil(hs as u32, 256);
        let device = model.device;

        let shared_mem = (256u32 * 4 * 2).max(hs as u32 * 4).max(super::SHARED_LPROJ_TOTAL);
        let func = module.get_function("megakernel_f32")?;
        let blocks_per_sm = func.max_active_blocks_per_sm(256, shared_mem as usize)?;
        let num_blocks = blocks_per_sm.max(1) as u32 * NUM_CUS;

        let mut instructions: Vec<Instruction> = Vec::new();
        instructions.push(D2dCopyInst::new(grid_x, act.hidden.as_write_ptr(),
            unsafe { prefill_bufs.hidden.as_ptr().add((n - 1) * hs) }, hs as i32).into_inst());
        instructions.push(D2dCopyInst::new(grid_x, act.normed.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
        instructions.push(RmsNormInst::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
            act.hidden.as_write_ptr(), act.normed.as_ptr(),
            model.final_norm_weight.as_ptr(), hs as i32, eps).into_inst());
        let lm_w_ptr = if cfg.tie_word_embeddings {
            model.embed_weight.as_ptr() as *const u8
        } else {
            model.lm_head_weight.as_ptr() as *const u8
        };
        instructions.push(LinearProjInst::new(OP_LM_HEAD, vs as u32,
            act.logits.as_write_ptr(), lm_w_ptr, act.hidden.as_ptr(),
            vs as i32, hs as i32, 0).into_inst());
        instructions.push(Instruction::new(OP_HALT, 0));

        let device_program = upload_program(device, &instructions)?;
        let flat_program: Vec<u64> = instructions.iter().flat_map(|i| i.words).collect();

        let watchdog = WatchdogThread::spawn();
        let wd_state_dev = watchdog.register(device)?;
        let wd_dev_ptr = wd_state_dev as *mut std::ffi::c_void;

        Ok(MegakernelProgram {
            instructions,
            device_program,
            module,
            num_blocks,
            shared_mem,
            device,
            embedding_inst_idx: 0,
            _mrope_inst_indices: Vec::new(),
            gqa_attn_inst_indices: Vec::new(),
            position_ids_dev_ptr: 0,
            kv: super::KvConfig {
                max_seq_len: cfg.max_seq_len as u32,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                kv_write_indices: Vec::new(),
                kv_base_ptrs: Vec::new(),
            },
            paged: false,
            paged_kv: None,
            quantized_kv: false,
            quant_kv: None,
            prefill_cache: None,
            dump_buffer: None,
            dump_counter: None,
            dump_capacity: 0,
            barrier_layer_map: Vec::new(),
            multi_gpu_attn_boundaries: Vec::new(),
            flat_program,
            _watchdog: watchdog,
            wd_dev_ptr,
            _not_send: std::marker::PhantomData,
        })
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
                let prev2_opcode = prev2.words[0] as u32;
                let normed_stage_ptr = act.normed_stage.as_ptr() as u64;
                if prev2_opcode == OP_D2D_COPY && prev2.words[1] == normed_stage_ptr {
                    if let Some(ref fc1) = moe.fc1_latent_proj {
                        // Emit fc1: normed(hs) → moe_latent(gupd)
                        let (lp_op, lp_w) = linear_proj_opcode_ptr(fc1);
                        prog.instructions[barrier_idx - 2] = LinearProjInst::new(
                            lp_op, gupd as u32,
                            act.moe_latent.as_write_ptr(), lp_w, act.normed.as_ptr(),
                            gupd as i32, hs as i32, 0,
                        ).into_inst();
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

            prog.instructions[barrier_idx] = MoeDispatchInst {
                opcode_gridx: OP_MOE_DISPATCH as u64,
                work_queue: p2p.work_queue.device_ptr() as u64,
                output_slots: p2p.output_slots.as_ptr() as u64,
                final_output: final_output_ptr,
                expert_ids: act.moe_expert_ids.as_ptr() as u64,
                expert_weights: act.moe_expert_weights.as_ptr() as u64,
                seq_counter: p2p.seq_counter.device_ptr() as u64,
                num_workers_hs: ((num_workers as u64) << 32) | (hs as u64),
                layer_k: ((layer_idx as u64) << 32) | (k as u64),
                eis_gate: ((eis as u64) << 32) | has_gate,
                activation: activation_ptr,
                layer_config_ptrs: p2p.gpu0_layer_config_ptrs.as_ptr() as u64,
                scratch_gate: p2p.gpu0_scratch_gate.as_ptr() as u64,
                scratch_up: p2p.gpu0_scratch_up.as_ptr() as u64,
                scratch_act: p2p.gpu0_scratch_act.as_ptr() as u64,
                num_gpus: num_gpus as u64,
                gate_up_in_dim: gupd as u64,
                _pad: 0,
            }.into_inst();
        }

        // Pass 2: rebuild instruction stream to insert OP_MOE_DISPATCH_POST after every
        // OP_MOE_DISPATCH, plus fc2+shared_expert+residual_add after POST for Nemotron-H
        // layers with fc2_latent_proj.
        //
        // Splitting OP_MOE_DISPATCH (PRE: zero+GPU0 experts) and OP_MOE_DISPATCH_POST (sum)
        // allows the CPU to fire workers concurrently with GPU 0's PRE batch, then wait for
        // both before firing POST. Restores ~25% of decode throughput on multi-GPU MoE.
        // (epic braidinfer-0hu Phase 7)
        //
        // compile_moe_ffn_multi_gpu emits NO post-barrier instructions for fc2_latent_proj layers
        // (emit_post_barrier=false), so there are no stale instructions to skip — only insertions.
        // The inserted_before map tracks index shift due to inserted instructions for attn-boundary remap.
        {
            let mut new_instructions =
                Vec::with_capacity(prog.instructions.len() + 8 * barrier_map.len());
            // inserted_before[i] = number of instructions inserted before old index i
            let mut inserted_before: Vec<usize> = vec![0usize; prog.instructions.len() + 1];
            for (i, inst) in prog.instructions.iter().enumerate() {
                inserted_before[i] = new_instructions.len() - i;
                new_instructions.push(inst.clone());
                // After OP_MOE_DISPATCH: insert OP_MOE_DISPATCH_POST (sum), then for
                // Nemotron-H, fc2_latent_proj + shared_expert + residual_add.
                if let Some(&layer_idx) = barrier_map.get(&i) {
                    let moe = model.moe_weights[layer_idx].as_ref().unwrap();
                    let dist = model.distributed_moe[layer_idx].as_ref();
                    let gupd = dist.map(|d| d.gate_up_in_dim).unwrap_or(hs);
                    let num_workers = p2p.workers.len();
                    let num_gpus = p2p.num_gpus;
                    let (k, eis) = match &cfg.layers[layer_idx].ffn_type {
                        crate::model::FfnType::MoE {
                            num_active,
                            expert_intermediate_size,
                            ..
                        } => (*num_active, *expert_intermediate_size),
                        _ => unreachable!(),
                    };
                    let has_gate = if moe.has_gate_proj { 1u64 } else { 0u64 };
                    let final_output_ptr = if moe.fc2_latent_proj.is_some() {
                        act.moe_latent.as_ptr() as u64
                    } else {
                        act.ffn_down_stage.as_ptr() as u64
                    };
                    let activation_ptr = if moe.fc1_latent_proj.is_some() {
                        act.moe_latent.as_ptr() as u64
                    } else {
                        act.normed.as_ptr() as u64
                    };

                    // OP_MOE_DISPATCH_POST: sums output_slots[0..num_gpus * hs] into final_output[0..gupd].
                    // Reuses MoeDispatchInst layout — same fields as OP_MOE_DISPATCH for ABI consistency.
                    new_instructions.push(MoeDispatchInst {
                        opcode_gridx: OP_MOE_DISPATCH_POST as u64,
                        work_queue: p2p.work_queue.device_ptr() as u64,
                        output_slots: p2p.output_slots.as_ptr() as u64,
                        final_output: final_output_ptr,
                        expert_ids: act.moe_expert_ids.as_ptr() as u64,
                        expert_weights: act.moe_expert_weights.as_ptr() as u64,
                        seq_counter: p2p.seq_counter.device_ptr() as u64,
                        num_workers_hs: ((num_workers as u64) << 32) | (hs as u64),
                        layer_k: ((layer_idx as u64) << 32) | (k as u64),
                        eis_gate: ((eis as u64) << 32) | has_gate,
                        activation: activation_ptr,
                        layer_config_ptrs: p2p.gpu0_layer_config_ptrs.as_ptr() as u64,
                        scratch_gate: p2p.gpu0_scratch_gate.as_ptr() as u64,
                        scratch_up: p2p.gpu0_scratch_up.as_ptr() as u64,
                        scratch_act: p2p.gpu0_scratch_act.as_ptr() as u64,
                        num_gpus: num_gpus as u64,
                        gate_up_in_dim: gupd as u64,
                        _pad: 0,
                    }.into_inst());

                    // The rest of the post-MoE insertions only apply to Nemotron-H
                    // (fc2_latent_proj-bearing layers).
                    if let Some(ref fc2) = moe.fc2_latent_proj {
                        // fc2: moe_latent(gupd) → ffn_down_stage(hs)
                        {
                            let (lp_op, lp_w) = linear_proj_opcode_ptr(fc2);
                            new_instructions.push(LinearProjInst::new(
                                lp_op, hs as u32,
                                act.ffn_down_stage.as_write_ptr(), lp_w, act.moe_latent.as_ptr(),
                                hs as i32, gupd as i32, 0,
                            ).into_inst());
                        }

                        // Shared expert (relu² path for Nemotron-H, no gate_proj)
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
                                    let (lp_op, lp_w) = linear_proj_opcode_ptr(&se.up_proj);
                                    new_instructions.push(LinearProjInst::new(
                                        lp_op, se_is as u32,
                                        act.moe_expert_up.as_write_ptr(), lp_w, act.normed.as_ptr(),
                                        se_is as i32, hs as i32, 0,
                                    ).into_inst());
                                }
                                new_instructions.push(ReluSqInst::new(
                                    div_ceil(se_is as u32, 256),
                                    act.moe_expert_act.as_write_ptr(),
                                    act.moe_expert_up.as_ptr(),
                                    se_is as i32,
                                ).into_inst());
                                {
                                    let (lp_op, lp_w) = linear_proj_opcode_ptr(&se.down_proj);
                                    new_instructions.push(LinearProjInst::new(
                                        lp_op, hs as u32,
                                        act.moe_expert_out.as_write_ptr(), lp_w, act.moe_expert_act.as_ptr(),
                                        hs as i32, se_is as i32, 0,
                                    ).into_inst());
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
                            // Note: has_gate_proj shared expert handled by compile_moe_ffn_multi_gpu.
                        }

                        // residual_add: hidden = residual + ffn_down_stage (Nemotron-H)
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
            inserted_before[prog.instructions.len()] = new_instructions.len() - prog.instructions.len();
            prog.instructions = new_instructions;

            // Remap multi_gpu_attn_boundaries: each old index shifts by inserted_before[i].
            prog.multi_gpu_attn_boundaries = prog.multi_gpu_attn_boundaries.iter()
                .map(|&(flush, resume)| {
                    (flush + inserted_before[flush], resume + inserted_before[resume])
                }).collect();

            // Remap barrier_layer_map indices (now point at OP_MOE_DISPATCH).
            prog.barrier_layer_map = prog.barrier_layer_map.iter()
                .map(|&(idx, layer)| (idx + inserted_before[idx], layer))
                .collect();
        }

        // KEEP barrier_layer_map: in the unified-worker design it identifies
        // OP_MOE_DISPATCH instruction indices so decode_step_p2p can wrap each
        // MoE op with worker dispatch (OP_MOE_FFN_REMOTE → wait_ack on each
        // worker) before firing the GPU 0 batch containing op_moe_dispatch.

        Ok(prog)
    }
}
