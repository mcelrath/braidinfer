//! Megakernel program compilation: translates model config + weights into instruction streams.
//! Extracted from megakernel.rs for maintainability.

use braidinfer_hip::HipResult;
use braidinfer_hip::module::Module;
use std::sync::Arc;

use super::compile_common::{AttentionVariant, div_ceil, emit_batched_linear_proj, linear_proj_opcode_ptr, rmsnorm_opcode};

use super::instructions::*;
use super::{CHUNK_TOKENS, Instruction, MegakernelProgram, NUM_CUS, PrefillBuffers};
#[allow(unused_imports)]
use super::{
    OP_ATTN_PAGED, OP_ATTN_PAGED_Q, OP_BARRIER, OP_CONV1D, OP_D2D_COPY,
    OP_DEINTERLEAVE, OP_EMBEDDING, OP_FFN_DOWN_RES, OP_FFN_DOWN_RES_RNF4, OP_FFN_GATE_UP,
    OP_FFN_GATE_UP_RNF4, OP_GDN_GATE, OP_GDN_RECUR, OP_HALT, OP_KV_QUANTIZE,
    OP_LINEAR_PROJ, OP_LINEAR_PROJ_PCG32, OP_LINEAR_PROJ_RNF4, OP_LM_HEAD, OP_MAMBA2_CONV1D,
    OP_MAMBA2_NORM_GATED, OP_MOE_DISPATCH, OP_MOE_DISPATCH_POST, OP_MOE_FFN, OP_MOE_GATE, OP_MROPE, OP_OUTPUT_GATE,
    OP_QK_NORM, OP_RELU_SQ, OP_RESIDUAL_ADD, OP_RMSNORM, OP_RMSNORM_GATE, OP_RMSNORM_WX,
    OP_SIGMOID_WEIGHTED_ADD, OP_SILU_MUL, OP_SSM_UPDATE,
};
use crate::model::Model;
use crate::weights::LayerWeights;

impl MegakernelProgram {
    pub fn compile_paged(model: &Model) -> HipResult<Self> {
        Self::compile_inner(model, true, false)
    }

    /// Compile for GPU-native P2P MoE dispatch (OP_MOE_DISPATCH).
    /// MoE layers emit OP_MOE_DISPATCH — handled entirely inside the megakernel by op_moe_dispatch.
    /// No CPU involvement in the hot path; workers on GPUs 1-3 run moe_worker_kernel.
    pub fn compile_multi_gpu_p2p(
        model: &Model,
        p2p: &mut crate::moe_p2p::MoeP2pContext,
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
            .any(|l| matches!(l.ffn_type, crate::config::FfnType::MoE { .. }));
        // OP_MOE_GATE needs 1024 floats = 4KB. GDN recurrent needs 2KB.
        // OP_LINEAR_PROJ_PCG32/RNF4 tiled-LDS: (8+7680+256)*4 = 31776 bytes per block.
        // 2 blocks/CU: 2*31776 = 63552 < 65536 ✓ — no occupancy reduction.
        let base_shared = if has_moe { 1024u32 * 4 } else { 256u32 * 4 * 2 };
        let shared_mem = base_shared.max(31776u32);
        let func = module.get_function("persistent_worker")?;
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
        let mut trace_probe_map: Vec<(usize, crate::tracer::Probe)> = Vec::new();

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
        trace_probe_map.push((embedding_inst_idx, crate::tracer::Probe::Embed));

        // Layers
        let mut attn_paged_inst_indices = Vec::new();
        let mut attn_quant_inst_indices = Vec::new();
        let mut attn_layer_count = 0usize;

        let mut gdn_idx = 0usize;
        let mut mamba2_idx = 0usize;
        for layer_i in 0..cfg.num_layers {
            use crate::config::LayerType;
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
                            attn_layer_count,
                            &mut instructions,
                            &mut multi_gpu_attn_boundaries,
                        );
                    }
                    // PostMixer probe at the last instruction emitted by the attention block
                    trace_probe_map.push((instructions.len() - 1, crate::tracer::Probe::PostMixer { layer: layer_i }));
                    attn_layer_count += 1;
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
                    // PostMixer probe at the ScaleAdd (residual) that ends compile_gdn_layer
                    trace_probe_map.push((instructions.len() - 1, crate::tracer::Probe::PostMixer { layer: layer_i }));
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
                    // PostMixer probe at the ResidualAdd that ends compile_mamba2_layer
                    trace_probe_map.push((instructions.len() - 1, crate::tracer::Probe::PostMixer { layer: layer_i }));
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
                crate::config::FfnType::Dense => {
                    Self::compile_ffn(cfg, &model.layers[layer_i], act, &mut instructions);
                    // PostFfn probe at the last instruction emitted by compile_ffn
                    trace_probe_map.push((instructions.len() - 1, crate::tracer::Probe::PostFfn { layer: layer_i }));
                }
                crate::config::FfnType::MoE { .. } => {
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
                        if std::env::var("XL4O_DBG").is_ok() {
                            eprintln!("[xl4o-dbg] compile_inner(mg=false) L{layer_i} MoE single-arm: expert_gate_up.num_elements()={}", moe.expert_gate_up.num_elements());
                        }
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
                            // ov5m.1 Phase 1a: PostFfn probe at the MoE block's final op.
                            // compile_moe_ffn ends on OP_RESIDUAL_ADD -> act.hidden (compile_moe.rs:183,
                            // dump-eligible). This is the SINGLE-GPU (-g1) decode path (compile_paged ->
                            // compile_inner(paged, multi_gpu=false)); it had NO MoE probe (Dense FFN got
                            // one @ the line above; MoE didn't) -> the per-layer decode trace truncated.
                            trace_probe_map.push((instructions.len() - 1, crate::tracer::Probe::PostFfn { layer: layer_i }));
                        }
                    }
                }
                crate::config::FfnType::None => {
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
        trace_probe_map.push((instructions.len() - 1, crate::tracer::Probe::FinalNorm));

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

        let watchdog = model.watchdog.clone();
        let wd_state_dev = watchdog.register(device)?;

        Ok(MegakernelProgram {
            instructions,
            num_blocks,
            device,
            embedding_inst_idx,
            _mrope_inst_indices: mrope_inst_indices,
            gqa_attn_inst_indices,
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
                    page_table_dirty: false,
                    kv_stride_paged: cfg.num_kv_heads * cfg.head_dim,
                })
            } else {
                None
            },
            quantized_kv: false,
            quant_kv: None,
            dump_buffer: None,
            dump_counter: None,
            dump_capacity: 0,
            trace_probe_map,
            barrier_layer_map,
            multi_gpu_attn_boundaries,
            _watchdog: watchdog,
            _not_send: std::marker::PhantomData,
            moe_act_d2d_indices: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compile_prefill_paged_persistent(
        model: &Model,
        module: Arc<Module>,
        tokens: &[u32],
        start_pos: u32,
        seq: &mut crate::paged_kv::SequenceState,
        allocator: &mut crate::paged_kv::PageAllocator,
        page_table_buf: &braidinfer_hip::memory::MappedHostBuffer<u64>,
        position_table_buf: &braidinfer_hip::memory::MappedHostBuffer<i32>,
        prefill_bufs: &mut PrefillBuffers,
    ) -> HipResult<Self> {
        let n = tokens.len();
        assert!(n > 0 && n <= CHUNK_TOKENS * 16, "prefill_paged_persistent: too many tokens");
        let cfg = &model.config;
        let device = model.device;
        let act = &model.activations;
        let shared_mem = (256u32 * 4 * 2)
            .max((cfg.hidden_size as u32) * 4)
            .max(31776u32);
        let func = module.get_function("persistent_worker")?;
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

        // 1. Pre-allocate N paged chunks (sub-memo Q8 invariant).
        // No host tier at compile time: host_alloc is threaded through at the
        // decode/prefill call sites that own the HostPageAllocator.  This compile
        // path does not have access to it; VRAM exhaustion returns OutOfMemory.
        for i in 0..n {
            let pos = start_pos as i32 + i as i32;
            seq.append_token(pos, allocator, None)?;
        }

        // 2. Write host-mapped page_table_buf with chunk base pointers (NOT slot
        //    indices — the kernel dereferences `(char*)(uintptr_t)page_table[idx]`).
        {
            let host_pt = page_table_buf.host_ptr();
            for (i, chunk) in seq.chunks.iter().enumerate() {
                let addr = allocator.slot_ptr(chunk.slot_index()) as u64;
                unsafe { host_pt.add(i).write_volatile(addr); }
            }
        }
        // 3. Write position_table_buf with mRoPE 3-tuples (temporal/height/width;
        //    text-only models write all three equal to position).
        {
            let host_pos = position_table_buf.host_ptr();
            for i in 0..n {
                let pos = (start_pos as i32) + i as i32;
                unsafe {
                    let base = host_pos.add(i * 3);
                    base.add(0).write_volatile(pos);
                    base.add(1).write_volatile(pos);
                    base.add(2).write_volatile(pos);
                }
            }
        }

        let page_table_ptr_u64 = page_table_buf.as_ptr() as u64;
        let position_table_ptr_u64 = position_table_buf.as_ptr() as u64;

        // 4. Embedding (writes prefill_bufs.hidden from tokens).
        //    Per compile_prefill_segment_with_module pattern (lazy: caller
        //    has already filled prefill_bufs.hidden via the embedding op).
        //    For paged-persistent first pass we follow the same convention:
        //    the megakernel does not emit embeddings; caller is responsible
        //    for prefill_bufs.hidden initialization before execute().

        // 5. Per-layer emission.
        let mut gdn_idx = 0usize;
        let mut mamba2_idx = 0usize;
        let mut attn_layer_idx = 0usize;
        let mut attn_paged_inst_indices: Vec<usize> = Vec::new();
        let mut trace_probe_map_pp: Vec<(usize, crate::tracer::Probe)> = Vec::new();

        for layer_i in 0..cfg.num_layers {
            use crate::config::LayerType;
            match cfg.layers[layer_i].layer_type {
                LayerType::Attention => {
                    let w = match &model.layers[layer_i] {
                        LayerWeights::Attention(w) => w,
                        _ => panic!("expected attention layer at {layer_i}"),
                    };
                    Self::emit_attention_layer(
                        cfg,
                        w,
                        act,
                        Some((prefill_bufs, n)),
                        &AttentionVariant::PrefillPagedKv {
                            attn_layer_index: attn_layer_idx,
                            start_pos,
                            n,
                            page_table_ptr: page_table_ptr_u64,
                            position_table_ptr: position_table_ptr_u64,
                        },
                        &mut instructions,
                        &mut Vec::new(),
                        &mut Vec::new(),
                        &mut Vec::new(),
                        &mut Vec::new(),
                        &mut attn_paged_inst_indices,
                        &mut Vec::new(),
                    );
                    // PostMixer probe at last instruction of attention block.
                    trace_probe_map_pp.push((instructions.len() - 1, crate::tracer::Probe::PostMixer { layer: layer_i }));
                    Self::compile_ffn_batched(cfg, layer_i, &model.layers[layer_i], prefill_bufs, n, &mut instructions);
                    if matches!(cfg.layers[layer_i].ffn_type, crate::config::FfnType::Dense) {
                        trace_probe_map_pp.push((instructions.len() - 1, crate::tracer::Probe::PostFfn { layer: layer_i }));
                    }
                    attn_layer_idx += 1;
                }
                LayerType::Gdn => {
                    let w = match &model.layers[layer_i] {
                        LayerWeights::Gdn(w) => w,
                        _ => panic!("expected GDN layer at {layer_i}"),
                    };
                    let conv_state = &model.gdn_conv_states[gdn_idx];
                    let gdn_state = &model.gdn_states[gdn_idx];

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

                    // PostMixer probe at last token ResidualAdd (end of GDN mixer, before FFN).
                    trace_probe_map_pp.push((instructions.len() - 1, crate::tracer::Probe::PostMixer { layer: layer_i }));
                    Self::compile_ffn_batched(cfg, layer_i, &model.layers[layer_i], prefill_bufs, n, &mut instructions);
                    if matches!(cfg.layers[layer_i].ffn_type, crate::config::FfnType::Dense) {
                        trace_probe_map_pp.push((instructions.len() - 1, crate::tracer::Probe::PostFfn { layer: layer_i }));
                    }
                    gdn_idx += 1;
                }
                LayerType::Mamba2 => {
                    let state = &model.mamba2_states[mamba2_idx];
                    let mut mamba_out_idx = 0usize;
                    for t in 0..n {
                        let hidden_t = unsafe { prefill_bufs.hidden.as_ptr().add(t * hs) };
                        let hidden_t_w = unsafe { prefill_bufs.hidden.as_write_ptr().add(t * hs) };
                        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), hidden_t, hs as i32).into_inst());
                        Self::compile_mamba2_layer(cfg, &model.layers[layer_i], act, state, &mut instructions);
                        // ov5m.4: compile_mamba2_layer ends on OP_SCALE_ADD -> act.hidden (hs,
                        // dump-eligible per dump.h:57). Probe THAT (last token), NOT the wrapping
                        // D2D copy-back below — OP_D2D_COPY isn't dump-eligible (and a blanket
                        // dump.h add hangs the GPU reading bad D2D dsts), so these Mamba2 layers
                        // were silently absent (~half the nemotron prefill trace).
                        mamba_out_idx = instructions.len() - 1;
                        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), hidden_t_w, act.hidden.as_ptr(), hs as i32).into_inst());
                    }
                    trace_probe_map_pp.push((mamba_out_idx, crate::tracer::Probe::PostMixer { layer: layer_i }));
                    mamba2_idx += 1;
                }
                LayerType::MoeFfn => {
                    // Handled by CPU path in prefill_mixed_chunk — skip here.
                    // (Matches compile_prefill_segment_with_module:957-958.)
                }
                LayerType::LfmConv => {
                    panic!("LfmConv layers not yet implemented in megakernel (braidinfer-aes.4)");
                }
            }
        }

        // 6. Final RMSNorm + LM head on last token's hidden.
        instructions.push(D2dCopyInst::new(
            div_ceil(hs as u32, 256),
            act.hidden.as_write_ptr(),
            unsafe { prefill_bufs.hidden.as_ptr().add((n - 1) * hs) },
            hs as i32,
        ).into_inst());
        instructions.push(D2dCopyInst::new(
            div_ceil(hs as u32, 256),
            act.normed.as_write_ptr(),
            act.hidden.as_ptr(),
            hs as i32,
        ).into_inst());
        instructions.push(RmsNormInst::new(
            rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
            act.hidden.as_write_ptr(), act.normed.as_ptr(),
            model.final_norm_weight.as_ptr(), hs as i32, eps,
        ).into_inst());
        trace_probe_map_pp.push((instructions.len() - 1, crate::tracer::Probe::FinalNorm));
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

        let watchdog = model.watchdog.clone();
        let _wd_state_dev = watchdog.register(device)?;
        let nkh = cfg.num_kv_heads;
        let hd = cfg.head_dim;

        Ok(MegakernelProgram {
            instructions,
            num_blocks,
            device,
            embedding_inst_idx: 0,
            _mrope_inst_indices: Vec::new(),
            gqa_attn_inst_indices: Vec::new(),
            kv: super::KvConfig {
                max_seq_len: cfg.max_seq_len as u32,
                num_kv_heads: nkh,
                head_dim: hd,
                kv_write_indices: Vec::new(),
                kv_base_ptrs: Vec::new(),
            },
            paged: true,
            paged_kv: Some(super::PagedKvState {
                page_table: None,
                position_table: None,
                attn_paged_inst_indices,
                attn_quant_inst_indices: Vec::new(),
                last_page_table_len: seq.chunks.len(),
                page_table_dirty: false,
                kv_stride_paged: nkh * hd,
            }),
            quantized_kv: false,
            quant_kv: None,
            dump_buffer: None,
            dump_counter: None,
            dump_capacity: 0,
            trace_probe_map: trace_probe_map_pp,
            barrier_layer_map: Vec::new(),
            multi_gpu_attn_boundaries: Vec::new(),
            _watchdog: watchdog,
            _not_send: std::marker::PhantomData,
            moe_act_d2d_indices: Vec::new(),
        })
    }

    /// bd srg6.10: paged variant of `compile_prefill_segment_with_module`.
    ///
    /// Compiles a megakernel program for ONE segment of a mixed (MoE) prefill,
    /// covering layers `layer_start..layer_end`. Attention layers in the segment
    /// emit paged KV writes (`AttentionVariant::PrefillPagedKv`). Used by
    /// `prefill_mixed_chunk` for single-GPU MoE prefill (bd srg6.10) and (via
    /// broadcast) for multi-GPU MoE prefill (bd srg6.15).
    ///
    /// Differences from `compile_prefill_paged_persistent`:
    ///   - Layer range is `layer_start..layer_end` (not all layers).
    ///   - Caller (`prefill_mixed_chunk`) has ALREADY called
    ///     `seq.append_token(...)` for ALL `n` prompt tokens BEFORE invoking
    ///     this function. This function does NOT touch `seq`/`allocator`.
    ///   - Each call writes the FULL `seq.chunks` slice into `page_table_buf`
    ///     (covers the whole prefill range so attention reads see all history).
    ///   - `is_last_segment` controls emission of final RMSNorm + LM head.
    ///   - MoeFfn layers in-range are skipped (CPU dispatches between segments).
    ///   - Caller must NOT cache the returned program (page_table_buf contents
    ///     change between prefills); compile fresh per call.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_prefill_segment_paged(
        model: &Model,
        module: Arc<Module>,
        tokens: &[u32],
        start_pos: u32,
        layer_start: usize,
        layer_end: usize,
        is_last_segment: bool,
        seq: &crate::paged_kv::SequenceState,
        allocator: &crate::paged_kv::PageAllocator,
        page_table_buf: &braidinfer_hip::memory::MappedHostBuffer<u64>,
        position_table_buf: &braidinfer_hip::memory::MappedHostBuffer<i32>,
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
        let func = module.get_function("persistent_worker")?;
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

        // Seed per-layer counters by walking 0..layer_start (mirror flat variant
        // at lines 808-820). attn_layer_idx must be GLOBAL across the model — the
        // PrefillPagedKv arm bakes attn_layer_index into the K/V offset
        // (layer_k_off = attn_layer_index * 2 * chunk_tokens * kv_stride * sizeof(f32))
        // which addresses into a shared per-chunk multi-layer slab.
        let mut gdn_idx = 0usize;
        let mut mamba2_idx = 0usize;
        let mut attn_layer_idx = 0usize;
        for i in 0..layer_start {
            use crate::config::LayerType;
            match cfg.layers[i].layer_type {
                LayerType::Gdn => gdn_idx += 1,
                LayerType::Mamba2 => mamba2_idx += 1,
                LayerType::Attention => attn_layer_idx += 1,
                _ => {}
            }
        }

        // Write host-mapped page_table_buf with the FULL seq.chunks slice
        // (covers all prefill tokens so far — not just this segment's n).
        // Each prefill_mixed_chunk iteration calls this function with the same
        // seq.chunks (populated once by the outer driver before any segment).
        {
            let host_pt = page_table_buf.host_ptr();
            for (i, chunk) in seq.chunks.iter().enumerate() {
                let addr = allocator.slot_ptr(chunk.slot_index()) as u64;
                unsafe { host_pt.add(i).write_volatile(addr); }
            }
        }
        // Write position_table_buf with mRoPE 3-tuples for the FULL prompt
        // range tokens 0..(start_pos + n). Attention reads at position
        // start_pos + t needs all prior positions visible.
        {
            let host_pos = position_table_buf.host_ptr();
            let total = start_pos as usize + n;
            for i in 0..total {
                unsafe {
                    let base = host_pos.add(i * 3);
                    base.add(0).write_volatile(i as i32);
                    base.add(1).write_volatile(i as i32);
                    base.add(2).write_volatile(i as i32);
                }
            }
        }

        let page_table_ptr_u64 = page_table_buf.as_ptr() as u64;
        let position_table_ptr_u64 = position_table_buf.as_ptr() as u64;

        let mut attn_paged_inst_indices: Vec<usize> = Vec::new();
        let mut trace_probe_map_seg: Vec<(usize, crate::tracer::Probe)> = Vec::new();

        for layer_i in layer_start..layer_end {
            use crate::config::LayerType;
            match cfg.layers[layer_i].layer_type {
                LayerType::Attention => {
                    let w = match &model.layers[layer_i] {
                        LayerWeights::Attention(w) => w,
                        _ => panic!("expected attention layer at {layer_i}"),
                    };
                    Self::emit_attention_layer(
                        cfg,
                        w,
                        act,
                        Some((prefill_bufs, n)),
                        &AttentionVariant::PrefillPagedKv {
                            attn_layer_index: attn_layer_idx,
                            start_pos,
                            n,
                            page_table_ptr: page_table_ptr_u64,
                            position_table_ptr: position_table_ptr_u64,
                        },
                        &mut instructions,
                        &mut Vec::new(),
                        &mut Vec::new(),
                        &mut Vec::new(),
                        &mut Vec::new(),
                        &mut attn_paged_inst_indices,
                        &mut Vec::new(),
                    );
                    // PostMixer probe at last instruction of attention block.
                    trace_probe_map_seg.push((instructions.len() - 1, crate::tracer::Probe::PostMixer { layer: layer_i }));
                    Self::compile_ffn_batched(cfg, layer_i, &model.layers[layer_i], prefill_bufs, n, &mut instructions);
                    if matches!(cfg.layers[layer_i].ffn_type, crate::config::FfnType::Dense) {
                        trace_probe_map_seg.push((instructions.len() - 1, crate::tracer::Probe::PostFfn { layer: layer_i }));
                    }
                    attn_layer_idx += 1;
                }
                LayerType::Gdn => {
                    let w = match &model.layers[layer_i] {
                        LayerWeights::Gdn(w) => w,
                        _ => panic!("expected GDN layer at {layer_i}"),
                    };
                    let conv_state = &model.gdn_conv_states[gdn_idx];
                    let gdn_state = &model.gdn_states[gdn_idx];

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

                    // PostMixer probe at last token ResidualAdd (end of GDN mixer, before FFN).
                    trace_probe_map_seg.push((instructions.len() - 1, crate::tracer::Probe::PostMixer { layer: layer_i }));
                    Self::compile_ffn_batched(cfg, layer_i, &model.layers[layer_i], prefill_bufs, n, &mut instructions);
                    if matches!(cfg.layers[layer_i].ffn_type, crate::config::FfnType::Dense) {
                        trace_probe_map_seg.push((instructions.len() - 1, crate::tracer::Probe::PostFfn { layer: layer_i }));
                    }
                    gdn_idx += 1;
                }
                LayerType::Mamba2 => {
                    let state = &model.mamba2_states[mamba2_idx];
                    let mut mamba_out_idx = 0usize;
                    for t in 0..n {
                        let hidden_t = unsafe { prefill_bufs.hidden.as_ptr().add(t * hs) };
                        let hidden_t_w = unsafe { prefill_bufs.hidden.as_write_ptr().add(t * hs) };
                        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), act.hidden.as_write_ptr(), hidden_t, hs as i32).into_inst());
                        Self::compile_mamba2_layer(cfg, &model.layers[layer_i], act, state, &mut instructions);
                        // ov5m.4: probe the OP_SCALE_ADD that ends compile_mamba2_layer (act.hidden,
                        // hs, dump-eligible per dump.h:57), NOT the wrapping D2D copy-back below
                        // (OP_D2D_COPY isn't dump-eligible -> these Mamba2 layers were missing).
                        mamba_out_idx = instructions.len() - 1;
                        instructions.push(D2dCopyInst::new(div_ceil(hs as u32, 256), hidden_t_w, act.hidden.as_ptr(), hs as i32).into_inst());
                    }
                    trace_probe_map_seg.push((mamba_out_idx, crate::tracer::Probe::PostMixer { layer: layer_i }));
                    mamba2_idx += 1;
                }
                LayerType::MoeFfn => {
                    // Handled by CPU path in prefill_mixed_chunk — skip here.
                }
                LayerType::LfmConv => {
                    panic!("LfmConv layers not yet implemented in megakernel (braidinfer-aes.4)");
                }
            }
        }

        if is_last_segment {
            instructions.push(D2dCopyInst::new(
                div_ceil(hs as u32, 256),
                act.hidden.as_write_ptr(),
                unsafe { prefill_bufs.hidden.as_ptr().add((n - 1) * hs) },
                hs as i32,
            ).into_inst());
            instructions.push(D2dCopyInst::new(
                div_ceil(hs as u32, 256),
                act.normed.as_write_ptr(),
                act.hidden.as_ptr(),
                hs as i32,
            ).into_inst());
            instructions.push(RmsNormInst::new(
                rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
                act.hidden.as_write_ptr(), act.normed.as_ptr(),
                model.final_norm_weight.as_ptr(), hs as i32, eps,
            ).into_inst());
            trace_probe_map_seg.push((instructions.len() - 1, crate::tracer::Probe::FinalNorm));
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
        }
        instructions.push(Instruction::new(OP_HALT, 0));

        let watchdog = model.watchdog.clone();
        let _wd_state_dev = watchdog.register(device)?;
        let nkh = cfg.num_kv_heads;
        let hd = cfg.head_dim;

        Ok(MegakernelProgram {
            instructions,
            num_blocks,
            device,
            embedding_inst_idx: 0,
            _mrope_inst_indices: Vec::new(),
            gqa_attn_inst_indices: Vec::new(),
            kv: super::KvConfig {
                max_seq_len: cfg.max_seq_len as u32,
                num_kv_heads: nkh,
                head_dim: hd,
                kv_write_indices: Vec::new(),
                kv_base_ptrs: Vec::new(),
            },
            paged: true,
            paged_kv: Some(super::PagedKvState {
                page_table: None,
                position_table: None,
                attn_paged_inst_indices,
                attn_quant_inst_indices: Vec::new(),
                last_page_table_len: seq.chunks.len(),
                page_table_dirty: false,
                kv_stride_paged: nkh * hd,
            }),
            quantized_kv: false,
            quant_kv: None,
            dump_buffer: None,
            dump_counter: None,
            dump_capacity: 0,
            trace_probe_map: trace_probe_map_seg,
            barrier_layer_map: Vec::new(),
            multi_gpu_attn_boundaries: Vec::new(),
            _watchdog: watchdog,
            _not_send: std::marker::PhantomData,
            moe_act_d2d_indices: Vec::new(),
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
        let func = module.get_function("persistent_worker")?;
        let blocks_per_sm = func.max_active_blocks_per_sm(256, shared_mem as usize)?;
        let num_blocks = blocks_per_sm.max(1) as u32 * NUM_CUS;

        let mut instructions: Vec<Instruction> = Vec::new();
        instructions.push(D2dCopyInst::new(grid_x, act.hidden.as_write_ptr(),
            unsafe { prefill_bufs.hidden.as_ptr().add((n - 1) * hs) }, hs as i32).into_inst());
        instructions.push(D2dCopyInst::new(grid_x, act.normed.as_write_ptr(), act.hidden.as_ptr(), hs as i32).into_inst());
        instructions.push(RmsNormInst::new(rmsnorm_opcode(cfg.rms_norm_one_plus_w), 1,
            act.hidden.as_write_ptr(), act.normed.as_ptr(),
            model.final_norm_weight.as_ptr(), hs as i32, eps).into_inst());
        let mut trace_probe_map_fn: Vec<(usize, crate::tracer::Probe)> = Vec::new();
        trace_probe_map_fn.push((instructions.len() - 1, crate::tracer::Probe::FinalNorm));
        let lm_w_ptr = if cfg.tie_word_embeddings {
            model.embed_weight.as_ptr() as *const u8
        } else {
            model.lm_head_weight.as_ptr() as *const u8
        };
        instructions.push(LinearProjInst::new(OP_LM_HEAD, vs as u32,
            act.logits.as_write_ptr(), lm_w_ptr, act.hidden.as_ptr(),
            vs as i32, hs as i32, 0).into_inst());
        instructions.push(Instruction::new(OP_HALT, 0));


        let watchdog = model.watchdog.clone();
        let wd_state_dev = watchdog.register(device)?;

        Ok(MegakernelProgram {
            instructions,
            num_blocks,
            device,
            embedding_inst_idx: 0,
            _mrope_inst_indices: Vec::new(),
            gqa_attn_inst_indices: Vec::new(),
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
            dump_buffer: None,
            dump_counter: None,
            dump_capacity: 0,
            trace_probe_map: trace_probe_map_fn,
            barrier_layer_map: Vec::new(),
            multi_gpu_attn_boundaries: Vec::new(),
            _watchdog: watchdog,
            _not_send: std::marker::PhantomData,
            moe_act_d2d_indices: Vec::new(),
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
    fn compile_inner_p2p(model: &Model, p2p: &mut crate::moe_p2p::MoeP2pContext) -> HipResult<Self> {
        // P2 (braidinfer-4n5.6): paged=true so GPU 0 runs the full local paged-KV
        // attention sequence (OP_ATTN_PAGED_Q + OP_ATTN_PAGED) instead of the
        // head-parallel broadcast path. compile_inner routes attention layers to
        // compile_attention_layer_paged when paged=true, regardless of multi_gpu.
        // multi_gpu_attn_boundaries will be empty → has_head_parallel=false in
        // decode_step_p2p_inner → dispatch_head_parallel_attention is never called.
        let mut prog = Self::compile_inner(model, true, true)?;
        // Allocate paged-KV host-mapped page/position tables for the p2p program.
        // decode_step_p2p_inner calls update_step_paged_no_upload which reads these.
        let max_chunks = (model.config.max_seq_len + super::CHUNK_TOKENS - 1) / super::CHUNK_TOKENS;
        prog.init_paged_buffers(max_chunks)?;

        let cfg = &model.config;
        let act = &model.activations;
        let hs = cfg.hidden_size;

        // Collect: (barrier_idx → layer_idx) for all MoE barriers
        // yef5.2 Step A: track D2D indices before Pass 2 reindexing.
        let mut moe_act_d2d_indices_pre_pass2: Vec<(usize, usize)> = Vec::new();
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
                crate::config::FfnType::MoE {
                    num_active,
                    expert_intermediate_size,
                    ..
                } => (*num_active, *expert_intermediate_size),
                _ => unreachable!(),
            };
            // gate_up_in_dim: expert input dimension (hs for standard MoE, moe_latent_size for Nemotron-H)
            let gupd = dist.map(|d| d.gate_up_in_dim).unwrap_or(hs);
            // (has_latent was used in the now-removed gpu0 self-expert path; the
            // barrier_idx-2 patch below accesses fc1_latent_proj directly.)

            // xl4o: compile_moe_ffn_multi_gpu now emits 2 routing placeholders AFTER the gate,
            // so the order is [..., D2D(normed→normed_stage)=barrier_idx-4, OP_MOE_GATE=barrier_idx-3,
            // routing_ids_ph=barrier_idx-2, routing_weights_ph=barrier_idx-1, OP_BARRIER].
            // The activation/fc1 staging patches barrier_idx-4 (normed_stage); the routing staging
            // patches barrier_idx-2/-1 (below, unconditional). (Guard read moved -2 → -4.)
            if barrier_idx >= 4 {
                let prev4 = &prog.instructions[barrier_idx - 4];
                let prev4_opcode = prev4.words[0] as u32;
                let normed_stage_ptr = act.normed_stage.as_ptr() as u64;
                if prev4_opcode == OP_D2D_COPY && prev4.words[1] == normed_stage_ptr {
                    if let Some(ref fc1) = moe.fc1_latent_proj {
                        // Emit fc1: normed(hs) → moe_latent(gupd)
                        let (lp_op, lp_w) = linear_proj_opcode_ptr(fc1);
                        prog.instructions[barrier_idx - 4] = LinearProjInst::new(
                            lp_op, gupd as u32,
                            act.moe_latent.as_write_ptr(), lp_w, act.normed.as_ptr(),
                            gupd as i32, hs as i32, 0,
                        ).into_inst();
                    } else {
                        // yef5.2 Step A H1 fix: stage `act.normed` into the
                        // GPU-0 VRAM staging buffer `activation_staging_vram`
                        // (NOT the host-mapped UC `moe_act_uc_handoff`). The
                        // host-UC handoff is ASYMMETRIC-STALE on gfx1100
                        // multi-GPU (GFX1100_ARCH.md §11.19(x)): GPU 0's vector
                        // stores enter GPU 0's L2 as write-back dirty lines; a
                        // worker GPU's GART read of that host-mapped page hits
                        // host DRAM before the dirty lines land — there is no
                        // GPU->GPU snoop path on gfx1100. Symptom: ~1/5 decode
                        // forward-pass divergence (MoE routing amplifies the
                        // stale activation). activation_staging_vram is GPU-0
                        // VRAM, and kernel patch 0001 maps it MTYPE_UC for peer
                        // contexts — worker reads bypass GPU 0's L2 and see
                        // fresh VRAM. Mirrors the proven cross-GPU UC-VRAM
                        // pattern documented at moe_p2p.rs:166-172 (bd 9gmh
                        // Phase 1). grid_x = ceil(hs/256) covers all `hs` elems.
                        // The sentinel stays host-UC (small hot glc+dlc read;
                        // only the BULK activation read suffered the asymmetry).
                        // yef5.2 Step A: allocate per-layer host-UC sentinel (if not yet done),
                        // then emit D2D with_signal so workers acquire-spin on it.
                        if p2p.moe_act_sentinel[layer_idx].is_none() {
                            let s = braidinfer_hip::memory::MappedHostBuffer::<u32>::alloc(1)
                                .expect("yef5.2: moe_act_sentinel alloc failed");
                            unsafe { s.host_ptr().write_volatile(0u32); }
                            p2p.moe_act_sentinel[layer_idx] = Some(s);
                        }
                        // xl4o: activation copy moves to barrier_idx-4; the sentinel is REMOVED
                        // here — it now lives on the routing_weights copy below (the last-executing
                        // staged copy), so the worker's acquire sees activation + both routing
                        // buffers staged. The sentinel is still ALLOCATED above (non-latent arm)
                        // and re-fetched by the routing patch.
                        prog.instructions[barrier_idx - 4] = D2dCopyInst::new(
                            div_ceil(hs as u32, 256),
                            p2p.activation_staging_vram.as_ptr() as *mut f32,
                            act.normed.as_ptr(),
                            hs as i32,
                        )
                        .into_inst();
                    }
                    // xl4o: routing staging — UNCONDITIONAL (both fc1-latent + non-latent arms; the
                    // gate writes act.moe_expert_ids/weights regardless). Patch the 2 routing
                    // placeholders (barrier_idx-2/-1) to D2dCopy into peer-UC-VRAM so the worker
                    // P2P-reads fresh routing, not the §11.19(x)-stale GPU-written host-UC.
                    debug_assert_eq!(prog.instructions[barrier_idx - 2].words[0] as u32, OP_D2D_COPY,
                        "xl4o: barrier_idx-2 is not the routing_ids placeholder");
                    debug_assert_eq!(prog.instructions[barrier_idx - 1].words[0] as u32, OP_D2D_COPY,
                        "xl4o: barrier_idx-1 is not the routing_weights placeholder");
                    prog.instructions[barrier_idx - 2] = D2dCopyInst::new(
                        div_ceil(k as u32, 256),
                        p2p.routing_ids_staging_vram.as_ptr() as *mut f32,
                        act.moe_expert_ids.as_ptr() as *const f32,
                        k as i32,
                    ).into_inst();
                    let mut weights_copy = D2dCopyInst::new(
                        div_ceil(k as u32, 256),
                        p2p.routing_weights_staging_vram.as_ptr() as *mut f32,
                        act.moe_expert_weights.as_ptr() as *const f32,
                        k as i32,
                    );
                    // option (b): sentinel only when allocated (non-latent arm); Nemotron-H routing
                    // stays unsignaled. The signal moved OFF the activation copy onto this
                    // routing_weights copy (last staged); record THIS index for the per-step seq
                    // bump (decode/mod.rs:571-574). barrier_idx-4 (activation) needs no record:
                    // signal_ptr==0 -> seq bump is a guarded no-op; Pass-2 in-order clone carries it.
                    if let Some(s) = p2p.moe_act_sentinel[layer_idx].as_ref() {
                        weights_copy = weights_copy.with_signal(s.host_ptr() as *mut u32, 1);
                        moe_act_d2d_indices_pre_pass2.push((barrier_idx - 1, layer_idx));
                    }
                    prog.instructions[barrier_idx - 1] = weights_copy.into_inst();
                }
            }

            // P3 (braidinfer-4n5.7): GPU 0 no longer runs expert compute.
            // Replace the former OP_MOE_FFN_REMOTE self-dispatch (gpu0 expert)
            // with a NOP: D2D_COPY(output_slots[0] ← gpu0_zero_buffer, gupd).
            // This zeroes slot 0 each step so future reads (e.g. debugging)
            // see clean state. OP_MOE_DISPATCH_POST starts summing at slot 1,
            // so slot 0 is never accumulated into the final output.
            // gupd is the expert input dimension (hs for standard MoE,
            // moe_latent_size for Nemotron-H). We zero gupd elements to match
            // the slot stride used by POST's sum loop.
            prog.instructions[barrier_idx] = D2dCopyInst::new(
                div_ceil(gupd as u32, 256),
                p2p.output_slots.dev_ptr(0) as *mut f32,
                p2p.gpu0_zero_buffer.as_ptr() as *const f32,
                gupd as i32,
            )
            .into_inst();

            // Suppress dead-code warnings for variables no longer used by the
            // (now-removed) gpu0 self-expert dispatch.
            let _ = (k, eis, hs, gupd, moe.has_gate_proj);

            // bd 1hik: populate the per-layer params table consumed by
            // `dispatch_moe_workers_decode_async`. CPU worker-dispatch path
            // no longer reads raw instruction words to recover these.
            let has_gate_bool = moe.has_gate_proj;
            p2p.decode_params[layer_idx] = Some(crate::moe_p2p::DecodeMoeParams {
                output_slots: p2p.output_slots.dev_ptr(0),
                // xl4o: worker reads routing from peer-UC-VRAM (staged above by the routing copies),
                // NOT the §11.19(x)-stale GPU-written host-UC act.moe_expert_ids/weights.
                expert_ids: p2p.routing_ids_staging_vram.as_ptr() as *const i32,
                expert_weights: p2p.routing_weights_staging_vram.as_ptr() as *const f32,
                hs: hs as u32,
                gupd: gupd as u32,
                k: k as u32,
                eis: eis as u32,
                has_gate_proj: has_gate_bool,
                relu_sq: !has_gate_bool,
            });
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
                // Nemotron-H (fc1_latent_proj present): fc1 writes its output to
                // `act.moe_latent` (cached GPU 0 VRAM) at barrier_idx-2 in Pass 1.
                // Workers read activation from `p2p.moe_act_uc_handoff_dev_ptrs[gpu_id]`
                // (host-mapped UC, see decode/mod.rs:1065 and the snl input-side fix
                // for the non-latent branch at megakernel_compile.rs:1573-1579).
                // Without this stage copy, the latent path leaves the handoff buffer
                // unwritten → workers read zeros/garbage → NaN logits downstream
                // (braidinfer-vo0). Stage moe_latent → handoff (gupd elements) just
                // before OP_MOE_DISPATCH fires; GPU 0's local-expert path keeps
                // reading activation_ptr=act.moe_latent (cached VRAM) unchanged.
                if let Some(&layer_idx) = barrier_map.get(&i) {
                    let moe_layer = model.moe_weights[layer_idx].as_ref().unwrap();
                    if moe_layer.fc1_latent_proj.is_some() {
                        let dist = model.distributed_moe[layer_idx].as_ref();
                        let gupd_stage = dist.map(|d| d.gate_up_in_dim)
                            .unwrap_or(hs);
                        new_instructions.push(
                            // yef5.2 Step A H1 fix: latent path stages into GPU-0
                            // VRAM activation_staging_vram (peer-UC via patch 0001),
                            // not host-UC moe_act_uc_handoff (§11.19(x) asymmetric-stale).
                            D2dCopyInst::new(
                                div_ceil(gupd_stage as u32, 256),
                                p2p.activation_staging_vram.as_ptr() as *mut f32,
                                act.moe_latent.as_ptr(),
                                gupd_stage as i32,
                            )
                            .into_inst(),
                        );
                    }
                }
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
                        crate::config::FfnType::MoE {
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
                        output_slots: p2p.output_slots.dev_ptr(0) as u64,
                        final_output: final_output_ptr,
                        expert_ids: act.moe_expert_ids.as_ptr() as u64,
                        expert_weights: act.moe_expert_weights.as_ptr() as u64,
                        seq_counter: 0, // unused by OP_MOE_DISPATCH_POST; field retained for layout compat
                        num_workers_hs: ((num_workers as u64) << 32) | (hs as u64),
                        layer_k: ((layer_idx as u64) << 32) | (k as u64),
                        eis_gate: ((eis as u64) << 32) | has_gate,
                        activation: activation_ptr,
                        // P3 (braidinfer-4n5.7): these fields are unused by
                        // OP_MOE_DISPATCH_POST kernel (see megakernel_moe_dispatch.hip);
                        // zero them now that the gpu0_* buffers are deleted.
                        layer_config_ptrs: 0,
                        scratch_gate: 0,
                        scratch_up: 0,
                        scratch_act: 0,
                        num_gpus: num_gpus as u64,
                        gate_up_in_dim: gupd as u64,
                        // OP_MOE_DISPATCH_POST doesn't use gpu0_acc, but
                        // the field is non-optional in the struct layout.
                        gpu0_acc: 0,
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
                                crate::config::FfnType::MoE { shared_intermediate_size, expert_intermediate_size, .. } => {
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

            // P2 (braidinfer-4n5.6) FIX: Pass 2 reindexed the instruction stream, but
            // only multi_gpu_attn_boundaries + barrier_layer_map were remapped above.
            // The p2p program is now paged=true, so update_step_paged_no_upload patches
            // instructions through SEVERAL recorded index vectors — ALL recorded against
            // the PRE-Pass-2 stream. Every one must shift by inserted_before, else the
            // per-step patch writes into the WRONG instruction. (The step-3 KV-write dst
            // patch is the dangerous one: at a stale index it overwrites an OP_ATTN_PAGED
            // instruction's fields with a D2dCopy dst pointer → null page_table → GPU
            // fault on address (nil).)
            prog.embedding_inst_idx += inserted_before[prog.embedding_inst_idx];
            prog.gqa_attn_inst_indices = prog
                .gqa_attn_inst_indices
                .iter()
                .map(|&idx| idx + inserted_before[idx])
                .collect();
            prog.kv.kv_write_indices = prog
                .kv
                .kv_write_indices
                .iter()
                .map(|layer| {
                    layer
                        .iter()
                        .map(|&(k, v)| (k + inserted_before[k], v + inserted_before[v]))
                        .collect()
                })
                .collect();
            if let Some(pk) = prog.paged_kv.as_mut() {
                pk.attn_paged_inst_indices = pk
                    .attn_paged_inst_indices
                    .iter()
                    .map(|&idx| idx + inserted_before[idx])
                    .collect();
                pk.attn_quant_inst_indices = pk
                    .attn_quant_inst_indices
                    .iter()
                    .map(|&idx| idx + inserted_before[idx])
                    .collect();
            }

            // CPU-side opcode verification (write-through: instructions live in host
            // memory, so this reads them directly — no GPU dispatch, no printf). Every
            // remapped index MUST now point at its expected opcode; a mismatch is
            // stale-index drift that would manifest as a GPU null-fault at runtime.
            // Panic on the CPU at compile time with the exact offending index instead.
            let opcode_at = |i: usize| prog.instructions[i].words[0] as u32;
            assert_eq!(
                opcode_at(prog.embedding_inst_idx),
                OP_EMBEDDING as u32,
                "p2p reindex: embedding_inst_idx {} → opcode {} (expected OP_EMBEDDING)",
                prog.embedding_inst_idx,
                opcode_at(prog.embedding_inst_idx),
            );
            for layer in &prog.kv.kv_write_indices {
                for &(k, v) in layer {
                    assert_eq!(opcode_at(k), OP_D2D_COPY as u32,
                        "p2p reindex: kv-write k-idx {} → opcode {} (expected OP_D2D_COPY)", k, opcode_at(k));
                    assert_eq!(opcode_at(v), OP_D2D_COPY as u32,
                        "p2p reindex: kv-write v-idx {} → opcode {} (expected OP_D2D_COPY)", v, opcode_at(v));
                }
            }
            if let Some(pk) = prog.paged_kv.as_ref() {
                for &i in &pk.attn_paged_inst_indices {
                    assert_eq!(opcode_at(i), OP_ATTN_PAGED as u32,
                        "p2p reindex: attn_paged idx {} → opcode {} (expected OP_ATTN_PAGED)", i, opcode_at(i));
                }
                for &i in &pk.attn_quant_inst_indices {
                    assert_eq!(opcode_at(i), OP_ATTN_PAGED_Q as u32,
                        "p2p reindex: attn_quant idx {} → opcode {} (expected OP_ATTN_PAGED_Q)", i, opcode_at(i));
                }
            }

            // yef5.2 Step A: remap D2D sentinel indices after Pass 2 insertion shifts.
            prog.moe_act_d2d_indices = moe_act_d2d_indices_pre_pass2
                .into_iter()
                .map(|(idx, layer)| (idx + inserted_before[idx], layer))
                .collect();
        }

        // KEEP barrier_layer_map: in the unified-worker design it identifies
        // OP_MOE_DISPATCH instruction indices so decode_step_p2p can wrap each
        // MoE op with worker dispatch (OP_MOE_FFN_REMOTE → wait_ack on each
        // worker) before firing the GPU 0 batch containing op_moe_dispatch.

        Ok(prog)
    }
}
