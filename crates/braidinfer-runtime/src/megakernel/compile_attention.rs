//! Attention layer compilation: emit_attention_layer and its three call-site wrappers.

use super::compile_common::{AttentionVariant, div_ceil, emit_batched_linear_proj, rmsnorm_opcode};
use super::instructions::*;
use super::{CHUNK_TOKENS, Instruction, MegakernelProgram, PrefillBuffers};
use crate::model::{ActivationBuffers, AttentionLayerWeights, KvCache, LayerWeights, ModelConfig};

impl MegakernelProgram {
    pub(super) fn emit_attention_layer(
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

    pub(super) fn compile_attention_layer(
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
    pub(super) fn compile_attention_layer_multi_gpu(
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
    pub(super) fn compile_attention_layer_paged(
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
}
