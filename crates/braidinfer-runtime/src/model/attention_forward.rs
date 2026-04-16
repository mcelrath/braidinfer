use braidinfer_hip::HipResult;
use braidinfer_hip::ffi;
use braidinfer_hip::memory::DeviceBuffer;

use super::Model;
use crate::weights::*;

impl Model {
    pub(crate) fn attention_forward(
        &mut self,
        layer_idx: usize,
        kv_cache_idx: usize,
        position: u32,
    ) -> HipResult<()> {
        let cfg = &self.config;
        let hs = cfg.hidden_size as u32;
        let nqh = cfg.num_q_heads as u32;
        let nkh = cfg.num_kv_heads as u32;
        let hd = cfg.head_dim as u32;
        let rd = cfg.rope_dim as u32;
        let s0 = cfg.mrope_section[0] as u32;
        let s1 = cfg.mrope_section[1] as u32;
        let s2 = cfg.mrope_section[2] as u32;
        let eps = cfg.rms_norm_eps;
        let sync_debug = std::env::var("SYNC_DEBUG").is_ok();

        macro_rules! sync_check {
            ($label:expr) => {
                if sync_debug {
                    if let Err(e) = self.stream.synchronize() {
                        eprintln!("SYNC_DEBUG: crash at L{}.{}", layer_idx, $label);
                        return Err(e);
                    }
                    eprintln!("SYNC_DEBUG: L{}.{} OK", layer_idx, $label);
                }
            };
        }

        // 1. RMSNorm
        // SAFETY: Raw pointer breaks borrow on self.layers for mutable self.activations access.
        // Pointer valid: layers not modified during attention_forward.
        let input_norm = match &self.layers[layer_idx] {
            LayerWeights::Attention(w) => &w.input_norm as *const DeviceBuffer<u16>,
            _ => panic!("expected attention layer"),
        };
        unsafe {
            self.kernels.rmsnorm.forward(
                &mut self.activations.normed,
                &self.activations.hidden,
                &*input_norm,
                1,
                hs,
                eps,
                cfg.rms_norm_one_plus_w,
                &self.stream,
            )?;
        }
        sync_check!("rmsnorm");

        // 2. Project Q+gate, K, V
        // Use raw pointers to LinearWeight to work around borrow checker
        // (self.layers borrows self, but we need &mut self.activations)
        let (w_q_gate_p, w_k_p, w_v_p, w_o_p, q_norm_w, k_norm_w) = match &self.layers[layer_idx] {
            LayerWeights::Attention(w) => (
                &w.w_q_gate as *const LinearWeight,
                &w.w_k as *const LinearWeight,
                &w.w_v as *const LinearWeight,
                &w.w_o as *const LinearWeight,
                &w.q_norm as *const DeviceBuffer<u16>,
                &w.k_norm as *const DeviceBuffer<u16>,
            ),
            _ => unreachable!(),
        };

        let q_mult = if cfg.has_output_gate { 2u32 } else { 1 };
        unsafe {
            (*w_q_gate_p).forward(
                &self.kernels.linear_proj,
                &mut self.activations.q_gate_attn,
                &self.activations.normed,
                nqh * hd * q_mult,
                hs,
                &self.stream,
            )?;
            sync_check!("q_proj");
            (*w_k_p).forward(
                &self.kernels.linear_proj,
                &mut self.activations.k_attn,
                &self.activations.normed,
                nkh * hd,
                hs,
                &self.stream,
            )?;
            sync_check!("k_proj");
            (*w_v_p).forward(
                &self.kernels.linear_proj,
                &mut self.activations.v_attn,
                &self.activations.normed,
                nkh * hd,
                hs,
                &self.stream,
            )?;
            sync_check!("v_proj");
        }

        // 3. Split q_gate_attn → q, gate (gated) or just copy (non-gated)
        let hd_usize = hd as usize;
        if cfg.has_output_gate {
            unsafe {
                for h in 0..nqh as usize {
                    let src_q = h * hd_usize * 2;
                    let src_g = h * hd_usize * 2 + hd_usize;
                    let dst = h * hd_usize;
                    d2d_copy_f32(
                        &mut self.activations.q_attn,
                        dst,
                        &self.activations.q_gate_attn,
                        src_q,
                        hd_usize,
                        &self.stream,
                    )?;
                    d2d_copy_f32(
                        &mut self.activations.gate_attn,
                        dst,
                        &self.activations.q_gate_attn,
                        src_g,
                        hd_usize,
                        &self.stream,
                    )?;
                }
            }
        } else {
            // Non-gated: q_gate_attn IS q_attn, just copy
            let total = nqh as usize * hd_usize;
            unsafe {
                d2d_copy_f32(
                    &mut self.activations.q_attn,
                    0,
                    &self.activations.q_gate_attn,
                    0,
                    total,
                    &self.stream,
                )?;
            }
        }
        sync_check!("q_copy");

        // 4a. Write K,V to paged KV BEFORE QK-norm so stored K preserves full dynamic range.
        {
            let seq = self
                .paged_seq
                .as_ref()
                .expect("paged sequence not initialized");
            let allocator = self
                .page_allocator
                .as_ref()
                .expect("paged allocator not initialized");
            let chunk_slot = seq.chunks.last().expect("missing paged chunk").slot_index();
            let chunk_base = allocator.slot_ptr(chunk_slot) as usize;
            let chunk_offset = seq.current_chunk_offset() as usize - 1;
            let kv_stride = self.config.num_kv_heads * self.config.head_dim;
            let layer_k_offset = kv_cache_idx
                * 2
                * crate::megakernel::CHUNK_TOKENS
                * kv_stride
                * std::mem::size_of::<f32>();
            let layer_v_offset = layer_k_offset
                + crate::megakernel::CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>();
            let chunk_head_stride = crate::megakernel::CHUNK_TOKENS * hd as usize;
            for h in 0..nkh as usize {
                let src_off = h * hd as usize;
                let head_byte_off = (h * chunk_head_stride + chunk_offset * hd as usize)
                    * std::mem::size_of::<f32>();
                unsafe {
                    braidinfer_hip::error::check(ffi::hipMemcpyAsync(
                        (chunk_base + layer_k_offset + head_byte_off) as *mut std::ffi::c_void,
                        self.activations.k_attn.as_ptr().add(src_off) as *const std::ffi::c_void,
                        hd as usize * std::mem::size_of::<f32>(),
                        ffi::hipMemcpyDeviceToDevice,
                        self.stream.raw(),
                    ))?;
                    braidinfer_hip::error::check(ffi::hipMemcpyAsync(
                        (chunk_base + layer_v_offset + head_byte_off) as *mut std::ffi::c_void,
                        self.activations.v_attn.as_ptr().add(src_off) as *const std::ffi::c_void,
                        hd as usize * std::mem::size_of::<f32>(),
                        ffi::hipMemcpyDeviceToDevice,
                        self.stream.raw(),
                    ))?;
                }
            }
        }

        // 4b. QK norm (in-place on q_attn, k_attn — for current token's attention computation)
        let per_head_qk_norm = cfg.has_qk_norm && unsafe { (*q_norm_w).len() == hd as usize };
        if cfg.has_qk_norm {
            if per_head_qk_norm {
                // Per-head QK norm (Qwen3.5 style): weight is [head_dim]
                unsafe {
                    self.kernels.qk_norm.forward(
                        &mut self.activations.q_attn,
                        &mut self.activations.k_attn,
                        &*q_norm_w,
                        &*k_norm_w,
                        nqh,
                        nkh,
                        hd,
                        eps,
                        &self.stream,
                    )?;
                }
            } else {
                // Full-hidden QK norm (OLMoE style): weight is [hidden_size], apply as RMSNorm
                // Use normed buffer as temp to avoid aliasing
                unsafe {
                    self.kernels.rmsnorm.forward(
                        &mut self.activations.normed,
                        &self.activations.q_attn,
                        &*q_norm_w,
                        1,
                        nqh * hd,
                        eps,
                        cfg.rms_norm_one_plus_w,
                        &self.stream,
                    )?;
                    d2d_copy_f32(
                        &mut self.activations.q_attn,
                        0,
                        &self.activations.normed,
                        0,
                        (nqh * hd) as usize,
                        &self.stream,
                    )?;
                    self.kernels.rmsnorm.forward(
                        &mut self.activations.normed,
                        &self.activations.k_attn,
                        &*k_norm_w,
                        1,
                        nkh * hd,
                        eps,
                        cfg.rms_norm_one_plus_w,
                        &self.stream,
                    )?;
                    d2d_copy_f32(
                        &mut self.activations.k_attn,
                        0,
                        &self.activations.normed,
                        0,
                        (nkh * hd) as usize,
                        &self.stream,
                    )?;
                }
            }
        }

        // 5. Apply RoPE (skip for Nemotron-H which has no rotary embeddings)
        if cfg.use_rope {
            let pos_data = [position as i32, position as i32, position as i32];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pos_data.as_ptr(),
                    self.activations.position_ids.host_ptr(),
                    pos_data.len(),
                )
            };

            self.kernels.mrope.forward(
                &mut self.activations.q_attn,
                &mut self.activations.k_attn,
                &self.activations.inv_freq,
                self.activations.position_ids.as_ptr(),
                nqh,
                nkh,
                hd,
                rd,
                s0,
                s1,
                s2,
                &self.stream,
            )?;
            sync_check!("mrope");
        } // end if cfg.use_rope

        // 6. KV write already done at step 4a (before QK-norm) for quantization quality.

        // 7. Paged attention
        let seq_len = position + 1;
        let seq = self
            .paged_seq
            .as_ref()
            .expect("paged sequence not initialized");
        let allocator = self
            .page_allocator
            .as_ref()
            .expect("paged allocator not initialized");
        let page_table_host: Vec<u64> = seq
            .chunks
            .iter()
            .map(|c| allocator.slot_ptr(c.slot_index()) as u64)
            .collect();
        self.paged_page_table
            .as_mut()
            .expect("paged page table not initialized")
            .copy_from_host(&page_table_host)?;
        self.paged_position_table
            .as_mut()
            .expect("paged position table not initialized")
            .copy_from_host(&seq.positions)?;
        let k_norm_weight = if per_head_qk_norm {
            unsafe { Some(&*k_norm_w) }
        } else {
            None
        };
        let layer_k_offset = (kv_cache_idx
            * 2
            * crate::megakernel::CHUNK_TOKENS
            * self.config.num_kv_heads
            * self.config.head_dim
            * std::mem::size_of::<f32>()) as u64;
        let layer_v_offset = layer_k_offset
            + (crate::megakernel::CHUNK_TOKENS
                * self.config.num_kv_heads
                * self.config.head_dim
                * std::mem::size_of::<f32>()) as u64;
        self.kernels.paged_attention.forward(
            &mut self.activations.attn_out,
            &self.activations.q_attn,
            self.paged_page_table.as_ref().unwrap(),
            self.paged_position_table.as_ref().unwrap(),
            &self.activations.inv_freq,
            nqh,
            nkh,
            hd,
            seq_len,
            crate::megakernel::CHUNK_TOKENS as u32,
            rd,
            layer_k_offset,
            layer_v_offset,
            k_norm_weight,
            &self.stream,
        )?;
        sync_check!("paged_attention");

        // 8. Output gate (Qwen3.5 only) or pass-through
        let final_attn = if cfg.has_output_gate {
            self.kernels.output_gate.forward(
                &mut self.activations.gated_out,
                &self.activations.attn_out,
                &self.activations.gate_attn,
                nqh * hd,
                &self.stream,
            )?;
            &self.activations.gated_out as *const DeviceBuffer<f32>
        } else {
            &self.activations.attn_out as *const DeviceBuffer<f32>
        };

        // 9. Output projection
        unsafe {
            (*w_o_p).forward(
                &self.kernels.linear_proj,
                &mut self.activations.out_proj,
                &*final_attn,
                hs,
                nqh * hd,
                &self.stream,
            )?;
        }

        // 10. Residual add
        unsafe {
            d2d_copy_f32(
                &mut self.activations.residual,
                0,
                &self.activations.hidden,
                0,
                hs as usize,
                &self.stream,
            )?;
        }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.out_proj,
            &self.activations.residual,
            hs,
            &self.stream,
        )?;

        Ok(())
    }
}
