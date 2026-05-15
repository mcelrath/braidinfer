use braidinfer_hip::memory::DeviceBuffer;

use super::super::{Model, ModelError};
use crate::config::{FfnType, LayerType};
use crate::gpu_utils::d2d_copy_f32;
use crate::quant::LinearWeight;
use crate::weights::LayerWeights;

impl Model {
    /// Single-GPU per-layer decode with activation trace checkpoints.
    /// Only reachable when trace.is_some() && multi_gpu.is_none().
    pub(crate) fn decode_step_trace(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        let hs = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps;
        let sync_debug = self.sync_debug;
        if self
            .config
            .layers
            .iter()
            .any(|layer| layer.layer_type == LayerType::Attention)
        {
            self.append_paged_decode_token(position)?;
        }

        macro_rules! sync_check_moe {
            ($label:expr) => {
                if sync_debug {
                    if let Err(e) = self.stream.synchronize() {
                        eprintln!("SYNC_DEBUG: crash at pos={}.{}", position, $label);
                        return Err(e.into());
                    }
                    eprintln!("SYNC_DEBUG: pos={}.{} OK", position, $label);
                }
            };
        }

        // Set position_ids for mRoPE/RoPE
        self.set_position(position).map_err(ModelError::Hip)?;

        // Embedding
        self.kernels.embedding.forward(
            &mut self.activations.hidden,
            &self.embed_weight,
            token_id as i32,
            hs,
            &self.stream,
        )?;
        sync_check_moe!("embed");

        if self.debug_nan {
            self.stream.synchronize()?;
            let mut buf = vec![0.0f32; self.config.hidden_size];
            self.activations.hidden.copy_to_host(&mut buf)?;
            let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "embed(tok={token_id}): max_abs={max_abs:.4e}, first5={:.4?}",
                &buf[..5]
            );
        }

        if self.trace.is_some() {
            self.stream.synchronize()?;
            let mut buf = vec![0.0f32; self.config.hidden_size];
            self.activations.hidden.copy_to_host(&mut buf)?;
            self.trace.as_mut().unwrap().write_checkpoint("embed", &buf);
        }

        // Process each layer
        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        let mut mamba2_idx = 0usize;
        for layer_i in 0..self.config.num_layers {
            match self.config.layers[layer_i].layer_type {
                LayerType::Attention => {
                    self.attention_forward(layer_i, kv_idx, position)?;
                    sync_check_moe!(format!("L{layer_i}.attn"));
                    kv_idx += 1;
                }
                LayerType::Gdn => {
                    self.gdn_forward(layer_i, gdn_idx)?;
                    sync_check_moe!(format!("L{layer_i}.gdn"));
                    gdn_idx += 1;
                }
                LayerType::Mamba2 => {
                    self.mamba2_forward(layer_i, mamba2_idx)?;
                    sync_check_moe!(format!("L{layer_i}.mamba2"));
                    mamba2_idx += 1;
                }
                LayerType::MoeFfn => {
                    // Standalone MoE FFN layer — just norm + MoE dispatch + residual
                    // The norm is applied inside moe_ffn_forward, skip to FFN below
                }
                LayerType::LfmConv => panic!("LfmConv not yet implemented"),
            }

            // Debug: check for NaN in hidden state after each layer
            if self.debug_nan {
                self.stream.synchronize()?;
                let mut buf = vec![0.0f32; self.config.hidden_size];
                self.activations.hidden.copy_to_host(&mut buf)?;
                let nan_count = buf.iter().filter(|x| x.is_nan()).count();
                let max_abs = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!(
                    "L{layer_i} ({:?}): {nan_count} NaN, max_abs={max_abs:.2e}",
                    self.config.layers[layer_i].layer_type
                );
            }

            if self.trace.is_some() {
                self.stream.synchronize()?;
                let mut buf = vec![0.0f32; self.config.hidden_size];
                self.activations.hidden.copy_to_host(&mut buf)?;
                self.trace
                    .as_mut()
                    .unwrap()
                    .write_checkpoint(&format!("L{layer_i}.post_mixer"), &buf);
            }

            // FFN: dense, MoE, or None (standalone layers like Nemotron M/*)
            if matches!(self.config.layers[layer_i].ffn_type, FfnType::MoE { .. }) {
                self.moe_ffn_forward(layer_i)?;
            } else if matches!(self.config.layers[layer_i].ffn_type, FfnType::None) {
                // No FFN for this layer (Nemotron M and * layers)
            } else {
                // Dense FFN: fused (bf16) or unfused (quantized)
                let hs = self.config.hidden_size;
                let is = self.config.intermediate_size;
                let eps = self.config.rms_norm_eps;

                // SAFETY: Raw pointers break borrow on self.layers for mutable self.activations.
                let (post_norm_p, w_gate_p, w_up_p, w_down_p) = match &self.layers[layer_i] {
                    LayerWeights::Attention(w) => (
                        &w.post_norm as *const DeviceBuffer<u16>,
                        &w.w_gate as *const LinearWeight,
                        &w.w_up as *const LinearWeight,
                        &w.w_down as *const LinearWeight,
                    ),
                    LayerWeights::Gdn(w) => (
                        &w.post_norm as *const DeviceBuffer<u16>,
                        &w.w_gate as *const LinearWeight,
                        &w.w_up as *const LinearWeight,
                        &w.w_down as *const LinearWeight,
                    ),
                    _ => panic!("dense FFN only for Attention/Gdn layers"),
                };

                let all_bf16 = unsafe {
                    matches!(&*w_gate_p, LinearWeight::Bf16(_))
                        && matches!(&*w_up_p, LinearWeight::Bf16(_))
                        && matches!(&*w_down_p, LinearWeight::Bf16(_))
                };

                if all_bf16 {
                    unsafe {
                        self.ffn_forward(
                            &*post_norm_p,
                            (*w_gate_p).as_bf16(),
                            (*w_up_p).as_bf16(),
                            (*w_down_p).as_bf16(),
                        )?;
                    }
                    sync_check_moe!(format!("L{layer_i}.ffn_bf16"));
                } else {
                    // Unfused path for quantized weights
                    unsafe {
                        d2d_copy_f32(
                            &mut self.activations.residual,
                            0,
                            &self.activations.hidden,
                            0,
                            hs,
                            &self.stream,
                        )?;
                    }
                    unsafe {
                        self.kernels.rmsnorm.forward(
                            &mut self.activations.normed,
                            &self.activations.hidden,
                            &*post_norm_p,
                            1,
                            hs as u32,
                            eps,
                            self.config.rms_norm_one_plus_w,
                            &self.stream,
                        )?;
                    }
                    sync_check_moe!(format!("L{layer_i}.ffn_norm"));
                    unsafe {
                        (*w_gate_p).forward(
                            &self.kernels.linear_proj,
                            &mut self.activations.ffn_gate,
                            &self.activations.normed,
                            is as u32,
                            hs as u32,
                            &self.stream,
                        )?;
                    }
                    sync_check_moe!(format!("L{layer_i}.ffn_gate"));
                    unsafe {
                        (*w_up_p).forward(
                            &self.kernels.linear_proj,
                            &mut self.activations.ffn_up,
                            &self.activations.normed,
                            is as u32,
                            hs as u32,
                            &self.stream,
                        )?;
                    }
                    sync_check_moe!(format!("L{layer_i}.ffn_up"));
                    self.kernels.silu_mul.forward(
                        &mut self.activations.ffn_act,
                        &self.activations.ffn_gate,
                        &self.activations.ffn_up,
                        is as u32,
                        &self.stream,
                    )?;
                    sync_check_moe!(format!("L{layer_i}.ffn_silu"));
                    unsafe {
                        (*w_down_p).forward(
                            &self.kernels.linear_proj,
                            &mut self.activations.ffn_down,
                            &self.activations.ffn_act,
                            hs as u32,
                            is as u32,
                            &self.stream,
                        )?;
                    }
                    sync_check_moe!(format!("L{layer_i}.ffn_down"));
                    self.kernels.residual_add.forward(
                        &mut self.activations.hidden,
                        &self.activations.ffn_down,
                        &self.activations.residual,
                        hs as u32,
                        &self.stream,
                    )?;
                }
            }

            if self.trace.is_some() {
                self.stream.synchronize()?;
                let mut buf = vec![0.0f32; self.config.hidden_size];
                self.activations.hidden.copy_to_host(&mut buf)?;
                self.trace
                    .as_mut()
                    .unwrap()
                    .write_checkpoint(&format!("L{layer_i}.post_ffn"), &buf);
            }
        }

        // Final RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed,
            &self.activations.hidden,
            &self.final_norm_weight,
            1,
            hs,
            eps,
            self.config.rms_norm_one_plus_w,
            &self.stream,
        )?;

        // LM head
        let lm_head_w = if self.config.tie_word_embeddings {
            &self.embed_weight
        } else {
            &self.lm_head_weight
        };
        self.kernels.linear_proj.forward(
            &mut self.activations.logits,
            lm_head_w,
            &self.activations.normed,
            self.config.vocab_size as u32,
            hs,
            &self.stream,
        )?;

        self.stream.synchronize()?;

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;

        if self.trace.is_some() {
            let mut hid_buf = vec![0.0f32; self.config.hidden_size];
            self.activations.hidden.copy_to_host(&mut hid_buf)?;
            self.trace
                .as_mut()
                .unwrap()
                .write_checkpoint("final_hidden", &hid_buf);

            let mut norm_buf = vec![0.0f32; self.config.hidden_size];
            self.activations.normed.copy_to_host(&mut norm_buf)?;
            self.trace
                .as_mut()
                .unwrap()
                .write_checkpoint("final_norm", &norm_buf);

            // Capture top-10 logits (token_id + value pairs as f32)
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top10: Vec<f32> = indexed
                .iter()
                .take(10)
                .flat_map(|&(id, val)| [id as f32, val])
                .collect();
            self.trace
                .as_mut()
                .unwrap()
                .write_checkpoint("top10_logits", &top10);
        }

        self.seq_len = position + 1;
        Ok(logits)
    }
}
