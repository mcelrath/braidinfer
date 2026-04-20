use braidinfer_hip::memory::DeviceBuffer;

use crate::paged_kv::{self, RecurrentCheckpointPool};

use super::Model;
use super::ModelError;
use crate::config::*;
use crate::weights::*;
use crate::gpu_utils::d2d_copy_f32;

impl Model {
    /// Save the current GDN recurrent states into a checkpoint pool slot.
    /// Lazy-initializes the pool on first call. Returns the slot index.
    pub fn save_recurrent_checkpoint(&mut self) -> Result<u32, ModelError> {
        if self.checkpoint_pool.is_none() {
            // Pool capacity 1: prefill uses ring-buffer overwrite (only most-recent needed).
            // Speculative decode (future) may increase this.
            self.checkpoint_pool =
                Some(RecurrentCheckpointPool::new(self.device, &self.config, 1)?);
        }
        // Free previous slot before allocating new one (ring buffer with capacity 1)
        if let Some(prev) = self.last_checkpoint_slot.take() {
            self.checkpoint_pool.as_mut().unwrap().free(prev);
        }
        let recurrent_bufs: Vec<&DeviceBuffer<f32>> =
            self.gdn_states.iter().map(|s| &s.recurrent).collect();
        let pool = self.checkpoint_pool.as_mut().unwrap();
        let slot = paged_kv::save_checkpoint(pool, &recurrent_bufs, self.stream.raw())?;
        self.last_checkpoint_slot = Some(slot);
        Ok(slot)
    }

    /// Process a sequence of tokens (prefill). Returns logits for the last token.
    /// In persistent mode (before worker starts): uses batched megakernel compile_prefill.
    /// Otherwise: sequential decode_step fallback.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::MissingWeight("empty token sequence".into()));
        }
        // Batched path: persistent mode or MoE model, worker not yet started (can still do hipMemcpy).
        if (self.persistent || self.has_moe) && self.persistent_workers.is_none() {
            return self.prefill_batched(tokens);
        }
        // Sequential fallback: use paged path if worker not started (coherent with decode_step_paged reference).
        let use_paged = self.persistent_workers.is_none();
        let mut logits = vec![];
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = self.seq_len + i as u32;
            logits = if use_paged {
                self.decode_step_paged(tok, pos)?
            } else {
                self.decode_step(tok, pos)?
            };
        }
        Ok(logits)
    }

    /// Process one chunk of tokens for a model that has MoE layers.
    /// Non-MoE layer spans are batched via compile_prefill_segment.
    /// MoE layers are processed per-token via moe_ffn_forward with d2d hidden-state handoff.
    /// Does NOT increment seq_len (caller does that once).
    fn prefill_mixed_chunk(
        &mut self,
        chunk: &[u32],
        start_pos: u32,
    ) -> Result<(), ModelError> {
        use crate::config::LayerType;
        use crate::megakernel::{CHUNK_TOKENS, MegakernelProgram, PrefillBuffers};

        if self.prefill_bufs.is_none() {
            self.prefill_bufs = Some(
                PrefillBuffers::alloc(self.device, &self.config, CHUNK_TOKENS)
                    .map_err(ModelError::Hip)?,
            );
        }

        // Start MoE worker GPUs lazily (no-op if single-GPU or already started).
        // Must happen before MoE layer processing and before the decode persistent worker launches.
        self.ensure_moe_workers_started()?;

        let n = chunk.len();
        let hs = self.config.hidden_size;
        let num_layers = self.config.num_layers;

        // Load megakernel module once and share across all segment compilations via Arc.
        let megakernel_module = std::sync::Arc::new(
            braidinfer_hip::module::Module::load(
                self.device,
                &crate::kernel::kernel_dir().join("megakernel.hsaco"),
            ).map_err(ModelError::Hip)?
        );

        // Emit embedding instructions via megakernel: one per token into prefill_bufs.hidden[t*hs]
        // We compile a tiny "embedding-only" segment by treating layer_start=0,layer_end=0.
        // Then for each span of non-MoE layers, compile_prefill_segment handles them.
        // For MoE layers, use moe_ffn_forward with d2d handoff.

        // Step 1: Embed all tokens into prefill_bufs.hidden via a tiny megakernel
        {
            use crate::megakernel::instructions::*;
            use crate::megakernel::{Instruction, NUM_CUS, OP_HALT};

            let grid_x = (hs as u32 + 255) / 256;
            let module = std::sync::Arc::clone(&megakernel_module);
            let shared_mem = (256u32 * 4 * 2).max(hs as u32 * 4).max(31776u32);
            let func = module.get_function("megakernel_f32").map_err(ModelError::Hip)?;
            let blocks_per_sm = func.max_active_blocks_per_sm(256, shared_mem as usize).map_err(ModelError::Hip)?;
            let num_blocks = blocks_per_sm.max(1) as u32 * NUM_CUS;

            let mut insts: Vec<Instruction> = Vec::new();
            let bufs = self.prefill_bufs.as_ref().unwrap();
            for t in 0..n {
                insts.push(EmbeddingInst::new(
                    grid_x,
                    unsafe { bufs.hidden.as_write_ptr().add(t * hs) },
                    self.embed_weight.as_ptr(),
                    chunk[t] as i32,
                    hs as i32,
                ).into_inst());
            }
            insts.push(Instruction::new(OP_HALT, 0));

            let flat: Vec<u64> = insts.iter().flat_map(|i| i.words).collect();
            let mut dev_prog = braidinfer_hip::memory::DeviceBuffer::alloc(self.device, flat.len()).map_err(ModelError::Hip)?;
            dev_prog.copy_from_host(&flat).map_err(ModelError::Hip)?;

            let mut prog_ptr: *const std::ffi::c_void = dev_prog.as_ptr().cast();
            let mut num_inst = insts.len() as i32;
            let mut args: [*mut std::ffi::c_void; 2] = [
                std::ptr::addr_of_mut!(prog_ptr).cast(),
                std::ptr::addr_of_mut!(num_inst).cast(),
            ];
            func.launch_cooperative(
                (num_blocks, 1, 1),
                (256, 1, 1),
                shared_mem,
                &self.stream,
                &mut args,
            ).map_err(ModelError::Hip)?;
            self.stream.synchronize().map_err(ModelError::Hip)?;
        }

        // Step 2: Walk layers, identify contiguous non-MoE spans vs MoE layers.
        // MoE layer types:
        //   - LayerType::MoeFfn: standalone MoE FFN (Nemotron 'E'), no attention — dispatch only
        //   - LayerType::Attention + ffn_type::MoE: attention + MoE FFN (Qwen) — 1-layer attention
        //     span (compile_prefill_segment skips FFN for MoE) then MoE FFN dispatch
        let mut layer_i = 0usize;

        while layer_i < num_layers {
            let lt = self.config.layers[layer_i].layer_type.clone();
            if lt == LayerType::MoeFfn {
                // Nemotron standalone MoE FFN layer (no attention part).
                if self.moe_p2p.is_some() {
                    // Multi-GPU batched path: dispatch all n tokens in one round-trip
                    let mut bufs = self.prefill_bufs.take().unwrap();
                    self.moe_ffn_forward_prefill_batched(layer_i, &mut bufs.hidden, n)
                        .map_err(ModelError::Hip)?;
                    self.prefill_bufs = Some(bufs);
                } else {
                    // Single-GPU batched path: one sync for all n gate computations
                    let mut bufs = self.prefill_bufs.take().unwrap();
                    self.moe_ffn_forward_prefill_single_gpu_batched(layer_i, &mut bufs.hidden, n)
                        .map_err(ModelError::Hip)?;
                    self.prefill_bufs = Some(bufs);
                }
                layer_i += 1;
            } else if matches!(self.config.layers[layer_i].ffn_type, FfnType::MoE { .. })
                && (lt == LayerType::Attention || lt == LayerType::Gdn) {
                // Attention/GDN + MoE FFN layer: run mixer via 1-layer segment,
                // then dispatch MoE FFN separately.
                // Never set is_last=true here: the segment would emit final_norm+LM_head
                // BEFORE the MoE FFN runs, producing logits from the pre-MoE hidden state.
                // Instead, we handle is_last after MoE below.
                let span_start = layer_i;
                let span_end = layer_i + 1;
                let is_truly_last = span_end == num_layers;

                // start_pos encodes RoPE positions baked into the compiled program.
                // Must be part of the key to avoid wrong positions on sequential calls.
                let key = (span_start, span_end, n, start_pos as usize);
                let mut bufs = self.prefill_bufs.take().unwrap();
                if !self.megakernel_prefill_segments.contains_key(&key) {
                    let mk = MegakernelProgram::compile_prefill_segment_with_module(
                        self, std::sync::Arc::clone(&megakernel_module), chunk, start_pos,
                        span_start, span_end, false, // never is_last: LM head runs after MoE below
                        &mut bufs,
                    ).map_err(ModelError::Hip)?;
                    self.megakernel_prefill_segments.insert(key, mk);
                }
                self.prefill_bufs = Some(bufs);
                let mk = self.megakernel_prefill_segments.get(&key).unwrap();
                mk.execute(&self.stream).map_err(ModelError::Hip)?;
                self.stream.synchronize().map_err(ModelError::Hip)?;

                // Dispatch MoE FFN for this layer
                if self.moe_p2p.is_some() {
                    let mut bufs = self.prefill_bufs.take().unwrap();
                    self.moe_ffn_forward_prefill_batched(layer_i, &mut bufs.hidden, n)
                        .map_err(ModelError::Hip)?;
                    self.prefill_bufs = Some(bufs);
                } else {
                    let mut bufs = self.prefill_bufs.take().unwrap();
                    self.moe_ffn_forward_prefill_single_gpu_batched(layer_i, &mut bufs.hidden, n)
                        .map_err(ModelError::Hip)?;
                    self.prefill_bufs = Some(bufs);
                }

                // If this is the last layer, emit final norm + LM head now (post-MoE).
                if is_truly_last {
                    let bufs = self.prefill_bufs.take().unwrap();
                    let mk = MegakernelProgram::compile_final_norm_lm_head(
                        self, std::sync::Arc::clone(&megakernel_module), &bufs, n,
                    ).map_err(ModelError::Hip)?;
                    self.prefill_bufs = Some(bufs);
                    mk.execute(&self.stream).map_err(ModelError::Hip)?;
                    self.stream.synchronize().map_err(ModelError::Hip)?;
                }
                layer_i += 1;
            } else {
                // Dense non-MoE span: accumulate contiguous dense layers
                let span_start = layer_i;
                while layer_i < num_layers {
                    let l = &self.config.layers[layer_i];
                    if l.layer_type == LayerType::MoeFfn { break; }
                    if matches!(l.ffn_type, FfnType::MoE { .. })
                        && (l.layer_type == LayerType::Attention || l.layer_type == LayerType::Gdn) { break; }
                    layer_i += 1;
                }
                let span_end = layer_i;
                let is_last = span_end == num_layers;

                let key = (span_start, span_end, n, start_pos as usize);
                let mut bufs = self.prefill_bufs.take().unwrap();

                if !self.megakernel_prefill_segments.contains_key(&key) {
                    let mk = MegakernelProgram::compile_prefill_segment_with_module(
                        self, std::sync::Arc::clone(&megakernel_module), chunk, start_pos,
                        span_start, span_end, is_last,
                        &mut bufs,
                    ).map_err(ModelError::Hip)?;
                    self.megakernel_prefill_segments.insert(key, mk);
                }
                self.prefill_bufs = Some(bufs);
                let mk = self.megakernel_prefill_segments.get(&key).unwrap();
                mk.execute(&self.stream).map_err(ModelError::Hip)?;
                self.stream.synchronize().map_err(ModelError::Hip)?;
            }
        }

        // When the last layer is a standalone MoeFfn, no dense span has is_last=true,
        // so the final norm + LM head was never emitted. Compute it now.
        if self.config.layers[num_layers - 1].layer_type == LayerType::MoeFfn {
            let bufs = self.prefill_bufs.take().unwrap();
            let mk = MegakernelProgram::compile_final_norm_lm_head(
                self, std::sync::Arc::clone(&megakernel_module), &bufs, n,
            ).map_err(ModelError::Hip)?;
            self.prefill_bufs = Some(bufs);
            mk.execute(&self.stream).map_err(ModelError::Hip)?;
            self.stream.synchronize().map_err(ModelError::Hip)?;
        }

        Ok(())
    }

    fn prefill_batched(&mut self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        use crate::megakernel::{CHUNK_TOKENS, MegakernelProgram, PrefillBuffers};

        // Lazy-alloc prefill buffers.
        if self.prefill_bufs.is_none() {
            self.prefill_bufs = Some(
                PrefillBuffers::alloc(self.device, &self.config, CHUNK_TOKENS)
                    .map_err(ModelError::Hip)?,
            );
        }

        let mut logits = vec![0.0f32; self.config.vocab_size];
        let total = tokens.len();
        let mut offset = 0;

        if self.has_moe {
            // Mixed path: batch non-MoE spans, sequential MoE layers
            while offset < total {
                let end = (offset + CHUNK_TOKENS).min(total);
                let chunk = &tokens[offset..end];
                let start_pos = self.seq_len + offset as u32;
                self.prefill_mixed_chunk(chunk, start_pos)?;
                offset = end;
            }
        } else {
            // Pure non-MoE path: use original compile_prefill
            while offset < total {
                let end = (offset + CHUNK_TOKENS).min(total);
                let chunk = &tokens[offset..end];
                let start_pos = self.seq_len + offset as u32;

                // take() avoids split-borrow: compile/update take &Model + &mut PrefillBuffers
                let mut bufs = self.prefill_bufs.take().unwrap();

                if chunk.len() == CHUNK_TOKENS {
                    // Full chunk: compile once and cache, then patch per chunk.
                    if self.megakernel_prefill.is_none() {
                        let mk = MegakernelProgram::compile_prefill(self, chunk, start_pos, &mut bufs)
                            .map_err(ModelError::Hip)?;
                        self.megakernel_prefill = Some(mk);
                    } else {
                        let mk = self.megakernel_prefill.as_mut().unwrap();
                        mk.update_prefill_chunk(chunk, start_pos, &mut bufs).map_err(ModelError::Hip)?;
                    }
                    self.prefill_bufs = Some(bufs);
                    let mk = self.megakernel_prefill.as_ref().unwrap();
                    mk.execute(&self.stream).map_err(ModelError::Hip)?;
                } else {
                    // Partial last chunk: cache by token count to avoid recompile on repeated prompts.
                    let n = chunk.len();
                    if self.megakernel_prefill_partial_n == n
                        && self.megakernel_prefill_partial.is_some()
                    {
                        let mk = self.megakernel_prefill_partial.as_mut().unwrap();
                        mk.update_prefill_chunk(chunk, start_pos, &mut bufs).map_err(ModelError::Hip)?;
                    } else {
                        let mk = MegakernelProgram::compile_prefill(self, chunk, start_pos, &mut bufs)
                            .map_err(ModelError::Hip)?;
                        self.megakernel_prefill_partial = Some(mk);
                        self.megakernel_prefill_partial_n = n;
                    }
                    self.prefill_bufs = Some(bufs);
                    let mk = self.megakernel_prefill_partial.as_ref().unwrap();
                    mk.execute(&self.stream).map_err(ModelError::Hip)?;
                }

                self.stream.synchronize().map_err(ModelError::Hip)?;
                offset = end;
            }
        }

        self.seq_len += total as u32;
        self.activations.logits.copy_to_host(&mut logits)?;
        Ok(logits)
    }

    /// Read all GDN recurrent state to host (for testing).
    pub fn read_gdn_state(&self) -> Result<Vec<Vec<f32>>, ModelError> {
        self.stream.synchronize()?;
        let mut result = Vec::with_capacity(self.gdn_states.len());
        for state in &self.gdn_states {
            let n = state.recurrent.len();
            let mut buf = vec![0.0f32; n];
            state.recurrent.copy_to_host(&mut buf)?;
            result.push(buf);
        }
        Ok(result)
    }

    /// Restore GDN recurrent states from a previously saved checkpoint slot.
    pub fn restore_recurrent_checkpoint(&mut self, slot: u32) -> Result<(), ModelError> {
        let pool = self
            .checkpoint_pool
            .as_ref()
            .ok_or_else(|| ModelError::MissingWeight("checkpoint_pool not initialized".into()))?;
        let mut recurrent_bufs: Vec<&mut DeviceBuffer<f32>> = self
            .gdn_states
            .iter_mut()
            .map(|s| &mut s.recurrent)
            .collect();
        let stream_raw = self.stream.raw();
        paged_kv::restore_checkpoint(pool, slot, &mut recurrent_bufs, stream_raw)?;
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn read_hidden(&self) -> Result<Vec<f32>, ModelError> {
        self.stream.synchronize()?;
        let mut buf = vec![0.0f32; self.config.hidden_size];
        self.activations.hidden.copy_to_host(&mut buf)?;
        Ok(buf)
    }

    pub fn decode_step_traced(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<(Vec<f32>, Vec<(String, Vec<f32>)>), ModelError> {
        let hs = self.config.hidden_size as u32;
        let vs = self.config.vocab_size as u32;
        let mut traces: Vec<(String, Vec<f32>)> = Vec::new();
        if self
            .config
            .layers
            .iter()
            .any(|layer| layer.layer_type == LayerType::Attention)
        {
            self.append_paged_decode_token(position)?;
        }

        self.kernels.embedding.forward(
            &mut self.activations.hidden,
            &self.embed_weight,
            token_id as i32,
            hs,
            &self.stream,
        )?;
        traces.push(("embed".into(), self.read_hidden()?));

        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        for i in 0..self.config.num_layers {
            if self.config.layers[i].layer_type == LayerType::Attention {
                self.attention_forward(i, kv_idx, position)?;
                kv_idx += 1;
            } else {
                self.gdn_forward(i, gdn_idx)?;
                gdn_idx += 1;
            }
            traces.push((format!("layer_{i}"), self.read_hidden()?));
        }

        unsafe {
            d2d_copy_f32(
                &mut self.activations.normed,
                0,
                &self.activations.hidden,
                0,
                hs as usize,
                &self.stream,
            )?;
        }
        self.kernels.rmsnorm.forward(
            &mut self.activations.hidden,
            &self.activations.normed,
            &self.final_norm_weight,
            1,
            hs,
            self.config.rms_norm_eps,
            self.config.rms_norm_one_plus_w,
            &self.stream,
        )?;
        traces.push(("final_norm".into(), self.read_hidden()?));

        self.kernels.lm_head.forward(
            &mut self.activations.logits,
            &self.embed_weight,
            &self.activations.hidden,
            vs,
            hs,
            &self.stream,
        )?;
        self.stream.synchronize()?;
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;
        self.seq_len = position + 1;
        Ok((logits, traces))
    }

    fn read_buf(&self, buf: &DeviceBuffer<f32>) -> Result<Vec<f32>, ModelError> {
        self.stream.synchronize()?;
        let mut v = vec![0.0f32; buf.len()];
        buf.copy_to_host(&mut v)?;
        Ok(v)
    }

    pub fn gdn_layer0_trace(
        &mut self,
        token_id: u32,
    ) -> Result<Vec<(String, Vec<f32>)>, ModelError> {
        let hs = self.config.hidden_size as u32;
        let nh = self.config.linear_num_heads as u32;
        let kd = self.config.linear_key_head_dim as u32;
        let vd = self.config.linear_value_head_dim as u32;
        let ck = self.config.linear_conv_kernel_dim as u32;
        let eps = self.config.rms_norm_eps;
        let mut traces: Vec<(String, Vec<f32>)> = Vec::new();

        // Embedding
        self.kernels.embedding.forward(
            &mut self.activations.hidden,
            &self.embed_weight,
            token_id as i32,
            hs,
            &self.stream,
        )?;
        traces.push(("embed".into(), self.read_hidden()?));

        let weights = match &self.layers[0] {
            LayerWeights::Gdn(w) => w as *const GdnLayerWeights,
            _ => panic!("layer 0 not GDN"),
        };
        let w = unsafe { &*weights };

        // RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed,
            &self.activations.hidden,
            &w.input_norm,
            1,
            hs,
            eps,
            self.config.rms_norm_one_plus_w,
            &self.stream,
        )?;
        traces.push(("normed".into(), self.read_buf(&self.activations.normed)?));

        let nvh_traced = self.config.linear_num_value_heads as u32;
        let gqa_traced = nvh_traced / nh;

        // QKV projection
        w.w_qkv.forward(
            &self.kernels.linear_proj,
            &mut self.activations.qkv,
            &self.activations.normed,
            nh * kd * 2 + nvh_traced * vd,
            hs,
            &self.stream,
        )?;
        traces.push(("qkv_pre_conv".into(), self.read_buf(&self.activations.qkv)?));

        // a, b, z projections
        w.w_a.forward(
            &self.kernels.linear_proj,
            &mut self.activations.a_proj,
            &self.activations.normed,
            nvh_traced,
            hs,
            &self.stream,
        )?;
        w.w_b.forward(
            &self.kernels.linear_proj,
            &mut self.activations.b_proj,
            &self.activations.normed,
            nvh_traced,
            hs,
            &self.stream,
        )?;
        w.w_z.forward(
            &self.kernels.linear_proj,
            &mut self.activations.z_proj,
            &self.activations.normed,
            nvh_traced * vd,
            hs,
            &self.stream,
        )?;
        traces.push(("a_proj".into(), self.read_buf(&self.activations.a_proj)?));
        traces.push(("b_proj".into(), self.read_buf(&self.activations.b_proj)?));
        traces.push(("z_proj".into(), self.read_buf(&self.activations.z_proj)?));

        // Conv1d: split qkv, run 3 separate convs, reassemble
        let conv_q_len = (nh * kd) as usize;
        let conv_k_len = (nh * kd) as usize;
        let conv_v_len = (nvh_traced * vd) as usize;
        let ck_usize = ck as usize;

        unsafe {
            d2d_copy_f32(
                &mut self.activations.q_gdn,
                0,
                &self.activations.qkv,
                0,
                conv_q_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.k_gdn,
                0,
                &self.activations.qkv,
                conv_q_len,
                conv_k_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.v_gdn,
                0,
                &self.activations.qkv,
                conv_q_len + conv_k_len,
                conv_v_len,
                &self.stream,
            )?;
        }

        let mut conv_w_q = DeviceBuffer::<u16>::alloc(self.device, conv_q_len * ck_usize)?;
        let mut conv_w_k = DeviceBuffer::<u16>::alloc(self.device, conv_k_len * ck_usize)?;
        let mut conv_w_v = DeviceBuffer::<u16>::alloc(self.device, conv_v_len * ck_usize)?;
        unsafe {
            let conv1d = w.conv1d_weight.as_ref().expect("conv1d_weight required for trace (set TRACE env var)");
            d2d_copy_u16(
                &mut conv_w_q,
                0,
                conv1d,
                0,
                conv_q_len * ck_usize,
                &self.stream,
            )?;
            d2d_copy_u16(
                &mut conv_w_k,
                0,
                conv1d,
                conv_q_len * ck_usize,
                conv_k_len * ck_usize,
                &self.stream,
            )?;
            d2d_copy_u16(
                &mut conv_w_v,
                0,
                conv1d,
                (conv_q_len + conv_k_len) * ck_usize,
                conv_v_len * ck_usize,
                &self.stream,
            )?;
        }

        let conv_state_q_len = conv_q_len * (ck_usize - 1);
        let conv_state_k_len = conv_k_len * (ck_usize - 1);
        let conv_state_v_len = conv_v_len * (ck_usize - 1);

        let mut cs_q = DeviceBuffer::<f32>::alloc(self.device, conv_state_q_len)?;
        let mut cs_k = DeviceBuffer::<f32>::alloc(self.device, conv_state_k_len)?;
        let mut cs_v = DeviceBuffer::<f32>::alloc(self.device, conv_state_v_len)?;
        unsafe {
            d2d_copy_f32(
                &mut cs_q,
                0,
                &self.gdn_conv_states[0],
                0,
                conv_state_q_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut cs_k,
                0,
                &self.gdn_conv_states[0],
                conv_state_q_len,
                conv_state_k_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut cs_v,
                0,
                &self.gdn_conv_states[0],
                conv_state_q_len + conv_state_k_len,
                conv_state_v_len,
                &self.stream,
            )?;
        }

        let mut conv_out_q = DeviceBuffer::<f32>::alloc(self.device, conv_q_len)?;
        let mut conv_out_k = DeviceBuffer::<f32>::alloc(self.device, conv_k_len)?;
        let mut conv_out_v = DeviceBuffer::<f32>::alloc(self.device, conv_v_len)?;

        self.kernels.causal_conv1d.forward(
            &mut cs_q,
            &self.activations.q_gdn,
            &conv_w_q,
            &mut conv_out_q,
            conv_q_len as u32,
            ck,
            &self.stream,
        )?;
        self.kernels.causal_conv1d.forward(
            &mut cs_k,
            &self.activations.k_gdn,
            &conv_w_k,
            &mut conv_out_k,
            conv_k_len as u32,
            ck,
            &self.stream,
        )?;
        self.kernels.causal_conv1d.forward(
            &mut cs_v,
            &self.activations.v_gdn,
            &conv_w_v,
            &mut conv_out_v,
            conv_v_len as u32,
            ck,
            &self.stream,
        )?;

        traces.push(("conv_out_q".into(), self.read_buf(&conv_out_q)?));
        traces.push(("conv_out_k".into(), self.read_buf(&conv_out_k)?));
        traces.push(("conv_out_v".into(), self.read_buf(&conv_out_v)?));

        // Copy conv outputs to q/k/v
        unsafe {
            d2d_copy_f32(
                &mut self.activations.q_gdn,
                0,
                &conv_out_q,
                0,
                conv_q_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.k_gdn,
                0,
                &conv_out_k,
                0,
                conv_k_len,
                &self.stream,
            )?;
            d2d_copy_f32(
                &mut self.activations.v_gdn,
                0,
                &conv_out_v,
                0,
                conv_v_len,
                &self.stream,
            )?;
        }

        // Gate
        self.kernels.gdn_gate.forward(
            &mut self.activations.gate_gdn,
            &w.a_log,
            &self.activations.a_proj,
            &w.dt_bias,
            nh,
            &self.stream,
        )?;
        traces.push(("gate".into(), self.read_buf(&self.activations.gate_gdn)?));

        // Recurrent
        self.kernels.gdn_recurrent_v2.forward(
            &self.activations.q_gdn,
            &self.activations.k_gdn,
            &self.activations.v_gdn,
            &self.activations.gate_gdn,
            &self.activations.b_proj,
            &mut self.gdn_states[0].recurrent,
            &mut self.activations.recurrent_out,
            nvh_traced,
            kd,
            vd,
            gqa_traced,
            &self.stream,
        )?;
        traces.push((
            "recurrent_out".into(),
            self.read_buf(&self.activations.recurrent_out)?,
        ));

        // RMSNormGated
        self.kernels.rmsnorm_gated.forward(
            &mut self.activations.normed_gated,
            &self.activations.recurrent_out,
            &self.activations.z_proj,
            &w.output_norm,
            nvh_traced,
            vd,
            eps,
            &self.stream,
        )?;
        traces.push((
            "normed_gated".into(),
            self.read_buf(&self.activations.normed_gated)?,
        ));

        // out_proj
        w.w_out.forward(
            &self.kernels.linear_proj,
            &mut self.activations.out_proj,
            &self.activations.normed_gated,
            hs,
            nvh_traced * vd,
            &self.stream,
        )?;
        traces.push((
            "out_proj".into(),
            self.read_buf(&self.activations.out_proj)?,
        ));

        // Residual
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
        traces.push(("after_residual".into(), self.read_hidden()?));

        Ok(traces)
    }

    pub fn reset_state(&mut self) -> Result<(), ModelError> {
        // Persistent worker holds all GPU CUs — must shut it down before any hipMemcpy.
        // It will be re-launched lazily on the next decode_step call.
        drop(self.persistent_workers.take());

        let nh = self.config.linear_num_heads;
        let kd = self.config.linear_key_head_dim;
        let vd = self.config.linear_value_head_dim;
        let ck = self.config.linear_conv_kernel_dim;
        let nvh_r = self.config.linear_num_value_heads;
        let qkv_out = nh * kd * 2 + nvh_r * vd;

        for state in &mut self.gdn_states {
            let zeros = vec![0.0f32; nvh_r * kd * vd];
            state.recurrent.copy_from_host(&zeros)?;
        }
        for conv_state in &mut self.gdn_conv_states {
            let zeros = vec![0.0f32; qkv_out * (ck - 1)];
            conv_state.copy_from_host(&zeros)?;
        }
        if let Some(caches) = self.legacy_kv_caches.as_mut() {
            let kv_size = self.config.max_seq_len * self.config.num_kv_heads * self.config.head_dim;
            let zeros_kv = vec![0.0f32; kv_size];
            for cache in caches {
                cache.k.copy_from_host(&zeros_kv)?;
                cache.v.copy_from_host(&zeros_kv)?;
            }
        }
        if let RecurrentLayerKind::Mamba2 { num_heads: m_nh, head_dim: m_hd, state_dim: m_sd, conv_kernel: m_ck, conv_dim: m_cd, .. } = &self.config.recurrent_kind {
            for state in &mut self.mamba2_states {
                state.ssm.copy_from_host(&vec![0.0f32; m_nh * m_hd * m_sd])?;
                state.conv.copy_from_host(&vec![0.0f32; m_cd * (m_ck - 1)])?;
            }
        }
        self.seq_len = 0;
        if let Some(seq) = self.paged_seq.as_mut() {
            if let Some(q_alloc) = self.quant_allocator.as_mut() {
                seq.free_quant_slots(q_alloc);
            }
            if let Some(alloc) = self.page_allocator.as_mut() {
                seq.reset(alloc);
            }
        }
        Ok(())
    }
}
