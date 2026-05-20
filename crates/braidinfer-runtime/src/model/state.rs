use braidinfer_hip::memory::DeviceBuffer;

use crate::paged_kv::{self, RecurrentCheckpointPool};

use super::Model;
use super::ModelError;
use crate::config::*;

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
        // Persistent + paged (non-MoE): use paged prefill so decode sees KV in paged chunks.
        // Worker not yet started so hipMemcpy is still allowed.
        if self.persistent && !self.has_moe && self.persistent_workers.is_none() {
            return self.prefill_paged(tokens);
        }
        // MoE + persistent: mixed batched path (compile_prefill_segment + moe_ffn_forward).
        if self.has_moe && self.persistent_workers.is_none() {
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

    /// Prefill using paged KV cache (for persistent single-GPU non-MoE models).
    /// Populates paged chunks via sequential decode_step_paged so that
    /// decode_step_persistent can attend to the full prefill context.
    ///
    /// Note: this is O(N^2) decode steps (not batched). Batched paged prefill
    /// is tracked as a follow-up optimization (TODO: braidinfer-8gz follow-up).
    fn prefill_paged(&mut self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        let start_pos = self.seq_len;
        let mut logits = vec![0.0f32; self.config.vocab_size];

        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i as u32;
            logits = self.decode_step_paged(tok, pos)?;
        }

        // seq_len is incremented by each decode_step_paged call internally,
        // so we don't need to increment it again here.
        // But decode_step_paged calls set seq_len = position + 1 after each step,
        // so after the last step it should equal start_pos + total.
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
            let wd = self.watchdog.clone();
            let wd_state_dev = wd.register(self.device).map_err(ModelError::Hip)?;
            let mut wd_ptr: *mut std::ffi::c_void = wd_state_dev as *mut std::ffi::c_void;
            // megakernel_f32 signature: (program, num_inst, watchdog, op_profile).
            // op_profile may be null when profiling is disabled.
            let mut op_profile_ptr: *mut std::ffi::c_void =
                crate::op_profile::get_global() as *mut std::ffi::c_void;
            let mut args: [*mut std::ffi::c_void; 4] = [
                std::ptr::addr_of_mut!(prog_ptr).cast(),
                std::ptr::addr_of_mut!(num_inst).cast(),
                std::ptr::addr_of_mut!(wd_ptr).cast(),
                std::ptr::addr_of_mut!(op_profile_ptr).cast(),
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
                // 5ax fix: position_ids is a SHARED buffer; cached segment programs
                // do NOT re-write it on execute. Refresh before every execute so the
                // FIRST run of a previously-compiled program reads the correct positions
                // (otherwise stale values from the LAST compiled program leak into
                // subsequent runs — manifests as MROPE reading wrong pos at token 0).
                bufs.write_positions(start_pos, n).map_err(ModelError::Hip)?;
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
                // 5ax fix: refresh position_ids before every execute (cached programs
                // share the prefill_bufs.position_ids buffer).
                bufs.write_positions(start_pos, n).map_err(ModelError::Hip)?;
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
        // Ensure prefill writes to legacy_kv_caches have completed on GPU 0
        // before the broadcast reads them. Safe — no cooperative kernel runs
        // on GPU 0 between prefill end and decode start (persistent_worker is
        // launched lazily on first decode call).
        self.stream.synchronize().map_err(ModelError::Hip)?;
        // Fix braidinfer-sew: multi-GPU prefill wrote K/V only to GPU 0's
        // legacy_kv_caches; broadcast positions 0..total to each worker's
        // attn_kv_caches so the head-parallel decode path reads valid
        // (non-uninit) keys/values for prefill positions.
        if let Some(mgpu) = self.multi_gpu.as_ref() {
            let has_attn_kv = !mgpu.workers.is_empty() && !mgpu.workers[0].attn_kv_caches.is_empty();
            if has_attn_kv {
                if let Some(legacy_kv) = self.legacy_kv_caches.as_ref() {
                    use crate::config::LayerType;
                    // Build attn_i → kv_i mapping. legacy_kv_caches is indexed by
                    // attention-layer order (kv_idx in compile_prefill); attn_kv_caches
                    // is indexed by attention-layer occurrence in cfg.layers.
                    let attn_to_kv: Vec<usize> = self
                        .config
                        .layers
                        .iter()
                        .enumerate()
                        .filter(|(_, l)| l.layer_type == LayerType::Attention)
                        .enumerate()
                        .map(|(_attn_i, (_layer_i, _))| _attn_i)
                        .collect();
                    mgpu.broadcast_prefill_kv_to_workers(
                        legacy_kv,
                        &attn_to_kv,
                        self.config.num_kv_heads,
                        self.config.head_dim,
                        self.config.max_seq_len,
                        total,
                    )
                    .map_err(ModelError::Hip)?;
                    // kv-unify Phase 1 (pc3h.1): legacy_kv_caches is dead memory
                    // on GPU 0 after broadcast in multi-GPU MoE — decode reads
                    // exclusively from worker.attn_kv_caches via head-parallel.
                    // Free to reclaim ~num_attn_layers × num_kv_heads × max_seq_len
                    // × head_dim × 8 bytes (K+V) on GPU 0 (~320MB for 35B-A3B at
                    // max_seq_len=8192).
                    self.legacy_kv_caches = None;
                    // kv-unify Phase 2 (pc3h.2): prefill_bufs scratch is dead
                    // after prefill on the multi-GPU MoE decode path — decode
                    // never invokes compile_prefill*. Free to reclaim per-chunk
                    // scratch (hidden, normed, qkv, ffn_act, etc).
                    self.prefill_bufs = None;
                    // Also drop cached prefill megakernel programs.
                    self.megakernel_prefill = None;
                    self.megakernel_prefill_partial = None;
                    self.megakernel_prefill_segments.clear();
                }
            }
        }
        self.activations.logits.copy_to_host(&mut logits)?;
        Ok(logits)
    }

    /// Read all GDN recurrent state to host (for testing).
    /// PRE-WORKER-ONLY: calls `copy_to_host` (hipMemcpy) — must NOT be called
    /// while the persistent cooperative worker is running (CUs held → deadlock).
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

    /// Read GDN conv1d state buffers (per-layer) for diagnostic inspection.
    pub fn read_gdn_conv_state(&self) -> Result<Vec<Vec<f32>>, ModelError> {
        self.stream.synchronize()?;
        let mut result = Vec::with_capacity(self.gdn_conv_states.len());
        for state in &self.gdn_conv_states {
            let n = state.len();
            let mut buf = vec![0.0f32; n];
            state.copy_to_host(&mut buf)?;
            result.push(buf);
        }
        Ok(result)
    }

    /// Read legacy_kv_caches K and V tensors per layer (diagnostic).
    /// Returns Vec<(K_layer, V_layer)>; empty Vec if legacy_kv_caches not initialized.
    /// Used by bench_coherence to compare K/V cache contents across runs (track 5ax).
    pub fn read_legacy_kv_caches(&self) -> Result<Vec<(Vec<f32>, Vec<f32>)>, ModelError> {
        self.stream.synchronize()?;
        let Some(caches) = self.legacy_kv_caches.as_ref() else {
            return Ok(Vec::new());
        };
        let mut result = Vec::with_capacity(caches.len());
        for cache in caches {
            let n = cache.k.len();
            let mut k_buf = vec![0.0f32; n];
            let mut v_buf = vec![0.0f32; n];
            cache.k.copy_to_host(&mut k_buf)?;
            cache.v.copy_to_host(&mut v_buf)?;
            result.push((k_buf, v_buf));
        }
        Ok(result)
    }

    /// Read per-GPU attn_kv_caches for the first attention layer, all KV heads,
    /// positions [0..max_pos). Returns Vec of (gpu_idx, k_slice, v_slice).
    /// Diagnostic for braidinfer-sew: compares against read_legacy_kv_caches[0]
    /// to verify the prefill broadcast reached every worker AND decode-time
    /// writes match the broadcast format.
    pub fn read_attn_kv_first_layer(
        &self,
        max_pos: usize,
    ) -> Result<Vec<(usize, Vec<f32>, Vec<f32>)>, ModelError> {
        self.stream.synchronize()?;
        let Some(mgpu) = self.multi_gpu.as_ref() else {
            return Ok(Vec::new());
        };
        let nkh = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_sl = self.config.max_seq_len;
        let mut result = Vec::with_capacity(mgpu.num_devices);
        for (gpu_i, worker) in mgpu.workers.iter().enumerate() {
            if worker.attn_kv_caches.is_empty() { continue; }
            let kv = &worker.attn_kv_caches[0];
            let k_full_len = kv.k.len();
            let mut k_full = vec![0.0f32; k_full_len];
            let mut v_full = vec![0.0f32; k_full_len];
            kv.k.copy_to_host(&mut k_full)?;
            kv.v.copy_to_host(&mut v_full)?;
            // Extract positions [0..max_pos) for each head into a flat slice.
            // Layout: [nkh, max_sl, hd]; we want [nkh, max_pos, hd].
            let mut k_slice = Vec::with_capacity(nkh * max_pos * hd);
            let mut v_slice = Vec::with_capacity(nkh * max_pos * hd);
            for h in 0..nkh {
                let base = h * max_sl * hd;
                let want = max_pos * hd;
                k_slice.extend_from_slice(&k_full[base..base + want]);
                v_slice.extend_from_slice(&v_full[base..base + want]);
            }
            result.push((gpu_i, k_slice, v_slice));
        }
        Ok(result)
    }

    /// Read KV chunk pool slot 0 contents (raw bytes) for diagnostic inspection.
    /// Returns empty Vec if page_allocator is not initialized (e.g., multi-GPU non-paged path).
    /// PRE-WORKER-ONLY: calls `memcpy_d2h` — must NOT be called while the persistent
    /// cooperative worker is running (CUs held → deadlock).
    pub fn read_kv_chunk_slot0(&self) -> Result<Vec<u8>, ModelError> {
        self.stream.synchronize()?;
        let Some(alloc) = self.page_allocator.as_ref() else {
            return Ok(Vec::new());
        };
        let chunk_bytes = alloc.chunk_bytes();
        let slot0_ptr = alloc.slot_ptr(0);
        let mut buf = vec![0u8; chunk_bytes];
        braidinfer_hip::memory::memcpy_d2h(&mut buf, slot0_ptr, chunk_bytes)
            .map_err(ModelError::Hip)?;
        Ok(buf)
    }

    /// Enable per-instruction activation dump on the paged megakernel.
    /// Call BEFORE the first decode_step_paged of each run. Capacity = max number of
    /// dump slots (one per OP per kernel-launch). For a 24-layer hybrid running 2 steps,
    /// ~50-100 ops per launch × 2 launches → 200 slots is generous.
    pub fn enable_paged_dump(&mut self, max_slots: i32) -> Result<(), ModelError> {
        let Some(mk) = self.megakernel_paged.as_mut() else {
            return Err(ModelError::MissingWeight(
                "megakernel_paged not initialized — call decode_step_paged once first".into(),
            ));
        };
        mk.enable_dump(max_slots).map_err(ModelError::Hip)
    }

    /// Read the per-instruction dump from the paged megakernel.
    /// Returns Vec<(opcode, inst_idx, output_data)> in the order they were dumped.
    pub fn read_paged_dump(&self) -> Result<Vec<(u32, u32, Vec<f32>)>, ModelError> {
        let Some(mk) = self.megakernel_paged.as_ref() else {
            return Err(ModelError::MissingWeight("megakernel_paged not initialized".into()));
        };
        mk.read_dump(&self.stream).map_err(ModelError::Hip)
    }

    /// Get the human-readable opcode name for a dumped op (for diagnostic printing).
    pub fn opcode_name(op: u32) -> String {
        crate::megakernel::opcode_name_str(op)
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

    /// Lane C Exp 1 / bd 4e2m: bring up persistent workers without running
    /// prefill, then dispatch one OP_D2D_COPY per live worker mailbox and
    /// verify the round-trip via host-mapped UC buffers.
    ///
    /// Returns Ok(per-worker-timing-string) on full success; Err(diagnostic)
    /// when any worker's verify fails. Caller is expected to drop+reload the
    /// Model on Err (same retry shape as full-decode warmup-discard).
    ///
    /// Goal: test whether prefill (which exercises MoE FFN dispatch through
    /// each worker's mailbox repeatedly per layer) is load-bearing for the
    /// cold-start cure, or whether a SINGLE op_d2d_copy round-trip per worker
    /// is sufficient. If 30/30 with sub-100ms cost: prefill is overkill and
    /// minimal-mailbox alone is the cure. If <30/30: prefill is doing more
    /// than just first-mailbox-transaction warming.
    ///
    /// Gated by BRAIDINFER_WARMUP_MODE=mailbox-only in generate.rs.
    pub fn minimal_mailbox_warmup_no_prefill(&mut self) -> Result<String, String> {
        use crate::megakernel::instructions::D2dCopyInst;
        use braidinfer_hip::memory::MappedHostBuffer;

        // bd 4e2m Lane 1 D1 (revised): mailbox-only warmup only applies to
        // multi-GPU mode. Single-GPU mode has no cross-GPU mailbox race
        // (no peer reads), and spawning the persistent_worker BEFORE prefill
        // breaks prefill's lazy paged-KV init (prefill_paged uses hipMemcpy
        // which deadlocks once a cooperative kernel is holding all CUs).
        // For single-GPU, the caller should fall back to full-decode warmup.
        if self.multi_gpu.is_none() {
            return Err("single-gpu-fallback".into());
        }

        let t_spawn = std::time::Instant::now();
        self.ensure_moe_workers_started()
            .map_err(|e| format!("ensure_moe_workers_started: {e:?}"))?;
        self.init_multi_gpu_persistent()
            .map_err(|e| format!("init_multi_gpu_persistent: {e:?}"))?;
        let spawn_ms = t_spawn.elapsed().as_secs_f64() * 1000.0;

        let dispatch = self
            .persistent_workers
            .as_mut()
            .ok_or_else(|| "no persistent_workers after spawn".to_string())?;

        let gpu_count = dispatch.workers.len();
        let live: Vec<usize> = (0..gpu_count).filter(|&i| dispatch.has_worker(i)).collect();
        if live.is_empty() {
            return Err("no live workers after spawn".into());
        }

        let src: MappedHostBuffer<f32> = MappedHostBuffer::alloc_portable_coherent(4)
            .map_err(|e| format!("alloc src: {e:?}"))?;
        let mut dst: MappedHostBuffer<f32> = MappedHostBuffer::alloc_portable_coherent(4)
            .map_err(|e| format!("alloc dst: {e:?}"))?;

        let expected = [1.0f32, 2.0, 3.0, 4.0];
        unsafe {
            std::slice::from_raw_parts_mut(src.host_ptr(), 4).copy_from_slice(&expected);
        }
        let sentinel = f32::from_bits(0xDEADBEEFu32);

        let mut diag = Vec::new();
        let mut per_worker = Vec::new();
        let t_dispatch = std::time::Instant::now();
        for &gpu_idx in &live {
            unsafe {
                for x in std::slice::from_raw_parts_mut(dst.host_ptr(), 4) {
                    *x = sentinel;
                }
            }
            let inst = D2dCopyInst::new(1, dst.as_mut_ptr(), src.device_ptr() as *const f32, 4)
                .into_inst();
            let t_one = std::time::Instant::now();
            let seq = dispatch.dispatch_batch_fire(gpu_idx, &[inst]);
            dispatch.wait_ack(gpu_idx, seq);
            let one_us = t_one.elapsed().as_micros();
            let got: [f32; 4] = unsafe {
                let s = std::slice::from_raw_parts(dst.host_ptr(), 4);
                [s[0], s[1], s[2], s[3]]
            };
            if got != expected {
                diag.push(format!(
                    "gpu{}: got [{:e},{:e},{:e},{:e}] want [1,2,3,4]",
                    gpu_idx, got[0], got[1], got[2], got[3]
                ));
            }
            per_worker.push(format!("gpu{}={}us", gpu_idx, one_us));
        }
        let dispatch_ms = t_dispatch.elapsed().as_secs_f64() * 1000.0;

        if !diag.is_empty() {
            return Err(format!(
                "spawn={:.1}ms; {}",
                spawn_ms,
                diag.join("; ")
            ));
        }
        Ok(format!(
            "spawn={:.1}ms dispatch={:.2}ms [{}]",
            spawn_ms,
            dispatch_ms,
            per_worker.join(",")
        ))
    }
}
