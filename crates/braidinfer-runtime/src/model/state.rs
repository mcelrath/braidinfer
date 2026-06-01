use braidinfer_hip::memory::DeviceBuffer;

use crate::paged_kv::{self, RecurrentCheckpointPool};

use super::Model;
use crate::weights::ModelError;
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
    ///
    /// bd 9gmh Phase 1: spawns the persistent worker at entry; all prefill paths
    /// route through the worker mailbox (compile_prefill* upload is a 1-element
    /// placeholder, mk.dispatch_via_worker reads instructions from CPU memory
    /// directly into WorkerQueue::inst[]). No hipMemcpy on GPU 0 once the worker
    /// is up — moe_forward.rs's three host-launched functions are migrated to
    /// mailbox-dispatched megakernel programs in this same commit.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::MissingWeight("empty token sequence".into()));
        }
        // Spawn the persistent worker before any prefill work.
        self.ensure_persistent_worker_spawned()?;
        if self.has_moe {
            self.prefill_batched(tokens)
        } else {
            self.prefill_paged(tokens)
        }
    }

    /// Prefill using paged KV cache (single-GPU dense / non-MoE).
    ///
    /// bd srg6.7 (Phase 3e): batched paged-prefill via
    /// `compile_prefill_paged_persistent`. Replaces the previous per-token
    /// decode_step loop (bd 0hp1, O(N) mailbox round-trips) with a single
    /// compiled paged-prefill program per CHUNK_TOKENS slab.
    fn prefill_paged(&mut self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        use crate::megakernel::instructions::*;
        use crate::megakernel::{CHUNK_TOKENS, Instruction, MegakernelProgram, PrefillBuffers};
        use braidinfer_hip::memory::MappedHostBuffer;
        use std::sync::Arc;

        let start_pos_full = self.seq_len;

        // Lazy-init paged page/position tables (DeviceBuffer ones, currently
        // dead — braidinfer-c7w2) and page_allocator/paged_seq.
        self.ensure_paged_decode_state(false)?;

        // Lazy-alloc prefill scratch (hidden / qkv / etc.).
        if self.prefill_bufs.is_none() {
            self.prefill_bufs = Some(
                PrefillBuffers::alloc(self.device, &self.config, CHUNK_TOKENS)
                    .map_err(ModelError::Hip)?,
            );
        }

        // Lazy-alloc host-mapped page/position tables for the paged-prefill writer.
        let max_chunks = self.max_paged_chunks();
        if self.prefill_paged_page_table.is_none() {
            self.prefill_paged_page_table = Some(
                MappedHostBuffer::<u64>::alloc(max_chunks).map_err(ModelError::Hip)?,
            );
        }
        if self.prefill_paged_position_table.is_none() {
            self.prefill_paged_position_table = Some(
                MappedHostBuffer::<i32>::alloc(3 * self.config.max_seq_len).map_err(ModelError::Hip)?,
            );
        }

        let megakernel_module = Arc::new(
            braidinfer_hip::module::Module::load(
                self.device,
                &crate::kernel::kernel_dir().join("megakernel.hsaco"),
            )
            .map_err(ModelError::Hip)?,
        );

        let hs = self.config.hidden_size;
        let mut offset = 0usize;
        while offset < tokens.len() {
            let end = (offset + CHUNK_TOKENS).min(tokens.len());
            let chunk = &tokens[offset..end];
            let chunk_start_pos = start_pos_full + offset as u32;
            let n = chunk.len();

            // Step 1: emit embeddings into prefill_bufs.hidden via mailbox
            // (mirror prefill_mixed_chunk:116-135 pattern).
            {
                let grid_x = (hs as u32 + 255) / 256;
                let mut insts: Vec<Instruction> = Vec::with_capacity(n);
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
                let dispatch = self.persistent_workers.as_mut()
                    .expect("persistent_workers must be initialized in Model::prefill");
                dispatch.dispatch_batch_slice(self.device.0 as usize, &insts);
            }

            // Step 2: compile + dispatch the paged-prefill program for this chunk.
            let mut bufs = self.prefill_bufs.take().unwrap();
            let mut seq = self.paged_seq.take().unwrap();
            let mut allocator = self.page_allocator.take().unwrap();

            let mut mk = {
                let page_table = self.prefill_paged_page_table.as_ref().unwrap();
                let pos_table = self.prefill_paged_position_table.as_ref().unwrap();
                MegakernelProgram::compile_prefill_paged_persistent(
                    self,
                    Arc::clone(&megakernel_module),
                    chunk,
                    chunk_start_pos,
                    &mut seq,
                    &mut allocator,
                    page_table,
                    pos_table,
                    &mut bufs,
                )
                .map_err(ModelError::Hip)?
            };

            self.prefill_bufs = Some(bufs);
            self.paged_seq = Some(seq);
            self.page_allocator = Some(allocator);

            // Enable dump buffer for tracing (no-op when tracer disabled).
            if self.tracer.enabled() && !mk.dump_active() {
                let max_slots = (mk.instructions.len() as i32).min(4096);
                if let Ok(()) = mk.enable_dump_persistent(max_slots) {
                    let dispatch = self.persistent_workers.as_mut().unwrap();
                    dispatch.set_trace_dump_ptrs(self.device.0 as usize, &mk);
                }
            }
            let dispatch = self.persistent_workers.as_mut()
                .expect("persistent_workers must be initialized in Model::prefill");
            mk.dispatch_via_worker(dispatch, self.device.0 as usize)
                .map_err(ModelError::Hip)?;
            // Drain trace dump after dispatch completes.
            if self.tracer.enabled() {
                let gpu_idx = self.device.0 as usize;
                let dispatch = self.persistent_workers.as_mut().unwrap();
                let _ = dispatch.drain_trace_dump(gpu_idx, &mk, &mut self.tracer);
            }

            offset = end;
        }

        self.seq_len += tokens.len() as u32;

        // Read last-token logits from act.logits via the persistent dispatch's
        // SDMA stream (CU-free; the persistent worker still holds compute CUs).
        // Mirror of prefill_batched:402-426.
        let mut logits = vec![0.0f32; self.config.vocab_size];
        let gpu_idx = self.device.0 as usize;
        let stream = self
            .persistent_workers
            .as_ref()
            .map(|d| d.sdma_stream(gpu_idx))
            .unwrap_or(std::ptr::null_mut());
        if stream.is_null() {
            self.activations.logits.copy_to_host(&mut logits)?;
        } else {
            braidinfer_hip::error::check(unsafe {
                braidinfer_hip::ffi::hipMemcpyAsync(
                    logits.as_mut_ptr() as *mut std::ffi::c_void,
                    self.activations.logits.as_ptr() as *const std::ffi::c_void,
                    logits.len() * std::mem::size_of::<f32>(),
                    braidinfer_hip::ffi::hipMemcpyDeviceToHost,
                    stream,
                )
            })
            .map_err(ModelError::Hip)?;
            braidinfer_hip::error::check(unsafe {
                braidinfer_hip::ffi::hipStreamSynchronize(stream)
            })
            .map_err(ModelError::Hip)?;
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
        use crate::megakernel::{CHUNK_TOKENS, PrefillBuffers};

        if self.prefill_bufs.is_none() {
            self.prefill_bufs = Some(
                PrefillBuffers::alloc(self.device, &self.config, CHUNK_TOKENS)
                    .map_err(ModelError::Hip)?,
            );
        }

        // Start MoE worker GPUs lazily (no-op if single-GPU or already started).
        // Must happen before MoE layer processing and before the decode persistent worker launches.
        self.ensure_moe_workers_started()?;

        // bd srg6.X3: MoE prefill uses paged KV segment compile path for
        // both single-GPU and multi-GPU. The flat path is retired; paged
        // broadcast to workers happens in prefill_batched after this returns.
        self.prefill_mixed_chunk_paged(chunk, start_pos)
    }

    /// MoE prefill via paged KV segment compile (srg6.10/srg6.15). Emits paged
    /// KV writes (AttentionVariant::PrefillPagedKv) for single- and multi-GPU MoE.
    ///
    /// Outer driver:
    ///   1. Ensure paged decode state (allocator + paged_seq).
    ///   2. append_token for ALL n prompt tokens (mirror srg6.5 invariant).
    ///   3. Embed all tokens into prefill_bufs.hidden via mailbox.
    ///   4. For each layer span (MoE-boundary-split): compile + dispatch a
    ///      paged segment program (NO cache — page_table contents would go
    ///      stale across prefills).
    ///   5. Between segments at MoE boundaries: CPU MoE dispatch
    ///      (moe_ffn_forward_prefill_batched), unchanged from flat path.
    fn prefill_mixed_chunk_paged(
        &mut self,
        chunk: &[u32],
        start_pos: u32,
    ) -> Result<(), ModelError> {
        use crate::config::LayerType;
        use crate::megakernel::{MegakernelProgram, instructions::*, Instruction};
        use braidinfer_hip::memory::MappedHostBuffer;

        // Lazy-init page_allocator + paged_seq.
        self.ensure_paged_decode_state(false)?;

        // Lazy-alloc host-mapped page/position tables (reuse the fields added
        // for single-GPU dense paged prefill in srg6.7).
        let max_chunks = self.max_paged_chunks();
        if self.prefill_paged_page_table.is_none() {
            self.prefill_paged_page_table = Some(
                MappedHostBuffer::<u64>::alloc(max_chunks).map_err(ModelError::Hip)?,
            );
        }
        if self.prefill_paged_position_table.is_none() {
            self.prefill_paged_position_table = Some(
                MappedHostBuffer::<i32>::alloc(3 * self.config.max_seq_len)
                    .map_err(ModelError::Hip)?,
            );
        }

        let n = chunk.len();
        let hs = self.config.hidden_size;
        let num_layers = self.config.num_layers;

        let megakernel_module = std::sync::Arc::new(
            braidinfer_hip::module::Module::load(
                self.device,
                &crate::kernel::kernel_dir().join("megakernel.hsaco"),
            ).map_err(ModelError::Hip)?,
        );

        // Step 1: append_token for ALL n prompt tokens BEFORE any segment
        // compile (light-review F2 lock).
        {
            let mut seq = self.paged_seq.take().unwrap();
            let mut allocator = self.page_allocator.take().unwrap();
            let mut host_alloc = self.host_page_allocator.take();
            for t in 0..n {
                let pos = start_pos as i32 + t as i32;
                seq.append_token(pos, &mut allocator, host_alloc.as_mut()).map_err(ModelError::Hip)?;
            }
            self.paged_seq = Some(seq);
            self.page_allocator = Some(allocator);
            self.host_page_allocator = host_alloc;
        }

        // Step 2: Embed all tokens into prefill_bufs.hidden via mailbox.
        {
            let grid_x = (hs as u32 + 255) / 256;
            let mut insts: Vec<Instruction> = Vec::with_capacity(n);
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
            let dispatch = self.persistent_workers.as_mut()
                .expect("persistent_workers must be initialized in Model::prefill");
            dispatch.dispatch_batch_slice(self.device.0 as usize, &insts);
        }

        // Step 3: Walk layers, splitting on MoE FFN boundaries. Each
        // contiguous non-MoE span compiles a paged segment program. MoE
        // layers dispatch CPU-side between segments.
        let mut layer_i = 0usize;
        while layer_i < num_layers {
            let lt = self.config.layers[layer_i].layer_type.clone();
            if lt == LayerType::MoeFfn {
                let mut bufs = self.prefill_bufs.take().unwrap();
                self.moe_ffn_forward_prefill_batched(layer_i, &mut bufs.hidden, n)
                    .map_err(ModelError::Hip)?;
                self.prefill_bufs = Some(bufs);
                // MoE Option-A: capture end-of-layer hidden (last token row) for MoeFfn layers.
                if self.tracer.enabled() {
                    let hs = self.config.hidden_size;
                    if let Some(ref bufs) = self.prefill_bufs {
                        let ptr = unsafe { bufs.hidden.as_ptr().add((n - 1) * hs) } as *const u8;
                        let _ = self.tracer.capture(0, crate::tracer::Probe::PostFfn { layer: layer_i }, ptr, hs * 4);
                    }
                }
                layer_i += 1;
            } else if matches!(self.config.layers[layer_i].ffn_type, FfnType::MoE { .. })
                && (lt == LayerType::Attention || lt == LayerType::Gdn)
            {
                // Attention/GDN with MoE FFN: 1-layer mixer segment, then CPU MoE.
                let span_start = layer_i;
                let span_end = layer_i + 1;
                let is_truly_last = span_end == num_layers;

                // Compile fresh (NO cache — light-review F1 lock).
                let mut bufs = self.prefill_bufs.take().unwrap();
                let seq = self.paged_seq.take().unwrap();
                let allocator = self.page_allocator.take().unwrap();
                let mut mk = {
                    let page_table = self.prefill_paged_page_table.as_ref().unwrap();
                    let pos_table = self.prefill_paged_position_table.as_ref().unwrap();
                    MegakernelProgram::compile_prefill_segment_paged(
                        self,
                        std::sync::Arc::clone(&megakernel_module),
                        chunk, start_pos,
                        span_start, span_end,
                        false, // never is_last: LM head runs after MoE below
                        &seq, &allocator,
                        page_table, pos_table,
                        &mut bufs,
                    ).map_err(ModelError::Hip)?
                };
                self.prefill_bufs = Some(bufs);
                self.paged_seq = Some(seq);
                self.page_allocator = Some(allocator);

                {
                    if self.tracer.enabled() && !mk.dump_active() {
                        let max_slots = (mk.instructions.len() as i32).min(4096);
                        if let Ok(()) = mk.enable_dump_persistent(max_slots) {
                            let dispatch = self.persistent_workers.as_mut().unwrap();
                            dispatch.set_trace_dump_ptrs(self.device.0 as usize, &mk);
                        }
                    }
                    let dispatch = self.persistent_workers.as_mut()
                        .expect("persistent_workers must be initialized in Model::prefill");
                    mk.dispatch_via_worker(dispatch, self.device.0 as usize)
                        .map_err(ModelError::Hip)?;
                    if self.tracer.enabled() {
                        let gpu_idx = self.device.0 as usize;
                        let dispatch = self.persistent_workers.as_mut().unwrap();
                        let _ = dispatch.drain_trace_dump(gpu_idx, &mk, &mut self.tracer);
                    }
                }

                // CPU MoE FFN dispatch.
                let mut bufs = self.prefill_bufs.take().unwrap();
                self.moe_ffn_forward_prefill_batched(layer_i, &mut bufs.hidden, n)
                    .map_err(ModelError::Hip)?;
                self.prefill_bufs = Some(bufs);

                // MoE Option-A: capture end-of-MoE-layer hidden (last token row).
                if self.tracer.enabled() {
                    let hs = self.config.hidden_size;
                    if let Some(ref bufs) = self.prefill_bufs {
                        let ptr = unsafe { bufs.hidden.as_ptr().add((n - 1) * hs) } as *const u8;
                        let _ = self.tracer.capture(0, crate::tracer::Probe::PostFfn { layer: layer_i }, ptr, hs * 4);
                    }
                }
                if is_truly_last {
                    let bufs = self.prefill_bufs.take().unwrap();
                    let mut mk = MegakernelProgram::compile_final_norm_lm_head(
                        self, std::sync::Arc::clone(&megakernel_module), &bufs, n,
                    ).map_err(ModelError::Hip)?;
                    self.prefill_bufs = Some(bufs);
                    if self.tracer.enabled() && !mk.dump_active() {
                        let max_slots = (mk.instructions.len() as i32).min(4096);
                        if let Ok(()) = mk.enable_dump_persistent(max_slots) {
                            let dispatch = self.persistent_workers.as_mut().unwrap();
                            dispatch.set_trace_dump_ptrs(self.device.0 as usize, &mk);
                        }
                    }
                    let dispatch = self.persistent_workers.as_mut()
                        .expect("persistent_workers must be initialized in Model::prefill");
                    mk.dispatch_via_worker(dispatch, self.device.0 as usize)
                        .map_err(ModelError::Hip)?;
                    if self.tracer.enabled() {
                        let gpu_idx = self.device.0 as usize;
                        let dispatch = self.persistent_workers.as_mut().unwrap();
                        let _ = dispatch.drain_trace_dump(gpu_idx, &mk, &mut self.tracer);
                    }
                }
                layer_i += 1;
            } else {
                // Dense non-MoE span.
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

                let mut bufs = self.prefill_bufs.take().unwrap();
                let seq = self.paged_seq.take().unwrap();
                let allocator = self.page_allocator.take().unwrap();
                let mut mk = {
                    let page_table = self.prefill_paged_page_table.as_ref().unwrap();
                    let pos_table = self.prefill_paged_position_table.as_ref().unwrap();
                    MegakernelProgram::compile_prefill_segment_paged(
                        self,
                        std::sync::Arc::clone(&megakernel_module),
                        chunk, start_pos,
                        span_start, span_end,
                        is_last,
                        &seq, &allocator,
                        page_table, pos_table,
                        &mut bufs,
                    ).map_err(ModelError::Hip)?
                };
                self.prefill_bufs = Some(bufs);
                self.paged_seq = Some(seq);
                self.page_allocator = Some(allocator);

                if self.tracer.enabled() && !mk.dump_active() {
                    let max_slots = (mk.instructions.len() as i32).min(4096);
                    if let Ok(()) = mk.enable_dump_persistent(max_slots) {
                        let dispatch = self.persistent_workers.as_mut().unwrap();
                        dispatch.set_trace_dump_ptrs(self.device.0 as usize, &mk);
                    }
                }
                let dispatch = self.persistent_workers.as_mut()
                    .expect("persistent_workers must be initialized in Model::prefill");
                mk.dispatch_via_worker(dispatch, self.device.0 as usize)
                    .map_err(ModelError::Hip)?;
                if self.tracer.enabled() {
                    let gpu_idx = self.device.0 as usize;
                    let dispatch = self.persistent_workers.as_mut().unwrap();
                    let _ = dispatch.drain_trace_dump(gpu_idx, &mk, &mut self.tracer);
                }
            }
        }

        // If last layer is standalone MoeFfn, emit final norm + LM head now.
        if self.config.layers[num_layers - 1].layer_type == LayerType::MoeFfn {
            // MoE Option-A: capture end-of-layer hidden for the last MoeFfn layer.
            if self.tracer.enabled() {
                let hs = self.config.hidden_size;
                let last_moe_layer = num_layers - 1;
                if let Some(ref bufs) = self.prefill_bufs {
                    let ptr = unsafe { bufs.hidden.as_ptr().add((n - 1) * hs) } as *const u8;
                    let _ = self.tracer.capture(0, crate::tracer::Probe::PostFfn { layer: last_moe_layer }, ptr, hs * 4);
                }
            }
            let bufs = self.prefill_bufs.take().unwrap();
            let mut mk = MegakernelProgram::compile_final_norm_lm_head(
                self, std::sync::Arc::clone(&megakernel_module), &bufs, n,
            ).map_err(ModelError::Hip)?;
            self.prefill_bufs = Some(bufs);
            if self.tracer.enabled() && !mk.dump_active() {
                let max_slots = (mk.instructions.len() as i32).min(4096);
                if let Ok(()) = mk.enable_dump_persistent(max_slots) {
                    let dispatch = self.persistent_workers.as_mut().unwrap();
                    dispatch.set_trace_dump_ptrs(self.device.0 as usize, &mk);
                }
            }
            let dispatch = self.persistent_workers.as_mut()
                .expect("persistent_workers must be initialized in Model::prefill");
            mk.dispatch_via_worker(dispatch, self.device.0 as usize)
                .map_err(ModelError::Hip)?;
            if self.tracer.enabled() {
                let gpu_idx = self.device.0 as usize;
                let dispatch = self.persistent_workers.as_mut().unwrap();
                let _ = dispatch.drain_trace_dump(gpu_idx, &mk, &mut self.tracer);
            }
        }

        Ok(())
    }

    fn prefill_batched(&mut self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        use crate::megakernel::{CHUNK_TOKENS, PrefillBuffers};

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

        // has_moe is always true when prefill_batched is called (prefill() routes
        // non-MoE to prefill_paged). Non-MoE path (compile_prefill) deleted in srg6.21.
        while offset < total {
            let end = (offset + CHUNK_TOKENS).min(total);
            let chunk = &tokens[offset..end];
            let start_pos = self.seq_len + offset as u32;
            self.prefill_mixed_chunk(chunk, start_pos)?;
            offset = end;
        }

        self.seq_len += total as u32;
        // Ensure prefill writes to the paged KV chunks have completed on GPU 0
        // before the broadcast reads them. Safe — no cooperative kernel runs
        // on GPU 0 between prefill end and decode start (persistent_worker is
        // launched lazily on first decode call).
        self.stream.synchronize().map_err(ModelError::Hip)?;
        // bd srg6.X3: broadcast paged KV chunks from GPU 0 to each worker.
        // prefill_mixed_chunk_paged (called above) populates GPU 0's paged_seq;
        // broadcast_paged_chunks_to_workers replicates every chunk to worker
        // paged_seq under GQA (KV heads replicated, not sliced).
        // The flat broadcast_prefill_kv_to_workers call is removed; decode now
        // reads exclusively from per-worker paged KV. Function body stays until
        // X5 deletes all attn_kv_caches sites.
        if let Some(mgpu) = self.multi_gpu.as_mut() {
            if !mgpu.workers.is_empty() {
                let seq = self.paged_seq.as_ref()
                    .expect("paged_seq must be initialized for multi-GPU MoE prefill");
                let alloc = self.page_allocator.as_ref()
                    .expect("page_allocator must be initialized for multi-GPU MoE prefill");
                mgpu.broadcast_paged_chunks_to_workers(seq, alloc)
                    .map_err(ModelError::Hip)?;
                // Free prefill scratch: paged decode never re-invokes compile_prefill*.
                self.prefill_bufs = None;
            }
        }
        // Persistent worker holds GPU 0 CUs — synchronous copy_to_host would
        // deadlock. Use the dispatch's SDMA stream (no CU contention).
        let gpu_idx = self.device.0 as usize;
        let stream = self
            .persistent_workers
            .as_ref()
            .map(|d| d.sdma_stream(gpu_idx))
            .unwrap_or(std::ptr::null_mut());
        if stream.is_null() {
            self.activations.logits.copy_to_host(&mut logits)?;
        } else {
            braidinfer_hip::error::check(unsafe {
                braidinfer_hip::ffi::hipMemcpyAsync(
                    logits.as_mut_ptr() as *mut std::ffi::c_void,
                    self.activations.logits.as_ptr() as *const std::ffi::c_void,
                    logits.len() * std::mem::size_of::<f32>(),
                    braidinfer_hip::ffi::hipMemcpyDeviceToHost,
                    stream,
                )
            })
            .map_err(ModelError::Hip)?;
            braidinfer_hip::error::check(unsafe {
                braidinfer_hip::ffi::hipStreamSynchronize(stream)
            })
            .map_err(ModelError::Hip)?;
        }
        Ok(logits)
    }

    /// Read all GDN recurrent state to host (for testing / checkpoint diagnostics).
    ///
    /// bd gnxs: under always-persistent mode the cooperative worker holds every
    /// GPU CU, so the prior `copy_to_host` path (synchronous `hipMemcpy` on the
    /// default stream) deadlocks. We now issue an async D2H copy on the
    /// persistent dispatch's per-GPU SDMA stream (no CU contention) and then
    /// synchronize that one stream. Mirrors the `kv_mirror_chunk` /
    /// `drain_kv_chunk_mirror` pattern used at chunk-seal in decode_step.
    ///
    /// Pre-worker callers (no decode/prefill yet) take the legacy path —
    /// `copy_to_host` is safe before the worker is spawned.
    pub fn read_gdn_state(&mut self) -> Result<Vec<Vec<f32>>, ModelError> {
        // Make sure the worker (and its SDMA stream) exist before we try the
        // async path. Idempotent.
        self.ensure_persistent_worker_spawned()?;

        let gpu_idx = self.device.0 as usize;
        let stream = self
            .persistent_workers
            .as_ref()
            .map(|d| d.sdma_stream(gpu_idx))
            .unwrap_or(std::ptr::null_mut());

        let mut result = Vec::with_capacity(self.gdn_states.len());
        if stream.is_null() {
            // Fallback: no SDMA stream available (pre-worker path). Safe to
            // use the synchronous copy here because the worker has not been
            // launched yet — the deadlock guard does nothing in that state.
            self.stream.synchronize()?;
            for state in &self.gdn_states {
                let n = state.recurrent.len();
                let mut buf = vec![0.0f32; n];
                state.recurrent.copy_to_host(&mut buf)?;
                result.push(buf);
            }
            return Ok(result);
        }

        // Issue async D2H copies onto the SDMA stream. Pinned-host buffering
        // isn't strictly required for correctness here (we synchronize before
        // returning), so we copy directly into the result Vecs.
        for state in &self.gdn_states {
            let n = state.recurrent.len();
            let mut buf = vec![0.0f32; n];
            braidinfer_hip::error::check(unsafe {
                braidinfer_hip::ffi::hipMemcpyAsync(
                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                    state.recurrent.as_ptr() as *const std::ffi::c_void,
                    n * std::mem::size_of::<f32>(),
                    braidinfer_hip::ffi::hipMemcpyDeviceToHost,
                    stream,
                )
            })
            .map_err(ModelError::Hip)?;
            result.push(buf);
        }
        // Synchronize SDMA stream — copies complete before we return.
        braidinfer_hip::error::check(unsafe {
            braidinfer_hip::ffi::hipStreamSynchronize(stream)
        })
        .map_err(ModelError::Hip)?;

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
        // Persistent worker holds all GPU CUs — must shut it down before any hipMemcpy
        // or hipHostFree.  It will be re-launched lazily on the next decode_step call.
        drop(self.persistent_workers.take());
        // Phase C (braidinfer-4n5): explicitly drop the HostPageAllocator's
        // ManuallyDrop<PinnedBuffer> pool AFTER the persistent worker has exited.
        // The pool uses ManuallyDrop to prevent automatic drop (which would call
        // hipHostFree while the worker held GPU CUs).  Now that the worker is
        // torn down, we call drop_pool() to free the pinned allocation.
        // This mirrors the ManuallyDrop<DeviceBuffer> unwrap in PersistentDispatch::drop.
        if let Some(ha) = self.host_page_allocator.take() {
            // SAFETY: the persistent worker is fully torn down (drop above completed).
            ha.drop_pool();
        }

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
            // Phase C: pass host_page_allocator so HostPinned chunk host slots
            // are freed back to the pool.  After this reset_state has taken the
            // host_page_allocator above (for explicit drop), so it is None here —
            // which is correct: we pass None, meaning HostPinned chunks won't be
            // freed into a pool that no longer exists.  Their host slots are leaked,
            // but the whole pool is being freed anyway via the ManuallyDrop::drop above.
            if let Some(alloc) = self.page_allocator.as_mut() {
                seq.reset(alloc, None);
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

    /// bd 4e2m Probe (a) variant (udi #3327): dispatch an empty packet
    /// (num_inst=0) per worker to trigger persistent_iter_poll_barrier
    /// first-iteration code path WITHOUT executing any opcode. Tests the
    /// hypothesis that "first poll-barrier success on the worker" is the
    /// cure mechanism, distinct from "any opcode dispatch via worker".
    /// If this cures cold-start NaN, the cure is poll-barrier specific.
    pub fn minimal_mailbox_warmup_empty_packet(&mut self) -> Result<String, String> {
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

        let mut per_worker = Vec::new();
        let t_dispatch = std::time::Instant::now();
        for &gpu_idx in &live {
            let t_one = std::time::Instant::now();
            // num_inst=0 batch: worker exits poll-barrier on seq > last_seq,
            // reads num_inst=0, skips the inner instruction loop, acks
            // immediately. Triggers the FIRST-ITERATION poll-barrier path
            // without running any opcode.
            let seq = dispatch.dispatch_batch_fire(gpu_idx, &[]);
            dispatch.wait_ack(gpu_idx, seq);
            let one_us = t_one.elapsed().as_micros();
            per_worker.push(format!("gpu{}={}us", gpu_idx, one_us));
        }
        let dispatch_ms = t_dispatch.elapsed().as_secs_f64() * 1000.0;

        Ok(format!(
            "spawn={:.1}ms empty-dispatch={:.2}ms [{}]",
            spawn_ms,
            dispatch_ms,
            per_worker.join(",")
        ))
    }
}
