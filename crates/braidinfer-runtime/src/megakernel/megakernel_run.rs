//! Megakernel runtime: per-step instruction patching, paged KV management, execution.
//! Extracted from megakernel.rs for maintainability.

use braidinfer_hip::HipResult;
use braidinfer_hip::memory::{DeviceBuffer, MappedHostBuffer};

use crate::paged_kv::{PageAllocator, SequenceState};
use crate::persistent_dispatch::PersistentDispatch;

use super::{
    CHUNK_TOKENS, Instruction, MegakernelProgram, OP_ATTN_PAGED_Q, OP_KV_QUANTIZE,
};
use super::instructions::{AttnPagedInst, AttnPagedQInst, D2dCopyInst, EmbeddingInst, KvQuantizeInst, make_opcode_gridx};

impl MegakernelProgram {
    fn patch_kv_write_offsets(&mut self, position: u32) {
        let hd = self.kv.head_dim;
        let max_sl = self.kv.max_seq_len as usize;
        let head_stride = max_sl * hd;
        for (layer_i, head_indices) in self.kv.kv_write_indices.iter().enumerate() {
            let (k_base, v_base) = self.kv.kv_base_ptrs[layer_i];
            for (h, &(k_idx, v_idx)) in head_indices.iter().enumerate() {
                let offset =
                    (h * head_stride + position as usize * hd) * std::mem::size_of::<f32>();
                unsafe {
                    let k_inst = self.instructions[k_idx].words.as_mut_ptr() as *mut D2dCopyInst;
                    (*k_inst).dst = (k_base + offset as u64) as *mut f32;
                    let v_inst = self.instructions[v_idx].words.as_mut_ptr() as *mut D2dCopyInst;
                    (*v_inst).dst = (v_base + offset as u64) as *mut f32;
                }
            }
        }
    }

    /// Update host-side instruction fields only (no GPU upload).
    /// For persistent worker path: worker reads instructions from host-mapped memory,
    /// not from device_program. Also writes position_ids via hipMemcpy (DMA, not SM).
    pub fn update_step_host_only(&mut self, token_id: u32, position: u32) -> HipResult<()> {
        assert!(position < self.kv.max_seq_len);

        unsafe {
            let inst = self.instructions[self.embedding_inst_idx].words.as_mut_ptr() as *mut EmbeddingInst;
            (*inst).token_id = token_id as u64;
        }

        // position_ids is now MappedHostBuffer — written via host_ptr by caller,
        // GPU reads through device_ptr. No hipMemcpy needed.

        self.patch_kv_write_offsets(position);

        Ok(())
    }

    /// Update per-step fields for the paged KV path. Patches host-mapped
    /// instruction buffer + page table; persistent_worker reads via mailbox.
    ///
    /// Performs steps 1-5 (host-side instruction patching + host-mapped page_table writes).
    /// The persistent caller dispatches the patched instructions via dispatch_batch_slice,
    /// which writes them directly to the host-mapped worker queue — no hipMemcpyAsync needed
    /// and no deadlock risk under the cooperative kernel.
    pub fn update_step_paged_no_upload(
        &mut self,
        token_id: u32,
        position: u32,
        seq: &SequenceState,
        allocator: &PageAllocator,
    ) -> HipResult<()> {
        assert!(
            position < self.kv.max_seq_len,
            "position {position} >= max_seq_len {}",
            self.kv.max_seq_len
        );
        assert!(self.paged, "update_step_paged_no_upload called on non-paged program");

        // 1. Patch embedding token_id
        unsafe {
            let inst = self.instructions[self.embedding_inst_idx].words.as_mut_ptr() as *mut EmbeddingInst;
            (*inst).token_id = token_id as u64;
        }

        // 2. Append per-token positions (3 ints: temporal, height, width) to position_table.
        {
            let seq_token_idx = (seq.seq_len as usize).saturating_sub(1);
            let pos_scalar = *seq
                .positions
                .get(seq_token_idx)
                .expect("position missing for appended paged token");
            let host_ptr = self
                .paged_kv
                .as_ref()
                .expect("paged_kv not initialized")
                .position_table
                .as_ref()
                .expect("position_table not allocated")
                .host_ptr();
            unsafe {
                let base = host_ptr.add(seq_token_idx * 3);
                base.add(0).write_volatile(pos_scalar);
                base.add(1).write_volatile(pos_scalar);
                base.add(2).write_volatile(pos_scalar);
            }
        }

        // 3. Patch KV write D2D_COPY destinations from paged chunk layout
        let chunk_offset = (seq.current_chunk_offset() as usize).saturating_sub(1);
        let kv_stride = self.paged_kv.as_ref().unwrap().kv_stride_paged;
        let hd = self.kv.head_dim;
        let chunk_head_stride = CHUNK_TOKENS * hd;

        for (layer_i, head_indices) in self.kv.kv_write_indices.iter().enumerate() {
            let chunk_slot = if seq.chunks.is_empty() {
                0
            } else {
                seq.chunks.last().unwrap().slot_index()
            };
            let chunk_base = allocator.slot_ptr(chunk_slot) as u64;
            let layer_k_offset =
                (layer_i * 2 * CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>()) as u64;
            let layer_v_offset =
                layer_k_offset + (CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>()) as u64;
            for (h, &(k_idx, v_idx)) in head_indices.iter().enumerate() {
                let head_byte_off =
                    (h * chunk_head_stride + chunk_offset * hd) * std::mem::size_of::<f32>();
                let k_ptr = chunk_base + layer_k_offset + head_byte_off as u64;
                let v_ptr = chunk_base + layer_v_offset + head_byte_off as u64;
                unsafe {
                    let k_inst = self.instructions[k_idx].words.as_mut_ptr() as *mut D2dCopyInst;
                    (*k_inst).dst = k_ptr as *mut f32;
                    let v_inst = self.instructions[v_idx].words.as_mut_ptr() as *mut D2dCopyInst;
                    (*v_inst).dst = v_ptr as *mut f32;
                }
            }
        }

        // 4. Patch attention instructions
        let total_seq_len = seq.seq_len as i32;
        let paged_kv = self.paged_kv.as_ref().unwrap();
        let page_table_ptr = paged_kv
            .page_table
            .as_ref()
            .expect("page_table not allocated")
            .as_ptr() as u64;
        let pos_table_ptr = paged_kv
            .position_table
            .as_ref()
            .expect("position_table not allocated")
            .as_ptr() as u64;
        let attn_paged_inst_indices = paged_kv.attn_paged_inst_indices.clone();
        let attn_quant_inst_indices = paged_kv.attn_quant_inst_indices.clone();

        if self.quantized_kv && seq.chunks.len() > 1 {
            let num_sealed = seq.chunks.len() - 1;
            let sealed_tokens = (num_sealed * CHUNK_TOKENS) as i32;
            let active_tokens = total_seq_len - sealed_tokens;
            let nqh = unsafe {
                let inst = self.instructions[attn_paged_inst_indices[0]].words.as_ptr() as *const AttnPagedInst;
                (*inst).nqh as u32
            };

            let quant_pt_ptr = self
                .quant_kv
                .as_ref()
                .expect("quant_kv not initialized")
                .quant_page_table
                .as_ref()
                .expect("quant_page_table not allocated")
                .device_ptr() as u64;

            for &idx in &attn_quant_inst_indices {
                unsafe {
                    let inst = self.instructions[idx].words.as_mut_ptr() as *mut AttnPagedQInst;
                    (*inst).opcode_gridx = make_opcode_gridx(OP_ATTN_PAGED_Q, nqh);
                }
                unsafe {
                    let inst = self.instructions[idx].words.as_mut_ptr() as *mut AttnPagedQInst;
                    (*inst).quant_page_table = quant_pt_ptr;
                    (*inst).pos_table = pos_table_ptr;
                    (*inst).quant_seq_len = sealed_tokens as u64;
                }
            }

            for &idx in &attn_paged_inst_indices {
                let active_pt_ptr =
                    page_table_ptr + (num_sealed * std::mem::size_of::<u64>()) as u64;
                unsafe {
                    let inst = self.instructions[idx].words.as_mut_ptr() as *mut AttnPagedInst;
                    (*inst).page_table = active_pt_ptr;
                    // pos_table layout is 3 ints per token (mRoPE-compatible).
                    (*inst).pos_table = pos_table_ptr
                        + (num_sealed * CHUNK_TOKENS * 3 * std::mem::size_of::<i32>()) as u64;
                    (*inst).seq_len = active_tokens as u64;
                }
            }
        } else {
            for &idx in &attn_quant_inst_indices {
                unsafe {
                    let inst = self.instructions[idx].words.as_mut_ptr() as *mut AttnPagedQInst;
                    (*inst).opcode_gridx = make_opcode_gridx(OP_ATTN_PAGED_Q, 0);
                }
            }
            for &idx in &attn_paged_inst_indices {
                unsafe {
                    let inst = self.instructions[idx].words.as_mut_ptr() as *mut AttnPagedInst;
                    (*inst).page_table = page_table_ptr;
                    (*inst).pos_table = pos_table_ptr;
                    (*inst).seq_len = total_seq_len as u64;
                    (*inst).partial_state = 0;
                }
            }
        }

        // 5. Upload page_table if chunk list changed (host-mapped: no HIP API call).
        if seq.chunks.len() != self.paged_kv.as_ref().unwrap().last_page_table_len {
            let page_table_buf = self.paged_kv.as_ref().unwrap().page_table.as_ref().expect("page_table not allocated");
            let host_ptr = page_table_buf.host_ptr();
            for (i, chunk) in seq.chunks.iter().enumerate() {
                let addr = allocator.slot_ptr(chunk.slot_index()) as u64;
                unsafe {
                    host_ptr.add(i).write_volatile(addr);
                }
            }
            self.paged_kv.as_mut().unwrap().last_page_table_len = seq.chunks.len();
        }

        // Step 6 (hipMemcpyAsync upload) intentionally SKIPPED.
        // The persistent caller dispatches via dispatch_batch_slice, which writes
        // instructions directly to the host-mapped worker queue.
        Ok(())
    }

    /// Allocate the next chunk if the current one just filled up.
    /// If quantized_kv is enabled, quantizes the sealed chunk via the
    /// persistent worker mailbox.
    /// Call after execute() + stream sync, before next update_step_paged().
    ///
    /// bd 9gmh Phase 2F: dispatch is now required (was Option). The legacy
    /// quantize_sealed_chunk (launch_cooperative) fallback has been deleted.
    pub fn post_step_paged(
        &mut self,
        position: u32,
        seq: &mut SequenceState,
        allocator: &mut PageAllocator,
        quant_allocator: Option<&mut PageAllocator>,
        cfg: &crate::config::ModelConfig,
        dispatch: &mut PersistentDispatch,
    ) -> HipResult<()> {
        if (position as usize + 1) % CHUNK_TOKENS == 0 {
            // Chunk just sealed
            if self.quantized_kv {
                if let Some(q_alloc) = quant_allocator {
                    // Get the f32 chunk that just sealed (last chunk before we append new one)
                    let sealed_chunk = seq.chunks.last().unwrap();
                    let f32_ptr = allocator.slot_ptr(sealed_chunk.slot_index());

                    // Allocate quantized chunk slot
                    let (q_slot, q_ptr) = q_alloc.alloc().ok_or(braidinfer_hip::HipError(
                        braidinfer_hip::ffi::hipErrorOutOfMemory,
                    ))?;

                    // Quantize via persistent worker mailbox (the sole code path).
                    self.quantize_sealed_chunk_via_worker(dispatch, 0, f32_ptr, q_ptr, cfg)?;

                    // Track slot for cleanup
                    seq.quant_slots.push(q_slot);

                    // Upload quantized page table — host-mapped, no HIP API call.
                    let num_sealed = seq.chunks.len();
                    let quant_kv = self.quant_kv.as_mut().expect("quant_kv not initialized");
                    let quant_pt = quant_kv
                        .quant_page_table
                        .as_mut()
                        .expect("quant_page_table not allocated");
                    unsafe {
                        quant_pt.host_ptr().add(num_sealed - 1).write_volatile(q_ptr as u64);
                    }
                    quant_kv.last_quant_page_table_len = num_sealed;
                }
            }
        }
        Ok(())
    }

    /// Quantize a sealed f32 chunk via the persistent worker mailbox.
    /// Builds the same OP_KV_QUANTIZE instruction list as `quantize_sealed_chunk`
    /// but dispatches via `dispatch_batch_fire` + `wait_ack` instead of
    /// `launch_cooperative` — safe while the persistent cooperative kernel holds all CUs.
    pub fn quantize_sealed_chunk_via_worker(
        &self,
        dispatch: &mut PersistentDispatch,
        gpu_idx: usize,
        f32_chunk_ptr: *const u8,
        quant_chunk_ptr: *mut u8,
        cfg: &crate::config::ModelConfig,
    ) -> HipResult<()> {
        use crate::paged_kv::quantized_kv_offsets;
        let nkh = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let num_attn_layers = cfg
            .layers
            .iter()
            .filter(|l| l.layer_type == crate::config::LayerType::Attention)
            .count();
        let kv_stride = nkh * hd;
        let f32_layer_bytes = 2 * CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>();

        let mut instructions: Vec<Instruction> = Vec::new();
        for layer_i in 0..num_attn_layers {
            let f32_base = f32_chunk_ptr as u64 + (layer_i * f32_layer_bytes) as u64;
            let f32_k = f32_base;
            let f32_v = f32_base + (CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>()) as u64;

            for (is_v, f32_src) in [(false, f32_k), (true, f32_v)] {
                let (q1d, q1s, rd, rs) = quantized_kv_offsets(cfg, CHUNK_TOKENS, layer_i, is_v);
                instructions.push(KvQuantizeInst {
                    opcode_gridx: make_opcode_gridx(OP_KV_QUANTIZE, (nkh * hd) as u32),
                    src:          f32_src as *const f32,
                    q1_data:      (quant_chunk_ptr as u64 + q1d as u64) as *mut u8,
                    q1_scale:     (quant_chunk_ptr as u64 + q1s as u64) as *mut f32,
                    r_data:       (quant_chunk_ptr as u64 + rd as u64) as *mut u8,
                    r_scale:      (quant_chunk_ptr as u64 + rs as u64) as *mut f32,
                    num_kv_heads: nkh as i32,
                    head_dim:     hd as i32,
                    chunk_tokens: CHUNK_TOKENS as i32,
                    _pad0:        0,
                    _pad:         [0; 11],
                }.into_inst());
            }
        }
        // NOTE: No HALT instruction — the persistent worker mailbox must NOT receive HALT
        // (it would terminate the worker). dispatch_batch_fire sends the batch and
        // wait_ack blocks until the worker has processed all instructions in the batch.
        let seq = dispatch.dispatch_batch_fire(gpu_idx, &instructions);
        dispatch.wait_ack(gpu_idx, seq);
        Ok(())
    }

    /// Lazily allocate the page_table and position_table device buffers.
    /// Must be called once before the first update_step_paged().
    pub fn init_paged_buffers(&mut self, max_chunks: usize) -> HipResult<()> {
        let paged_kv = self.paged_kv.as_mut().expect("init_paged_buffers called on non-paged program");
        if paged_kv.page_table.is_none() {
            // Host-mapped: GPU reads via device_ptr; CPU writes via host_ptr without
            // hipMemcpyAsync. Required for persistent cooperative-kernel safety.
            paged_kv.page_table = Some(MappedHostBuffer::alloc(max_chunks)?);
        }
        if paged_kv.position_table.is_none() {
            // 3 ints per token (temporal, height, width) for mRoPE compatibility.
            // For text-only models all 3 are written to the same value and op_attn_paged
            // collapses to standard RoPE via mrope_section sizes.
            paged_kv.position_table = Some(
                braidinfer_hip::memory::MappedHostBuffer::alloc(self.kv.max_seq_len as usize * 3)?,
            );
        }
        Ok(())
    }

    /// Enable quantized KV cache. Allocates scratch buffer and quantized page table.
    /// Call after init_paged_buffers, before first decode step.
    pub fn enable_quantized_kv(
        &mut self,
        max_chunks: usize,
        cfg: &crate::config::ModelConfig,
    ) -> HipResult<()> {
        let nqh = cfg.num_q_heads;
        let hd = cfg.head_dim;
        let num_attn_layers = cfg
            .layers
            .iter()
            .filter(|l| l.layer_type == crate::config::LayerType::Attention)
            .count();
        // Scratch: [nqh × (2+hd)] per attention layer (each layer gets its own scratch region)
        let scratch_per_layer = nqh * (2 + hd);
        let total_scratch = num_attn_layers * scratch_per_layer;
        let quant_scratch = DeviceBuffer::alloc(self.device, total_scratch)?;
        // bd 4qh/8gz Phase 5 (deferred 2026-05-01, attempted 2026-05-20): host-mapped
        // quant_page_table so chunk-seal updates use host_ptr write_volatile instead of
        // hipMemcpy. Required for KV_QUANT on the persistent path (the deferred phase
        // had a NaN regression; this attempt re-tries with explicit init to zero).
        let quant_page_table = MappedHostBuffer::<u64>::alloc(max_chunks)?;
        // Initialize all slots to 0 — the kernel reads pages up to last_quant_page_table_len
        // but defensive zeroing avoids any uninitialized-bits class issues that may have
        // caused the prior NaN regression.
        unsafe {
            let host = quant_page_table.host_ptr();
            for i in 0..max_chunks {
                host.add(i).write_volatile(0u64);
            }
        }

        // Patch OP_ATTN_PAGED_Q scratch pointers and OP_ATTN_PAGED partial_state pointers
        let scratch_base = quant_scratch.as_ptr() as u64;
        let attn_quant_inst_indices = self.paged_kv.as_ref().unwrap().attn_quant_inst_indices.clone();
        let attn_paged_inst_indices = self.paged_kv.as_ref().unwrap().attn_paged_inst_indices.clone();
        for (layer_i, &q_idx) in attn_quant_inst_indices.iter().enumerate() {
            let scratch_ptr =
                scratch_base + (layer_i * scratch_per_layer * std::mem::size_of::<f32>()) as u64;
            unsafe {
                let inst = self.instructions[q_idx].words.as_mut_ptr() as *mut AttnPagedQInst;
                (*inst).scratch = scratch_ptr;
            }
        }
        for (layer_i, &p_idx) in attn_paged_inst_indices.iter().enumerate() {
            let scratch_ptr =
                scratch_base + (layer_i * scratch_per_layer * std::mem::size_of::<f32>()) as u64;
            unsafe {
                let inst = self.instructions[p_idx].words.as_mut_ptr() as *mut AttnPagedInst;
                (*inst).partial_state = scratch_ptr;
            }
        }
        self.quant_kv = Some(super::QuantizedKvState {
            quant_scratch: Some(quant_scratch),
            quant_page_table: Some(quant_page_table),
            last_quant_page_table_len: 0,
        });
        self.quantized_kv = true;
        Ok(())
    }

    // bd 9gmh Phase 2F: quantize_sealed_chunk (legacy launch_cooperative path) deleted.
    // The mailbox-only quantize_sealed_chunk_via_worker above is the sole code path.

}

