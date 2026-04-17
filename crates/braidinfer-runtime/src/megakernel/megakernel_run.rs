//! Megakernel runtime: per-step instruction patching, paged KV management, execution.
//! Extracted from megakernel.rs for maintainability.

use std::ffi::c_void;

use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;

use crate::paged_kv::{PageAllocator, SequenceState};

use super::{
    CHUNK_TOKENS, INST_SIZE, Instruction, MegakernelProgram, OP_ATTN_PAGED_Q, OP_KV_QUANTIZE,
    upload_program,
};
use super::instructions::{AttnPagedInst, AttnPagedQInst, EmbeddingInst, GqaAttnInst, HaltInst, KvQuantizeInst, make_opcode_gridx};

impl MegakernelProgram {
    fn patch_kv_write_offsets(&mut self, position: u32) {
        let hd = self.head_dim_attn;
        let max_sl = self.max_seq_len as usize;
        let head_stride = max_sl * hd;
        for (layer_i, head_indices) in self.kv_write_indices.iter().enumerate() {
            let (k_base, v_base) = self.kv_base_ptrs[layer_i];
            for (h, &(k_idx, v_idx)) in head_indices.iter().enumerate() {
                let offset =
                    (h * head_stride + position as usize * hd) * std::mem::size_of::<f32>();
                self.instructions[k_idx].words[1] = k_base + offset as u64;
                self.instructions[v_idx].words[1] = v_base + offset as u64;
            }
        }
    }

    pub fn update_step(&mut self, token_id: u32, position: u32, stream: &Stream) -> HipResult<()> {
        assert!(
            position < self.max_seq_len,
            "position {position} >= max_seq_len {}",
            self.max_seq_len
        );

        // Update embedding token_id
        unsafe {
            let inst = self.instructions[self.embedding_inst_idx].words.as_mut_ptr() as *mut EmbeddingInst;
            (*inst).token_id = token_id as u64;
        }

        // Update position_ids on device ([temporal, height, width] = all equal position for text)
        // Use synchronous hipMemcpy because pos_data is a stack local
        let pos_data = [position as i32, position as i32, position as i32];
        braidinfer_hip::error::check(unsafe {
            braidinfer_hip::ffi::hipMemcpy(
                self.position_ids_dev_ptr as *mut std::ffi::c_void,
                pos_data.as_ptr().cast(),
                3 * std::mem::size_of::<i32>(),
                braidinfer_hip::ffi::hipMemcpyHostToDevice,
            )
        })?;

        // Update KV cache write offsets (position-dependent, [H,T,D] layout)
        self.patch_kv_write_offsets(position);

        // Update GQA attention seq_len
        let seq_len = position + 1;
        for &idx in &self.gqa_attn_inst_indices {
            unsafe {
                let inst = self.instructions[idx].words.as_mut_ptr() as *mut GqaAttnInst;
                (*inst).seq_len = seq_len as u64;
            }
        }

        // Upload entire instruction buffer in one hipMemcpyAsync call.
        // ~500 instructions × 128 bytes = ~64KB; one 64KB copy is cheaper than 24× 128-byte copies.
        let flat: Vec<u64> = self.instructions.iter().flat_map(|i| i.words).collect();
        // When dump is active, device_program has an OP_NOP header at offset 0;
        // instructions start at offset INST_SIZE words.
        let offset_words = if self.dump_buffer.is_some() {
            INST_SIZE
        } else {
            0
        };
        let dev_ptr = unsafe { self.device_program.as_mut_ptr().add(offset_words) };
        let size = flat.len() * std::mem::size_of::<u64>();
        braidinfer_hip::error::check(unsafe {
            braidinfer_hip::ffi::hipMemcpyAsync(
                dev_ptr.cast(),
                flat.as_ptr().cast(),
                size,
                braidinfer_hip::ffi::hipMemcpyHostToDevice,
                stream.raw(),
            )
        })?;
        Ok(())
    }

    /// Update host-side instruction fields only (no GPU upload).
    /// For persistent worker path: worker reads instructions from host-mapped memory,
    /// not from device_program. Also writes position_ids via hipMemcpy (DMA, not SM).
    pub fn update_step_host_only(&mut self, token_id: u32, position: u32) -> HipResult<()> {
        assert!(position < self.max_seq_len);

        unsafe {
            let inst = self.instructions[self.embedding_inst_idx].words.as_mut_ptr() as *mut EmbeddingInst;
            (*inst).token_id = token_id as u64;
        }

        // position_ids is now MappedHostBuffer — written via host_ptr by caller,
        // GPU reads through device_ptr. No hipMemcpy needed.

        self.patch_kv_write_offsets(position);

        let seq_len = position + 1;
        for &idx in &self.gqa_attn_inst_indices {
            unsafe {
                let inst = self.instructions[idx].words.as_mut_ptr() as *mut GqaAttnInst;
                (*inst).seq_len = seq_len as u64;
            }
        }

        Ok(())
    }

    /// Update per-step fields for the paged KV path.
    /// Must be called before `execute()` each decode step.
    pub fn update_step_paged(
        &mut self,
        token_id: u32,
        position: u32,
        seq: &SequenceState,
        allocator: &PageAllocator,
        stream: &Stream,
    ) -> HipResult<()> {
        assert!(
            position < self.max_seq_len,
            "position {position} >= max_seq_len {}",
            self.max_seq_len
        );
        assert!(self.paged, "update_step_paged called on non-paged program");

        // 1. Patch embedding token_id
        unsafe {
            let inst = self.instructions[self.embedding_inst_idx].words.as_mut_ptr() as *mut EmbeddingInst;
            (*inst).token_id = token_id as u64;
        }

        // 2. Append scalar logical position to position_table in sequence order.
        // MappedHostBuffer: write via host_ptr (no hipMemcpy — GPU reads through device_ptr).
        {
            let seq_token_idx = (seq.seq_len as usize).saturating_sub(1);
            let pos_scalar = *seq
                .positions
                .get(seq_token_idx)
                .expect("position missing for appended paged token");
            let host_ptr = self
                .position_table
                .as_ref()
                .expect("position_table not allocated")
                .host_ptr();
            unsafe {
                host_ptr.add(seq_token_idx).write_volatile(pos_scalar);
            }
        }

        // position_ids for mRoPE: written by caller via set_position() (MappedHostBuffer).
        // No hipMemcpy needed — GPU reads through device_ptr.

        // 3. Patch KV write D2D_COPY destinations from paged chunk layout [H,T,D]
        // current_chunk_offset() returns len (post-increment from append_token).
        // The write target is len-1 (the slot just reserved).
        let chunk_offset = (seq.current_chunk_offset() as usize).saturating_sub(1);
        let kv_stride = self.kv_stride_paged;
        let _nkh = self.num_kv_heads_attn;
        let hd = self.head_dim_attn;
        let chunk_head_stride = CHUNK_TOKENS * hd; // elements between heads within chunk

        for (layer_i, head_indices) in self.kv_write_indices.iter().enumerate() {
            let chunk_slot = if seq.chunks.is_empty() {
                0
            } else {
                seq.chunks.last().unwrap().slot_index()
            };
            let chunk_base = allocator.slot_ptr(chunk_slot) as u64;
            // layout: [layer0_K[nkh, chunk_tokens, hd], layer0_V[...], layer1_K, ...]
            let layer_k_offset =
                (layer_i * 2 * CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>()) as u64;
            let layer_v_offset =
                layer_k_offset + (CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>()) as u64;
            for (h, &(k_idx, v_idx)) in head_indices.iter().enumerate() {
                let head_byte_off =
                    (h * chunk_head_stride + chunk_offset * hd) * std::mem::size_of::<f32>();
                let k_ptr = chunk_base + layer_k_offset + head_byte_off as u64;
                let v_ptr = chunk_base + layer_v_offset + head_byte_off as u64;
                self.instructions[k_idx].words[1] = k_ptr;
                self.instructions[v_idx].words[1] = v_ptr;
            }
        }

        // 4. Patch attention instructions
        let total_seq_len = seq.seq_len as i32;
        let page_table_ptr = self
            .page_table
            .as_ref()
            .expect("page_table not allocated")
            .as_ptr() as u64;
        let pos_table_ptr = self
            .position_table
            .as_ref()
            .expect("position_table not allocated")
            .as_ptr() as u64;

        if self.quantized_kv && seq.chunks.len() > 1 {
            // Two-phase: quantized sealed chunks + f32 active chunk
            let num_sealed = seq.chunks.len() - 1;
            let sealed_tokens = (num_sealed * CHUNK_TOKENS) as i32;
            let active_tokens = total_seq_len - sealed_tokens;
            let nqh = self.instructions[self.attn_paged_inst_indices[0]].words[6] as u32;

            let quant_pt_ptr = self
                .quant_page_table
                .as_ref()
                .expect("quant_page_table not allocated")
                .as_ptr() as u64;

            // Patch OP_ATTN_PAGED_Q: enable (grid_x=nqh), quant page table, sealed seq_len
            for &idx in &self.attn_quant_inst_indices {
                self.instructions[idx].words[0] = (OP_ATTN_PAGED_Q as u64) | ((nqh as u64) << 32);
                unsafe {
                    let inst = self.instructions[idx].words.as_mut_ptr() as *mut AttnPagedQInst;
                    (*inst).quant_page_table = quant_pt_ptr;
                    (*inst).pos_table = pos_table_ptr;
                    (*inst).quant_seq_len = sealed_tokens as u64;
                }
            }

            // Patch OP_ATTN_PAGED: f32 page table (only active chunk), active seq_len
            // The active chunk is the last one in seq.chunks. We put its pointer
            // at offset `sealed_tokens/CHUNK_TOKENS` in the f32 page table, but simpler:
            // point to a single-entry table with just the active chunk.
            // We reuse the main page_table — the active chunk ptr is at index `num_sealed`.
            for &idx in &self.attn_paged_inst_indices {
                // Point page_table at the last entry (active chunk)
                let active_pt_ptr =
                    page_table_ptr + (num_sealed * std::mem::size_of::<u64>()) as u64;
                unsafe {
                    let inst = self.instructions[idx].words.as_mut_ptr() as *mut AttnPagedInst;
                    (*inst).page_table = active_pt_ptr;
                    (*inst).pos_table = pos_table_ptr + (num_sealed * CHUNK_TOKENS * std::mem::size_of::<i32>()) as u64;
                    (*inst).seq_len = active_tokens as u64;
                }
            }
        } else {
            // No quantized chunks yet (or quantized_kv not enabled): all f32
            // Disable OP_ATTN_PAGED_Q (grid_x=0)
            for &idx in &self.attn_quant_inst_indices {
                self.instructions[idx].words[0] = OP_ATTN_PAGED_Q as u64; // grid_x=0
            }
            // OP_ATTN_PAGED sees all chunks, no partial_state
            for &idx in &self.attn_paged_inst_indices {
                unsafe {
                    let inst = self.instructions[idx].words.as_mut_ptr() as *mut AttnPagedInst;
                    (*inst).page_table = page_table_ptr;
                    (*inst).pos_table = pos_table_ptr;
                    (*inst).seq_len = total_seq_len as u64;
                }
                if !self.quantized_kv {
                    unsafe {
                        let inst = self.instructions[idx].words.as_mut_ptr() as *mut AttnPagedInst;
                        (*inst).partial_state = 0;
                    }
                }
            }
        }

        // 5. Upload page_table if chunk list changed
        if seq.chunks.len() != self.last_page_table_len {
            let page_table_dev = self.page_table.as_mut().expect("page_table not allocated");
            let host_ptrs: Vec<u64> = seq
                .chunks
                .iter()
                .map(|c| allocator.slot_ptr(c.slot_index()) as u64)
                .collect();
            let dst = page_table_dev.as_mut_ptr() as *mut u8;
            let bytes = host_ptrs.len() * std::mem::size_of::<u64>();
            braidinfer_hip::error::check(unsafe {
                braidinfer_hip::ffi::hipMemcpyAsync(
                    dst.cast(),
                    host_ptrs.as_ptr().cast(),
                    bytes,
                    braidinfer_hip::ffi::hipMemcpyHostToDevice,
                    stream.raw(),
                )
            })?;
            self.last_page_table_len = seq.chunks.len();
        }

        // 6. Upload entire instruction buffer in one hipMemcpyAsync call.
        let flat: Vec<u64> = self.instructions.iter().flat_map(|i| i.words).collect();
        let offset_words = if self.dump_buffer.is_some() {
            INST_SIZE
        } else {
            0
        };
        let dev_ptr = unsafe { self.device_program.as_mut_ptr().add(offset_words) };
        let size = flat.len() * std::mem::size_of::<u64>();
        braidinfer_hip::error::check(unsafe {
            braidinfer_hip::ffi::hipMemcpyAsync(
                dev_ptr.cast(),
                flat.as_ptr().cast(),
                size,
                braidinfer_hip::ffi::hipMemcpyHostToDevice,
                stream.raw(),
            )
        })?;
        Ok(())
    }

    /// Allocate the next chunk if the current one just filled up.
    /// If quantized_kv is enabled, quantizes the sealed chunk.
    /// Call after execute() + stream sync, before next update_step_paged().
    pub fn post_step_paged(
        &mut self,
        position: u32,
        seq: &mut SequenceState,
        allocator: &mut PageAllocator,
        quant_allocator: Option<&mut PageAllocator>,
        cfg: &crate::model::ModelConfig,
        stream: &Stream,
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

                    // Run quantize kernel
                    self.quantize_sealed_chunk(f32_ptr, q_ptr, cfg, stream)?;
                    stream.synchronize()?;

                    // Track slot for cleanup
                    seq.quant_slots.push(q_slot);

                    // Upload quantized page table
                    let num_sealed = seq.chunks.len();
                    let quant_pt = self
                        .quant_page_table
                        .as_mut()
                        .expect("quant_page_table not allocated");
                    let q_ptr_val = q_ptr as u64;
                    let offset = (num_sealed - 1) * std::mem::size_of::<u64>();
                    braidinfer_hip::error::check(unsafe {
                        braidinfer_hip::ffi::hipMemcpy(
                            (quant_pt.as_mut_ptr() as *mut u8).add(offset).cast(),
                            std::ptr::addr_of!(q_ptr_val).cast(),
                            std::mem::size_of::<u64>(),
                            braidinfer_hip::ffi::hipMemcpyHostToDevice,
                        )
                    })?;
                    self.last_quant_page_table_len = num_sealed;
                }
            }
        }
        Ok(())
    }

    /// Lazily allocate the page_table and position_table device buffers.
    /// Must be called once before the first update_step_paged().
    pub fn init_paged_buffers(&mut self, max_chunks: usize) -> HipResult<()> {
        if self.page_table.is_none() {
            self.page_table = Some(DeviceBuffer::alloc(self.device, max_chunks)?);
        }
        if self.position_table.is_none() {
            self.position_table =
                Some(braidinfer_hip::memory::MappedHostBuffer::alloc(self.max_seq_len as usize)?);
        }
        Ok(())
    }

    /// Enable quantized KV cache. Allocates scratch buffer and quantized page table.
    /// Call after init_paged_buffers, before first decode step.
    pub fn enable_quantized_kv(
        &mut self,
        max_chunks: usize,
        cfg: &crate::model::ModelConfig,
    ) -> HipResult<()> {
        let nqh = cfg.num_q_heads;
        let hd = cfg.head_dim;
        let num_attn_layers = cfg
            .layers
            .iter()
            .filter(|l| l.layer_type == crate::model::LayerType::Attention)
            .count();
        // Scratch: [nqh × (2+hd)] per attention layer (each layer gets its own scratch region)
        let scratch_per_layer = nqh * (2 + hd);
        let total_scratch = num_attn_layers * scratch_per_layer;
        self.quant_scratch = Some(DeviceBuffer::alloc(self.device, total_scratch)?);
        self.quant_page_table = Some(DeviceBuffer::alloc(self.device, max_chunks)?);

        // Patch OP_ATTN_PAGED_Q scratch pointers and OP_ATTN_PAGED partial_state pointers
        let scratch_base = self.quant_scratch.as_ref().unwrap().as_ptr() as u64;
        for (layer_i, &q_idx) in self.attn_quant_inst_indices.iter().enumerate() {
            let scratch_ptr =
                scratch_base + (layer_i * scratch_per_layer * std::mem::size_of::<f32>()) as u64;
            self.instructions[q_idx].words[1] = scratch_ptr;
        }
        for (layer_i, &p_idx) in self.attn_paged_inst_indices.iter().enumerate() {
            let scratch_ptr =
                scratch_base + (layer_i * scratch_per_layer * std::mem::size_of::<f32>()) as u64;
            self.instructions[p_idx].words[14] = scratch_ptr;
        }
        self.quantized_kv = true;
        Ok(())
    }

    /// Quantize a sealed f32 chunk. Call from post_step_paged when a chunk fills up.
    /// Launches OP_KV_QUANTIZE for each layer's K and V via the megakernel.
    pub fn quantize_sealed_chunk(
        &self,
        f32_chunk_ptr: *const u8,
        quant_chunk_ptr: *mut u8,
        cfg: &crate::model::ModelConfig,
        stream: &Stream,
    ) -> HipResult<()> {
        use crate::paged_kv::quantized_kv_offsets;
        let nkh = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let num_attn_layers = cfg
            .layers
            .iter()
            .filter(|l| l.layer_type == crate::model::LayerType::Attention)
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
                    _pad:         [0; 10],
                }.into_inst());
            }
        }
        instructions.push(HaltInst::new().into_inst());

        // Upload and execute
        let prog_buf = upload_program(self.device, &instructions)?;

        let func = self.module.get_function("megakernel_f32")?;
        let mut prog_ptr: *const c_void = prog_buf.as_ptr().cast();
        let mut num_inst = instructions.len() as i32;
        let mut args: [*mut c_void; 2] = [
            std::ptr::addr_of_mut!(prog_ptr).cast(),
            std::ptr::addr_of_mut!(num_inst).cast(),
        ];
        func.launch_cooperative(
            (self.num_blocks, 1, 1),
            (256, 1, 1),
            self.shared_mem,
            stream,
            &mut args,
        )
    }

    /// Patch a cached prefill MegakernelProgram for a new chunk (tokens, start_pos).
    /// Faster than recompiling: only updates token IDs, KV write pointers, AttnPrefillInst, and position IDs.
    pub(crate) fn update_prefill_chunk(
        &mut self,
        tokens: &[u32],
        start_pos: u32,
        prefill_bufs: &mut super::PrefillBuffers,
    ) -> HipResult<()> {
        use super::instructions::{AttnPrefillInst, D2dCopyInst};
        let n = tokens.len();
        assert_eq!(n, self.prefill_n, "update_prefill_chunk: token count must match template size {}", self.prefill_n);

        // 1. Patch embedding token IDs
        for (i, &tok) in tokens.iter().enumerate() {
            let idx = self.prefill_embedding_start + i;
            unsafe {
                let inst = self.instructions[idx].words.as_mut_ptr() as *mut EmbeddingInst;
                (*inst).token_id = tok as u64;
            }
        }

        // 2. Patch KV write D2dCopy destinations
        let hd = self.head_dim_attn;
        let max_sl = self.max_seq_len as usize;
        for entry in &self.prefill_kv_entries {
            let (k_base, v_base) = self.kv_base_ptrs[entry.layer_kv_idx];
            let token_offset = start_pos as usize + entry.t;
            let byte_offset = (entry.h * max_sl * hd + token_offset * hd) * std::mem::size_of::<f32>();
            unsafe {
                let k_inst = self.instructions[entry.k_inst_idx].words.as_mut_ptr() as *mut D2dCopyInst;
                (*k_inst).dst = (k_base + byte_offset as u64) as *mut f32;
                let v_inst = self.instructions[entry.v_inst_idx].words.as_mut_ptr() as *mut D2dCopyInst;
                (*v_inst).dst = (v_base + byte_offset as u64) as *mut f32;
            }
        }

        // 3. Patch AttnPrefillInst start_pos fields
        for &idx in &self.prefill_attn_inst_indices {
            unsafe {
                let inst = self.instructions[idx].words.as_mut_ptr() as *mut AttnPrefillInst;
                (*inst).start_pos = start_pos as u64;
            }
        }

        // 4. Upload updated position IDs
        let mut pos_data = vec![0i32; n * 3];
        for t in 0..n {
            let pos = (start_pos + t as u32) as i32;
            pos_data[t * 3] = pos;
            pos_data[t * 3 + 1] = pos;
            pos_data[t * 3 + 2] = pos;
        }
        prefill_bufs.position_ids.copy_from_host(&pos_data)?;

        // 5. Re-upload modified instructions to device
        let total_words = self.instructions.len() * INST_SIZE;
        let mut flat: Vec<u64> = Vec::with_capacity(total_words);
        for inst in &self.instructions {
            flat.extend_from_slice(&inst.words);
        }
        self.device_program.copy_from_host(&flat)
    }
}
