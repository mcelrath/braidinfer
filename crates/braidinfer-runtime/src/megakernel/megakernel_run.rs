//! Megakernel runtime: per-step instruction patching, paged KV management, execution.
//! Extracted from megakernel.rs for maintainability.

use std::ffi::c_void;

use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::HipResult;

use crate::paged_kv::{PageAllocator, SequenceState};

use super::{Instruction, MegakernelProgram, INST_SIZE, CHUNK_TOKENS,
    OP_ATTN_PAGED_Q, OP_KV_QUANTIZE, OP_HALT};

impl MegakernelProgram {
    pub fn update_step(&mut self, token_id: u32, position: u32, stream: &Stream) -> HipResult<()> {
        assert!(position < self.max_seq_len, "position {position} >= max_seq_len {}", self.max_seq_len);


        // Update embedding token_id
        self.instructions[self.embedding_inst_idx].set_int(3, token_id as i32);

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
        let _nkh = self.num_kv_heads_attn;
        let hd = self.head_dim_attn;
        let max_sl = self.max_seq_len as usize;
        let head_stride = max_sl * hd; // elements between consecutive heads
        for (layer_i, head_indices) in self.kv_write_indices.iter().enumerate() {
            let (k_base, v_base) = self.kv_base_ptrs[layer_i];
            for (h, &(k_idx, v_idx)) in head_indices.iter().enumerate() {
                let offset = (h * head_stride + position as usize * hd) * std::mem::size_of::<f32>();
                self.instructions[k_idx].words[1] = k_base + offset as u64;
                self.instructions[v_idx].words[1] = v_base + offset as u64;
            }
        }

        // Update GQA attention seq_len
        let seq_len = position + 1;
        for &idx in &self.gqa_attn_inst_indices {
            self.instructions[idx].set_int(8, seq_len as i32);
        }

        // Upload entire instruction buffer in one hipMemcpyAsync call.
        // ~500 instructions × 128 bytes = ~64KB; one 64KB copy is cheaper than 24× 128-byte copies.
        let flat: Vec<u64> = self.instructions.iter().flat_map(|i| i.words).collect();
        // When dump is active, device_program has an OP_NOP header at offset 0;
        // instructions start at offset INST_SIZE words.
        let offset_words = if self.dump_buffer.is_some() { INST_SIZE } else { 0 };
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
        assert!(position < self.max_seq_len, "position {position} >= max_seq_len {}", self.max_seq_len);
        assert!(self.paged, "update_step_paged called on non-paged program");

        // 1. Patch embedding token_id
        self.instructions[self.embedding_inst_idx].set_int(3, token_id as i32);

        // 2. Append scalar position to position_table on device at offset [position]
        // Use synchronous hipMemcpy (not Async) because source is a stack local
        {
            let pos_scalar = position as i32;
            let pos_table_ptr = self.position_table.as_ref().expect("position_table not allocated").as_ptr();
            let dst = unsafe { (pos_table_ptr as *mut u8).add(position as usize * std::mem::size_of::<i32>()) };
            braidinfer_hip::error::check(unsafe {
                braidinfer_hip::ffi::hipMemcpy(
                    dst.cast(),
                    std::ptr::addr_of!(pos_scalar).cast(),
                    std::mem::size_of::<i32>(),
                    braidinfer_hip::ffi::hipMemcpyHostToDevice,
                )
            })?;
        }

        // Also update position_ids for mRoPE (same as flat path)
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

        // 3. Patch KV write D2D_COPY destinations from paged chunk layout [H,T,D]
        // current_chunk_offset() returns len (post-increment from append_token).
        // The write target is len-1 (the slot just reserved).
        let chunk_offset = (seq.current_chunk_offset() as usize).saturating_sub(1);
        let kv_stride = self.kv_stride_paged;
        let _nkh = self.num_kv_heads_attn;
        let hd = self.head_dim_attn;
        let chunk_head_stride = CHUNK_TOKENS * hd; // elements between heads within chunk

        for (layer_i, head_indices) in self.kv_write_indices.iter().enumerate() {
            let chunk_slot = if seq.chunks.is_empty() { 0 } else {
                seq.chunks.last().unwrap().slot_index()
            };
            let chunk_base = allocator.slot_ptr(chunk_slot) as u64;
            // layout: [layer0_K[nkh, chunk_tokens, hd], layer0_V[...], layer1_K, ...]
            let layer_k_offset = (layer_i * 2 * CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>()) as u64;
            let layer_v_offset = layer_k_offset + (CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>()) as u64;
            for (h, &(k_idx, v_idx)) in head_indices.iter().enumerate() {
                let head_byte_off = (h * chunk_head_stride + chunk_offset * hd) * std::mem::size_of::<f32>();
                let k_ptr = chunk_base + layer_k_offset + head_byte_off as u64;
                let v_ptr = chunk_base + layer_v_offset + head_byte_off as u64;
                self.instructions[k_idx].words[1] = k_ptr;
                self.instructions[v_idx].words[1] = v_ptr;
            }
        }

        // 4. Patch attention instructions
        let total_seq_len = (position + 1) as i32;
        let page_table_ptr = self.page_table.as_ref().expect("page_table not allocated").as_ptr() as u64;
        let pos_table_ptr = self.position_table.as_ref().expect("position_table not allocated").as_ptr() as u64;

        if self.quantized_kv && seq.chunks.len() > 1 {
            // Two-phase: quantized sealed chunks + f32 active chunk
            let num_sealed = seq.chunks.len() - 1;
            let sealed_tokens = (num_sealed * CHUNK_TOKENS) as i32;
            let active_tokens = total_seq_len - sealed_tokens;
            let nqh = self.instructions[self.attn_paged_inst_indices[0]].words[6] as u32;

            let quant_pt_ptr = self.quant_page_table.as_ref()
                .expect("quant_page_table not allocated").as_ptr() as u64;

            // Patch OP_ATTN_PAGED_Q: enable (grid_x=nqh), quant page table, sealed seq_len
            for &idx in &self.attn_quant_inst_indices {
                self.instructions[idx].words[0] =
                    (OP_ATTN_PAGED_Q as u64) | ((nqh as u64) << 32);
                self.instructions[idx].words[3] = quant_pt_ptr;
                self.instructions[idx].words[4] = pos_table_ptr;
                self.instructions[idx].set_int(9, sealed_tokens);
            }

            // Patch OP_ATTN_PAGED: f32 page table (only active chunk), active seq_len
            // The active chunk is the last one in seq.chunks. We put its pointer
            // at offset `sealed_tokens/CHUNK_TOKENS` in the f32 page table, but simpler:
            // point to a single-entry table with just the active chunk.
            // We reuse the main page_table — the active chunk ptr is at index `num_sealed`.
            for &idx in &self.attn_paged_inst_indices {
                // Point page_table at the last entry (active chunk)
                let active_pt_ptr = page_table_ptr + (num_sealed * std::mem::size_of::<u64>()) as u64;
                self.instructions[idx].words[3] = active_pt_ptr;
                self.instructions[idx].words[4] = pos_table_ptr
                    + (num_sealed * CHUNK_TOKENS * std::mem::size_of::<i32>()) as u64;
                self.instructions[idx].set_int(9, active_tokens);
            }
        } else {
            // No quantized chunks yet (or quantized_kv not enabled): all f32
            // Disable OP_ATTN_PAGED_Q (grid_x=0)
            for &idx in &self.attn_quant_inst_indices {
                self.instructions[idx].words[0] = OP_ATTN_PAGED_Q as u64; // grid_x=0
            }
            // OP_ATTN_PAGED sees all chunks, no partial_state
            for &idx in &self.attn_paged_inst_indices {
                self.instructions[idx].words[3] = page_table_ptr;
                self.instructions[idx].words[4] = pos_table_ptr;
                self.instructions[idx].set_int(9, total_seq_len);
                if !self.quantized_kv {
                    self.instructions[idx].words[14] = 0; // no partial_state
                }
            }
        }

        // 5. Upload page_table if chunk list changed
        if seq.chunks.len() != self.last_page_table_len {
            let page_table_dev = self.page_table.as_mut().expect("page_table not allocated");
            let host_ptrs: Vec<u64> = seq.chunks.iter()
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
        let offset_words = if self.dump_buffer.is_some() { INST_SIZE } else { 0 };
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
                    let (q_slot, q_ptr) = q_alloc.alloc()
                        .ok_or(braidinfer_hip::HipError(braidinfer_hip::ffi::hipErrorOutOfMemory))?;

                    // Run quantize kernel
                    self.quantize_sealed_chunk(f32_ptr, q_ptr, cfg, stream)?;
                    stream.synchronize()?;

                    // Track slot for cleanup
                    seq.quant_slots.push(q_slot);

                    // Upload quantized page table
                    let num_sealed = seq.chunks.len();
                    let quant_pt = self.quant_page_table.as_mut()
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
            // Allocate next f32 chunk for continued writing
            seq.append_token(allocator)?;
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
            self.position_table = Some(DeviceBuffer::alloc(self.device, self.max_seq_len as usize)?);
        }
        Ok(())
    }

    /// Enable quantized KV cache. Allocates scratch buffer and quantized page table.
    /// Call after init_paged_buffers, before first decode step.
    pub fn enable_quantized_kv(&mut self, max_chunks: usize, cfg: &crate::model::ModelConfig) -> HipResult<()> {
        let nqh = cfg.num_q_heads;
        let hd = cfg.head_dim;
        let num_attn_layers = cfg.layers.iter().filter(|l| l.layer_type == crate::model::LayerType::Attention).count();
        // Scratch: [nqh × (2+hd)] per attention layer (each layer gets its own scratch region)
        let scratch_per_layer = nqh * (2 + hd);
        let total_scratch = num_attn_layers * scratch_per_layer;
        self.quant_scratch = Some(DeviceBuffer::alloc(self.device, total_scratch)?);
        self.quant_page_table = Some(DeviceBuffer::alloc(self.device, max_chunks)?);

        // Patch OP_ATTN_PAGED_Q scratch pointers and OP_ATTN_PAGED partial_state pointers
        let scratch_base = self.quant_scratch.as_ref().unwrap().as_ptr() as u64;
        for (layer_i, &q_idx) in self.attn_quant_inst_indices.iter().enumerate() {
            let scratch_ptr = scratch_base + (layer_i * scratch_per_layer * std::mem::size_of::<f32>()) as u64;
            self.instructions[q_idx].words[1] = scratch_ptr;
        }
        for (layer_i, &p_idx) in self.attn_paged_inst_indices.iter().enumerate() {
            let scratch_ptr = scratch_base + (layer_i * scratch_per_layer * std::mem::size_of::<f32>()) as u64;
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
        let num_attn_layers = cfg.layers.iter().filter(|l| l.layer_type == crate::model::LayerType::Attention).count();
        let kv_stride = nkh * hd;
        let f32_layer_bytes = 2 * CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>();

        let mut instructions: Vec<Instruction> = Vec::new();
        for layer_i in 0..num_attn_layers {
            let f32_base = f32_chunk_ptr as u64 + (layer_i * f32_layer_bytes) as u64;
            let f32_k = f32_base;
            let f32_v = f32_base + (CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>()) as u64;

            for (is_v, f32_src) in [(false, f32_k), (true, f32_v)] {
                let (q1d, q1s, rd, rs) = quantized_kv_offsets(cfg, CHUNK_TOKENS, layer_i, is_v);
                let mut inst = Instruction::new(OP_KV_QUANTIZE, (nkh * hd) as u32);
                inst.words[1] = f32_src;
                inst.words[2] = quant_chunk_ptr as u64 + q1d as u64;
                inst.words[3] = quant_chunk_ptr as u64 + q1s as u64;
                inst.words[4] = quant_chunk_ptr as u64 + rd as u64;
                inst.words[5] = quant_chunk_ptr as u64 + rs as u64;
                inst.set_int(6, nkh as i32);
                inst.set_int(7, hd as i32);
                inst.set_int(8, CHUNK_TOKENS as i32);
                instructions.push(inst);
            }
        }
        instructions.push(Instruction::new(OP_HALT, 0));

        // Upload and execute
        let flat: Vec<u64> = instructions.iter().flat_map(|i| i.words).collect();
        let mut prog_buf = DeviceBuffer::<u64>::alloc(self.device, flat.len())?;
        prog_buf.copy_from_host(&flat)?;

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
}
