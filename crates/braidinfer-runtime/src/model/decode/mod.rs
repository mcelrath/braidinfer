use crate::megakernel::{CHUNK_TOKENS, MegakernelProgram, OP_HALT, OP_LM_HEAD, SHARED_LPROJ_TOTAL};
use crate::persistent_dispatch::BatchDispatcher;
use crate::weights::LayerWeights;

use super::Model;
use crate::weights::ModelError;

fn print_moe_stats_decode(
    tracer: &crate::tracer::Tracer,
    label: &str,
    output_slots_host: *const f32,
    num_gpus: usize,
    hidden_size: usize,
) {
    use crate::tracer::Probe;
    let stat = |name: &str, slice: &[f32]| {
        let mut n_nan = 0usize;
        let mut n_inf = 0usize;
        let mut max_abs = 0.0f32;
        for &x in slice {
            if x.is_nan() { n_nan += 1; }
            else if x.is_infinite() { n_inf += 1; }
            else if x.abs() > max_abs { max_abs = x.abs(); }
        }
        eprintln!(
            "[moe {label}] {name}: n={} nan={} inf={} max_abs={:.4} first4={:?}",
            slice.len(), n_nan, n_inf, max_abs, &slice[..slice.len().min(4)]
        );
    };
    for gpu_id in 0..num_gpus {
        let slice = unsafe { std::slice::from_raw_parts(output_slots_host.add(gpu_id * hidden_size), hidden_size) };
        stat(&format!("output_slots[gpu{}]", gpu_id), slice);
    }
    let empty: &[f32] = &[];
    let num_workers = num_gpus.saturating_sub(1);
    for w_idx in 0..num_workers {
        let buf = tracer.read_f32(Probe::WorkerFfnOut { worker: w_idx, layer: 0 }).unwrap_or(empty);
        stat(&format!("worker{}_ffn_out(gpu{})", w_idx, w_idx + 1), buf);
    }
}

impl Model {
    /// Persistent worker decode using paged KV cache.
    /// On first call: compiles paged megakernel, initializes page allocator + sequence,
    /// then launches persistent worker.
    /// Lazy-init the persistent worker + paged megakernel + tracer. Originally
    /// extracted by bd 9gmh Phase 2D; that phase is reverted, but the helper
    /// remains so decode_step_persistent can call it idempotently.
    ///
    /// bd a2dk: pass quantized=true when KV_QUANT=1 so the quant_allocator is
    /// created. Otherwise post_step_paged silently no-ops at chunk-seal and
    /// quant_page_table[0] stays 0, causing OP_ATTN_PAGED_Q to dereference
    /// `chunk_base + k_q1scale_off = 0 + 0x4000` and page-fault.
    pub(crate) fn ensure_persistent_worker_spawned(&mut self) -> Result<(), ModelError> {
        use crate::persistent_dispatch::PersistentDispatch;
        // bd 174k Phase C: single-GPU MoE may have already initialized
        // persistent_workers (and moe_p2p) via ensure_moe_workers_started
        // during enable_multi_gpu. In that case megakernel_paged still
        // needs lazy-init; only the dispatch creation is skipped.
        let need_megakernel_paged = self.megakernel_paged.is_none();
        let need_dispatch = self.persistent_workers.is_none();
        if !need_megakernel_paged && !need_dispatch {
            return Ok(());
        }

        let max_chunks = self.max_paged_chunks();

        if need_megakernel_paged {
            let mut mk = MegakernelProgram::compile_paged(self)?;
            mk.init_paged_buffers(max_chunks).map_err(ModelError::Hip)?;
            if std::env::var("KV_QUANT").as_deref() == Ok("1") {
                mk.enable_quantized_kv(max_chunks, &self.config).map_err(ModelError::Hip)?;
            }
            self.megakernel_paged = Some(mk);
        }

        // Patch LM head instruction to write to logits_mapped (host-mapped)
        // so CPU can read without hipMemcpy. Idempotent.
        {
            let mk = self.megakernel_paged.as_mut().unwrap();
            let lm_head_idx = mk
                .instructions
                .iter()
                .rposition(|inst| (inst.words[0] as u32) == OP_LM_HEAD)
                .expect("lm_head not found in paged megakernel");
            mk.instructions[lm_head_idx].words[1] =
                self.activations.logits_mapped.as_write_ptr() as u64;
        }

        let quantized = std::env::var("KV_QUANT").as_deref() == Ok("1");
        self.ensure_paged_decode_state(quantized)?;

        if need_dispatch {
            let shared_mem = SHARED_LPROJ_TOTAL as u32;
            let mut dispatch =
                PersistentDispatch::init(&[self.device], shared_mem, self.config.hidden_size, self.watchdog.clone()).map_err(ModelError::Hip)?;
            dispatch.ensure_sdma_stream(self.device).map_err(ModelError::Hip)?;
            self.persistent_workers = Some(dispatch);
        } else {
            // bd 174k Phase C: single-GPU MoE path created the dispatch in
            // ensure_moe_workers_started (with empty worker_devices, so GPU 0
            // wasn't launched). Add GPU 0 now — first decode step is when the
            // persistent_worker is allowed to hold CUs.
            let shared_mem = SHARED_LPROJ_TOTAL as u32;
            let dispatch = self.persistent_workers.as_mut().unwrap();
            if !dispatch.has_worker(self.device.0 as usize) {
                dispatch.add_device(self.device, shared_mem)
                    .map_err(ModelError::Hip)?;
            }
        }

        if !self.tracer.enabled() {
            let sdma = self.persistent_workers.as_ref().unwrap()
                .sdma_stream(self.device.0 as usize);
            match crate::tracer::Tracer::from_env(vec![sdma]) {
                Ok(t) => self.tracer = t,
                Err(e) => eprintln!("[braidinfer] tracer init failed: {e:?}"),
            }
        }
        Ok(())
    }

    pub(super) fn decode_step_persistent(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<Vec<f32>, ModelError> {
        // bd vaaf: lazy-init was duplicated here and in ensure_persistent_worker_spawned
        // after bd 174k Phase C decoupled megakernel_paged from dispatch. Both are
        // idempotent now — consolidate to the single helper.
        self.ensure_persistent_worker_spawned()?;

        // Enable dump buffer on first traced step. enable_dump_persistent allocates VRAM
        // buffers only (no NOP header / device_program rebuild — persistent path reads
        // dump pointers from WorkerQueue). set_trace_dump_ptrs writes them into the queue.
        //
        // CAPACITY SIZING: the kernel-side dump pipeline (kernels/dump.h) fires
        // unconditionally on EVERY dump-eligible opcode (OP_RMSNORM, OP_LINEAR_PROJ,
        // OP_RESIDUAL_ADD, OP_SCALE_ADD, OP_FFN_DOWN_RES, etc.), not only the ones
        // listed in MegakernelProgram::trace_probe_map. Drain-side filtering
        // (PersistentDispatch::drain_trace_dump) maps inst_idx → Probe via
        // trace_probe_map and discards slots that don't match. Therefore the slot
        // capacity must accommodate ALL dump-eligible instructions in the program.
        // We use instructions.len() as a safe upper bound (worst case: every
        // instruction is dump-eligible). Bounded at 4096 to cap the VRAM footprint
        // at 4096 × 32KB = 128MB.
        //
        // Follow-up bd k357 will add a kernel-side trace_mask filter so dumps fire
        // ONLY at trace_probe_map sites — eliminating the bandwidth waste and
        // allowing capacity = trace_probe_map.len() + small safety.
        if self.tracer.enabled() {
            let mk = self.megakernel_paged.as_mut().unwrap();
            if !mk.dump_active() {
                let max_slots = (mk.instructions.len() as i32).min(4096);
                if let Err(e) = mk.enable_dump_persistent(max_slots) {
                    eprintln!("[braidinfer] enable_dump_persistent failed: {e:?}");
                } else {
                    let dispatch = self.persistent_workers.as_mut().unwrap();
                    dispatch.set_trace_dump_ptrs(self.device.0 as usize, mk);
                }
            }
        }

        // Write position_ids directly to host-mapped memory (no hipMemcpy)
        self.set_position(position).map_err(ModelError::Hip)?;

        // Append token to paged sequence state (allocates chunk slot if needed).
        // Phase C: pass host_page_allocator so the fallback to HostPinned tier
        // fires when VRAM is exhausted and the host tier is enabled.
        {
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            let host_alloc = self.host_page_allocator.as_mut();
            seq_mut.append_token(position as i32, alloc_mut, host_alloc).map_err(ModelError::Hip)?;
        }

        // Host-side patching only: update instructions without hipMemcpyAsync.
        // Persistent caller will dispatch via dispatch_batch_slice instead.
        {
            let mk = self.megakernel_paged.as_mut().unwrap();
            let seq = self.paged_seq.as_ref().unwrap();
            let allocator = self.page_allocator.as_ref().unwrap();
            mk.update_step_paged_no_upload(token_id, position, seq, allocator)
                .map_err(ModelError::Hip)?;
        }

        // Dispatch: send all instructions (excluding HALT) via persistent worker mailbox.
        // HALT EXCLUSION (CRITICAL): the persistent cooperative kernel loops forever
        // waiting for the next batch; HALT would cause it to exit. We must never
        // send HALT over the mailbox — only send to halt_idx (exclusive).
        let mk = self.megakernel_paged.as_ref().unwrap();
        let halt_idx = mk
            .instructions
            .iter()
            .position(|inst| (inst.words[0] as u32 as u64) == OP_HALT as u64)
            .unwrap_or(mk.instructions.len());
        let dispatch = self.persistent_workers.as_mut().unwrap();
        dispatch.dispatch_batch_slice(0, &mk.instructions[..halt_idx]);

        // Drain trace dump after ack (dispatch_batch_slice is synchronous: wait_ack
        // already returned). SDMA copies slot data from VRAM to host shadows in tracer.
        if self.tracer.enabled() {
            let mk_ref = self.megakernel_paged.as_ref().unwrap();
            let gpu_idx = self.device.0 as usize;
            let dispatch = self.persistent_workers.as_mut().unwrap();
            if let Err(e) = dispatch.drain_trace_dump(gpu_idx, mk_ref, &mut self.tracer) {
                eprintln!("[braidinfer] drain_trace_dump failed: {e:?}");
            }
        }

        // Read logits directly from host-mapped memory (no hipMemcpy needed)
        let logits = unsafe {
            std::slice::from_raw_parts(
                self.activations.logits_mapped.host_ptr(),
                self.config.vocab_size,
            )
        }
        .to_vec();

        // Tracer: per-layer probes (embed, L{i}.post_mixer, L{i}.post_ffn, final_norm)
        // are now populated by drain_trace_dump above. Top10 logits are host-side only.
        if self.tracer.enabled() {
            use crate::tracer::Probe;
            // FinalNorm is already in shadows from drain_trace_dump; no SDMA needed.
            // Keep the explicit capture as a fallback if trace_probe_map is incomplete.
            let _ = self.tracer.capture_f32(0, Probe::FinalNorm, &self.activations.normed);
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1)); // NaN-safe (was partial_cmp().unwrap() — panicked on NaN logits)
            let top10: Vec<f32> = indexed.iter().take(10)
                .flat_map(|&(id, val)| [id as f32, val])
                .collect();
            self.tracer.record_host_f32(Probe::Logits { top_k: 10 }, &top10);
            let _ = self.tracer.drain();
        }

        // Post-step: handle chunk-seal lifecycle (including KV quantization via mailbox).
        // quantize_sealed_chunk_via_worker dispatches OP_KV_QUANTIZE instructions over
        // the persistent worker mailbox — safe under the cooperative kernel (no hipMemcpy).
        {
            let mk = self.megakernel_paged.as_mut().unwrap();
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            let q_alloc = self.quant_allocator.as_mut();
            let dispatch = self.persistent_workers.as_mut()
                .expect("persistent_workers must be initialized in decode_step_persistent");
            mk.post_step_paged(position, seq_mut, alloc_mut, q_alloc, &self.config, dispatch)
                .map_err(ModelError::Hip)?;
        }

        // Chunk-seal mirror hook (wt1 P2-c): enqueue SDMA copy of the just-sealed
        // chunk to pinned host memory. The persistent_workers SDMA stream is used so
        // the copy runs out-of-band without blocking the CPU decode loop.
        if (position as usize + 1) % CHUNK_TOKENS == 0 {
            if let Some(dispatch) = self.persistent_workers.as_mut() {
                let alloc = self.page_allocator.as_ref().unwrap();
                let seq = self.paged_seq.as_ref().unwrap();
                if let Some(sealed) = seq.chunks.last() {
                    let vram_ptr = alloc.slot_ptr(sealed.slot_index());
                    dispatch.kv_mirror_chunk(0, vram_ptr).map_err(ModelError::Hip)?;
                    dispatch.drain_kv_chunk_mirror(0, position).map_err(ModelError::Hip)?;
                }
            }
        }

        self.seq_len = position + 1;
        Ok(logits)
    }

    /// Multi-GPU persistent worker decode for MoE models.
    /// Initializes P2P workers on first call, then delegates to decode_step_p2p.
    pub(super) fn decode_step_persistent_multi_gpu(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<Vec<f32>, ModelError> {
        // Lazy-init: compile P2P megakernel + launch workers on ALL GPUs.
        // For MoE multi-GPU, ensure_moe_workers_started already populated GPUs 1..N
        // during prefill, but GPU 0's persistent worker is launched here on the first
        // decode call (after prefill kbk launches on GPU 0 are complete). For non-MoE
        // multi-GPU, persistent_workers is None until this point.
        let needs_gpu0 = self
            .persistent_workers
            .as_ref()
            .map(|d| !d.has_worker(0))
            .unwrap_or(true);
        if needs_gpu0 {
            self.init_multi_gpu_persistent()?;
        }

        // P2P megakernel is always initialized when has_moe && num_gpus > 1.
        // enable_multi_gpu() in model_load.rs silently no-ops on !has_moe;
        // the cli.rs auto-rule hard-errors on dense-too-big-for-one-GPU before
        // reaching here. Belt-and-suspenders InvalidConfig in case a future
        // load-rule change ever lets a non-MoE multi-GPU Model construct.
        if self.megakernel_multi_gpu_p2p.is_none() {
            return Err(ModelError::InvalidConfig(
                "decode_step_persistent_multi_gpu reached without an MoE-P2P \
                 megakernel program. This indicates a non-MoE multi-GPU Model \
                 was constructed, which is not a supported configuration in \
                 braidinfer (no tensor-parallel/pipeline-parallel dispatch for \
                 dense models). cli.rs::apply_auto_modes should have rejected \
                 this at startup; this is a defensive guard.".into(),
            ));
        }
        self.decode_step_p2p(token_id, position)
    }

    /// Lazily initialize the MoE mailbox dispatch context. On multi-GPU this
    /// launches peer-GPU persistent workers and compiles the P2P decode
    /// megakernel. On single-GPU MoE (bd 174k Phase C) it only initializes
    /// the moe_p2p host-mapped UC routing buffers and skips both worker
    /// launches and compile_multi_gpu_p2p — single-GPU decode keeps using
    /// megakernel_paged.
    ///
    /// Safe to call during prefill (no cooperative kernel running on GPU 0 yet,
    /// so hipMalloc is allowed).
    pub(crate) fn ensure_moe_workers_started(&mut self) -> Result<(), ModelError> {
        use crate::persistent_dispatch::PersistentDispatch;
        if self.moe_p2p.is_some() {
            return Ok(());
        }
        if !self.has_moe {
            return Ok(());
        }
        // bd 174k Phase C: lift the multi_gpu==None and num_gpus<=1 early-returns.
        // Single-GPU MoE still initializes moe_p2p (with num_workers=0) so the
        // unified mailbox-routed prefill path works.
        let num_gpus = self.multi_gpu.as_ref().map_or(1, |m| m.num_devices);
        let is_single_gpu = num_gpus <= 1;
        // bd ntz6: the moe_gemv_worker.hip kernel was orphaned (never compiled,
        // OP_EXPERT_FFN had no producer/consumer). Workers run persistent_worker.hsaco
        // via OP_MOE_FFN_REMOTE. shared_mem floor is just SHARED_LPROJ_TOTAL.
        let shared_mem_persistent = SHARED_LPROJ_TOTAL;
        let hs = self.config.hidden_size;
        let max_eis = self
            .config
            .layers
            .iter()
            .filter_map(|l| match &l.ffn_type {
                crate::config::FfnType::MoE { expert_intermediate_size, .. } => {
                    Some(*expert_intermediate_size)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let worker_devices: Vec<_> = (1..num_gpus)
            .map(|i| braidinfer_core::types::DeviceId(i as u32))
            .collect();
        let num_total_layers = self.config.layers.len();
        let dist_refs: Vec<Option<&crate::weights::DistributedMoeWeights>> =
            self.distributed_moe.iter().map(|d| d.as_ref()).collect();
        let gate_up_in_dim = self.config.moe_latent_size.unwrap_or(hs);
        let mut p2p = crate::moe_p2p::MoeP2pContext::init(
            self.device,
            &worker_devices,
            hs,
            gate_up_in_dim,
            max_eis,
            num_total_layers,
            &dist_refs,
            shared_mem_persistent,
        )
        .map_err(ModelError::Hip)?;
        // bd 174k Phase C: compile_multi_gpu_p2p produces the P2P decode
        // megakernel (patches OP_BARRIER -> OP_MOE_DISPATCH). Single-GPU
        // decode keeps using megakernel_paged via decode_step — skip the
        // P2P compile for num_gpus=1 to leave megakernel_multi_gpu_p2p
        // None.
        if !is_single_gpu {
            let mk_p2p = MegakernelProgram::compile_multi_gpu_p2p(self, &mut p2p)
                .map_err(ModelError::Hip)?;
            self.megakernel_multi_gpu_p2p = Some(mk_p2p);
        }
        self.moe_p2p = Some(p2p);

        // wt1 P2-a: PersistentDispatch now owns SDMA streams (one per GPU,
        // indexed by DeviceId.0). Stream allocation for worker GPUs (1..N-1)
        // happens inside `add_device` immediately BEFORE the cooperative
        // launch (hipStreamCreate on a GPU with an active coop kernel can
        // wedge — see sdma_under_coop_fork probe). GPU 0's stream is
        // allocated here explicitly because GPU 0's persistent worker is
        // added later in `init_multi_gpu_persistent` on first decode call.
        //
        // Order matters: init_with_total → worker GPU streams + workers
        // launched; then ensure_sdma_stream(gpu0) while GPU 0 still has no
        // persistent worker; then DecodeMirror::alloc borrowing all streams.
        {
            let mut dispatch = PersistentDispatch::init_with_total(
                num_gpus,
                &worker_devices,
                shared_mem_persistent,
                hs,
                self.watchdog.clone(),
            )
            .map_err(ModelError::Hip)?;
            // GPU 0 stream — safe to allocate now (GPU 0 has no persistent
            // worker yet; still in kbk-launch phase).
            dispatch
                .ensure_sdma_stream(self.device)
                .map_err(ModelError::Hip)?;
            self.persistent_workers = Some(dispatch);
            eprintln!(
                "  MoE P2P dispatch initialized: {} worker GPUs (prefill path)",
                num_gpus - 1
            );
        }

        // Tracer: lazy-init from env vars. Borrows SDMA streams from PersistentDispatch
        // (owned there; destroyed in PersistentDispatch::Drop after workers exit).
        if !self.tracer.enabled() {
            let dispatch = self
                .persistent_workers
                .as_ref()
                .expect("PersistentDispatch must be initialized before Tracer");
            let mut streams: Vec<braidinfer_hip::ffi::hipStream_t> =
                Vec::with_capacity(1 + worker_devices.len());
            streams.push(dispatch.sdma_stream(self.device.0 as usize));
            for &dev in &worker_devices {
                streams.push(dispatch.sdma_stream(dev.0 as usize));
            }
            self.tracer = crate::tracer::Tracer::from_env(streams).map_err(ModelError::Hip)?;
        }
        Ok(())
    }

    /// DEBUG_P2P_HIDDEN probe: copy first 16 floats of `activations.hidden` via SDMA
    /// (GPU 0 stream) and print a one-line diagnostic. Safe under the persistent
    /// cooperative kernel — SDMA operates independently of the held CUs.
    /// No-op when `debug_p2p_hidden` is false or tracer is disabled.
    fn probe_hidden_after_segment(&mut self, label: &str) {
        if !self.debug_p2p_hidden || !self.tracer.enabled() {
            return;
        }
        use braidinfer_hip::device::DeviceGuard;
        use crate::tracer::Probe;
        let hidden_ptr = self.activations.hidden.as_ptr() as *const f32;
        let device = self.device;
        let hidden_size = self.config.hidden_size;
        let _guard = match DeviceGuard::switch_to(device) {
            Ok(g) => g,
            Err(e) => { eprintln!("DBG hidden[{label}] DeviceGuard error: {e:?}"); return; }
        };
        let n = 16usize.min(hidden_size);
        let copy_bytes = n * std::mem::size_of::<f32>();
        if let Err(e) = self.tracer.capture(0, Probe::Hidden { gpu: 0, head_only: true }, hidden_ptr as *const u8, copy_bytes) {
            eprintln!("DBG hidden[{label}] capture error: {e:?}");
            return;
        }
        if let Err(e) = self.tracer.drain() {
            eprintln!("DBG hidden[{label}] drain error: {e:?}");
            return;
        }
        let buf = self.tracer.read_f32(Probe::Hidden { gpu: 0, head_only: true }).unwrap_or(&[]);
        let buf = &buf[..n.min(buf.len())];
        if buf.len() < 4 { return; }
        let mut nan = false;
        let mut inf = false;
        let mut max_abs = 0.0f32;
        for &v in buf { if v.is_nan() { nan = true; } if v.is_infinite() { inf = true; } if v.abs() > max_abs { max_abs = v.abs(); } }
        eprintln!(
            "DBG hidden[{label}] nan={nan} inf={inf} max_abs={max_abs:.3e} h[0..4]={:.3e},{:.3e},{:.3e},{:.3e}",
            buf[0], buf[1], buf[2], buf[3]
        );
    }

    pub(crate) fn init_multi_gpu_persistent(&mut self) -> Result<(), ModelError> {
        use crate::persistent_dispatch::PersistentDispatch;
        let _ = PersistentDispatch::init; // silence unused import in single-gpu-skip path

        let num_gpus = self.multi_gpu.as_ref().unwrap().num_devices;
        // bd ntz6: moe_gemv_worker.hip orphan deleted; shared_mem floor is SHARED_LPROJ_TOTAL.
        let shared_mem = SHARED_LPROJ_TOTAL;
        let hs = self.config.hidden_size;
        // For MoE multi-GPU: workers (GPUs 1..N-1) were already launched by
        // ensure_moe_workers_started during prefill. Add GPU 0 now (after
        // prefill kbk completes — its persistent kernel can hold all CUs).
        // For non-MoE multi-GPU or single-GPU: nothing exists yet, allocate
        // a fresh PersistentDispatch with the required slots and launch only
        // GPU 0 (workers don't need persistent_worker — there's no MoE).
        self.ensure_moe_workers_started()?;

        if self.persistent_workers.is_none() {
            // Non-MoE or single-GPU path: create a single-slot dispatcher.
            let total = num_gpus.max(1);
            let dispatch = PersistentDispatch::init_with_total(
                total,
                &[self.device],
                shared_mem,
                hs,
                self.watchdog.clone(),
            ).map_err(ModelError::Hip)?;
            self.persistent_workers = Some(dispatch);
        } else {
            // MoE path: workers already up; add GPU 0.
            let dispatch = self.persistent_workers.as_mut().unwrap();
            if !dispatch.has_worker(0) {
                dispatch.add_device(self.device, shared_mem)
                    .map_err(ModelError::Hip)?;
            }
        }
        Ok(())
    }

    /// Snapshot MoE output_slots (host-mapped, direct read) and per-worker FFN_REMOTE
    /// local_output (SDMA copy) via tracer, then print stats. No-op if tracer is
    /// disabled or moe_p2p is None.
    fn decode_mirror_moe_snapshot(&mut self, layer_idx: usize) {
        if !self.tracer.enabled() { return; }
        let Some(p2p) = self.moe_p2p.as_ref() else { return; };
        use braidinfer_hip::device::{Device, DeviceGuard};
        use crate::tracer::Probe;
        let output_slots_host = p2p.output_slots.host_ptr() as *const f32;
        let worker_local_outputs: Vec<*const f32> = p2p.workers.iter()
            .map(|w| w.local_output.as_ptr() as *const f32).collect();
        let worker_devs: Vec<_> = p2p.workers.iter().map(|w| w.device).collect();
        let num_gpus = p2p.num_gpus;
        let hidden_size = p2p.hidden_size;
        let copy_bytes = hidden_size * std::mem::size_of::<f32>();
        let _guard = match worker_devs.first().copied() {
            Some(d) => match DeviceGuard::switch_to(d) {
                Ok(g) => Some(g),
                Err(e) => { eprintln!("[moe mirror L{layer_idx}] DeviceGuard: {e:?}"); return; }
            },
            None => None,
        };
        for (w_idx, &dev) in worker_devs.iter().enumerate() {
            if w_idx >= worker_local_outputs.len() { break; }
            if let Err(e) = Device::set_current(dev) { eprintln!("[moe mirror L{layer_idx}] set_current: {e:?}"); return; }
            let src = worker_local_outputs[w_idx];
            if let Err(e) = self.tracer.capture(w_idx + 1, Probe::WorkerFfnOut { worker: w_idx, layer: 0 }, src as *const u8, copy_bytes) {
                eprintln!("[moe mirror L{layer_idx}] capture failed: {e:?}"); return;
            }
        }
        if let Err(e) = self.tracer.drain() {
            eprintln!("[moe mirror L{layer_idx}] drain failed: {e:?}"); return;
        }
        let label = format!("L{layer_idx}");
        print_moe_stats_decode(&self.tracer, &label, output_slots_host, num_gpus, hidden_size);
    }

    pub(super) fn decode_step_p2p(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        self.decode_step_p2p_inner(token_id, position)
    }

    pub(super) fn decode_step_p2p_inner(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        // 5ax-decode probe: BRAIDINFER_MTYPE_AUDIT=1 dumps the MTYPE table
        // for all cross-agent and reused buffers in the decode path. Runs
        self.set_position(position).map_err(ModelError::Hip)?;

        // β'''' sentinel monotonic-seq (bd braidinfer-sm16, udi #3189): use
        // per-step monotonic value (position+1) so no CPU reset is needed.
        // Avoids the race where CPU 0-write to normed_seq doesn't drain to
        // peer-GPU UTCL2 before workers spin-wait → workers saw stale 1
        // from prior step → exited spin too early → read stale normed_stage
        // → all-workers simultaneous NaN at later pos (Sig B canonical T6).
        // Patch producer D2dCopy signal_seq at each attn flush boundary.
        let seq_value = (position as u64) + 1;
        let mk = self.megakernel_multi_gpu_p2p.as_mut().unwrap();
        let boundaries: Vec<usize> = mk
            .multi_gpu_attn_boundaries
            .iter()
            .map(|(flush_idx, _)| *flush_idx)
            .collect();
        for flush_idx in &boundaries {
            // D2dCopyInst layout: words[7] = signal_seq (see instructions.rs).
            mk.instructions[*flush_idx].words[7] = seq_value;
        }
        // yef5.2 Step A: patch moe_act D2D signal_seq to (position+1).
        // D2dCopyInst layout: words[7] = signal_seq.
        let moe_act_d2d: Vec<usize> = mk.moe_act_d2d_indices.iter().map(|(i, _)| *i).collect();
        for d2d_idx in &moe_act_d2d {
            mk.instructions[*d2d_idx].words[7] = seq_value;
        }
        // yef5.2 P1d (option c): patch each OP_MOE_DISPATCH_POST seq_counter to (position+1) so the
        // POST's acquire-spin on the worker result sentinels matches the workers' release value.
        // MoeDispatchInst layout: words[6] = seq_counter (instructions.rs). Empty unless YEF52_OPTION_C.
        let moe_post_seq: Vec<usize> = mk.moe_post_seq_indices.clone();
        for post_idx in &moe_post_seq {
            mk.instructions[*post_idx].words[6] = seq_value;
        }

        // Update per-step state in p2p megakernel (embedding ptr, mRoPE positions, etc.)
        mk.update_step_host_only(token_id, position)?;
        // Search for OP_LM_HEAD by opcode rather than using a hardcoded offset.
        // words[1] = output pointer (LinearProjInst layout; see instructions.rs:556).
        let lm_head_idx = mk
            .instructions
            .iter()
            .rposition(|inst| (inst.words[0] as u32) == OP_LM_HEAD)
            .expect("lm_head not found in p2p megakernel");
        mk.instructions[lm_head_idx].words[1] = self.activations.logits_mapped.as_write_ptr() as u64;

        let _hs = self.config.hidden_size;
        // bd srg6.X2: head-parallel is now gated on per-worker paged KV state
        // (workers[0].paged_seq.is_some()), not on legacy attn_kv_caches.
        let has_head_parallel = self
            .multi_gpu
            .as_ref()
            .map(|m| !m.workers.is_empty() && m.workers[0].paged_seq.is_some())
            .unwrap_or(false);
        let use_distributed_qkv = has_head_parallel && {
            !self
                .megakernel_multi_gpu_p2p
                .as_ref()
                .unwrap()
                .multi_gpu_attn_boundaries
                .is_empty()
        };
        let attn_boundaries: Vec<(usize, usize)> = if has_head_parallel {
            let mk_ref = self.megakernel_multi_gpu_p2p.as_ref().unwrap();
            if use_distributed_qkv {
                mk_ref.multi_gpu_attn_boundaries.clone()
            } else {
                mk_ref
                    ._mrope_inst_indices
                    .iter()
                    .zip(mk_ref.gqa_attn_inst_indices.iter())
                    .map(|(&m, &g)| (m, g))
                    .collect()
            }
        } else {
            Vec::new()
        };
        // MoE dispatch boundaries: instruction indices of OP_MOE_DISPATCH (post-Pass-2 remap).
        // For each, CPU dispatches OP_MOE_FFN_REMOTE on every worker BEFORE firing the
        // GPU 0 batch containing the OP_MOE_DISPATCH instruction.
        let moe_boundaries: Vec<(usize, usize)> = self.megakernel_multi_gpu_p2p
            .as_ref().unwrap().barrier_layer_map.clone();
        let n_inst = self
            .megakernel_multi_gpu_p2p
            .as_ref()
            .unwrap()
            .instructions
            .len();

        // bd srg6.X2: per-worker paged KV state update — append current token
        // position to each worker's paged_seq (allocates a new chunk slot when
        // the current chunk fills). Must run BEFORE any attention layer dispatch
        // so that dispatch_head_parallel_attention sees the correct chunk/slot.
        // Also update per-worker position_table (3 i32s per token) and page_table
        // via write_volatile (MappedHostBuffer — no HIP API, safe under persistent worker).
        // bd 4n5 P3b: disaggregated decode — GPU 0 runs ALL attention (the p2p
        // megakernel's OP_ATTN_PAGED, patched below from self.paged_seq via
        // update_step_paged_no_upload). Workers do NOT attend in decode (P2 removed
        // head-parallel dispatch for MoE decode), so the old per-worker KV maintenance
        // is VESTIGIAL — it wrote worker page/position tables nothing reads AND did
        // worker.paged_seq.append_token, a per-step worker-pool VRAM alloc that could
        // return OutOfMemory at runtime for KV nobody uses. Removed (both the wasted
        // work and that gratuitous OOM path). Only GPU 0's self.paged_seq grows here.
        //
        // The worker KV buffers stay ALLOCATED: they're reused by the next
        // head-parallel prefill, and they cannot be freed mid-session anyway (the
        // worker cooperative kernels run for the model's lifetime, so hipFree would
        // deadlock). Pools are pre-allocated once (max_chunks = max_seq_len/CHUNK_TOKENS)
        // and reused — no per-request grow/free, hence no fragmentation and no surprise
        // runtime OOM within max_seq_len; past it, a clean hipErrorOutOfMemory (not a
        // fragmentation crash). GPU 0's append below is the only runtime KV alloc.
        if has_head_parallel {
            let seq = self.paged_seq.as_mut()
                .expect("self.paged_seq must be initialized for GPU 0 paged decode");
            let alloc = self.page_allocator.as_mut()
                .expect("self.page_allocator must be initialized for GPU 0");
            let host_alloc = self.host_page_allocator.as_mut();
            seq.append_token(position as i32, alloc, host_alloc)
                .map_err(ModelError::Hip)?;
        }

        // P2 (braidinfer-4n5.6): GPU 0 runs the full local paged-KV attention
        // sequence (OP_ATTN_PAGED_Q + OP_ATTN_PAGED) baked into the p2p megakernel.
        // The p2p program now has paged=true (compile_inner_p2p uses paged=true,multi_gpu=true),
        // so it has attn_paged_inst_indices + paged_kv.page_table that must be patched
        // each step — identical to decode_step_persistent's update_step_paged_no_upload path.
        //
        // self.paged_seq and self.page_allocator are initialized by prefill (ensure_paged_decode_state
        // called from prefill_mixed_chunk_paged) and the token was appended above in the
        // has_head_parallel block (or we must do it here if has_head_parallel is false).
        //
        // NOTE: when has_head_parallel=false (empty multi_gpu_attn_boundaries in P2),
        // self.paged_seq.append_token must be called here since the has_head_parallel
        // block was skipped. When has_head_parallel=true, append already happened above.
        if !has_head_parallel {
            if let (Some(seq), Some(alloc)) = (self.paged_seq.as_mut(), self.page_allocator.as_mut()) {
                let host_alloc = self.host_page_allocator.as_mut();
                seq.append_token(position as i32, alloc, host_alloc).map_err(ModelError::Hip)?;
            }
        }
        // Patch the p2p program's attn_paged_inst_indices + page_table buffer.
        if self.megakernel_multi_gpu_p2p.as_ref().map(|mk| mk.paged).unwrap_or(false) {
            if let (Some(seq), Some(alloc)) = (self.paged_seq.as_ref(), self.page_allocator.as_ref()) {
                let mk_p2p = self.megakernel_multi_gpu_p2p.as_mut().unwrap();
                mk_p2p.update_step_paged_no_upload(token_id, position, seq, alloc)
                    .map_err(ModelError::Hip)?;
            }
        }

        let mut attn_i = 0usize;
        let mut moe_i = 0usize;
        let mut i = 0usize;
        let mut seg_start = 0usize; // start of current segment in mk.instructions
        // Gather instructions returned from dispatch_head_parallel_attention, prepended
        // to the next segment to save one dispatch round-trip per attention layer.
        let mut pending_gather: Vec<crate::megakernel::Instruction> = Vec::new();

        while i < n_inst {
            let opcode = self.megakernel_multi_gpu_p2p.as_ref().unwrap().instructions[i].words[0] as u32 as u64;
            if opcode == OP_HALT as u64 {
                break;
            }

            // MoE boundary (Phase 7 split): OP_MOE_DISPATCH at index i is the PRE op
            // (zero output_slots[0] + GPU 0 local experts). The instruction at i+1
            // is OP_MOE_DISPATCH_POST (sum across slots). To restore parallelism
            // between GPU 0's local experts and worker remote experts:
            //   1. Flush GPU 0 segment [seg_start..=i] including PRE — fire async.
            //   2. Concurrently fire OP_MOE_FFN_REMOTE on each worker (async).
            //   3. Wait for ALL acks (GPU 0 PRE + workers).
            //   4. Resume segment at i+1 (OP_MOE_DISPATCH_POST runs as part of the
            //      next batch, which starts only after wait_ack guarantees workers
            //      have written output_slots).
            if moe_i < moe_boundaries.len() && i == moe_boundaries[moe_i].0 {
                let layer_idx = moe_boundaries[moe_i].1;
                // Build combined GPU 0 segment [seg_start..=i] (including PRE).
                let seg_end_inclusive = i + 1;
                let mk_insts = &self.megakernel_multi_gpu_p2p.as_ref().unwrap().instructions[seg_start..seg_end_inclusive];
                let combined: Vec<crate::megakernel::Instruction> = if pending_gather.is_empty() {
                    mk_insts.to_vec()
                } else {
                    let mut c = std::mem::take(&mut pending_gather);
                    c.extend_from_slice(mk_insts);
                    c
                };
                // yef5.2 Step A: fire PRE async (workers spin on moe_act_sentinel);
                // workers acquire-spin on activation_staging_vram sentinel so they
                // cannot race the D2D copy. Capture GPU0 PRE ack-seq and include it
                // in try_wait_acks_many (reverse ack unchanged).
                let gpu0_pre_seq: u32;
                {
                    let dispatch: &mut dyn BatchDispatcher =
                        self.persistent_workers.as_mut().unwrap();
                    // Fire all chunks; last chunk's seq is the one to wait on.
                    let mut last_seq = 0u32;
                    for chunk in combined.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS) {
                        last_seq = dispatch.dispatch_batch_fire(0, chunk);
                    }
                    gpu0_pre_seq = last_seq;
                }
                self.probe_hidden_after_segment(&format!("moe-pre L{}", layer_idx));
                let gpu0_seq: Option<u32> = Some(gpu0_pre_seq);
                // Dispatch OP_MOE_FFN_REMOTE on each worker. Workers acquire-spin
                // on moe_act_sentinel before reading activation_staging_vram.
                self.dispatch_moe_workers_decode_async(layer_idx, gpu0_seq, seq_value)?;
                // MoE mirror: snapshot output_slots + worker FFN outputs after ack.
                if self.tracer.enabled() {
                    self.decode_mirror_moe_snapshot(layer_idx);
                }
                // Resume segment AT i+1 (OP_MOE_DISPATCH_POST and beyond).
                seg_start = i + 1;
                moe_i += 1;
                i = seg_start;
                continue;
            }

            // Head-parallel attention boundary: flush segment, dispatch parallel QKV+GQA
            if has_head_parallel && attn_i < attn_boundaries.len() {
                let (flush_idx, resume_idx) = attn_boundaries[attn_i];
                if i == flush_idx {
                    // Include this instruction in the segment, then flush.
                    // Prepend any pending gather from previous attention layer.
                    let seg_end = i + 1;
                    {
                        let mk_insts = &self.megakernel_multi_gpu_p2p.as_ref().unwrap().instructions[seg_start..seg_end];
                        let dispatch: &mut dyn BatchDispatcher =
                            self.persistent_workers.as_mut().unwrap();
                        if pending_gather.is_empty() {
                            dispatch.dispatch_batch_slice(0, mk_insts);
                        } else {
                            // Fuse pending gather with this segment: one combined dispatch
                            let mut combined = std::mem::take(&mut pending_gather);
                            combined.extend_from_slice(mk_insts);
                            dispatch.dispatch_batch_slice(0, &combined);
                        }
                    }
                    self.probe_hidden_after_segment(&format!("attn-pre #{}", attn_i));
                    pending_gather = self.dispatch_head_parallel_attention(attn_i, position)?;
                    attn_i += 1;
                    i = if use_distributed_qkv { resume_idx } else { resume_idx + 1 };
                    seg_start = i;
                    continue;
                }
            }

            i += 1;
        }

        // Dispatch remaining segment, prepending any pending gather
        if seg_start < i || !pending_gather.is_empty() {
            let mk_insts = &self.megakernel_multi_gpu_p2p.as_ref().unwrap().instructions[seg_start..i];
            // Combine pending gather + remaining segment into one owned Vec
            // so the dispatcher borrow doesn't conflict with mk_insts.
            let combined: Vec<crate::megakernel::Instruction> = if pending_gather.is_empty() {
                mk_insts.to_vec()
            } else {
                let mut c = pending_gather;
                c.extend_from_slice(mk_insts);
                c
            };
            let dispatch: &mut dyn BatchDispatcher =
                self.persistent_workers.as_mut().unwrap();
            for chunk in combined.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS) {
                dispatch.dispatch_batch(0, chunk);
            }
            self.probe_hidden_after_segment("tail");
        }

        let logits = unsafe {
            std::slice::from_raw_parts(
                self.activations.logits_mapped.host_ptr(),
                self.config.vocab_size,
            )
        }
        .to_vec();

        if self.tracer.enabled() {
            // Persistent path: only top10_logits captured here (from host-mapped logits_mapped).
            // Per-layer probes (embed, L{i}.post_mixer, etc.) are emitted via the SDMA dump pipeline.
            use crate::tracer::Probe;
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1)); // NaN-safe (was partial_cmp().unwrap() — panicked on NaN logits)
            let top10: Vec<f32> = indexed
                .iter()
                .take(10)
                .flat_map(|&(id, val)| [id as f32, val])
                .collect();
            self.tracer.record_host_f32(Probe::Logits { top_k: 10 }, &top10);
            let _ = self.tracer.drain();
        }

        // (MoE worker timing report removed with the unified-worker cutover —
        //  per-op timing is now visible via DISPATCH_RTT in persistent_dispatch.)
        self.seq_len = position + 1;
        Ok(logits)
    }

    /// Head-parallel GQA attention. Two modes:
    ///
    /// **Distributed QKV** (use_distributed_qkv=true, triggered from RMSNorm boundary):
    ///   Each GPU receives normed (P2P broadcast from GPU 0) and projects its own Q/K/V slices.
    ///   After GQA, gate slices are collected to GPU 0 for output-gate (run later in megakernel).
    ///
    /// **Legacy** (use_distributed_qkv=false, triggered from mRoPE boundary):
    ///   GPU 0 has already projected Q/K/V. Only KV write + GQA are distributed.
    ///
    /// After this returns, activations.attn_out[0..nqh*hd] contains the concatenated GQA outputs,
    /// and (for distributed mode) activations.gate_attn[0..nqh*hd] contains the full gate.
    /// Head-parallel attention dispatch.
    /// Returns gather instructions (OP_D2D_COPY from GPUs 1+ to GPU 0) to be prepended
    /// to the next GPU 0 segment, saving one dispatch round-trip per attention layer.
    /// GPU 1+ streams are synchronized before return, so gather is safe to fuse.
    fn dispatch_head_parallel_attention(
        &mut self,
        attn_i: usize,
        position: u32,
    ) -> Result<Vec<crate::megakernel::Instruction>, ModelError> {
        use crate::megakernel::instructions::{
            AttnPagedInst, D2dCopyInst, DeinterleaveInst, LinearProjInst,
        };
        use crate::megakernel::{
            CHUNK_TOKENS, Instruction, OP_LINEAR_PROJ, OP_LINEAR_PROJ_PCG32, OP_LINEAR_PROJ_RNF4,
        };
        use crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS;
        use crate::quant::{LinearWeight, WeightFormat};

        fn emit_linear_proj_inst(
            batch: &mut Vec<Instruction>,
            w: &LinearWeight,
            out_ptr: *mut f32,
            in_ptr: *const f32,
            out_dim: usize,
            in_dim: usize,
        ) {
            let (opcode, w_ptr) = match w.weight_format() {
                WeightFormat::PcG32Q4 => (OP_LINEAR_PROJ_PCG32, w.raw_data_ptr()),
                WeightFormat::Rnf4G128 => (OP_LINEAR_PROJ_RNF4, w.raw_data_ptr()),
                WeightFormat::Bf16 => (OP_LINEAR_PROJ, w.raw_data_ptr()),
            };
            batch.push(LinearProjInst::new(opcode, out_dim as u32, out_ptr, w_ptr, in_ptr, out_dim as i32, in_dim as i32, 0).into_inst());
        }

        let num_gpus = self.multi_gpu.as_ref().unwrap().num_devices;
        let nqh = self.config.num_q_heads;
        let nkh = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let hs = self.config.hidden_size;
        let max_sl = self.config.max_seq_len;
        let local_nqh = nqh / num_gpus;
        let local_nkh = nkh; // GQA: KV heads replicated on every GPU, not split
        // bd 6u4l: partition-validity guard for the absolute-vs-local head-index
        // OOB class (srg6.15: op_attn_paged indexed q/output by absolute head into
        // worker-LOCAL local_nqh-sized buffers → GPU wedge). The kernel-side fix
        // (1824f51) made op_attn_paged index by q_head_local; these host asserts
        // catch a partition/config regression at dispatch instead of as a
        // kfd_wait_on_events wedge.
        debug_assert!(
            nqh % num_gpus == 0,
            "nqh={} not divisible by num_gpus={}: head partition will OOB",
            nqh, num_gpus
        );
        debug_assert!(local_nqh > 0, "local_nqh=0: degenerate head partition");
        let head_stride = max_sl * hd;
        let q_mult = if self.config.has_output_gate { 2 } else { 1 };
        let has_gate = self.config.has_output_gate;
        let use_distributed_qkv = !self
            .megakernel_multi_gpu_p2p
            .as_ref()
            .unwrap()
            .multi_gpu_attn_boundaries
            .is_empty();

        // GPU 0 base pointers (P2P-accessible)
        // Use normed_stage (GART/MappedHostBuffer) for broadcast source, NOT normed (device VRAM).
        // On RDNA3 PCIe, P2P reads bypass GPU 0's L2 and hit VRAM — which may be stale since
        // op_rmsnorm_wx writes go through L2. normed_stage is write-through to system RAM,
        // so GPU 1-3's peer_copy_kernel reads the correct value.
        let normed_base = self.activations.normed_stage.device_ptr() as u64;
        let k_attn_base = self.activations.k_attn.as_ptr() as u64;
        let v_attn_base = self.activations.v_attn.as_ptr() as u64;
        let q_attn_base = self.activations.q_attn.as_ptr() as u64;
        let attn_out_base = self.activations.attn_out.as_write_ptr() as u64;
        let gate_attn_base = self.activations.gate_attn.as_write_ptr() as u64;

        // bd srg6.X2: paged KV layout constants.
        // Per chunk: [num_attn_layers * 2 * CHUNK_TOKENS * nkh * hd] f32s.
        // Layer K offset: attn_i * 2 * CHUNK_TOKENS * nkh * hd * 4 bytes.
        // Layer V offset: layer_k_offset + CHUNK_TOKENS * nkh * hd * 4 bytes.
        let kv_stride = local_nkh * hd;
        let paged_layer_k_offset =
            (attn_i * 2 * CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>()) as u64;
        let paged_layer_v_offset =
            paged_layer_k_offset + (CHUNK_TOKENS * kv_stride * std::mem::size_of::<f32>()) as u64;
        let chunk_head_stride = CHUNK_TOKENS * hd;

        // D7-A: k_norm / inv_freq pointers from GPU 0's weight tensors are
        // P2P-readable by all worker GPUs via gfx11 mtype-UC (kernel patch 0001 +
        // hsa-rocr-p2p-mtype-uc-gfx11.patch). Read-only tensors → UC write-through
        // is correct. Assert P2P is available: MultiGpuContext::init logged
        // "P2P: GPU i→j enabled" for each pair; paged_seq.is_some() implies
        // init_attn_buffers ran, which runs only after P2P init.
        let layer_idx_for_attn = self
            .config
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| l.layer_type == crate::config::LayerType::Attention)
            .nth(attn_i)
            .map(|(i, _)| i)
            .unwrap();
        let (k_norm_ptr_gpu0, inv_freq_ptr_gpu0) = {
            let kn = if self.config.has_qk_norm {
                match &self.layers[layer_idx_for_attn] {
                    LayerWeights::Attention(w) => w.k_norm.as_ptr(),
                    _ => panic!("expected attention layer"),
                }
            } else {
                std::ptr::null()
            };
            (kn, self.activations.inv_freq.as_ptr())
        };

        let mut seq_nums: Vec<(usize, u32)> = Vec::with_capacity(num_gpus);

        for gpu_i in 0..num_gpus {
            let mut batch: Vec<Instruction> = Vec::new();

            // Resolve per-GPU buffer pointers (paged path — no attn_kv_caches)
            let (q_ptr, out_ptr) = {
                let mgpu = self.multi_gpu.as_ref().unwrap();
                let q = if gpu_i == 0 {
                    q_attn_base
                } else {
                    mgpu.workers[gpu_i].attn_q.as_ref().unwrap().as_ptr() as u64
                };
                let out = if gpu_i == 0 {
                    attn_out_base
                } else {
                    mgpu.workers[gpu_i].attn_out_dev_self.unwrap() as u64
                };
                (q, out)
            };

            // Resolve per-worker paged KV pointers (chunk slot + offsets within chunk).
            // GPU 0: `broadcast_paged_chunks_to_workers` skips GPU 0, so workers[0]'s
            // paged_seq is NOT populated with prefill chunks — only the per-decode-step
            // append. Source paged_seq + page_allocator + page_table from `self` for gpu_i==0;
            // pos_table stays per-worker (workers[0].paged_position_table IS updated by
            // the per-decode-step write_volatile loop earlier in this function).
            let (paged_chunk_k_ptr, paged_chunk_v_ptr, page_table_ptr, pos_table_ptr, chunk_offset) = {
                let mgpu = self.multi_gpu.as_ref().unwrap();
                if gpu_i == 0 {
                    let seq = self.paged_seq.as_ref()
                        .expect("self.paged_seq required for gpu_i==0 head-parallel paged dispatch");
                    let alloc = self.page_allocator.as_ref()
                        .expect("self.page_allocator required for gpu_i==0");
                    let chunk_slot = seq.chunks.last()
                        .expect("self.paged_seq must have at least one chunk after append_token")
                        .slot_index();
                    let chunk_base = alloc.slot_ptr(chunk_slot) as u64;
                    let chunk_offset = (seq.current_chunk_offset() as usize).saturating_sub(1);
                    let k_ptr = chunk_base + paged_layer_k_offset;
                    let v_ptr = chunk_base + paged_layer_v_offset;
                    let pt = mgpu.workers[0].paged_page_table.as_ref()
                        .expect("workers[0].paged_page_table required for gpu_i==0 (mirrored from self.paged_seq)").as_ptr() as u64;
                    let pos = mgpu.workers[0].paged_position_table.as_ref()
                        .expect("workers[0].paged_position_table required").as_ptr() as u64;
                    (k_ptr, v_ptr, pt, pos, chunk_offset)
                } else {
                    let worker = &mgpu.workers[gpu_i];
                    let seq = worker.paged_seq.as_ref()
                        .expect("worker.paged_seq required for head-parallel paged dispatch");
                    let alloc = worker.page_allocator.as_ref()
                        .expect("worker.page_allocator required");
                    let chunk_slot = seq.chunks.last()
                        .expect("paged_seq must have at least one chunk after append_token")
                        .slot_index();
                    let chunk_base = alloc.slot_ptr(chunk_slot) as u64;
                    let chunk_offset = (seq.current_chunk_offset() as usize).saturating_sub(1);
                    let k_ptr = chunk_base + paged_layer_k_offset;
                    let v_ptr = chunk_base + paged_layer_v_offset;
                    let pt = worker.paged_page_table.as_ref().expect("paged_page_table required").as_ptr() as u64;
                    let pos = worker.paged_position_table.as_ref().expect("paged_position_table required").as_ptr() as u64;
                    (k_ptr, v_ptr, pt, pos, chunk_offset)
                }
            };

            if use_distributed_qkv {
                // ── Distributed QKV mode ──────────────────────────────────────────────────
                let (normed_local, q_gate_ptr, k_local_ptr, v_local_ptr, gate_ptr) = {
                    let mgpu = self.multi_gpu.as_ref().unwrap();
                    if gpu_i == 0 {
                        let q_gate = self.activations.q_gate_attn.as_write_ptr() as u64;
                        let k = k_attn_base;
                        let v = v_attn_base;
                        let gate = gate_attn_base;
                        (normed_base, q_gate, k, v, gate)
                    } else {
                        let w = &mgpu.workers[gpu_i];
                        let normed = w.attn_normed.as_ref().unwrap().as_write_ptr() as u64;
                        let q_gate = w.attn_q_gate.as_ref().unwrap().as_write_ptr() as u64;
                        let k = w.attn_k.as_ref().unwrap().as_write_ptr() as u64;
                        let v = w.attn_v.as_ref().unwrap().as_write_ptr() as u64;
                        let gate = w.attn_gate.as_ref().map(|b| b.as_write_ptr() as u64).unwrap_or(0);
                        (normed, q_gate, k, v, gate)
                    }
                };

                // 0. Broadcast normed to GPUs 1..n (β'''' sentinel wait)
                if gpu_i > 0 {
                    let normed_seq_slot = unsafe {
                        (self.activations.normed_seq.device_ptr() as *const u32).add(attn_i)
                    };
                    let wait_value = (position as u32) + 1;
                    batch.push(
                        D2dCopyInst::new((hs as u32 + 255) / 256, normed_local as *mut f32, normed_base as *const f32, hs as i32)
                            .with_wait(normed_seq_slot, wait_value)
                            .into_inst()
                    );
                }

                // 1-3. QKV projections (unchanged from flat path).
                if gpu_i == 0 {
                    let aw = match &self.layers[layer_idx_for_attn] {
                        LayerWeights::Attention(w) => w,
                        _ => panic!("expected attention layer"),
                    };
                    emit_linear_proj_inst(&mut batch, &aw.w_q_gate, q_gate_ptr as *mut f32, normed_local as *const f32, local_nqh * hd * q_mult, hs);
                    emit_linear_proj_inst(&mut batch, &aw.w_k, k_local_ptr as *mut f32, normed_local as *const f32, local_nkh * hd, hs);
                    emit_linear_proj_inst(&mut batch, &aw.w_v, v_local_ptr as *mut f32, normed_local as *const f32, local_nkh * hd, hs);
                } else {
                    let w_q = unsafe { &*(&self.multi_gpu.as_ref().unwrap().workers[gpu_i].attn_w_q_gate[attn_i] as *const LinearWeight) };
                    let w_k = unsafe { &*(&self.multi_gpu.as_ref().unwrap().workers[gpu_i].attn_w_k[attn_i] as *const LinearWeight) };
                    let w_v = unsafe { &*(&self.multi_gpu.as_ref().unwrap().workers[gpu_i].attn_w_v[attn_i] as *const LinearWeight) };
                    emit_linear_proj_inst(&mut batch, w_q, q_gate_ptr as *mut f32, normed_local as *const f32, local_nqh * hd * q_mult, hs);
                    emit_linear_proj_inst(&mut batch, w_k, k_local_ptr as *mut f32, normed_local as *const f32, local_nkh * hd, hs);
                    emit_linear_proj_inst(&mut batch, w_v, v_local_ptr as *mut f32, normed_local as *const f32, local_nkh * hd, hs);
                }

                // 4. Deinterleave Q+gate → q_attn, gate_attn (only for gated Q)
                if has_gate {
                    let total = local_nqh * hd;
                    batch.push(DeinterleaveInst::new((total as u32 + 255) / 256, q_ptr as *mut f32, gate_ptr as *mut f32, q_gate_ptr as *const f32, local_nqh as i32, hd as i32, 1).into_inst());
                } else {
                    batch.push(D2dCopyInst::new(((local_nqh * hd) as u32 + 255) / 256, q_ptr as *mut f32, q_gate_ptr as *const f32, (local_nqh * hd) as i32).into_inst());
                }

                // Steps 5 (QkNorm) and 6 (mRoPE) on K REMOVED for paged path.
                // Paged KV cache stores PRE-QKNorm, PRE-mRoPE K (for quantization quality).
                // AttnPagedInst applies QkNorm + mRoPE internally to Q at attention time.

                // 7. Paged KV write: per KV head D2D_COPY to current chunk slot.
                // Source: k_local (pre-norm) → chunk slot at [attn_i layer, head h, chunk_offset].
                for h in 0..local_nkh {
                    let src_k = (k_local_ptr + (h * hd * 4) as u64) as *const f32;
                    let src_v = (v_local_ptr + (h * hd * 4) as u64) as *const f32;
                    let dst_k = (paged_chunk_k_ptr + (h * chunk_head_stride * 4 + chunk_offset * hd * 4) as u64) as *mut f32;
                    let dst_v = (paged_chunk_v_ptr + (h * chunk_head_stride * 4 + chunk_offset * hd * 4) as u64) as *mut f32;
                    batch.push(D2dCopyInst::new((hd as u32 + 255) / 256, dst_k, src_k, hd as i32).into_inst());
                    batch.push(D2dCopyInst::new((hd as u32 + 255) / 256, dst_v, src_v, hd as i32).into_inst());
                }

                // 8. AttnPagedInst: paged attention with per-worker head slice.
                // k_norm + inv_freq from GPU 0 (P2P-readable via mtype-UC patch 0001).
                {
                    let rd = self.config.rope_dim;
                    let eps = self.config.rms_norm_eps;
                    let ms = self.config.mrope_sections();
                    let seq_len = (position + 1) as i32;
                    let local_q_head_start = (gpu_i * local_nqh) as u16;
                    let local_nqh_u16 = local_nqh as u16;
                    // bd 6u4l: u16-cast sanity (not a tautology — catches a future
                    // divergence between the cast and the size computation).
                    debug_assert!(
                        local_nqh_u16 as usize == local_nqh,
                        "local_nqh u16 cast overflow: local_nqh={} u16={}",
                        local_nqh, local_nqh_u16
                    );
                    // bd 6u4l defense-in-depth: catches a FUTURE regression that
                    // passes a gpu0-absolute slice pointer for a worker instead of
                    // the worker-local buffer. Does NOT test the kernel-indexing
                    // invariant (the srg6.15 bug, already fixed kernel-side in 1824f51).
                    debug_assert!(
                        gpu_i == 0
                            || q_ptr != q_attn_base + (gpu_i * local_nqh * hd * 4) as u64,
                        "gpu_i={} q_ptr is a gpu0-absolute slice offset", gpu_i
                    );
                    let mut inst = AttnPagedInst::new(
                        local_nqh as u32,
                        out_ptr as *mut f32,
                        q_ptr as *const f32,
                        inv_freq_ptr_gpu0,
                        nqh as i32, nkh as i32, hd as i32,
                        seq_len,
                        CHUNK_TOKENS as i32,
                        rd as i32,
                        paged_layer_k_offset,
                        paged_layer_v_offset,
                        k_norm_ptr_gpu0,
                        eps,
                        ms[0] as i32,
                        ms[1] as i32,
                        local_q_head_start,
                        local_nqh_u16,
                    );
                    inst.page_table = page_table_ptr;
                    inst.pos_table = pos_table_ptr;
                    batch.push(inst.into_inst());
                }
            } else {
                // ── Legacy mode (QKV already projected on GPU 0) ─────────────────────────
                // For GPU i > 0: copy Q slice from GPU 0's q_attn to local attn_q
                if gpu_i > 0 {
                    let src_q = q_attn_base + (gpu_i * local_nqh * hd * 4) as u64;
                    batch.push(D2dCopyInst::new(((local_nqh * hd) as u32 + 255) / 256, q_ptr as *mut f32, src_q as *const f32, (local_nqh * hd) as i32).into_inst());
                }

                // Paged KV write from GPU 0's k/v_attn (pre-norm for paged path).
                // Heads are replicated across workers (GQA: local_nkh = nkh).
                for h in 0..local_nkh {
                    let src_k = (k_attn_base + (h * hd * 4) as u64) as *const f32;
                    let src_v = (v_attn_base + (h * hd * 4) as u64) as *const f32;
                    let dst_k = (paged_chunk_k_ptr + (h * chunk_head_stride * 4 + chunk_offset * hd * 4) as u64) as *mut f32;
                    let dst_v = (paged_chunk_v_ptr + (h * chunk_head_stride * 4 + chunk_offset * hd * 4) as u64) as *mut f32;
                    batch.push(D2dCopyInst::new((hd as u32 + 255) / 256, dst_k, src_k, hd as i32).into_inst());
                    batch.push(D2dCopyInst::new((hd as u32 + 255) / 256, dst_v, src_v, hd as i32).into_inst());
                }

                // AttnPagedInst for legacy path.
                {
                    let rd = self.config.rope_dim;
                    let eps = self.config.rms_norm_eps;
                    let ms = self.config.mrope_sections();
                    let seq_len = (position + 1) as i32;
                    let q_src = if gpu_i == 0 { q_attn_base } else { q_ptr };
                    let local_q_head_start = (gpu_i * local_nqh) as u16;
                    let local_nqh_u16 = local_nqh as u16;
                    // bd 6u4l: u16-cast sanity (not a tautology — catches a future
                    // divergence between the cast and the size computation).
                    debug_assert!(
                        local_nqh_u16 as usize == local_nqh,
                        "local_nqh u16 cast overflow: local_nqh={} u16={}",
                        local_nqh, local_nqh_u16
                    );
                    // bd 6u4l defense-in-depth: catches a FUTURE regression that
                    // passes a gpu0-absolute slice pointer for a worker instead of
                    // the worker-local buffer. Does NOT test the kernel-indexing
                    // invariant (the srg6.15 bug, already fixed kernel-side in 1824f51).
                    debug_assert!(
                        gpu_i == 0
                            || q_ptr != q_attn_base + (gpu_i * local_nqh * hd * 4) as u64,
                        "gpu_i={} q_ptr is a gpu0-absolute slice offset", gpu_i
                    );
                    let mut inst = AttnPagedInst::new(
                        local_nqh as u32,
                        out_ptr as *mut f32,
                        q_src as *const f32,
                        inv_freq_ptr_gpu0,
                        nqh as i32, nkh as i32, hd as i32,
                        seq_len,
                        CHUNK_TOKENS as i32,
                        rd as i32,
                        paged_layer_k_offset,
                        paged_layer_v_offset,
                        k_norm_ptr_gpu0,
                        eps,
                        ms[0] as i32,
                        ms[1] as i32,
                        local_q_head_start,
                        local_nqh_u16,
                    );
                    inst.page_table = page_table_ptr;
                    inst.pos_table = pos_table_ptr;
                    batch.push(inst.into_inst());
                }
            }

            // All GPUs (including workers 1..N-1) dispatch via the
            // persistent_worker mailbox (PersistentDispatch).
            assert!(
                batch.len() <= MAX_BATCH_INSTRUCTIONS,
                "attn batch overflow gpu={} len={}",
                gpu_i, batch.len()
            );
            let dispatch: &mut dyn BatchDispatcher =
                self.persistent_workers.as_mut().unwrap();
            let seq = dispatch.dispatch_batch_fire(gpu_i, &batch);
            seq_nums.push((gpu_i, seq));
            // β'''' diagnostic probes (ea #3147, post-reboot timing test):
            // BRAIDINFER_RACE_PROBE=1 blocks host on GPU 0 batch ack before
            // workers dispatch — adds synchronization (not just delay).
            // BRAIDINFER_USLEEP_PROBE=N adds an N-microsecond sleep after GPU 0
            // dispatch — pure timing delay without ack semantics.
            // Both env-gated, off by default. Diagnostic only; not shippable.
            if gpu_i == 0 {
                if std::env::var("BRAIDINFER_RACE_PROBE").is_ok() {
                    let dispatch_ro: &dyn BatchDispatcher =
                        self.persistent_workers.as_ref().unwrap();
                    dispatch_ro.wait_ack(0, seq);
                }
                if let Ok(us) = std::env::var("BRAIDINFER_USLEEP_PROBE")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .map(|us| Result::<u64, ()>::Ok(us))
                    .unwrap_or(Ok(0))
                {
                    if us > 0 {
                        std::thread::sleep(std::time::Duration::from_micros(us));
                    }
                }
            }
        }

        // Wait for all GPUs' dispatchers to complete attention.
        let dispatch: &dyn BatchDispatcher =
            self.persistent_workers.as_ref().unwrap();
        for &(gpu_i, seq) in &seq_nums {
            dispatch.wait_ack(gpu_i, seq);
        }

        // Per-layer snapshot (BRAIDINFER_DECODE_MIRROR + first decode step
        // + first attn layer). Surfaces whether worker.attn_normed differs
        // from CPU-view of normed_stage — diagnosing L2-staleness on read.
        // bd 4e2m sig-B-upstream probe (udi #3197): instrument ALL attn layers
        // (not just attn_i==0) to find FIRST layer where normed_stage(CPU-view)
        // first turns NaN. attn_i==0 alone catches downstream propagation but
        // misses which attn layer originated the NaN.
        // Gather GPU 1..num_gpus attn_out + gate_attn via persistent worker OP_D2D_COPY.
        // MUST NOT use peer_copy_async (kernel launch on GPU 0) while persistent cooperative
        // worker holds all CUs. Route all GPU-0 copies through persistent worker protocol.
        // These instructions are returned to the caller to be prepended to the next segment,
        // fusing them into one dispatch round-trip (safe: GPU 1+ streams already synchronized).
        let mut gather_batch: Vec<Instruction> = Vec::new();
        let n_elems = local_nqh * hd;
        let grid_x = ((n_elems as u32) + 255) / 256;

        // attn_out gather: GPU i → act.attn_out[i*n_elems..]
        // Reads worker's host-mapped UC via GPU 0's dev_ptr (no peer-VRAM read).
        for gpu_i in 1..num_gpus {
            let src = self.multi_gpu.as_ref().unwrap().workers[gpu_i]
                .attn_out_dev_gpu0
                .unwrap() as *const f32;
            let dst =
                unsafe { (self.activations.attn_out.as_write_ptr()).add(gpu_i * n_elems) };
            gather_batch.push(D2dCopyInst::new(grid_x, dst, src, n_elems as i32).into_inst());
        }

        // gate_attn gather: GPU i → act.gate_attn[i*n_elems..]
        // GPU 0's gate was written directly to act.gate_attn[0..n_elems] by deinterleave.
        if use_distributed_qkv && has_gate {
            for gpu_i in 1..num_gpus {
                let src = self.multi_gpu.as_ref().unwrap().workers[gpu_i]
                    .attn_gate
                    .as_ref()
                    .unwrap()
                    .as_ptr() as *const f32;
                let dst = unsafe {
                    self.activations
                        .gate_attn
                        .as_write_ptr()
                        .add(gpu_i * n_elems)
                };
                gather_batch.push(D2dCopyInst::new(grid_x, dst, src, n_elems as i32).into_inst());
            }
        }

        assert!(
            gather_batch.len() <= MAX_BATCH_INSTRUCTIONS,
            "gather batch overflow len={}",
            gather_batch.len()
        );

        Ok(gather_batch)
    }

    /// CPU-orchestrated MoE dispatch on worker GPUs (1..N-1) for one decode token.
    ///
    /// Phase 7 split: this is called AFTER GPU 0's PRE batch (containing
    /// OP_MOE_DISPATCH) has been fired async. We dispatch OP_MOE_FFN_REMOTE
    /// on each worker async, then wait for both worker acks AND `gpu0_seq` ack
    /// before returning. After return, the caller fires the next GPU 0 batch
    /// starting with OP_MOE_DISPATCH_POST (sum), which runs only after this
    /// wait completes — guaranteeing all output_slots are populated and visible.
    fn dispatch_moe_workers_decode_async(&mut self, layer_idx: usize, gpu0_seq: Option<u32>, seq_value: u64) -> Result<(), ModelError> {
        // bd 1hik: per-layer routing parameters are populated by
        // `compile_multi_gpu_p2p` into `MoeP2pContext::decode_params[layer_idx]`
        // from the same source of truth used to emit OP_MOE_DISPATCH. The
        // CPU worker-dispatch path no longer reads raw `MoeDispatchInst`
        // words — decoupling worker dispatch from the GPU-0 PRE opcode's
        // wire layout (unblocks bd 0hu.3 option (b)).
        let p2p = self.moe_p2p.as_ref().expect("moe_p2p must be initialized for MoE decode");
        // yef5.2.5 mechanism confirmation (YEF52_TRACE): CPU-side — survives the GPU abort.
        // At each MoE-layer dispatch print the seq the CPU is FIRING and the CURRENT
        // sentinel values (GPU0's activation progress + the workers' result progress). If the
        // firing seq climbs while a sentinel stalls -> the CPU is racing ahead of a stuck GPU,
        // confirming the cross-step overwrite + showing the lead (gap) and which side is stuck.
        if std::env::var("YEF52_TRACE").is_ok() {
            let act = p2p.moe_act_sentinel[layer_idx].as_ref()
                .map(|b| unsafe { b.host_ptr().read_volatile() });
            let res: Vec<u32> = p2p.moe_result_sentinel[layer_idx].as_ref()
                .map(|ws| ws.iter().map(|b| unsafe { b.host_ptr().read_volatile() }).collect())
                .unwrap_or_default();
            eprintln!("[yef52trace] L{layer_idx} firing seq={seq_value} | act_sentinel={act:?} result={res:?}");
        }
        let params = p2p.decode_params[layer_idx]
            .as_ref()
            .expect("decode_params not populated for MoE layer (compile_multi_gpu_p2p must run first)");
        let output_slots = params.output_slots;
        let expert_ids = params.expert_ids;
        let expert_weights = params.expert_weights;
        let hs = params.hs as usize;
        let k = params.k as usize;
        let eis = params.eis as usize;
        let has_gate = params.has_gate_proj;
        let gupd = if params.gupd == 0 { hs } else { params.gupd as usize };
        let relu_sq = params.relu_sq;
        let num_gpus = p2p.num_gpus;
        let num_workers = p2p.workers.len();
        // Token index 0 for decode (single-token); per-worker output slot:
        //   output_slots + (0 * num_gpus + gpu_id) * hs == output_slots + gpu_id * hs
        // Worker GPU id = worker_idx + 1 (workers are GPUs 1..N-1).

        // Build per-worker instructions then dispatch_batch_fire on each, then wait.
        // We borrow p2p immutably for build; persistent_workers borrowed mutably for dispatch.
        let insts: Vec<(usize, crate::megakernel::Instruction)> = (0..num_workers).map(|w| {
            let gpu_id = w + 1;
            // yef5.2 P1c (option c): this worker's RESULT sentinel (host-UC), null until
            // compile_multi_gpu_p2p fills moe_result_sentinel. Its presence GATES the WHOLE
            // option-c path, so the working tree stays SAFE (the pre-yef5.2 output_slots +
            // CPU-ack path) until the kernel result-store + POST peer-read + compile-fill
            // all land together. When filled: the worker writes its OWN UC-VRAM + AGENT-
            // RELEASE-signals; GPU0's POST acquire-spins + peer-reads it fresh (UC bypasses
            // all caches, §5.3 — fixes the B1 §11.19(x)-stale host-UC POST-read).
            let result_sentinel: *const u32 = p2p.moe_result_sentinel[layer_idx]
                .as_ref()
                .map(|v| v[w].host_ptr() as *const u32)
                .unwrap_or(std::ptr::null());
            let out_slot = if result_sentinel.is_null() {
                unsafe { output_slots.add(gpu_id * hs) }  // pre-yef5.2 path (host-UC + CPU ack)
            } else {
                p2p.local_output_uc_ptr_for(w).as_raw()    // option c: worker-own UC-VRAM peer-read
            };
            // yef5.2 Step A H1 fix: workers P2P-read the activation from GPU-0
            // VRAM `activation_staging_vram` (peer-mapped MTYPE_UC via kernel
            // patch 0001) instead of host-mapped UC `moe_act_uc_handoff`, which
            // is asymmetric-stale on gfx1100 multi-GPU (§11.19(x): no GPU->GPU
            // snoop; worker GART read hits host DRAM before GPU-0 L2 dirty lines
            // land -> ~1/5 forward-pass divergence). The GPU-0 VRAM VA is
            // directly peer-readable from each worker context (same raw-ptr
            // handoff as k_norm_ptr_gpu0 / normed_base in
            // dispatch_head_parallel_attention). Single source buffer at offset
            // 0; no per-gpu peer-view accessor or gpu_id offset needed.
            let activation = p2p.activation_staging_vram.as_ptr() as *const f32;
            // bd el1f: decode is single-token; the multi-token Step 1 drain
            // race doesn't apply, so wait_ptr=null (no acquire). The decode
            // path uses OP_MOE_DISPATCH (megakernel-internal) which handles
            // its own ordering.
            // yef5.2 Step A: acquire-spin on the per-layer activation sentinel.
            // sentinel_ptr is host-UC (MappedHostBuffer<u32>); wait_seq = (position+1).
            let sentinel_ptr: *const u32 = p2p.moe_act_sentinel[layer_idx]
                .as_ref()
                .map(|s| s.host_ptr() as *const u32)
                .unwrap_or(std::ptr::null());
            let inst = p2p.build_ffn_remote_inst(
                w,
                layer_idx,
                activation,
                out_slot,
                expert_ids,
                expert_weights,
                k, eis, hs, gupd, has_gate, relu_sq,
                sentinel_ptr,
                seq_value,
                result_sentinel,
                seq_value,
            );
            (gpu_id, inst)
        }).collect();
        // Sanity: gpu_id maps to persistent_workers index = gpu_id (workers are
        // [GPU0, GPU1, ..., GPU(num_gpus-1)] in PersistentDispatch::workers).
        let _ = num_gpus;

        let dispatch: &mut dyn BatchDispatcher =
            self.persistent_workers.as_mut().unwrap();
        let mut seq_per_gpu: Vec<(usize, u32)> = Vec::with_capacity(num_workers + 1);
        for (gpu_idx, inst) in &insts {
            // dispatch_batch_fire takes a slice; one OP_MOE_FFN_REMOTE per worker.
            let single = std::slice::from_ref(inst);
            let seq = dispatch.dispatch_batch_fire(*gpu_idx, single);
            seq_per_gpu.push((*gpu_idx, seq));
        }
        // Wait for every worker's ack AND for GPU 0's PRE batch ack (if provided).
        // The next GPU 0 batch — starting with OP_MOE_DISPATCH_POST — fires only
        // after this wait completes, ensuring output_slots are fully populated.
        // braidinfer-wks Phase 1: use parallel-poll helper instead of
        // sequential wait_ack so the polling thread can service all GPUs
        // in one shared loop (foundation of the daemon's 1-core dispatcher).
        // yef5.2.5: the GPU0-PRE ack is LOAD-BEARING for the FORWARD activation signal — the
        // iter-3 host-buffer dump showed that without it the workers never observe the activation
        // sentinel (act obs=0, spin-forever / self-break to garbage). So under YEF52_DROP_ACK we
        // wait ONLY on GPU0's PRE and drop the per-worker acks (the reverse rides its result
        // sentinel, validated). That reclaims the worker-fanout wait while keeping the forward
        // CPU-synced. Default (neither flag): full ack — the proven old path. YEF52_OPTION_C=1
        // alone (no DROP_ACK) = the validated peer-read config with the full ack net.
        if std::env::var("YEF52_DROP_ALL_ACK").is_ok() {
            // PW1 (yef5.2.8): fully async — drop BOTH the worker acks AND the gpu0_pre wait.
            // Tests whether the forward activation signal survives without the gpu0_pre CPU ack
            // (the iter-3 "act obs=0 without gpu0_pre" finding may be broken-pool-confounded).
            // CLEAN POOL ONLY. If it wedges clean, the gpu0_pre ack is a real forward dependency;
            // if it survives, double-buffer can drop the gpu0_pre wait outright.
            let _ = (gpu0_seq, &seq_per_gpu, &dispatch);
        } else if std::env::var("YEF52_DROP_ACK").is_ok() {
            if let Some(seq) = gpu0_seq {
                if let Err(e) = dispatch.try_wait_acks_many(&[(0, seq)]) {
                    panic!("{e}");
                }
            }
        } else {
            if let Some(seq) = gpu0_seq {
                seq_per_gpu.push((0, seq));
            }
            if let Err(e) = dispatch.try_wait_acks_many(&seq_per_gpu) {
                panic!("{e}");
            }
        }
        Ok(())
    }


}
