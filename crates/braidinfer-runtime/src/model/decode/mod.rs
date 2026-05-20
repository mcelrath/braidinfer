mod trace;

use crate::megakernel::{CHUNK_TOKENS, MegakernelProgram, OP_HALT, OP_LM_HEAD, SHARED_LPROJ_TOTAL};
use crate::persistent_dispatch::BatchDispatcher;
use crate::weights::LayerWeights;

use super::Model;
use super::ModelError;

// ---- Decode-mirror print helpers (Phase 2b) --------------------------------

fn print_stats_decode(
    tracer: &crate::tracer::Tracer,
    label: &str,
    position: u32,
    num_workers: usize,
    max_seq_len: usize,
    head_dim: usize,
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
            "[snap {label} pos={position}] {name}: n={} nan={} inf={} max_abs={:.4} first4={:?}",
            slice.len(), n_nan, n_inf, max_abs, &slice[..slice.len().min(4)]
        );
    };
    let empty: &[f32] = &[];
    stat("act.hidden", tracer.read_f32(Probe::Hidden { gpu: 0, head_only: false }).unwrap_or(empty));
    stat("act.attn_out", tracer.read_f32(Probe::Custom(std::borrow::Cow::Borrowed("act.attn_out"))).unwrap_or(empty));
    for w_idx in 0..num_workers {
        stat(&format!("g{}.attn_normed", w_idx + 1), tracer.read_f32(Probe::AttnNormed { gpu: w_idx + 1 }).unwrap_or(empty));
    }
    for w_idx in 0..num_workers {
        stat(&format!("g{}.attn_q_gate", w_idx + 1), tracer.read_f32(Probe::AttnQGate { gpu: w_idx + 1 }).unwrap_or(empty));
    }
    for w_idx in 0..num_workers {
        stat(&format!("g{}.attn_k", w_idx + 1), tracer.read_f32(Probe::AttnK { gpu: w_idx + 1 }).unwrap_or(empty));
    }
    let used = (position as usize + 1).min(max_seq_len);
    let slice_len = used * head_dim;
    let num_gpus = 1 + num_workers;
    for gpu_i in 0..num_gpus {
        for layer_i in 0..2usize {
            if let Some(k_full) = tracer.read_f32(Probe::KvCache { gpu: gpu_i, attn_layer: layer_i, k: true, head: 0 }) {
                let k_slice = &k_full[..slice_len.min(k_full.len())];
                stat(&format!("g{gpu_i}.kv[{layer_i}].k(h0,p0..{used})"), k_slice);
            }
            if let Some(v_full) = tracer.read_f32(Probe::KvCache { gpu: gpu_i, attn_layer: layer_i, k: false, head: 0 }) {
                let v_slice = &v_full[..slice_len.min(v_full.len())];
                stat(&format!("g{gpu_i}.kv[{layer_i}].v(h0,p0..{used})"), v_slice);
            }
        }
    }
}

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

fn print_normed_stage_stats(label: &str, position: u32, normed_stage_host: &[f32]) {
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut n_denorm = 0usize;
    let mut max_abs = 0.0f32;
    for &x in normed_stage_host {
        let bits = x.to_bits();
        let exp = (bits >> 23) & 0xFF;
        let mant = bits & 0x7F_FFFF;
        if x.is_nan() { n_nan += 1; }
        else if x.is_infinite() { n_inf += 1; }
        else if exp == 0 && mant != 0 { n_denorm += 1; }
        else if x.abs() > max_abs { max_abs = x.abs(); }
    }
    eprintln!(
        "[snap {label}] normed_stage(CPU-view): n={} nan={} inf={} denorm={} max_abs={:.4} first4={:?}",
        normed_stage_host.len(), n_nan, n_inf, n_denorm, max_abs,
        &normed_stage_host[..normed_stage_host.len().min(4)]
    );
    let _ = position;
}

impl Model {
    /// Persistent worker decode using paged KV cache.
    /// On first call: compiles paged megakernel, initializes page allocator + sequence,
    /// then launches persistent worker.
    pub(super) fn decode_step_persistent(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<Vec<f32>, ModelError> {
        use crate::persistent_dispatch::PersistentDispatch;

        // Lazy-init: compile PAGED megakernel FIRST (needs GPU queries),
        // then launch persistent worker (occupies all SMs).
        if self.persistent_workers.is_none() {
            let max_chunks = self.max_paged_chunks();

            if self.megakernel_paged.is_none() {
                let mut mk = MegakernelProgram::compile_paged(self)?;
                mk.init_paged_buffers(max_chunks).map_err(ModelError::Hip)?;
                self.megakernel_paged = Some(mk);
            }

            // Patch LM head instruction to write to logits_mapped (host-mapped)
            // so CPU can read without hipMemcpy (which deadlocks the cooperative kernel).
            // This must be done whether megakernel_paged was just compiled or
            // pre-compiled by prefill_paged (which doesn't patch logits_mapped).
            {
                let mk = self.megakernel_paged.as_mut().unwrap();
                // Search for OP_LM_HEAD by opcode rather than using a hardcoded
                // offset (n_inst-2). words[0] encodes opcode in the low 32 bits.
                let lm_head_idx = mk
                    .instructions
                    .iter()
                    .rposition(|inst| (inst.words[0] as u32) == OP_LM_HEAD)
                    .expect("lm_head not found in paged megakernel");
                // words[1] = output pointer (LinearProjInst layout; see instructions.rs:556).
                mk.instructions[lm_head_idx].words[1] =
                    self.activations.logits_mapped.as_write_ptr() as u64;
            }

            // Ensure paged decode state (page_allocator + paged_seq) is initialized.
            self.ensure_paged_decode_state(false)?;

            // PCG32 full kernel requires SHARED_LPROJ_TOTAL (31776B) for its LDS tile.
            let shared_mem = SHARED_LPROJ_TOTAL as u32;
            let dispatch =
                PersistentDispatch::init(&[self.device], shared_mem, self.config.hidden_size, self.watchdog.clone()).map_err(ModelError::Hip)?;
            self.persistent_workers = Some(dispatch);
        }

        // Write position_ids directly to host-mapped memory (no hipMemcpy)
        self.set_position(position).map_err(ModelError::Hip)?;

        // Append token to paged sequence state (allocates chunk slot if needed).
        {
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            seq_mut.append_token(position as i32, alloc_mut).map_err(ModelError::Hip)?;
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

        // Read logits directly from host-mapped memory (no hipMemcpy needed)
        let logits = unsafe {
            std::slice::from_raw_parts(
                self.activations.logits_mapped.host_ptr(),
                self.config.vocab_size,
            )
        }
        .to_vec();

        // Post-step: handle chunk-seal lifecycle. For unquantized persistent, this is
        // a no-op (post_step_paged early-returns when self.quantized_kv is false).
        // For future persistent+quant wiring, this is where quantization would fire —
        // but quantize_sealed_chunk + stream.synchronize() are HIP API calls that would
        // deadlock under the cooperative kernel. The PERSISTENT+KV_QUANT combination
        // is therefore guarded with InvalidConfig in decode_step (mod.rs); when that
        // combination is properly wired, this call site will be the integration point
        // (and quantize_sealed_chunk will need a cooperative-safe variant).
        {
            let mk = self.megakernel_paged.as_mut().unwrap();
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            mk.post_step_paged(position, seq_mut, alloc_mut, None, &self.config, &self.stream)
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

        // P2P megakernel is always initialized above when has_moe && num_gpus > 1.
        // For non-MoE multi-GPU models, fall through to decode_step_paged.
        if self.megakernel_multi_gpu_p2p.is_some() {
            return self.decode_step_p2p(token_id, position);
        }
        self.decode_step_paged(token_id, position)
    }

    /// Lazily start MoE expert workers (GPUs 1-3) without launching the GPU 0 decode
    /// persistent cooperative kernel. Safe to call during prefill (no cooperative kernel
    /// running on GPU 0 yet, so hipMalloc is allowed).
    pub(crate) fn ensure_moe_workers_started(&mut self) -> Result<(), ModelError> {
        use crate::persistent_dispatch::PersistentDispatch;
        if self.moe_p2p.is_some() {
            return Ok(());
        }
        if !self.has_moe {
            return Ok(());
        }
        let num_gpus = match self.multi_gpu.as_ref() {
            Some(m) => m.num_devices,
            None => return Ok(()),
        };
        if num_gpus <= 1 {
            return Ok(());
        }
        // moe_gemv_worker LDS layout: 1024 f32 elements (expert accumulator tile)
        // + 256 bytes overhead (header / sync primitives in the kernel).
        const MOE_WORKER_LDS_ELEMS: u32 = 1024;
        const MOE_WORKER_LDS_OVERHEAD_BYTES: u32 = 256;
        let moe_worker_shared_mem = MOE_WORKER_LDS_ELEMS * 4 + MOE_WORKER_LDS_OVERHEAD_BYTES;
        let shared_mem_persistent = moe_worker_shared_mem.max(SHARED_LPROJ_TOTAL);
        let hs = self.config.hidden_size;
        let max_eis = self
            .config
            .layers
            .iter()
            .filter_map(|l| match &l.ffn_type {
                crate::model::FfnType::MoE { expert_intermediate_size, .. } => {
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
        let p2p = crate::moe_p2p::MoeP2pContext::init(
            self.device,
            &worker_devices,
            hs,
            gate_up_in_dim,
            max_eis,
            num_total_layers,
            &dist_refs,
            moe_worker_shared_mem,
        )
        .map_err(ModelError::Hip)?;
        let mk_p2p = MegakernelProgram::compile_multi_gpu_p2p(self, &p2p)
            .map_err(ModelError::Hip)?;
        self.moe_p2p = Some(p2p);
        self.megakernel_multi_gpu_p2p = Some(mk_p2p);

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
        // BRAIDINFER_DECODE_MIRROR=1 is a deprecated compat shim: if BRAIDINFER_TRACE is
        // unset, construct with ProbeFilter::All. TODO Phase 5: consolidate env vars.
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
            let tracer = if std::env::var("BRAIDINFER_TRACE").is_err()
                && std::env::var("BRAIDINFER_DECODE_MIRROR").is_ok()
            {
                // Deprecated compat: BRAIDINFER_DECODE_MIRROR=1, BRAIDINFER_TRACE unset.
                eprintln!("  tracer: BRAIDINFER_DECODE_MIRROR compat — ProbeFilter::All ({} streams)", streams.len());
                crate::tracer::Tracer::with_filter_and_streams(streams, crate::tracer::ProbeFilter::All)
            } else {
                crate::tracer::Tracer::from_env(streams).map_err(ModelError::Hip)?
            };
            self.tracer = tracer;
        }
        Ok(())
    }

    /// 5ax-decode MTYPE audit: dump (memory_type, alloc_flags) for every
    /// cross-agent or reused buffer in the multi-GPU decode path. Per
    /// GFX1100_ARCH.md §5.5 Rule 5 — `mem_type=2 alloc_flags=0x0` ==
    /// cached device buffer == L2-stale candidate. `0x3` = UC. `1` = host.
    fn dump_mtype_audit(&self) {
        eprintln!("=== MTYPE audit (5ax-decode) ===");
        eprintln!("Legend: mem_type 1=Host 2=Device | alloc_flags 0x0=cached 0x1=fine-grained 0x3=UC");
        let dev = |b: &braidinfer_hip::DeviceBuffer<f32>, name: &str| {
            match b.pointer_attributes() {
                Ok((t, f)) => eprintln!("  {name:46} mem_type={t} alloc_flags=0x{f:x}"),
                Err(e) => eprintln!("  {name:46} ERR {e:?}"),
            }
        };
        let host = |b: &braidinfer_hip::MappedHostBuffer<f32>, name: &str| {
            match b.pointer_attributes() {
                Ok((t, f)) => eprintln!("  {name:46} mem_type={t} alloc_flags=0x{f:x}"),
                Err(e) => eprintln!("  {name:46} ERR {e:?}"),
            }
        };

        eprintln!("-- activations (GPU 0) --");
        dev(&self.activations.hidden, "activations.hidden");
        dev(&self.activations.normed, "activations.normed");
        host(&self.activations.normed_stage, "activations.normed_stage");
        dev(&self.activations.q_attn, "activations.q_attn");
        dev(&self.activations.k_attn, "activations.k_attn");
        dev(&self.activations.v_attn, "activations.v_attn");
        dev(&self.activations.gate_attn, "activations.gate_attn");
        dev(&self.activations.attn_out, "activations.attn_out");
        dev(&self.activations.gated_out, "activations.gated_out");
        dev(&self.activations.residual, "activations.residual");

        if let Some(legacy) = self.legacy_kv_caches.as_ref() {
            eprintln!("-- legacy_kv_caches (GPU 0, prefill K/V) --");
            for (i, kv) in legacy.iter().enumerate() {
                dev(&kv.k, &format!("legacy_kv_caches[{i}].k"));
                if i == 0 { dev(&kv.v, &format!("legacy_kv_caches[{i}].v")); }
                if i >= 2 { eprintln!("  ... ({} total layers)", legacy.len()); break; }
            }
        }

        if let Some(mgpu) = self.multi_gpu.as_ref() {
            for (gpu_i, w) in mgpu.workers.iter().enumerate() {
                eprintln!("-- worker[{gpu_i}] (device {}) --", w.device.0);
                if let Some(b) = w.attn_normed.as_ref() { dev(b, &format!("workers[{gpu_i}].attn_normed")); }
                if let Some(b) = w.attn_q_gate.as_ref() { dev(b, &format!("workers[{gpu_i}].attn_q_gate")); }
                if let Some(b) = w.attn_k.as_ref()      { dev(b, &format!("workers[{gpu_i}].attn_k")); }
                if let Some(b) = w.attn_v.as_ref()      { dev(b, &format!("workers[{gpu_i}].attn_v")); }
                if let Some(b) = w.attn_gate.as_ref()   { dev(b, &format!("workers[{gpu_i}].attn_gate")); }
                if let Some(b) = w.attn_out.as_ref()    { host(b, &format!("workers[{gpu_i}].attn_out")); }
                for (i, kv) in w.attn_kv_caches.iter().enumerate() {
                    if i < 2 {
                        dev(&kv.k, &format!("workers[{gpu_i}].attn_kv_caches[{i}].k"));
                        dev(&kv.v, &format!("workers[{gpu_i}].attn_kv_caches[{i}].v"));
                    }
                }
                if w.attn_kv_caches.len() > 2 {
                    eprintln!("  ... ({} attn_kv_cache layers)", w.attn_kv_caches.len());
                }
            }
        }

        if let Some(p2p) = self.moe_p2p.as_ref() {
            eprintln!("-- moe_p2p --");
            host(&p2p.output_slots, "moe_p2p.output_slots");
            host(&p2p.activation_staging, "moe_p2p.activation_staging");
        }
        eprintln!("=== end MTYPE audit ===");
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
        // moe_gemv_worker LDS layout: 1024 f32 elements (expert accumulator tile)
        // + 256 bytes overhead (header / sync primitives in the kernel).
        const MOE_WORKER_LDS_ELEMS: u32 = 1024;
        const MOE_WORKER_LDS_OVERHEAD_BYTES: u32 = 256;
        let moe_worker_shared_mem = MOE_WORKER_LDS_ELEMS * 4 + MOE_WORKER_LDS_OVERHEAD_BYTES;
        let shared_mem = moe_worker_shared_mem.max(SHARED_LPROJ_TOTAL);
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

    /// SDMA-based snapshot of cross-GPU debug-relevant tensors (KV caches +
    /// GPU 0 activations) into pinned host buffers via tracer, with per-buffer
    /// stats printed to stderr. No-op if tracer is disabled or multi_gpu is None.
    pub(super) fn decode_mirror_snapshot(&mut self, label: &str, position: u32) {
        if !self.tracer.enabled() || self.multi_gpu.is_none() {
            return;
        }
        use braidinfer_hip::device::{Device, DeviceGuard};
        use crate::tracer::Probe;
        let gpu0 = self.device;
        let hs = self.config.hidden_size;
        let local_nkh = self.config.num_kv_heads;
        let max_seq_len = self.config.max_seq_len;
        let head_dim = self.config.head_dim;
        let hs_bytes = hs * 4;
        let mgpu = self.multi_gpu.as_ref().unwrap();
        let worker_devices: Vec<_> = mgpu.workers.iter().skip(1).map(|w| w.device).collect();
        let workers_kv_refs: Vec<Vec<(*const u8, *const u8)>> =
            mgpu.workers.iter().map(|w| {
                w.attn_kv_caches.iter().map(|kv| (kv.k.as_ptr() as *const u8, kv.v.as_ptr() as *const u8)).collect()
            }).collect();
        let workers_normed: Vec<Option<*const f32>> = mgpu.workers.iter()
            .map(|w| w.attn_normed.as_ref().map(|b| b.as_ptr() as *const f32)).collect();
        let workers_q_gate: Vec<Option<(*const f32, usize)>> = mgpu.workers.iter()
            .map(|w| w.attn_q_gate.as_ref().map(|b| (b.as_ptr() as *const f32, b.len()))).collect();
        let workers_k: Vec<Option<*const f32>> = mgpu.workers.iter()
            .map(|w| w.attn_k.as_ref().map(|b| b.as_ptr() as *const f32)).collect();
        let num_workers = worker_devices.len();
        let nqh_total = self.config.num_q_heads;
        let attn_out_floats = nqh_total * head_dim;
        let attn_out_bytes = attn_out_floats * 4;
        let kv_bytes = local_nkh * max_seq_len * head_dim * 4;
        let dispatch = self.persistent_workers.as_ref().unwrap();

        let _guard = match DeviceGuard::switch_to(gpu0) {
            Ok(g) => g,
            Err(e) => { eprintln!("[snap {label}] DeviceGuard error: {e:?}"); return; }
        };
        // GPU 0: act.hidden + act.attn_out
        if let Err(e) = self.tracer.capture(0, Probe::Hidden { gpu: 0, head_only: false }, self.activations.hidden.as_ptr() as *const u8, hs_bytes) {
            eprintln!("[snap {label}] FAILED: {e:?}"); return;
        }
        if let Err(e) = self.tracer.capture(0, Probe::Custom(std::borrow::Cow::Borrowed("act.attn_out")), self.activations.attn_out.as_ptr() as *const u8, attn_out_bytes) {
            eprintln!("[snap {label}] FAILED: {e:?}"); return;
        }
        // Per-GPU attn_kv (GPU 0 first, then workers).
        for (gpu_i, kv_layers) in workers_kv_refs.iter().enumerate() {
            let dev = if gpu_i == 0 { gpu0 } else { worker_devices[gpu_i - 1] };
            if let Err(e) = Device::set_current(dev) { eprintln!("[snap {label}] set_current: {e:?}"); return; }
            if gpu_i > 0 {
                if let Err(e) = dispatch.record_kv_event(dev.0 as usize) { eprintln!("[snap {label}] record_kv_event: {e:?}"); return; }
                if let Err(e) = dispatch.wait_kv_event_on_sdma(dev.0 as usize) { eprintln!("[snap {label}] wait_kv_event: {e:?}"); return; }
            }
            for (layer_i, &(k_ptr, v_ptr)) in kv_layers.iter().enumerate() {
                if let Err(e) = self.tracer.capture(gpu_i, Probe::KvCache { gpu: gpu_i, attn_layer: layer_i, k: true, head: 0 }, k_ptr, kv_bytes) {
                    eprintln!("[snap {label}] FAILED: {e:?}"); return;
                }
                if let Err(e) = self.tracer.capture(gpu_i, Probe::KvCache { gpu: gpu_i, attn_layer: layer_i, k: false, head: 0 }, v_ptr, kv_bytes) {
                    eprintln!("[snap {label}] FAILED: {e:?}"); return;
                }
            }
        }
        // Per-worker attn_normed / attn_q_gate / attn_k
        let local_nqh = attn_out_floats / head_dim / (1 + num_workers);
        for (w_idx, &dev) in worker_devices.iter().enumerate() {
            if let Err(e) = Device::set_current(dev) { eprintln!("[snap {label}] set_current: {e:?}"); return; }
            if let Some(Some(p)) = workers_normed.get(w_idx + 1).copied() {
                if let Err(e) = self.tracer.capture(w_idx + 1, Probe::AttnNormed { gpu: w_idx + 1 }, p as *const u8, hs_bytes) {
                    eprintln!("[snap {label}] FAILED: {e:?}"); return;
                }
            }
            if let Some(Some((p, n))) = workers_q_gate.get(w_idx + 1).copied() {
                let cap = local_nqh * head_dim * 2;
                let copy_bytes = n.min(cap) * 4;
                if let Err(e) = self.tracer.capture(w_idx + 1, Probe::AttnQGate { gpu: w_idx + 1 }, p as *const u8, copy_bytes) {
                    eprintln!("[snap {label}] FAILED: {e:?}"); return;
                }
            }
            if let Some(Some(p)) = workers_k.get(w_idx + 1).copied() {
                let copy_bytes = local_nkh * head_dim * 4;
                if let Err(e) = self.tracer.capture(w_idx + 1, Probe::AttnK { gpu: w_idx + 1 }, p as *const u8, copy_bytes) {
                    eprintln!("[snap {label}] FAILED: {e:?}"); return;
                }
            }
        }
        if let Err(e) = self.tracer.drain() {
            eprintln!("[snap {label}] drain FAILED: {e:?}"); return;
        }
        print_stats_decode(&self.tracer, label, position, num_workers, max_seq_len, head_dim);
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
        let max_steps: u32 = std::env::var("BRAIDINFER_DECODE_MIRROR_STEPS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        let do_snap = self.tracer.enabled() && (position < (self.seq_len + max_steps));
        if do_snap {
            self.decode_mirror_snapshot(&format!("step-begin pos={position}"), position);
        }
        let r = self.decode_step_p2p_inner(token_id, position);
        if do_snap {
            self.decode_mirror_snapshot(&format!("step-end pos={position}"), position);
        }
        r
    }

    pub(super) fn decode_step_p2p_inner(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        // 5ax-decode probe: BRAIDINFER_MTYPE_AUDIT=1 dumps the MTYPE table
        // for all cross-agent and reused buffers in the decode path. Runs
        // once on the first decode step. Surfaces buffers that should be
        // UC but aren't (mem_type=2 device + alloc_flags=0 means cached,
        // the canonical L2-stale candidate). Uses static AtomicBool to
        // run only once even though decode_step_p2p is called per step.
        static AUDITED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if std::env::var("BRAIDINFER_MTYPE_AUDIT").is_ok()
            && !AUDITED.swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            self.dump_mtype_audit();
        }
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
        let has_head_parallel = self
            .multi_gpu
            .as_ref()
            .map(|m| !m.workers[0].attn_kv_caches.is_empty())
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
                // snl 2026-05-15 ordering fix: workers P2P-read from the
                // host-mapped `moe_act_uc_handoff` buffer; the D2D copy that
                // populates it runs inside GPU 0's PRE batch. If workers
                // fire concurrently with GPU 0 (via dispatch_batch_fire),
                // they can race the D2D and read uninitialized/stale data
                // → NaN propagation. Force GPU 0's batch to ack BEFORE
                // dispatching workers. Sacrifices the PRE/worker overlap
                // documented above but is required for correctness; the
                // overlap optimization can be restored once an in-megakernel
                // signal-then-fire mechanism replaces CPU-side fan-out.
                let dispatch: &mut dyn BatchDispatcher =
                    self.persistent_workers.as_mut().unwrap();
                for chunk in combined.chunks(crate::persistent_dispatch::MAX_BATCH_INSTRUCTIONS) {
                    dispatch.dispatch_batch(0, chunk);
                }
                self.probe_hidden_after_segment(&format!("moe-pre L{}", layer_idx));
                let gpu0_seq: Option<u32> = None;
                // Dispatch OP_MOE_FFN_REMOTE on each worker. Workers run in
                // parallel with each other (their dispatches are still async)
                // but only after GPU 0's PRE has completed.
                self.dispatch_moe_workers_decode_async(layer_idx, gpu0_seq)?;
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

        if self.trace.is_some() {
            // Multi-GPU trace: only top10_logits available. hidden/normed are in GPU VRAM
            // and inaccessible while the persistent cooperative worker holds all CUs.
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top10: Vec<f32> = indexed
                .iter()
                .take(10)
                .flat_map(|&(id, val)| [id as f32, val])
                .collect();
            self.trace.as_mut().unwrap().write_checkpoint("top10_logits", &top10);
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
            D2dCopyInst, DeinterleaveInst, GqaAttnInst as GqaAttnInstLocal, LinearProjInst,
            MropeInst, QkNormInst,
        };
        use crate::megakernel::{
            Instruction, OP_LINEAR_PROJ, OP_LINEAR_PROJ_PCG32, OP_LINEAR_PROJ_RNF4,
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

        let mut seq_nums: Vec<(usize, u32)> = Vec::with_capacity(num_gpus);

        for gpu_i in 0..num_gpus {
            let mut batch: Vec<Instruction> = Vec::new();

            // Resolve per-GPU buffer pointers
            let (kv_k_base, kv_v_base, q_ptr, out_ptr) = {
                let mgpu = self.multi_gpu.as_ref().unwrap();
                let kc = &mgpu.workers[gpu_i].attn_kv_caches[attn_i];
                let q = if gpu_i == 0 {
                    q_attn_base
                } else {
                    mgpu.workers[gpu_i].attn_q.as_ref().unwrap().as_ptr() as u64
                };
                let out = if gpu_i == 0 {
                    attn_out_base
                } else {
                    // Worker writes via its own dev_ptr to the host-mapped buffer.
                    mgpu.workers[gpu_i].attn_out_dev_self.unwrap() as u64
                };
                (
                    kc.k.as_write_ptr() as u64,
                    kc.v.as_write_ptr() as u64,
                    q,
                    out,
                )
            };

            if use_distributed_qkv {
                // ── Distributed QKV mode ──────────────────────────────────────────────────
                // GPU 0 has normed in act.normed (from megakernel RMSNorm).
                // GPUs 1..n need normed via P2P broadcast.
                // Get this attention layer's weights (GPU 0 VRAM, P2P-accessible from all GPUs).
                let layer_idx_for_attn = self
                    .config
                    .layers
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.layer_type == crate::config::LayerType::Attention)
                    .nth(attn_i)
                    .map(|(i, _)| i)
                    .unwrap();

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

                // 0. Broadcast normed to GPUs 1..n
                if gpu_i > 0 {
                    // β'''' sentinel wait (bd braidinfer-sm16): acquire-load
                    // on act.normed_seq[attn_i] == 1 forces this D2dCopy's
                    // peer-read of normed_stage to be ordered after the
                    // producer's GPU 0 D2dCopy at compile_attention.rs that
                    // writes act.normed → act.normed_stage and signals.
                    let normed_seq_slot = unsafe {
                        (self.activations.normed_seq.device_ptr() as *const u32).add(attn_i)
                    };
                    // β'''' monotonic seq: wait for (position+1) per #3189
                    let wait_value = (position as u32) + 1;
                    batch.push(
                        D2dCopyInst::new((hs as u32 + 255) / 256, normed_local as *mut f32, normed_base as *const f32, hs as i32)
                            .with_wait(normed_seq_slot, wait_value)
                            .into_inst()
                    );
                }

                // 1-3. QKV projections.
                // GPU 0: use original layer weights (no copy, row_start=0).
                // GPUs 1+: use pre-copied slice in ctx.workers[gpu_i].attn_w_*.
                if gpu_i == 0 {
                    let aw = match &self.layers[layer_idx_for_attn] {
                        LayerWeights::Attention(w) => w,
                        _ => panic!("expected attention layer"),
                    };
                    emit_linear_proj_inst(
                        &mut batch,
                        &aw.w_q_gate,
                        q_gate_ptr as *mut f32,
                        normed_local as *const f32,
                        local_nqh * hd * q_mult,
                        hs,
                    );
                    emit_linear_proj_inst(
                        &mut batch,
                        &aw.w_k,
                        k_local_ptr as *mut f32,
                        normed_local as *const f32,
                        local_nkh * hd,
                        hs,
                    );
                    emit_linear_proj_inst(
                        &mut batch,
                        &aw.w_v,
                        v_local_ptr as *mut f32,
                        normed_local as *const f32,
                        local_nkh * hd,
                        hs,
                    );
                } else {
                    let w_q = unsafe {
                        &*(&self.multi_gpu.as_ref().unwrap().workers[gpu_i].attn_w_q_gate[attn_i]
                            as *const LinearWeight)
                    };
                    let w_k = unsafe {
                        &*(&self.multi_gpu.as_ref().unwrap().workers[gpu_i].attn_w_k[attn_i]
                            as *const LinearWeight)
                    };
                    let w_v = unsafe {
                        &*(&self.multi_gpu.as_ref().unwrap().workers[gpu_i].attn_w_v[attn_i]
                            as *const LinearWeight)
                    };
                    emit_linear_proj_inst(
                        &mut batch,
                        w_q,
                        q_gate_ptr as *mut f32,
                        normed_local as *const f32,
                        local_nqh * hd * q_mult,
                        hs,
                    );
                    emit_linear_proj_inst(
                        &mut batch,
                        w_k,
                        k_local_ptr as *mut f32,
                        normed_local as *const f32,
                        local_nkh * hd,
                        hs,
                    );
                    emit_linear_proj_inst(
                        &mut batch,
                        w_v,
                        v_local_ptr as *mut f32,
                        normed_local as *const f32,
                        local_nkh * hd,
                        hs,
                    );
                }

                // 4. Deinterleave Q+gate → q_attn, gate_attn (only for gated Q)
                if has_gate {
                    let total = local_nqh * hd;
                    batch.push(DeinterleaveInst::new((total as u32 + 255) / 256, q_ptr as *mut f32, gate_ptr as *mut f32, q_gate_ptr as *const f32, local_nqh as i32, hd as i32, 1).into_inst());
                } else {
                    // Non-gated: q_gate IS q, just copy
                    batch.push(D2dCopyInst::new(((local_nqh * hd) as u32 + 255) / 256, q_ptr as *mut f32, q_gate_ptr as *const f32, (local_nqh * hd) as i32).into_inst());
                }

                // 5. QK-norm on local k (only for models with QK-norm weights — e.g. Qwen3, not Nemotron-H)
                if self.config.has_qk_norm {
                    let (q_norm_ptr, k_norm_ptr, qk_norm_eps) = {
                        match &self.layers[layer_idx_for_attn] {
                            LayerWeights::Attention(w) => (
                                w.q_norm.as_ptr(),
                                w.k_norm.as_ptr(),
                                self.config.rms_norm_eps,
                            ),
                            _ => panic!("expected attention layer"),
                        }
                    };
                    batch.push(QkNormInst::new((local_nqh + local_nkh) as u32, q_ptr as *mut f32, k_local_ptr as *mut f32, q_norm_ptr, k_norm_ptr, local_nqh as i32, local_nkh as i32, hd as i32, qk_norm_eps, 0).into_inst());
                }

                // 6. mRoPE on local Q+K — only for models that use RoPE.
                // MUST run BEFORE the KV write so the cache stores POST-MROPE K
                // (op_gqa_attn at step 8 reads cache K without re-applying MROPE).
                // This also matches legacy_kv_caches's layout (post-MROPE K written
                // by emit_attention_layer Prefill variant), so the sew prefill
                // broadcast is consistent with what decode-time KV writes produce.
                //
                // CRITICAL: use the per-worker position_ids_local pointer, NOT
                // self.activations.position_ids — the latter is a non-portable
                // host-mapped buffer whose device_ptr is only valid on GPU 0.
                // Workers reading via that pointer get garbage → wrong rotation
                // → wrong K → broken attention.
                if self.config.use_rope {
                    let rd = self.config.rope_dim;
                    let ms = self.config.mrope_sections();
                    let pos_ptr = self.multi_gpu.as_ref().unwrap().workers[gpu_i]
                        .position_ids_local
                        .as_ptr();
                    batch.push(MropeInst::new((local_nqh + local_nkh) as u32, q_ptr as *mut f32, k_local_ptr as *mut f32, self.activations.inv_freq.as_ptr(), pos_ptr, local_nqh as i32, local_nkh as i32, hd as i32, rd as i32, ms[0] as i32, ms[1] as i32, ms[2] as i32, 0).into_inst());
                }

                // 7. KV write (local — from local k/v to local KV cache)
                for h_local in 0..local_nkh {
                    let src_k = k_local_ptr + (h_local * hd * 4) as u64;
                    let src_v = v_local_ptr + (h_local * hd * 4) as u64;
                    let dst_k =
                        kv_k_base + ((h_local * head_stride + position as usize * hd) * 4) as u64;
                    let dst_v =
                        kv_v_base + ((h_local * head_stride + position as usize * hd) * 4) as u64;
                    batch.push(D2dCopyInst::new(((hd as u32) + 255) / 256, dst_k as *mut f32, src_k as *const f32, hd as i32).into_inst());
                    batch.push(D2dCopyInst::new(((hd as u32) + 255) / 256, dst_v as *mut f32, src_v as *const f32, hd as i32).into_inst());
                }

                // 8. GQA (same as legacy path)
                let seq_len = (position + 1) as i32;
                {
                    let mut inst = GqaAttnInstLocal::new(local_nqh as u32, out_ptr as *mut f32, q_ptr as *const f32, kv_k_base as *const f32, kv_v_base as *const f32, nqh as i32, nkh as i32, hd as i32, seq_len, max_sl as i32);
                    inst.q_head_start = (gpu_i * local_nqh) as u64;
                    batch.push(inst.into_inst());
                }
            } else {
                // ── Legacy mode (QKV already projected on GPU 0) ─────────────────────────
                // KV write: per KV head, from GPU 0's k/v_attn to this GPU's KV cache
                for h_local in 0..local_nkh {
                    let h_global = gpu_i * local_nkh + h_local;
                    let src_k = k_attn_base + (h_global * hd * 4) as u64;
                    let src_v = v_attn_base + (h_global * hd * 4) as u64;
                    let dst_k =
                        kv_k_base + ((h_local * head_stride + position as usize * hd) * 4) as u64;
                    let dst_v =
                        kv_v_base + ((h_local * head_stride + position as usize * hd) * 4) as u64;
                    batch.push(D2dCopyInst::new(((hd as u32) + 255) / 256, dst_k as *mut f32, src_k as *const f32, hd as i32).into_inst());
                    batch.push(D2dCopyInst::new(((hd as u32) + 255) / 256, dst_v as *mut f32, src_v as *const f32, hd as i32).into_inst());
                }

                // For GPU i > 0: copy Q slice from GPU 0's q_attn to local attn_q
                if gpu_i > 0 {
                    let src_q = q_attn_base + (gpu_i * local_nqh * hd * 4) as u64;
                    batch.push(D2dCopyInst::new(((local_nqh * hd) as u32 + 255) / 256, q_ptr as *mut f32, src_q as *const f32, (local_nqh * hd) as i32).into_inst());
                }

                // GQA attention
                let seq_len = (position + 1) as i32;
                let q_src = if gpu_i == 0 { q_attn_base } else { q_ptr };
                {
                    let mut inst = GqaAttnInstLocal::new(local_nqh as u32, out_ptr as *mut f32, q_src as *const f32, kv_k_base as *const f32, kv_v_base as *const f32, nqh as i32, nkh as i32, hd as i32, seq_len, max_sl as i32);
                    inst.q_head_start = (gpu_i * local_nqh) as u64;
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
        let all_attn = std::env::var("BRAIDINFER_DECODE_MIRROR_ALL_ATTN").is_ok();
        if self.tracer.enabled() && (attn_i == 0 || all_attn) {
            let normed_stage_host: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    self.activations.normed_stage.host_ptr() as *const f32,
                    hs,
                )
            }.to_vec();
            let label = format!("attn{attn_i}-post-ack pos={position}");
            self.decode_mirror_snapshot(&label, position);
            print_normed_stage_stats(&label, position, &normed_stage_host);
        }

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
    fn dispatch_moe_workers_decode_async(&mut self, layer_idx: usize, gpu0_seq: Option<u32>) -> Result<(), ModelError> {
        // Find the OP_MOE_DISPATCH instruction for this layer.
        let mk = self.megakernel_multi_gpu_p2p.as_ref().unwrap();
        let moe_inst_idx = mk.barrier_layer_map.iter()
            .find(|&&(_, l)| l == layer_idx)
            .map(|&(i, _)| i)
            .expect("layer_idx not in barrier_layer_map");
        let inst = &mk.instructions[moe_inst_idx];

        // Decode MoeDispatchInst layout (see kernels/megakernel_moe_dispatch.hip header).
        // words[2]=output_slots, [4]=expert_ids, [5]=expert_weights, [7]=(num_workers<<32)|hs,
        // [8]=(layer_idx<<32)|k, [9]=(eis<<32)|has_gate, [10]=activation, [16]=gate_up_in_dim
        let output_slots = inst.words[2] as *mut f32;
        let expert_ids = inst.words[4] as *const i32;
        let expert_weights = inst.words[5] as *const f32;
        let hs = (inst.words[7] & 0xFFFFFFFF) as usize;
        let k = (inst.words[8] & 0xFFFFFFFF) as usize;
        let eis = (inst.words[9] >> 32) as usize;
        let has_gate = (inst.words[9] & 0xFFFFFFFF) != 0;
        // inst.words[10] holds act.normed.as_ptr() — cached GPU 0 VRAM.
        // Workers previously P2P-read from there (cross-GPU peer-VRAM read
        // of cached memory) which pressures GPU 0's L2 + PCIe non-posted
        // path at 4+ GPUs (snl follow-up: ea bridge #242 mechanism).
        // Workers now read from a per-worker device pointer to the
        // host-mapped UC handoff buffer; the megakernel program stages
        // `act.normed → moe_act_uc_handoff` via OP_D2D_COPY at MoE entry.
        let _activation_cached = inst.words[10] as *const f32;
        let mut gupd = inst.words[16] as usize;
        if gupd == 0 { gupd = hs; }
        // Standard MoE has gate→silu_mul; non-gated path uses relu_squared.
        let relu_sq = !has_gate;

        let p2p = self.moe_p2p.as_ref().expect("moe_p2p must be initialized for MoE decode");
        let num_gpus = p2p.num_gpus;
        let num_workers = p2p.workers.len();
        // Token index 0 for decode (single-token); per-worker output slot:
        //   output_slots + (0 * num_gpus + gpu_id) * hs == output_slots + gpu_id * hs
        // Worker GPU id = worker_idx + 1 (workers are GPUs 1..N-1).

        // Build per-worker instructions then dispatch_batch_fire on each, then wait.
        // We borrow p2p immutably for build; persistent_workers borrowed mutably for dispatch.
        let insts: Vec<(usize, crate::megakernel::Instruction)> = (0..num_workers).map(|w| {
            let gpu_id = w + 1;
            let out_slot = unsafe { output_slots.add(gpu_id * hs) };
            // Per-worker device pointer to the host-mapped activation handoff.
            let activation = p2p.moe_act_uc_handoff_dev_ptrs[gpu_id] as *const f32;
            let inst = p2p.build_ffn_remote_inst(
                w,
                layer_idx,
                activation,
                out_slot,
                expert_ids,
                expert_weights,
                k, eis, hs, gupd, has_gate, relu_sq,
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
        if let Some(seq) = gpu0_seq {
            seq_per_gpu.push((0, seq));
        }
        if let Err(e) = dispatch.try_wait_acks_many(&seq_per_gpu) {
            panic!("{e}");
        }
        Ok(())
    }


    /// Run a single decode step using the paged KV cache path.
    /// Returns logits [vocab_size].
    pub fn decode_step_paged(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<Vec<f32>, ModelError> {
        self.decode_step_paged_inner(token_id, position, false)
    }

    /// Run a single decode step with quantized KV cache (4-bit residual_pc).
    /// Sealed chunks are quantized to int4; active chunk stays f32.
    pub fn decode_step_paged_quantized(
        &mut self,
        token_id: u32,
        position: u32,
    ) -> Result<Vec<f32>, ModelError> {
        self.decode_step_paged_inner(token_id, position, true)
    }

    fn decode_step_paged_inner(
        &mut self,
        token_id: u32,
        position: u32,
        quantized: bool,
    ) -> Result<Vec<f32>, ModelError> {
        // KV quantization is single-GPU only: multi-GPU paged dispatch not yet implemented.
        if quantized && self.multi_gpu.is_some() {
            return Err(ModelError::InvalidConfig(
                "KV_QUANT is not supported in multi-GPU mode".into(),
            ));
        }

        let max_chunks = self.max_paged_chunks();

        // Lazy-init: compile paged megakernel
        if self.megakernel_paged.is_none() {
            let mut mk = MegakernelProgram::compile_paged(self)?;
            mk.init_paged_buffers(max_chunks)?;
            if quantized {
                mk.enable_quantized_kv(max_chunks, &self.config)?;
            }
            self.megakernel_paged = Some(mk);
        } else {
            let mk = self.megakernel_paged.as_ref().unwrap();
            assert_eq!(
                mk.quantized_kv, quantized,
                "cannot mix decode_step_paged and decode_step_paged_quantized on the same model"
            );
        }

        // Lazy-init: f32 PageAllocator (staging) and SequenceState
        self.ensure_paged_decode_state(quantized)?;

        // append_token
        {
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            seq_mut.append_token(position as i32, alloc_mut)?;
        }

        // Write position_ids to host-mapped memory before paged step (no hipMemcpy).
        self.set_position(position).map_err(ModelError::Hip)?;

        let stream = &self.stream;
        let mk = self.megakernel_paged.as_mut().unwrap();
        let seq = self.paged_seq.as_ref().unwrap();
        let allocator = self.page_allocator.as_ref().unwrap();

        mk.update_step_paged(token_id, position, seq, allocator, stream)?;
        mk.execute(stream)?;
        stream.synchronize()?;

        // Post-step: handle chunk seal + quantization
        {
            let mk = self.megakernel_paged.as_mut().unwrap();
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            let q_alloc = self.quant_allocator.as_mut();
            mk.post_step_paged(
                position,
                seq_mut,
                alloc_mut,
                q_alloc,
                &self.config,
                &self.stream,
            )?;
        }

        // Chunk-seal mirror hook (wt1 P2-c): same as persistent path but for
        // the non-persistent paged decode route.
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

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;
        self.seq_len = position + 1;
        Ok(logits)
    }
}
