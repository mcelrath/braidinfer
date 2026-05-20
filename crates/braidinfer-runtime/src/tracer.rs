//! Unified SDMA-based tracer (bd tmo8 Phase 1).
//!
//! Replaces three legacy diagnostic paths:
//!   - `decode/trace.rs` per-layer checkpoints (BTRC writer)
//!   - `mirror.rs::DecodeMirror` multi-GPU coherence snapshots
//!
//! All capture issues `hipMemcpyAsync(DeviceToHost)` on a borrowed per-GPU SDMA
//! stream (owned by `PersistentDispatch::sdma_streams`, wt1 P2-a). SDMA engines
//! run independently of compute CUs, so capture is safe under the cooperative
//! persistent worker — proven by wt1 panel (3-model × 200 token mirror-on).
//!
//! Persistent-worker safety model:
//!   - `Tracer` borrows raw `hipStream_t` handles; never owns them.
//!   - `PersistentDispatch` outlives `Tracer` via struct-field declaration order
//!     in `Model` (same pattern as the retired `DecodeMirror`).
//!   - `capture*` MUST NOT call `hipStreamCreate`, `hipMemcpy` (sync), or any
//!     other HIP API that would launch a kernel on a CU-holding GPU.
//!
//! Zero-overhead-when-disabled guarantee (PLAN-tmo8 N5):
//!   - `capture*` early-returns on `ProbeFilter::None` BEFORE evaluating
//!     `Probe::name()` or constructing any `Cow::Owned`.
//!   - Callers that build runtime-formatted Custom probes MUST gate their
//!     `format!()` with `if self.tracer.enabled() { ... }` to avoid the
//!     allocation. See R4 in the plan.
//!
//! # Dump pipeline contract (production decode path)
//!
//! Per-layer probes on the production `decode_step_persistent` path are emitted
//! via the megakernel's in-kernel dump pipeline (`kernels/dump.h` +
//! `kernels/megakernel_dispatch.hip`). When `Tracer::enabled()` returns true on
//! a model's first traced step, `enable_dump_persistent` allocates a VRAM
//! `dump_buffer` + host-mapped `dump_counter`, and
//! `PersistentDispatch::set_trace_dump_ptrs` plumbs them into the worker's
//! `WorkerQueue`. After each batch ack, the dispatcher calls
//! `drain_trace_dump` which SDMA-copies populated slots from `dump_buffer` to
//! pinned host, decodes each slot via [`decode_dump_slot`] (pure, unit-tested),
//! looks up the instruction's `Probe` in `MegakernelProgram::trace_probe_map`,
//! and inserts the payload into [`Tracer::insert_shadow`].
//!
//! ## Important: drain-side filtering, not kernel-side
//!
//! The kernel's `dump_instruction_output` fires on EVERY dump-eligible opcode
//! (see the `switch` block in `kernels/dump.h`: `OP_RMSNORM`, `OP_LINEAR_PROJ`,
//! `OP_RESIDUAL_ADD`, `OP_SCALE_ADD`, `OP_FFN_DOWN_RES`, `OP_EMBEDDING`,
//! `OP_GDN_GATE`, `OP_QK_NORM`, `OP_MROPE`, `OP_OUTPUT_GATE`, `OP_MOE_GATE`,
//! `OP_MOE_FFN`, plus FFN and LINEAR_PROJ variants). It does NOT consult
//! `trace_probe_map`. Filtering happens drain-side in
//! `PersistentDispatch::drain_trace_dump`: slots whose `inst_idx` doesn't match
//! a probe entry are discarded.
//!
//! Consequence: `dump_buffer` capacity must accommodate ALL dump-eligible
//! instructions in the program, not just the probe sites. The decode path
//! sizes `max_slots = min(instructions.len(), 4096)` for that reason.
//!
//! A future optimization (bd k357) will add a kernel-side `trace_mask` so the
//! kernel skips non-probe ops, allowing exact-sized capacity.
//!
//! ## Probe → opcode mapping
//!
//! `MegakernelProgram::trace_probe_map` is populated by `compile_inner` at
//! these sites:
//!
//! | Probe variant       | Compile site                                            | Underlying opcode |
//! |---------------------|---------------------------------------------------------|-------------------|
//! | `Embed`             | After `EmbeddingInst` push                              | `OP_EMBEDDING`    |
//! | `PostMixer{layer}`  | Last inst of `compile_attention_layer*` / `_gdn_layer` / `_mamba2_layer` | `OP_RESIDUAL_ADD` (attn) / `OP_SCALE_ADD` (GDN/Mamba2) |
//! | `PostFfn{layer}`    | Last inst of `compile_ffn` (Dense FFN)                  | `OP_FFN_DOWN_RES` |
//! | `FinalNorm`         | Final RMSNorm before LM head                            | `OP_RMSNORM*`     |
//!
//! Every opcode in this table must be present in `kernels/dump.h`'s switch
//! statement. If a new mixer/FFN variant lands with a non-dump-eligible final
//! opcode, the probe will silently fail to capture. To detect this regression,
//! add the new opcode to dump.h's switch.

use std::borrow::Cow;
use std::collections::HashMap;

use braidinfer_hip::{HipError, HipResult, error, ffi};
use braidinfer_hip::memory::{DeviceBuffer, PinnedBuffer};

// ─────────────────────────────────────────────────────────────────────────────
// Probe — hybrid enum + free-form fallback
// ─────────────────────────────────────────────────────────────────────────────

/// What to capture. Stable variants give type-checked names for the common
/// per-layer / per-GPU probes; `Custom` handles ad-hoc / GDN-deep-dive labels.
///
/// `Custom` takes `Cow<'static, str>`:
///   - `Cow::Borrowed("static_name")` — zero allocation.
///   - `Cow::Owned(format!("L{i}.{tensor}"))` — one allocation. Callers MUST
///     check `tracer.enabled()` before constructing these or the
///     zero-overhead-when-disabled guarantee is broken (PLAN-tmo8 R4).
#[derive(Debug, Clone)]
pub enum Probe {
    /// Token embedding output.
    Embed,
    /// After attention/GDN/SSM block of layer `layer`.
    PostMixer { layer: usize },
    /// After FFN block of layer `layer`.
    PostFfn { layer: usize },
    /// Final RMSNorm before LM head.
    FinalNorm,
    /// Top-K logits after LM head.
    Logits { top_k: usize },
    /// Per-GPU `act.normed` (post-attn-RMSNorm staging).
    AttnNormed { gpu: usize },
    /// Per-GPU `act.q_gate_attn` (Q+gate interleaved post-projection).
    AttnQGate { gpu: usize },
    /// Per-GPU `act.k_attn` (K post-projection).
    AttnK { gpu: usize },
    /// Per-GPU hidden state; `head_only=true` captures the first 16 floats.
    Hidden { gpu: usize, head_only: bool },
    /// Per-(gpu, attn_layer, k_or_v, head) KV cache slice.
    KvCache { gpu: usize, attn_layer: usize, k: bool, head: usize },
    /// MoE per-token output slot, indexed by (gpu, layer, token).
    MoeOutputSlots { gpu: usize, layer: usize, token: usize },
    /// Per-worker FFN_REMOTE local_output for layer `layer`.
    WorkerFfnOut { worker: usize, layer: usize },
    /// Ad-hoc probe. Prefer `Cow::Borrowed` for stable strings.
    Custom(Cow<'static, str>),
}

impl Probe {
    /// Canonical name used for filter matching + sink keys.
    pub fn name(&self) -> Cow<'static, str> {
        match self {
            Probe::Embed => Cow::Borrowed("embed"),
            Probe::PostMixer { layer } => Cow::Owned(format!("L{layer}.post_mixer")),
            Probe::PostFfn { layer } => Cow::Owned(format!("L{layer}.post_ffn")),
            Probe::FinalNorm => Cow::Borrowed("final_norm"),
            Probe::Logits { top_k } => Cow::Owned(format!("top{top_k}_logits")),
            Probe::AttnNormed { gpu } => Cow::Owned(format!("g{gpu}.attn_normed")),
            Probe::AttnQGate { gpu } => Cow::Owned(format!("g{gpu}.attn_q_gate")),
            Probe::AttnK { gpu } => Cow::Owned(format!("g{gpu}.attn_k")),
            Probe::Hidden { gpu, head_only } => {
                if *head_only {
                    Cow::Owned(format!("g{gpu}.hidden.head16"))
                } else {
                    Cow::Owned(format!("g{gpu}.hidden"))
                }
            }
            Probe::KvCache { gpu, attn_layer, k, head } => {
                let kv = if *k { "k" } else { "v" };
                Cow::Owned(format!("g{gpu}.kv[{attn_layer}].{kv}(h{head})"))
            }
            Probe::MoeOutputSlots { gpu, layer, token } => {
                Cow::Owned(format!("moe.L{layer}.output_slots[g{gpu}][t{token}]"))
            }
            Probe::WorkerFfnOut { worker, layer } => {
                Cow::Owned(format!("moe.L{layer}.worker{worker}_ffn_out"))
            }
            Probe::Custom(s) => s.clone(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProbeFilter
// ─────────────────────────────────────────────────────────────────────────────

pub enum ProbeFilter {
    /// No probes fire. Zero overhead.
    None,
    /// All probes fire.
    All,
    /// Regex match against `Probe::name()`. Acceptable for an operator-set
    /// dev/debug env var; the regex crate's RE2 engine has no catastrophic
    /// backtracking.
    Regex(regex::Regex),
}

impl ProbeFilter {
    /// Parse `BRAIDINFER_TRACE` env var into a filter.
    ///   unset / empty  → None
    ///   "1"            → All
    ///   any other      → Regex
    pub fn from_env() -> HipResult<Self> {
        let val = std::env::var("BRAIDINFER_TRACE").ok();
        match val.as_deref() {
            None | Some("") => Ok(ProbeFilter::None),
            Some("1") => Ok(ProbeFilter::All),
            Some(pattern) => {
                let re = regex::Regex::new(pattern).map_err(|e| {
                    eprintln!("[braidinfer] WARN: invalid BRAIDINFER_TRACE regex '{pattern}': {e} — disabling tracer");
                    HipError(ffi::hipErrorInvalidValue)
                })?;
                Ok(ProbeFilter::Regex(re))
            }
        }
    }

    #[inline(always)]
    pub fn is_none(&self) -> bool {
        matches!(self, ProbeFilter::None)
    }

    /// Does this filter fire on `name`? Callers SHOULD check `is_none()` first
    /// to avoid constructing the name string when disabled.
    pub fn matches(&self, name: &str) -> bool {
        match self {
            ProbeFilter::None => false,
            ProbeFilter::All => true,
            ProbeFilter::Regex(re) => re.is_match(name),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dump slot decoder — pure function, testable without GPU
// ─────────────────────────────────────────────────────────────────────────────

/// Byte size of a single dump slot. Must match `DUMP_SLOT_BYTES` in
/// `kernels/dump.h`: 4 u32 header + 8192 f32 payload = 32784 bytes.
pub const DUMP_SLOT_BYTES: usize = 16 + 8192 * 4;
pub const DUMP_HEADER_INTS: usize = 4;
pub const DUMP_MAX_FLOATS: usize = 8192;

/// Parsed view of one dump slot. `payload` borrows from the input buffer.
#[derive(Debug, Clone, Copy)]
pub struct DumpSlot<'a> {
    pub opcode: u32,
    pub inst_idx: u32,
    /// Number of f32 floats in the payload. May be 0 if the opcode wasn't
    /// dump-eligible (kernel writes header only). May be capped at
    /// `DUMP_MAX_FLOATS` if the op's natural output exceeded the slot.
    pub size: u32,
    pub payload: &'a [f32],
}

/// Decode one dump slot from raw bytes. Returns `None` if `bytes` is too short
/// for a full slot. Format mirrors `kernels/dump.h`:
///   [opcode:u32 | inst_idx:u32 | size:u32 | pad:u32] [data:f32[size]]
pub fn decode_dump_slot(bytes: &[u8]) -> Option<DumpSlot<'_>> {
    if bytes.len() < DUMP_HEADER_INTS * 4 {
        return None;
    }
    let opcode = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let inst_idx = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let size = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    // pad at bytes[12..16] — reserved.
    let payload_floats = (size as usize).min(DUMP_MAX_FLOATS);
    let payload_bytes = payload_floats * 4;
    let payload_start = DUMP_HEADER_INTS * 4;
    if bytes.len() < payload_start + payload_bytes {
        return None;
    }
    let payload = unsafe {
        std::slice::from_raw_parts(
            bytes[payload_start..].as_ptr() as *const f32,
            payload_floats,
        )
    };
    Some(DumpSlot { opcode, inst_idx, size, payload })
}

// ─────────────────────────────────────────────────────────────────────────────
// TraceSink — BTRC binary format (moved from trace.rs, byte-for-byte identical)
// ─────────────────────────────────────────────────────────────────────────────

/// Streaming binary sink. Format:
///   header: "BTRC" + u32 version (1)
///   record: u32 name_len | name bytes | u32 num_elements | f32[num_elements]
///   tail: u32 count
///
/// Compatible with `scripts/compare_traces.py` (BTRC v1 reader).
pub struct TraceSink {
    writer: std::io::BufWriter<std::fs::File>,
    count: u32,
}

impl TraceSink {
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        use std::io::Write;
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(b"BTRC")?;
        let version: u32 = 1;
        writer.write_all(&version.to_le_bytes())?;
        Ok(TraceSink { writer, count: 0 })
    }

    pub fn write_checkpoint(&mut self, name: &str, data: &[f32]) -> std::io::Result<()> {
        use std::io::Write;
        let name_bytes = name.as_bytes();
        self.writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        self.writer.write_all(name_bytes)?;
        self.writer.write_all(&(data.len() as u32).to_le_bytes())?;
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.writer.write_all(bytes)?;
        self.count += 1;
        Ok(())
    }

    pub fn close(mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.writer.write_all(&self.count.to_le_bytes())?;
        self.writer.flush()?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tracer
// ─────────────────────────────────────────────────────────────────────────────

pub struct Tracer {
    /// One SDMA stream per GPU, BORROWED from `PersistentDispatch::sdma_streams`.
    /// Tracer must outlive any compute that holds CUs, but PersistentDispatch
    /// outlives Tracer via Model field-drop order (same model as DecodeMirror).
    streams: Vec<ffi::hipStream_t>,
    /// Per-probe pinned-host shadow, lazy-allocated on first capture by canonical name.
    shadows: HashMap<String, PinnedBuffer<u8>>,
    /// Optional BTRC file sink. None = in-memory only.
    sink: Option<TraceSink>,
    filter: ProbeFilter,
}

unsafe impl Send for Tracer {}
unsafe impl Sync for Tracer {}

impl Tracer {
    /// Construct from env vars (`BRAIDINFER_TRACE`, `BRAIDINFER_TRACE_FILE`,
    /// legacy `TRACE`). Pass per-GPU SDMA stream handles borrowed from
    /// `PersistentDispatch::sdma_streams`.
    pub fn from_env(streams: Vec<ffi::hipStream_t>) -> HipResult<Self> {
        let mut filter = ProbeFilter::from_env()?;
        let mut sink_path = std::env::var("BRAIDINFER_TRACE_FILE").ok();

        // Legacy TRACE=path alias.
        if filter.is_none() && sink_path.is_none() {
            if let Ok(legacy_path) = std::env::var("TRACE") {
                if !legacy_path.is_empty() {
                    eprintln!(
                        "[braidinfer] WARN: TRACE env var is deprecated; use BRAIDINFER_TRACE=1 + BRAIDINFER_TRACE_FILE={legacy_path}"
                    );
                    filter = ProbeFilter::All;
                    sink_path = Some(legacy_path);
                }
            }
        }

        let sink = sink_path.and_then(|p| match TraceSink::open(&p) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("[braidinfer] WARN: TraceSink::open({p}) failed: {e}");
                None
            }
        });

        Ok(Tracer {
            streams,
            shadows: HashMap::new(),
            sink,
            filter,
        })
    }

    /// Test-only constructor: filter is None, no streams. All capture calls no-op.
    #[doc(hidden)]
    pub fn disabled() -> Self {
        Tracer {
            streams: Vec::new(),
            shadows: HashMap::new(),
            sink: None,
            filter: ProbeFilter::None,
        }
    }

    /// Construct with an explicit filter and borrowed SDMA streams. Used by
    /// `DecodeMirror` to wrap its hipMemcpyAsync sites without changing the
    /// `BRAIDINFER_DECODE_MIRROR` env-var gate (Phase 2a facade).
    #[doc(hidden)]
    pub fn with_filter_and_streams(streams: Vec<ffi::hipStream_t>, filter: ProbeFilter) -> Self {
        Tracer {
            streams,
            shadows: HashMap::new(),
            sink: None,
            filter,
        }
    }

    #[inline(always)]
    pub fn enabled(&self) -> bool {
        !self.filter.is_none()
    }

    /// Issue an SDMA D2H copy of `bytes` from device `gpu_idx`'s `ptr` into the
    /// pinned-host shadow keyed by `probe.name()`. No-op if the filter doesn't
    /// match (early-returns BEFORE constructing the name when filter is None).
    ///
    /// # Safety
    /// `ptr` must point to at least `bytes` bytes of VRAM owned by GPU `gpu_idx`.
    /// Caller is responsible for matching the device — passing a GPU-1 ptr with
    /// `gpu_idx=0` issues the copy on GPU 0's SDMA stream and will fault.
    pub fn capture(
        &mut self,
        gpu_idx: usize,
        probe: Probe,
        ptr: *const u8,
        bytes: usize,
    ) -> HipResult<()> {
        if self.filter.is_none() {
            return Ok(());
        }
        let name = probe.name();
        if !self.filter.matches(&name) {
            return Ok(());
        }
        let stream = self.streams.get(gpu_idx).copied().ok_or_else(|| {
            eprintln!(
                "[braidinfer] Tracer::capture: gpu_idx={gpu_idx} out of range (have {} streams)",
                self.streams.len()
            );
            HipError(ffi::hipErrorInvalidValue)
        })?;
        if stream.is_null() {
            eprintln!("[braidinfer] Tracer::capture: SDMA stream for gpu_idx={gpu_idx} is null — PersistentDispatch::ensure_sdma_stream not called");
            return Err(HipError(ffi::hipErrorInvalidValue));
        }
        let key = name.into_owned();
        let shadow = match self.shadows.get_mut(&key) {
            Some(s) if s.len() >= bytes => s,
            _ => {
                let buf = PinnedBuffer::<u8>::alloc(bytes)?;
                self.shadows.insert(key.clone(), buf);
                self.shadows.get_mut(&key).unwrap()
            }
        };
        error::check(unsafe {
            ffi::hipMemcpyAsync(
                shadow.as_mut_ptr() as *mut std::ffi::c_void,
                ptr as *const std::ffi::c_void,
                bytes,
                ffi::hipMemcpyDeviceToHost,
                stream,
            )
        })?;
        Ok(())
    }

    pub fn capture_f32(
        &mut self,
        gpu_idx: usize,
        probe: Probe,
        buf: &DeviceBuffer<f32>,
    ) -> HipResult<()> {
        if self.filter.is_none() {
            return Ok(());
        }
        self.capture(gpu_idx, probe, buf.as_ptr() as *const u8, buf.size_bytes())
    }

    pub fn capture_bf16(
        &mut self,
        gpu_idx: usize,
        probe: Probe,
        buf: &DeviceBuffer<u16>,
    ) -> HipResult<()> {
        if self.filter.is_none() {
            return Ok(());
        }
        self.capture(gpu_idx, probe, buf.as_ptr() as *const u8, buf.size_bytes())
    }

    /// Synchronize all SDMA streams, then flush sink (if set). Call once per
    /// logical step boundary.
    pub fn drain(&mut self) -> HipResult<()> {
        if self.filter.is_none() {
            return Ok(());
        }
        for &s in &self.streams {
            if !s.is_null() {
                error::check(unsafe { ffi::hipStreamSynchronize(s) })?;
            }
        }
        if let Some(sink) = self.sink.as_mut() {
            for (name, shadow) in &self.shadows {
                if shadow.len() % 4 == 0 {
                    let n = shadow.len() / 4;
                    let slice = unsafe {
                        std::slice::from_raw_parts(shadow.as_ptr() as *const f32, n)
                    };
                    if let Err(e) = sink.write_checkpoint(name, slice) {
                        eprintln!("[braidinfer] TraceSink write failed for {name}: {e}");
                    }
                }
            }
        }
        Ok(())
    }

    /// Read a probe's shadow contents after `drain()`. Returns None if the probe
    /// was never captured (or filter rejected it).
    pub fn read(&self, probe: Probe) -> Option<&[u8]> {
        let name = probe.name();
        self.shadows.get(name.as_ref()).map(|s| s.as_slice())
    }

    pub fn read_f32(&self, probe: Probe) -> Option<&[f32]> {
        self.read(probe).and_then(|bytes| {
            if bytes.len() % 4 == 0 {
                Some(unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
                })
            } else {
                None
            }
        })
    }

    /// Write host-side f32 data directly to the BTRC sink (if set). Use this
    /// when data is already on the CPU (e.g., host-mapped logits) and no SDMA
    /// copy is needed. No-op if filter doesn't match or sink is absent.
    pub fn record_host_f32(&mut self, probe: Probe, data: &[f32]) {
        if self.filter.is_none() {
            return;
        }
        let name = probe.name();
        if !self.filter.matches(&name) {
            return;
        }
        if let Some(sink) = self.sink.as_mut() {
            if let Err(e) = sink.write_checkpoint(&name, data) {
                eprintln!("[braidinfer] Tracer::record_host_f32: sink write failed for {name}: {e}");
            }
        }
    }

    /// Expose the filter's match predicate for callers that have already
    /// constructed the name string (e.g., drain_trace_dump in persistent_dispatch).
    #[inline(always)]
    pub fn filter_matches(&self, name: &str) -> bool {
        self.filter.matches(name)
    }

    /// Insert pre-decoded host bytes directly into the shadow map. Used by
    /// `PersistentDispatch::drain_trace_dump` which reads dump slots from VRAM
    /// via SDMA and has already copied the payload to the CPU.
    ///
    /// Allocates or reuses a `PinnedBuffer<u8>` sized to `bytes.len()` keyed by
    /// `name`. The buffer is allocated via `hipHostMalloc` (pinned DRAM), NOT via
    /// SDMA — this is safe because `drain_trace_dump` is called from CPU-only
    /// code that runs outside the persistent worker's batch dispatch window.
    pub fn insert_shadow(&mut self, name: String, bytes: &[u8]) {
        use braidinfer_hip::memory::PinnedBuffer;
        let shadow = match self.shadows.get_mut(&name) {
            Some(s) if s.len() >= bytes.len() => s,
            _ => {
                // hipHostMalloc is allowed here: drain_trace_dump is called after
                // wait_ack, outside any kernel dispatch, so no cooperative kernel
                // holds CUs at this point in the decode loop.
                let buf = match PinnedBuffer::<u8>::alloc(bytes.len()) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("[braidinfer] Tracer::insert_shadow: alloc failed for {name}: {e:?}");
                        return;
                    }
                };
                self.shadows.insert(name.clone(), buf);
                self.shadows.get_mut(&name).unwrap()
            }
        };
        // SAFETY: PinnedBuffer<u8> owns len bytes of pinned DRAM; copying into it is safe.
        let dst = unsafe { shadow.as_mut_slice() };
        dst[..bytes.len()].copy_from_slice(bytes);
    }

    /// Close + flush the sink (if any). Safe to call multiple times; subsequent
    /// calls are no-ops.
    pub fn close_sink(&mut self) -> std::io::Result<()> {
        if let Some(sink) = self.sink.take() {
            sink.close()?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_name_stable_strings() {
        assert_eq!(&*Probe::Embed.name(), "embed");
        assert_eq!(&*Probe::FinalNorm.name(), "final_norm");
        assert_eq!(&*Probe::PostMixer { layer: 5 }.name(), "L5.post_mixer");
        assert_eq!(&*Probe::PostFfn { layer: 42 }.name(), "L42.post_ffn");
        assert_eq!(&*Probe::Logits { top_k: 10 }.name(), "top10_logits");
        assert_eq!(&*Probe::AttnNormed { gpu: 1 }.name(), "g1.attn_normed");
        assert_eq!(
            &*Probe::Hidden { gpu: 0, head_only: true }.name(),
            "g0.hidden.head16"
        );
        assert_eq!(
            &*Probe::KvCache { gpu: 1, attn_layer: 0, k: true, head: 0 }.name(),
            "g1.kv[0].k(h0)"
        );
        assert_eq!(
            &*Probe::Custom(Cow::Borrowed("normed")).name(),
            "normed"
        );
        assert_eq!(
            &*Probe::Custom(Cow::Owned("L0.qkv_pre_conv".into())).name(),
            "L0.qkv_pre_conv"
        );
    }

    #[test]
    fn filter_none_skips_everything() {
        let f = ProbeFilter::None;
        assert!(f.is_none());
        assert!(!f.matches("embed"));
        assert!(!f.matches("L5.post_mixer"));
    }

    #[test]
    fn filter_all_matches_everything() {
        let f = ProbeFilter::All;
        assert!(!f.is_none());
        assert!(f.matches("embed"));
        assert!(f.matches("anything"));
    }

    #[test]
    fn filter_regex_matches_pattern() {
        let f = ProbeFilter::Regex(regex::Regex::new(r"L\d+\.post_mixer").unwrap());
        assert!(f.matches("L5.post_mixer"));
        assert!(f.matches("L42.post_mixer"));
        assert!(!f.matches("L5.post_ffn"));
        assert!(!f.matches("embed"));
    }

    #[test]
    fn disabled_tracer_is_zero_overhead_noop() {
        let mut t = Tracer::disabled();
        assert!(!t.enabled());
        // capture with out-of-range gpu_idx + null ptr would be a hard error if
        // the early-return guard were missing. The fact that it returns Ok and
        // does not allocate a shadow proves the early-return path is taken
        // BEFORE name evaluation and gpu_idx bounds check.
        let r = t.capture(99, Probe::Embed, std::ptr::null(), 0);
        assert!(r.is_ok());
        let r = t.capture(99, Probe::Custom(Cow::Borrowed("xyz")), std::ptr::null(), 0);
        assert!(r.is_ok());
        let r = t.drain();
        assert!(r.is_ok());
        assert_eq!(t.shadows.len(), 0);
    }

    #[test]
    fn btrc_format_round_trip() {
        use std::io::Read;
        let path = std::env::temp_dir().join("braidinfer_tracer_btrc_round_trip.btrc");
        {
            let mut sink = TraceSink::open(&path).unwrap();
            sink.write_checkpoint("hello", &[1.0_f32, 2.0, 3.0]).unwrap();
            sink.write_checkpoint("world", &[4.0_f32]).unwrap();
            sink.close().unwrap();
        }
        let mut bytes = Vec::new();
        std::fs::File::open(&path).unwrap().read_to_end(&mut bytes).unwrap();
        // Header
        assert_eq!(&bytes[0..4], b"BTRC");
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            1,
            "BTRC version"
        );
        // Tail u32 = count = 2
        let n = bytes.len();
        let count = u32::from_le_bytes([bytes[n - 4], bytes[n - 3], bytes[n - 2], bytes[n - 1]]);
        assert_eq!(count, 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn dump_slot_decode_well_formed() {
        // Build a slot the way kernels/dump.h would: [opcode, inst_idx, size, pad]
        // header then `size` f32 payload, padded out to DUMP_SLOT_BYTES.
        let mut bytes = vec![0u8; DUMP_SLOT_BYTES];
        bytes[0..4].copy_from_slice(&36u32.to_le_bytes()); // OP_SCALE_ADD
        bytes[4..8].copy_from_slice(&17u32.to_le_bytes()); // inst_idx 17
        bytes[8..12].copy_from_slice(&3u32.to_le_bytes()); // size 3
        // pad at [12..16] = zero
        let payload = [1.5_f32, -2.25, 3.125];
        for (i, &v) in payload.iter().enumerate() {
            bytes[16 + i * 4..16 + (i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        let slot = decode_dump_slot(&bytes).expect("decode well-formed slot");
        assert_eq!(slot.opcode, 36);
        assert_eq!(slot.inst_idx, 17);
        assert_eq!(slot.size, 3);
        assert_eq!(slot.payload, &[1.5_f32, -2.25, 3.125]);
    }

    #[test]
    fn dump_slot_decode_caps_oversized_size() {
        // Kernel writes size > DUMP_MAX_FLOATS (truncated by dump.h:133 to
        // DUMP_MAX_FLOATS). The decoder's payload slice should also cap at
        // DUMP_MAX_FLOATS to match what the kernel actually wrote.
        let mut bytes = vec![0u8; DUMP_SLOT_BYTES];
        bytes[8..12].copy_from_slice(&((DUMP_MAX_FLOATS as u32) * 2).to_le_bytes());
        let slot = decode_dump_slot(&bytes).unwrap();
        assert_eq!(slot.payload.len(), DUMP_MAX_FLOATS);
    }

    #[test]
    fn dump_slot_decode_zero_size() {
        // size=0 = opcode wasn't dump-eligible (kernel returns early before
        // writing payload). Decoder returns empty payload slice.
        let bytes = vec![0u8; DUMP_SLOT_BYTES];
        let slot = decode_dump_slot(&bytes).unwrap();
        assert_eq!(slot.size, 0);
        assert_eq!(slot.payload.len(), 0);
    }

    #[test]
    fn dump_slot_decode_short_buffer() {
        // Truncated buffer (less than header bytes) → None.
        let bytes = vec![0u8; 8];
        assert!(decode_dump_slot(&bytes).is_none());
    }

    #[test]
    fn filter_from_env_respects_unset() {
        // SAFETY: tests run sequentially within a binary; we set then unset.
        unsafe {
            std::env::remove_var("BRAIDINFER_TRACE");
        }
        let f = ProbeFilter::from_env().unwrap();
        assert!(f.is_none());
    }
}
