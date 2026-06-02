//! Weight types, loading, and activation buffer allocation.
//! Extracted from model.rs for maintainability.

use braidinfer_core::safetensors::SafeTensorSet;
use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::{DeviceBuffer, MappedHostBuffer};
use safetensors::Dtype;

pub use crate::quant::{
    LinearWeight, PackedWeights, WeightFormat, WeightQuantMode, quantize_pc_g32_q4,
    quantize_rnf4_g128,
};

// ---- Layer weight structs ----

pub struct GdnLayerWeights {
    pub input_norm: DeviceBuffer<u16>, // bf16: (1+w) pattern, zeros init
    pub w_qkv: LinearWeight,           // [6144, 1024]
    pub w_a: LinearWeight,             // [16, 1024]
    pub w_b: LinearWeight,             // [16, 1024]
    pub w_z: LinearWeight,             // [2048, 1024]
    pub conv1d_weight_q: DeviceBuffer<u16>, // bf16 [nh*kd, ck] pre-split Q slice
    pub conv1d_weight_k: DeviceBuffer<u16>, // bf16 [nh*kd, ck] pre-split K slice
    pub conv1d_weight_v: DeviceBuffer<u16>, // bf16 [nh*vd, ck] pre-split V slice
    pub a_log: DeviceBuffer<f32>,      // f32 (special: log space)
    pub dt_bias: DeviceBuffer<u16>,    // bf16 [16]
    pub output_norm: DeviceBuffer<f32>, // f32 [128] (QK-norm, (1+w) pattern)
    pub w_out: LinearWeight,           // [1024, 2048]
    pub post_norm: DeviceBuffer<u16>,  // bf16
    pub w_gate: LinearWeight,          // [3584, 1024]
    pub w_up: LinearWeight,            // [3584, 1024]
    pub w_down: LinearWeight,          // [1024, 3584]
}

pub struct AttentionLayerWeights {
    pub input_norm: DeviceBuffer<u16>, // bf16: stays bf16
    pub w_q_gate: LinearWeight,        // [nqh*hd*q_mult, hs]
    pub w_k: LinearWeight,             // [nkh*hd, hs]
    pub w_v: LinearWeight,             // [nkh*hd, hs]
    pub w_o: LinearWeight,             // [hs, nqh*hd]
    pub q_norm: DeviceBuffer<u16>,     // bf16: stays bf16
    pub k_norm: DeviceBuffer<u16>,     // bf16: stays bf16
    pub post_norm: DeviceBuffer<u16>,  // bf16: stays bf16
    pub w_gate: LinearWeight,          // MLP gate
    pub w_up: LinearWeight,            // MLP up
    pub w_down: LinearWeight,          // MLP down
}

pub struct Mamba2LayerWeights {
    pub input_norm: DeviceBuffer<u16>, // bf16 rmsnorm weight [hidden_size]
    pub in_proj: LinearWeight,         // [hidden_size, in_proj_size]
    pub conv1d_weight: DeviceBuffer<u16>, // bf16 [conv_dim, 1, conv_kernel] (depthwise)
    pub conv1d_bias: DeviceBuffer<f32>, // f32 [conv_dim]
    pub dt_bias: DeviceBuffer<f32>,    // f32 [num_heads]
    pub a_log: DeviceBuffer<f32>,      // f32 [num_heads]
    pub d: DeviceBuffer<f32>,          // f32 [num_heads]
    pub norm_weight: DeviceBuffer<f32>, // f32 rmsnorm_gated weight [intermediate]
    pub out_proj: LinearWeight,        // [intermediate, hidden_size]
}

/// Standalone MoE FFN layer (Nemotron-H 'E' layers) — just norm + MoE dispatch
pub struct MoeFfnLayerWeights {
    pub input_norm: DeviceBuffer<u16>, // bf16 rmsnorm weight [hidden_size]
}

pub enum LayerWeights {
    Gdn(GdnLayerWeights),
    Attention(AttentionLayerWeights),
    Mamba2(Mamba2LayerWeights),
    MoeFfn(MoeFfnLayerWeights),
}

/// Dense FFN weights (gate_proj + up_proj + down_proj)
pub struct DenseFfnWeights {
    pub gate_proj: LinearWeight,
    pub up_proj: LinearWeight,
    pub down_proj: LinearWeight,
}

/// MoE FFN weights for one layer
pub struct MoeWeights {
    pub gate: DeviceBuffer<u16>, // [num_experts, hidden_size] — router, MUST stay bf16
    pub expert_gate_up: LinearWeight, // SwiGLU: [ne, 2*eis, hs] fused; relu²: [ne, eis, hs] (up only)
    pub expert_down: LinearWeight,    // [num_experts, hidden_size, expert_is]
    pub shared_expert: Option<DenseFfnWeights>, // always-on shared expert
    pub shared_expert_gate: Option<DeviceBuffer<u16>>, // [1, hidden_size] gate for shared expert
    pub has_gate_proj: bool,          // false for relu² (Nemotron), true for SwiGLU
    pub score_correction_bias: Option<Vec<f32>>, // [num_experts] f32, added to scores before top-k
    pub score_correction_bias_gpu: Option<DeviceBuffer<f32>>, // GPU copy of correction_bias
    pub num_experts: usize,
    pub expert_intermediate_size: usize,
    /// Input dimension of the expert gate/up weight matrices.
    /// For models with moe_latent_size (e.g. Nemotron-H), this is moe_latent_size (<hidden_size).
    pub gate_up_in_dim: usize,
    /// fc1_latent_proj: [moe_latent_size, hidden_size] — projects hidden→latent before expert dispatch.
    /// None for models without moe_latent_size.
    pub fc1_latent_proj: Option<LinearWeight>,
    /// fc2_latent_proj: [hidden_size, moe_latent_size] — projects accumulated latent→hidden after experts.
    /// None for models without moe_latent_size.
    pub fc2_latent_proj: Option<LinearWeight>,
}

/// Per-GPU expert weight buffer for distributed MoE.
pub struct GpuExpertBuffer {
    pub device: DeviceId,
    pub gate_up: DeviceBuffer<u8>, // packed expert weights on this GPU
    pub down: DeviceBuffer<u8>,    // packed down_proj weights on this GPU
    pub local_expert_count: usize, // how many experts on this GPU
    /// Maps global expert_id → local slot index (None if not on this GPU).
    /// Indexed by global expert_id, len = num_experts.
    pub slot_map: Vec<Option<usize>>,
}

/// Distributed expert weight buffers across GPUs.
/// Gate, shared expert, and metadata stay in MoeWeights on GPU 0.
/// This struct holds only the per-GPU expert copies.
pub struct DistributedMoeWeights {
    pub expert_buffers: Vec<GpuExpertBuffer>, // [num_devices]
    pub expert_device: Vec<usize>,            // [num_experts] → device index
    pub has_gate_proj: bool,
    pub num_experts: usize,
    pub expert_intermediate_size: usize,
    /// Input dimension of the expert gate/up weight matrices.
    /// For models with moe_latent_size (e.g. Nemotron-H), this is moe_latent_size (< hidden_size).
    pub gate_up_in_dim: usize,
    pub gate_up_expert_stride: usize,
    pub down_expert_stride: usize,
    pub gate_up_row_stride: usize,
    pub weight_format: WeightFormat,
    // GPU 0 uses original packed buffer (no extra copy)
    pub gpu0_gate_up_base: *const u8,
    pub gpu0_down_base: *const u8,
}

pub struct GdnState {
    pub recurrent: DeviceBuffer<f32>, // [16, 128, 128]
}

pub struct Mamba2State {
    pub ssm: DeviceBuffer<f32>,  // [num_heads, head_dim, state_size]
    pub conv: DeviceBuffer<f32>, // [conv_dim, conv_kernel]
}

pub struct KvCache {
    pub k: DeviceBuffer<f32>, // [num_kv_heads, max_seq_len, head_dim]
    pub v: DeviceBuffer<f32>,
}

pub struct ActivationBuffers {
    pub hidden: DeviceBuffer<f32>, // [hidden_size]
    pub normed: DeviceBuffer<f32>, // [hidden_size]
    // GDN temporaries
    pub qkv: DeviceBuffer<f32>,           // [6144]
    pub q_gdn: DeviceBuffer<f32>,         // [16*128] = [2048]
    pub k_gdn: DeviceBuffer<f32>,         // [16*128] = [2048]
    pub v_gdn: DeviceBuffer<f32>,         // [16*128] = [2048]
    pub a_proj: DeviceBuffer<f32>,        // [16]
    pub b_proj: DeviceBuffer<f32>,        // [16]
    pub z_proj: DeviceBuffer<f32>,        // [2048]
    pub gate_gdn: DeviceBuffer<f32>,      // [16]
    pub recurrent_out: DeviceBuffer<f32>, // [2048]
    pub normed_gated: DeviceBuffer<f32>,  // [2048]
    pub out_proj: DeviceBuffer<f32>,      // [1024]
    // Attention temporaries
    pub q_gate_attn: DeviceBuffer<f32>, // [4096] Q+gate
    pub q_attn: DeviceBuffer<f32>,      // [2048]
    pub gate_attn: DeviceBuffer<f32>,   // [2048]
    pub k_attn: DeviceBuffer<f32>,      // [512]
    pub v_attn: DeviceBuffer<f32>,      // [512]
    pub attn_out: DeviceBuffer<f32>,    // [2048]
    pub gated_out: DeviceBuffer<f32>,   // [2048]
    // FFN temporaries
    pub ffn_gate: DeviceBuffer<f32>, // [3584]
    pub ffn_up: DeviceBuffer<f32>,   // [3584]
    pub ffn_act: DeviceBuffer<f32>,  // [3584]
    pub ffn_down: DeviceBuffer<f32>, // [1024]
    // Shared
    pub residual: DeviceBuffer<f32>, // [1024]
    // Final
    pub logits: DeviceBuffer<f32>, // [vocab_size]
    pub logits_mapped: braidinfer_hip::MappedHostBuffer<f32>, // [vocab_size] for persistent worker path
    // inv_freq and position_ids for mRoPE
    pub inv_freq: DeviceBuffer<f32>, // [rope_dim/2]
    pub position_ids: braidinfer_hip::MappedHostBuffer<i32>, // [3] — host-mapped for persistent worker path
    // conv states per GDN layer (allocated separately)
    // Pre-allocated GDN conv state temp buffers (reused each gdn_forward call)
    // Multi-GPU barrier staging: written by megakernel before OP_BARRIER, read by CPU dispatch.
    // MappedHostBuffer = GPU-writable (GART/uncached) + CPU-readable without hipMemcpy.
    pub normed_stage: MappedHostBuffer<f32>, // [hidden_size] — copy of normed for CPU broadcast
    // bd braidinfer-sm16 / udi #2740: producer/consumer sequence number for
    // normed_stage. op_rmsnorm_wx writes (position+1) on completion; workers'
    // op_d2d_copy spin-waits on this value before peer-reading normed_stage.
    // Forces PCIe-posted writes to drain through host-mapped UC GART page
    // before consumer reads. Initial value = 0 (position+1 starts at 1).
    pub normed_seq: MappedHostBuffer<u32>, // [1] — sequence number
    pub ffn_down_stage: MappedHostBuffer<f32>, // [hidden_size] — CPU writes gathered expert output
    // MoE scratch buffers (pre-allocated to avoid hipMalloc in hot path)
    pub moe_scores: DeviceBuffer<f32>,         // [max_num_experts]
    pub moe_expert_ids: MappedHostBuffer<i32>, // [max_k] — GPU writes, CPU reads via host_ptr
    pub moe_expert_weights: MappedHostBuffer<f32>, // [max_k] — GPU writes, CPU reads via host_ptr
    pub moe_expert_gate: DeviceBuffer<f32>,    // [max_expert_intermediate_size]
    pub moe_expert_up: DeviceBuffer<f32>,      // [max_expert_intermediate_size]
    pub moe_expert_act: DeviceBuffer<f32>,     // [max_expert_intermediate_size]
    pub moe_expert_out: DeviceBuffer<f32>,     // [hidden_size]
    pub moe_latent: DeviceBuffer<f32>,         // [moe_latent_size or hidden_size] — fc1/fc2 staging
    // Prefill MoE batched scratch (CHUNK_TOKENS × max dims) — avoids hipMalloc per layer call
    pub prefill_moe_normed: DeviceBuffer<f32>,        // [CHUNK_TOKENS × max_latent_size]
    pub prefill_moe_expert_input: DeviceBuffer<f32>,  // [CHUNK_TOKENS × max_latent_size]
    pub prefill_moe_gate_out: DeviceBuffer<f32>,      // [CHUNK_TOKENS × max_eis]
    pub prefill_moe_up_out: DeviceBuffer<f32>,        // [CHUNK_TOKENS × max_eis]
    pub prefill_moe_act_out: DeviceBuffer<f32>,       // [CHUNK_TOKENS × max_eis]
    pub prefill_moe_down_out: DeviceBuffer<f32>,      // [CHUNK_TOKENS × max_hs]
    pub prefill_moe_ffn_out: DeviceBuffer<f32>,       // [CHUNK_TOKENS × max_hs]
    pub prefill_moe_residual: DeviceBuffer<f32>,      // [CHUNK_TOKENS × max_hs]
    pub prefill_moe_ids_dev: DeviceBuffer<i32>,       // [CHUNK_TOKENS × max_k]
    pub prefill_moe_weights_dev: DeviceBuffer<f32>,   // [CHUNK_TOKENS × max_k]
    pub prefill_moe_token_indices: DeviceBuffer<i32>, // [CHUNK_TOKENS]
    pub prefill_moe_token_weights: DeviceBuffer<f32>, // [CHUNK_TOKENS]
    // Mamba2 scratch buffers
    pub mamba2_in_proj: DeviceBuffer<f32>, // [in_proj_size] (gate + xBC + dt)
    pub mamba2_conv_out: DeviceBuffer<f32>, // [conv_dim] (after conv1d + activation)
    pub mamba2_ssm_out: DeviceBuffer<f32>, // [intermediate] (SSM output y)
    // GPU-resident argmax
    pub argmax_result: DeviceBuffer<i32>, // [1] — single token ID
}

// ---- Error type ----

#[derive(Debug)]
pub enum ModelError {
    Hip(braidinfer_hip::HipError),
    SafeTensors(braidinfer_core::safetensors::SafeTensorsError),
    MissingWeight(String),
    InvalidConfig(String),
    Io(std::io::Error),
}

impl From<braidinfer_hip::HipError> for ModelError {
    fn from(e: braidinfer_hip::HipError) -> Self {
        ModelError::Hip(e)
    }
}

impl From<braidinfer_core::safetensors::SafeTensorsError> for ModelError {
    fn from(e: braidinfer_core::safetensors::SafeTensorsError) -> Self {
        ModelError::SafeTensors(e)
    }
}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        ModelError::Io(e)
    }
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Hip(e) => write!(f, "HIP error: {e:?}"),
            ModelError::SafeTensors(e) => write!(f, "SafeTensors error: {e}"),
            ModelError::MissingWeight(s) => write!(f, "Missing weight: {s}"),
            ModelError::InvalidConfig(s) => write!(f, "Invalid model configuration: {s}"),
            ModelError::Io(e) => write!(f, "IO error: {e}"),
        }
    }
}

impl std::error::Error for ModelError {}

// ---- Helper: load a tensor by name, convert to f32, upload to GPU ----

/// Try multiple name patterns, return first that exists in safetensors.
pub fn find_weight_name(
    st: &SafeTensorSet,
    bqnt: Option<&MmapBqnt>,
    candidates: &[String],
) -> Result<String, ModelError> {
    for name in candidates {
        // bd 4ayf A3.2.3b: a self-contained bqnt has the names; st is the legacy fallback.
        if bqnt.map(|b| b.entry(name).is_some()).unwrap_or(false) || st.tensor_data(name).is_ok() {
            return Ok(name.clone());
        }
    }
    Err(ModelError::MissingWeight(format!(
        "none of {:?} found",
        candidates
    )))
}

/// Load a bf16 tensor, returning a typed DeviceBuffer<u16>.
/// The underlying data is copied directly from mmap — zero conversion.
pub fn load_weight_bf16(
    st: &SafeTensorSet,
    name: &str,
    device: DeviceId,
    expected_len: usize,
) -> Result<DeviceBuffer<u16>, ModelError> {
    let raw = st
        .tensor_data(name)
        .map_err(|_| ModelError::MissingWeight(name.to_string()))?;
    assert_eq!(
        raw.len(),
        expected_len * 2,
        "weight {name}: expected {} bytes, got {}",
        expected_len * 2,
        raw.len()
    );
    let mut buf = DeviceBuffer::<u16>::alloc(device, expected_len)?;
    let data: &[u16] =
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u16, expected_len) };
    buf.copy_from_host(data)?;
    Ok(buf)
}

// --- Weight Quantization (types re-exported from quant module) ---

// LinearWeight impl is in crate::quant

// NF4 constants and quantization functions are in crate::quant

/// Load a weight tensor, optionally quantizing at load time.
pub fn load_weight_quantized(
    st: &SafeTensorSet,
    name: &str,
    device: DeviceId,
    out_dim: usize,
    in_dim: usize,
    format: WeightFormat,
) -> Result<PackedWeights, ModelError> {
    let expected_len = out_dim * in_dim;
    let raw = st
        .tensor_data(name)
        .map_err(|_| ModelError::MissingWeight(name.to_string()))?;
    assert_eq!(
        raw.len(),
        expected_len * 2,
        "weight {name}: expected {} bytes, got {}",
        expected_len * 2,
        raw.len()
    );
    let bf16_data: &[u16] =
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u16, expected_len) };

    match format {
        WeightFormat::Bf16 => {
            let mut buf = DeviceBuffer::<u8>::alloc(device, expected_len * 2)?;
            buf.copy_from_host(raw)?;
            Ok(PackedWeights {
                data: buf,
                format,
                out_dim,
                in_dim,
            })
        }
        WeightFormat::Rnf4G128 => {
            let packed = quantize_rnf4_g128(bf16_data, out_dim, in_dim);
            let mut buf = DeviceBuffer::<u8>::alloc(device, packed.len())?;
            buf.copy_from_host(&packed)?;
            Ok(PackedWeights {
                data: buf,
                format,
                out_dim,
                in_dim,
            })
        }
        WeightFormat::PcG32Q4 => {
            let packed = quantize_pc_g32_q4(bf16_data, out_dim, in_dim);
            let mut buf = DeviceBuffer::<u8>::alloc(device, packed.len())?;
            buf.copy_from_host(&packed)?;
            Ok(PackedWeights {
                data: buf,
                format,
                out_dim,
                in_dim,
            })
        }
    }
}

/// Determine weight format for a given weight name under a quantization mode.
pub fn weight_format_for(name: &str, mode: WeightQuantMode) -> WeightFormat {
    match mode {
        WeightQuantMode::Bf16 => WeightFormat::Bf16,
        WeightQuantMode::Rnf4 => WeightFormat::Rnf4G128,
        WeightQuantMode::Mixed => {
            // MLP weights at PcG32Q4, everything else at Rnf4G128
            if name.contains("mlp.")
                || name.contains("gate_proj")
                || name.contains("up_proj")
                || name.contains("down_proj")
            {
                // But NOT the MoE router gate
                if name.contains("mlp.gate.weight")
                    || name.contains("block_sparse_moe.gate")
                    || name.contains("mlp.router")
                {
                    WeightFormat::Bf16 // router stays bf16
                } else {
                    WeightFormat::PcG32Q4
                }
            } else {
                WeightFormat::Rnf4G128
            }
        }
    }
}

/// Load a linear weight, quantizing at load time based on format.
pub fn load_linear_weight(
    st: &SafeTensorSet,
    name: &str,
    device: DeviceId,
    out_dim: usize,
    in_dim: usize,
    mode: WeightQuantMode,
) -> Result<LinearWeight, ModelError> {
    let format = weight_format_for(name, mode);
    match format {
        WeightFormat::Bf16 => {
            let buf = load_weight_bf16(st, name, device, out_dim * in_dim)?;
            Ok(LinearWeight::Bf16(buf))
        }
        _ => {
            let pw = load_weight_quantized(st, name, device, out_dim, in_dim, format)?;
            Ok(LinearWeight::Packed(pw))
        }
    }
}

/// Quantize a host bf16 buffer to a LinearWeight on GPU.
pub fn host_bf16_to_linear_weight(
    host_buf: &[u16],
    out_dim: usize,
    in_dim: usize,
    fmt: WeightFormat,
    device: DeviceId,
) -> Result<LinearWeight, ModelError> {
    match fmt {
        WeightFormat::Bf16 => {
            let mut buf = DeviceBuffer::<u16>::alloc(device, host_buf.len())?;
            buf.copy_from_host(host_buf)?;
            Ok(LinearWeight::Bf16(buf))
        }
        fmt => {
            let packed = if fmt == WeightFormat::Rnf4G128 {
                quantize_rnf4_g128(host_buf, out_dim, in_dim)
            } else {
                quantize_pc_g32_q4(host_buf, out_dim, in_dim)
            };
            let mut buf = DeviceBuffer::<u8>::alloc(device, packed.len())?;
            buf.copy_from_host(&packed)?;
            Ok(LinearWeight::Packed(PackedWeights {
                data: buf,
                format: fmt,
                out_dim,
                in_dim,
            }))
        }
    }
}

/// Quantize host bf16 buffer, optionally write to bqnt writer, upload to GPU.
/// Used by MoE loading paths that fuse per-expert tensors into a single packed buffer.
pub(crate) fn host_bf16_quantize_upload_cache(
    host_buf: &[u16],
    out_dim: usize,
    in_dim: usize,
    fmt: WeightFormat,
    device: DeviceId,
    cache_name: &str,
    writer: Option<&mut crate::bqnt::BqntWriter>,
) -> Result<LinearWeight, ModelError> {
    let packed = match fmt {
        WeightFormat::Rnf4G128 => quantize_rnf4_g128(host_buf, out_dim, in_dim),
        WeightFormat::PcG32Q4 => quantize_pc_g32_q4(host_buf, out_dim, in_dim),
        WeightFormat::Bf16 => host_buf.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    if let Some(w) = writer {
        if let Err(e) = w.write_tensor(cache_name, crate::bqnt::StorageDtype::from_weight_format(fmt), out_dim as u32, in_dim as u32, 2, &packed) {
            eprintln!("bqnt: cache write failed for {cache_name}: {e}");
        }
    }
    let mut buf = DeviceBuffer::<u8>::alloc(device, packed.len())?;
    buf.copy_from_host(&packed)?;
    Ok(LinearWeight::Packed(PackedWeights {
        data: buf,
        format: fmt,
        out_dim,
        in_dim,
    }))
}

// --- BQNT (pre-quantized) loading ---

use crate::bqnt::{MmapBqnt, code_to_format};
pub use crate::moe_weights::*;

/// Like load_linear_weight but also writes packed bytes to a BqntWriter for caching.
/// Only called when quantizing from safetensors for the first time (no bqnt found).
pub fn load_linear_weight_cached(
    st: &SafeTensorSet,
    name: &str,
    device: DeviceId,
    out_dim: usize,
    in_dim: usize,
    mode: WeightQuantMode,
    writer: &mut crate::bqnt::BqntWriter,
) -> Result<LinearWeight, ModelError> {
    let format = weight_format_for(name, mode);
    match format {
        WeightFormat::Bf16 => {
            // Bf16 weights not quantized, don't cache in bqnt
            let buf = load_weight_bf16(st, name, device, out_dim * in_dim)?;
            Ok(LinearWeight::Bf16(buf))
        }
        _ => {
            let expected_len = out_dim * in_dim;
            let raw = st
                .tensor_data(name)
                .map_err(|_| ModelError::MissingWeight(name.to_string()))?;
            let bf16_data: &[u16] =
                unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u16, expected_len) };
            let packed = match format {
                WeightFormat::Rnf4G128 => quantize_rnf4_g128(bf16_data, out_dim, in_dim),
                WeightFormat::PcG32Q4 => quantize_pc_g32_q4(bf16_data, out_dim, in_dim),
                WeightFormat::Bf16 => unreachable!(),
            };
            writer
                .write_tensor(name, crate::bqnt::StorageDtype::from_weight_format(format), out_dim as u32, in_dim as u32, 2, &packed)
                .map_err(|e| ModelError::MissingWeight(format!("bqnt write {name}: {e}")))?;
            let mut buf = DeviceBuffer::<u8>::alloc(device, packed.len())?;
            buf.copy_from_host(&packed)?;
            Ok(LinearWeight::Packed(PackedWeights {
                data: buf,
                format,
                out_dim,
                in_dim,
            }))
        }
    }
}

/// Load a linear weight directly from a pre-quantized .bqnt file.
/// Zero quantization cost — packed bytes go straight from mmap to GPU.
pub fn load_linear_weight_bqnt(
    bqnt: &MmapBqnt,
    name: &str,
    device: DeviceId,
    // bd 4ayf B1: (arena_base_ptr, data_start). When present, a Packed weight becomes a
    // non-owning VIEW into the bulk-load arena at arena_base + (data_offset - data_start) —
    // no per-tensor copy. None = per-tensor alloc+copy (multi-GPU/fused/quantize-at-load).
    arena: Option<(*const u8, u64)>,
) -> Result<LinearWeight, ModelError> {
    let entry = bqnt
        .entry(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    let format = code_to_format(entry.format)
        .and_then(|s| s.to_weight_format())
        .ok_or_else(|| {
            ModelError::MissingWeight(format!(
                "{name}: not a linear-weight format code {} (bd 4ayf: F32/non-linear tensors \
                 load via load_weight_f32_bqnt, not the linear path)",
                entry.format
            ))
        })?;
    let data = bqnt
        .tensor_data(name)
        .ok_or_else(|| ModelError::MissingWeight(format!("{name}: data out of bounds")))?;

    match format {
        WeightFormat::Bf16 => {
            let n_elements = entry.out_features as usize * entry.in_features as usize;
            let bf16_data: &[u16] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, n_elements) };
            let mut buf = DeviceBuffer::<u16>::alloc(device, n_elements)?;
            buf.copy_from_host(bf16_data)?;
            Ok(LinearWeight::Bf16(buf))
        }
        _ => {
            // bd 4ayf B1: non-owning view into the bulk-load arena when present (no copy),
            // else per-tensor alloc + copy_from_host.
            let data_buf = if let Some((arena_ptr, data_start)) = arena {
                let off = (entry.data_offset - data_start) as usize;
                unsafe {
                    DeviceBuffer::<u8>::view(device, arena_ptr.add(off), entry.data_bytes as usize)
                }
            } else {
                let mut buf = DeviceBuffer::<u8>::alloc(device, data.len())?;
                buf.copy_from_host(data)?;
                buf
            };
            Ok(LinearWeight::Packed(PackedWeights {
                data: data_buf,
                format,
                out_dim: entry.out_features as usize,
                in_dim: entry.in_features as usize,
            }))
        }
    }
}

/// Load a bf16 weight (non-quantized, e.g. layernorm) from .bqnt file.
/// Falls back to safetensors if not found in bqnt (norms, biases may not be quantized).
pub fn load_weight_bf16_bqnt(
    bqnt: &MmapBqnt,
    name: &str,
    device: DeviceId,
    expected_len: usize,
    // bd 4ayf.12: (arena_base_ptr, data_start). When present, return a non-owning VIEW into the
    // bulk-load arena (no copy, no duplication) — bf16 bytes are stored as-is, so a u16 view works.
    arena: Option<(*const u8, u64)>,
) -> Result<DeviceBuffer<u16>, ModelError> {
    // bd 4ayf (multi-GPU regression fix): only read if the tensor is actually Bf16-stored at
    // the expected size. A v1 bqnt may carry this tensor QUANTIZED — e.g. embed_tokens.weight
    // as Q4, a dead pre-A2 copy the loader never used. The caller (load_bf16) wants bf16, so
    // return Err to fall back to safetensors rather than misreading Q4 bytes as bf16 (was an
    // assert -> panic on the v1 qwen35_35b_a3b -g2 load).
    let entry = match bqnt.entry(name) {
        Some(e) => e,
        None => return Err(ModelError::MissingWeight(name.to_string())),
    };
    if crate::bqnt::code_to_format(entry.format) != Some(crate::bqnt::StorageDtype::Bf16) {
        return Err(ModelError::MissingWeight(name.to_string()));
    }
    let data = match bqnt.tensor_data(name) {
        Some(d) if d.len() == expected_len * 2 => d,
        _ => return Err(ModelError::MissingWeight(name.to_string())),
    };
    // bd 4ayf.12: arena view (no copy) when present — bf16 stored as-is, u16 view at the offset.
    if let Some((arena_base, data_start)) = arena {
        let off = (entry.data_offset - data_start) as usize;
        let ptr = unsafe { arena_base.add(off) } as *const u16;
        return Ok(unsafe { DeviceBuffer::<u16>::view(device, ptr, expected_len) });
    }
    let bf16_data: &[u16] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, expected_len) };
    let mut buf = DeviceBuffer::<u16>::alloc(device, expected_len)?;
    buf.copy_from_host(bf16_data)?;
    Ok(buf)
}

/// bd 4ayf A3: load an F32-loaded tensor (GDN/Mamba2 recurrent state, norms read as f32)
/// from the bqnt. Mirrors `load_weight_f32`'s dtype-flexibility: a tensor stored `F32` is
/// read direct; a tensor stored `Bf16` is widened bf16->f32. Used for every `load_weight_f32`
/// call site when the bqnt is present (safetensors is the legacy fallback).
pub fn load_weight_f32_bqnt(
    bqnt: &MmapBqnt,
    name: &str,
    device: DeviceId,
    expected_len: usize,
    // bd 4ayf.12: (arena_base_ptr, data_start). An F32-stored tensor can be a non-owning VIEW
    // into the arena (no copy). A Bf16-stored tensor is WIDENED bf16->f32 (a conversion, not a
    // reinterpret), so it cannot be a view — it always copies.
    arena: Option<(*const u8, u64)>,
) -> Result<DeviceBuffer<f32>, ModelError> {
    let entry = bqnt
        .entry(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    let sdt = crate::bqnt::code_to_format(entry.format).ok_or_else(|| {
        ModelError::MissingWeight(format!("{name}: unknown bqnt storage code {}", entry.format))
    })?;
    let data = bqnt
        .tensor_data(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    // bd 4ayf.12: F32-stored + arena -> direct f32 view (no copy).
    if let (crate::bqnt::StorageDtype::F32, Some((arena_base, data_start))) = (sdt, arena) {
        if data.len() == expected_len * 4 {
            let off = (entry.data_offset - data_start) as usize;
            let ptr = unsafe { arena_base.add(off) } as *const f32;
            return Ok(unsafe { DeviceBuffer::<f32>::view(device, ptr, expected_len) });
        }
    }
    let f32_data: Vec<f32> = match sdt {
        crate::bqnt::StorageDtype::F32 => {
            // bd 4ayf: size mismatch (e.g. a v1 bqnt) -> Err -> st fallback (was assert/panic).
            if data.len() != expected_len * 4 {
                return Err(ModelError::MissingWeight(name.to_string()));
            }
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        crate::bqnt::StorageDtype::Bf16 => {
            // bd 4ayf: size mismatch (e.g. a v1 bqnt) -> Err -> st fallback (was assert/panic).
            if data.len() != expected_len * 2 {
                return Err(ModelError::MissingWeight(name.to_string()));
            }
            data.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes(b.try_into().unwrap());
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }
        other => {
            return Err(ModelError::MissingWeight(format!(
                "{name}: bqnt storage {other:?} is not f32-loadable"
            )));
        }
    };
    let mut buf = DeviceBuffer::<f32>::alloc(device, expected_len)?;
    buf.copy_from_host(&f32_data)?;
    Ok(buf)
}

pub fn load_weight_f32(
    st: &SafeTensorSet,
    name: &str,
    device: DeviceId,
    expected_len: usize,
) -> Result<DeviceBuffer<f32>, ModelError> {
    let info = st
        .tensor_info(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    let raw = st
        .tensor_data(name)
        .map_err(|_| ModelError::MissingWeight(name.to_string()))?;
    let data: Vec<f32> = match info.dtype {
        Dtype::F32 => {
            assert_eq!(raw.len(), expected_len * 4, "weight {name}: size mismatch");
            raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        Dtype::BF16 => {
            assert_eq!(raw.len(), expected_len * 2, "weight {name}: size mismatch");
            raw.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes(b.try_into().unwrap());
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }
        other => panic!("load_weight_f32: unsupported dtype {other:?} for {name}"),
    };
    let mut buf = DeviceBuffer::<f32>::alloc(device, expected_len)?;
    buf.copy_from_host(&data)?;
    Ok(buf)
}

// ---- Precompute inv_freq ----

pub fn compute_inv_freq(rope_dim: usize, rope_theta: f32) -> Vec<f32> {
    let num_pairs = rope_dim / 2;
    (0..num_pairs)
        .map(|i| {
            let exp = 2.0 * i as f32 / rope_dim as f32;
            1.0 / rope_theta.powf(exp)
        })
        .collect()
}

#[cfg(test)]
mod bqnt_reader_compat_tests {
    use super::*;
    use crate::bqnt::{BqntWriter, MmapBqnt, StorageDtype, packed_size};

    // bd 4ayf v1-compat regression (found by the multi-GPU test, fixed in 067e11a): a tensor
    // stored QUANTIZED (Q4) in a (v1) bqnt — e.g. embed_tokens, a dead pre-A2 copy — must make
    // load_weight_bf16_bqnt return Err so the loader falls back to safetensors, NOT panic by
    // misreading the Q4 bytes as bf16 (the original assert at weights.rs:624). CPU-only: the
    // Err path returns before any DeviceBuffer allocation, so no GPU is required to run this.
    #[test]
    fn bf16_reader_rejects_quantized_tensor() {
        let path = std::env::temp_dir().join("braidinfer_test_bf16_reject_q4.bqnt");
        {
            let mut w = BqntWriter::create(&path, 1).unwrap();
            let packed = vec![0u8; packed_size(StorageDtype::PcG32Q4, 4, 32)];
            w.write_tensor("embed", StorageDtype::PcG32Q4, 4, 32, 2, &packed)
                .unwrap();
            w.finish("{}").unwrap();
        }
        let bqnt = MmapBqnt::open(&path).unwrap();
        let r = load_weight_bf16_bqnt(&bqnt, "embed", DeviceId(0), 4 * 32, None);
        assert!(
            r.is_err(),
            "a Q4-stored tensor must Err from the bf16 reader (-> safetensors fallback), not panic"
        );
        let _ = std::fs::remove_file(&path);
    }

    // bd 4ayf: a bf16 tensor present at the WRONG size must Err (-> st fallback), not OOB/panic.
    #[test]
    fn bf16_reader_rejects_size_mismatch() {
        let path = std::env::temp_dir().join("braidinfer_test_bf16_size.bqnt");
        {
            let mut w = BqntWriter::create(&path, 1).unwrap();
            w.write_tensor("norm", StorageDtype::Bf16, 8, 1, 1, &vec![0u8; 16])
                .unwrap();
            w.finish("{}").unwrap();
        }
        let bqnt = MmapBqnt::open(&path).unwrap();
        let r = load_weight_bf16_bqnt(&bqnt, "norm", DeviceId(0), 999, None);
        assert!(r.is_err(), "a bf16 size-mismatch must Err, not OOB/panic");
        let _ = std::fs::remove_file(&path);
    }

    // bd 4ayf: the f32 reader rejects a quantized tensor (-> st fallback), not panic.
    #[test]
    fn f32_reader_rejects_quantized_tensor() {
        let path = std::env::temp_dir().join("braidinfer_test_f32_reject_q4.bqnt");
        {
            let mut w = BqntWriter::create(&path, 1).unwrap();
            let packed = vec![0u8; packed_size(StorageDtype::PcG32Q4, 4, 32)];
            w.write_tensor("a_log", StorageDtype::PcG32Q4, 4, 32, 2, &packed)
                .unwrap();
            w.finish("{}").unwrap();
        }
        let bqnt = MmapBqnt::open(&path).unwrap();
        let r = load_weight_f32_bqnt(&bqnt, "a_log", DeviceId(0), 4 * 32, None);
        assert!(r.is_err(), "a Q4-stored tensor must Err from the f32 reader, not panic");
        let _ = std::fs::remove_file(&path);
    }
}
