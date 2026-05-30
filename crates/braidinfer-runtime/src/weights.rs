//! Weight types, loading, and activation buffer allocation.
//! Extracted from model.rs for maintainability.

use braidinfer_core::safetensors::SafeTensorSet;
use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::{DeviceBuffer, MappedHostBuffer};
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{HipResult, ffi};
use safetensors::Dtype;

use crate::config::*;
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

fn checked_hip_memcpy_h2d(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    size: usize,
) -> Result<(), ModelError> {
    braidinfer_hip::error::check(unsafe {
        ffi::hipMemcpy(dst, src, size, ffi::hipMemcpyHostToDevice)
    })?;
    Ok(())
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
fn host_bf16_quantize_upload_cache(
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
            let mut buf = DeviceBuffer::<u8>::alloc(device, data.len())?;
            buf.copy_from_host(data)?;
            Ok(LinearWeight::Packed(PackedWeights {
                data: buf,
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
) -> Result<DeviceBuffer<u16>, ModelError> {
    if let Some(data) = bqnt.tensor_data(name) {
        assert_eq!(
            data.len(),
            expected_len * 2,
            "bqnt weight {name}: expected {} bytes, got {}",
            expected_len * 2,
            data.len()
        );
        let bf16_data: &[u16] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, expected_len) };
        let mut buf = DeviceBuffer::<u16>::alloc(device, expected_len)?;
        buf.copy_from_host(bf16_data)?;
        Ok(buf)
    } else {
        Err(ModelError::MissingWeight(name.to_string()))
    }
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
    let f32_data: Vec<f32> = match sdt {
        crate::bqnt::StorageDtype::F32 => {
            assert_eq!(
                data.len(),
                expected_len * 4,
                "bqnt f32 weight {name}: expected {} bytes, got {}",
                expected_len * 4,
                data.len()
            );
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        crate::bqnt::StorageDtype::Bf16 => {
            assert_eq!(
                data.len(),
                expected_len * 2,
                "bqnt bf16->f32 weight {name}: expected {} bytes, got {}",
                expected_len * 2,
                data.len()
            );
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

/// Load all MoE weights for a single layer (gate, experts, shared expert).
pub fn load_moe_weights(
    st: &SafeTensorSet,
    prefix: &str,
    config: &ModelConfig,
    ffn_type: &FfnType,
    device: DeviceId,
    wq: WeightQuantMode,
    bqnt: Option<&MmapBqnt>,
) -> Result<MoeWeights, ModelError> {
    load_moe_weights_inner(st, prefix, config, ffn_type, device, wq, bqnt, false, None)
}

/// Like load_moe_weights_lite but writes packed bytes to writer for bqnt caching.
/// skip_experts=true so expert weights are not loaded (multi-GPU loads them directly).
pub fn load_moe_weights_lite_cached(
    st: &SafeTensorSet,
    prefix: &str,
    config: &ModelConfig,
    ffn_type: &FfnType,
    device: DeviceId,
    wq: WeightQuantMode,
    bqnt: Option<&MmapBqnt>,
    writer: &mut crate::bqnt::BqntWriter,
) -> Result<MoeWeights, ModelError> {
    load_moe_weights_inner(
        st,
        prefix,
        config,
        ffn_type,
        device,
        wq,
        bqnt,
        true,
        Some(writer),
    )
}

/// Like load_moe_weights but writes packed bytes to writer for bqnt caching.
pub fn load_moe_weights_cached(
    st: &SafeTensorSet,
    prefix: &str,
    config: &ModelConfig,
    ffn_type: &FfnType,
    device: DeviceId,
    wq: WeightQuantMode,
    bqnt: Option<&MmapBqnt>,
    writer: &mut crate::bqnt::BqntWriter,
) -> Result<MoeWeights, ModelError> {
    load_moe_weights_inner(
        st,
        prefix,
        config,
        ffn_type,
        device,
        wq,
        bqnt,
        false,
        Some(writer),
    )
}

/// Load MoE weights, optionally skipping expert weights (for multi-GPU direct loading).
/// When skip_experts=true, expert_gate_up and expert_down are empty placeholders.
pub fn load_moe_weights_lite(
    st: &SafeTensorSet,
    prefix: &str,
    config: &ModelConfig,
    ffn_type: &FfnType,
    device: DeviceId,
    wq: WeightQuantMode,
    bqnt: Option<&MmapBqnt>,
) -> Result<MoeWeights, ModelError> {
    load_moe_weights_inner(st, prefix, config, ffn_type, device, wq, bqnt, true, None)
}

/// Load an optional latent projection weight (fc1_latent_proj or fc2_latent_proj).
/// Tries BQNT first, then safetensors. Returns None if not found in either source.
fn load_latent_proj(
    prefix: &str,
    proj_name: &str,
    device: DeviceId,
    bqnt: Option<&MmapBqnt>,
    st: &SafeTensorSet,
    wq: WeightQuantMode,
) -> Result<Option<LinearWeight>, ModelError> {
    let tensor_name = format!("{prefix}{proj_name}.weight");

    // Try BQNT first
    if let Some(b) = bqnt {
        if let Some(entry) = b.entry(&tensor_name) {
            let fmt = crate::bqnt::code_to_format(entry.format)
                .and_then(|s| s.to_weight_format())
                .ok_or_else(|| {
                ModelError::InvalidConfig(format!("{tensor_name}: not a linear bqnt format"))
            })?;
            let out_dim = entry.out_features as usize;
            let in_dim = entry.in_features as usize;
            if let Some(data) = b.tensor_data(&tensor_name) {
                let mut buf = DeviceBuffer::<u8>::alloc(device, data.len())?;
                buf.copy_from_host(data)?;
                return Ok(Some(LinearWeight::Packed(PackedWeights { data: buf, format: fmt, out_dim, in_dim })));
            }
        }
    }

    // Try safetensors
    if let Ok(raw) = st.tensor_data(&tensor_name) {
        let shape = st.tensor_info(&tensor_name).map(|i| i.shape.clone()).unwrap_or_default();
        let out_dim = shape.first().copied().unwrap_or(0);
        let in_dim = shape.get(1).copied().unwrap_or(0);
        let slice_u16 = unsafe {
            std::slice::from_raw_parts(raw.as_ptr() as *const u16, out_dim * in_dim)
        };
        let lw = host_bf16_to_linear_weight(slice_u16, out_dim, in_dim, weight_format_for(&tensor_name, wq), device)?;
        return Ok(Some(lw));
    }

    Ok(None)
}

fn load_moe_weights_inner(
    st: &SafeTensorSet,
    prefix: &str,
    config: &ModelConfig,
    ffn_type: &FfnType,
    device: DeviceId,
    wq: WeightQuantMode,
    bqnt: Option<&MmapBqnt>,
    skip_experts: bool,
    writer: Option<&mut crate::bqnt::BqntWriter>,
) -> Result<MoeWeights, ModelError> {
    let FfnType::MoE {
        num_experts,
        expert_intermediate_size,
        num_shared,
        shared_intermediate_size,
        ..
    } = ffn_type
    else {
        unreachable!("load_moe_weights called on non-MoE layer")
    };
    let ne = *num_experts;
    let eis = *expert_intermediate_size;
    let hs = config.hidden_size;

    // Helper: try bqnt first, then safetensors (with optional bqnt cache write).
    // RefCell allows the closure and outer code to share mutable writer access.
    let writer_cell = std::cell::RefCell::new(writer);
    let load_lw = |name: &str, out_dim: usize, in_dim: usize| -> Result<LinearWeight, ModelError> {
        if let Some(b) = bqnt {
            if let Ok(lw) = load_linear_weight_bqnt(b, name, device) {
                return Ok(lw);
            }
        }
        if writer_cell.borrow().is_some() {
            let mut guard = writer_cell.borrow_mut();
            load_linear_weight_cached(
                st,
                name,
                device,
                out_dim,
                in_dim,
                wq,
                guard.as_mut().unwrap(),
            )
        } else {
            load_linear_weight(st, name, device, out_dim, in_dim, wq)
        }
    };

    // Router gate: try mlp.gate, gate (Nemotron), block_sparse_moe.gate, mlp.router (always bf16)
    let gate_name = [
        format!("{prefix}mlp.gate.weight"),
        format!("{prefix}gate.weight"),
        format!("{prefix}block_sparse_moe.gate.weight"),
        format!("{prefix}mlp.router.weight"),
    ]
    .into_iter()
    .find(|n| st.tensor_data(n).is_ok())
    .ok_or_else(|| ModelError::MissingWeight(format!("{prefix}mlp.gate.weight (or variants)")))?;
    let gate = load_weight_bf16(st, &gate_name, device, ne * hs)?;

    // Detect whether experts have gate_proj (SwiGLU) or just up_proj (relu²)
    // Check per-expert gate_proj OR fused gate_up_proj (which implies SwiGLU)
    let fused_name_check = format!("{prefix}mlp.experts.gate_up_proj");
    let has_fused_gate_up = st.tensor_data(&fused_name_check).is_ok()
        || bqnt.map_or(false, |b| b.entry(&fused_name_check).is_some());
    let has_gate_proj = has_fused_gate_up
        || [
            format!("{prefix}mlp.experts.0.gate_proj.weight"),
            format!("{prefix}experts.0.gate_proj.weight"),
            format!("{prefix}block_sparse_moe.experts.0.w1.weight"),
        ]
        .iter()
        .any(|n| st.tensor_data(n).is_ok());

    let expert_fmt = if has_gate_proj {
        weight_format_for(&format!("{prefix}mlp.experts.0.gate_proj.weight"), wq)
    } else {
        // Try Nemotron naming: experts.0.up_proj.weight (under mixer. prefix)
        weight_format_for(&format!("{prefix}experts.0.up_proj.weight"), wq)
    };

    // Expert weights: skip when loading lite (multi-GPU loads directly to per-GPU buffers)
    let (expert_gate_up, expert_down) = if skip_experts {
        let empty_gu = LinearWeight::Packed(PackedWeights {
            data: DeviceBuffer::<u8>::alloc(device, 0)?,
            format: WeightFormat::PcG32Q4,
            out_dim: 0,
            in_dim: 0,
        });
        let empty_d = LinearWeight::Packed(PackedWeights {
            data: DeviceBuffer::<u8>::alloc(device, 0)?,
            format: WeightFormat::PcG32Q4,
            out_dim: 0,
            in_dim: 0,
        });
        (empty_gu, empty_d)
    } else {
        // Expert gate+up: try bqnt fused, then safetensors fused, else per-expert fuse on host
        let fused_name = format!("{prefix}mlp.experts.gate_up_proj");
        let bqnt_fused = bqnt.and_then(|b| load_linear_weight_bqnt(b, &fused_name, device).ok());
        let expert_gate_up = if let Some(lw) = bqnt_fused {
            lw
        } else if st.tensor_data(&fused_name).is_ok() {
            if writer_cell.borrow().is_some() {
                let mut guard = writer_cell.borrow_mut();
                load_linear_weight_cached(
                    st,
                    &fused_name,
                    device,
                    ne * 2 * eis,
                    hs,
                    wq,
                    guard.as_mut().unwrap(),
                )?
            } else {
                load_linear_weight(st, &fused_name, device, ne * 2 * eis, hs, wq)?
            }
        } else if has_gate_proj {
            // SwiGLU: fuse gate_proj + up_proj per expert
            let expert_elems = 2 * eis * hs;
            let mut host_buf = vec![0u16; ne * expert_elems];
            for e in 0..ne {
                let (gp, up) = [
                    (
                        format!("{prefix}mlp.experts.{e}.gate_proj.weight"),
                        format!("{prefix}mlp.experts.{e}.up_proj.weight"),
                    ),
                    (
                        format!("{prefix}block_sparse_moe.experts.{e}.w1.weight"),
                        format!("{prefix}block_sparse_moe.experts.{e}.w3.weight"),
                    ),
                ]
                .into_iter()
                .find(|(g, _)| st.tensor_data(g).is_ok())
                .ok_or_else(|| {
                    ModelError::MissingWeight(format!(
                        "{prefix}experts.{e}.gate_proj.weight (or variants)"
                    ))
                })?;
                let g_raw = st
                    .tensor_data(&gp)
                    .map_err(|_| ModelError::MissingWeight(gp))?;
                let u_raw = st
                    .tensor_data(&up)
                    .map_err(|_| ModelError::MissingWeight(up))?;
                let dst_off = e * expert_elems;
                let g_slice =
                    unsafe { std::slice::from_raw_parts(g_raw.as_ptr() as *const u16, eis * hs) };
                let u_slice =
                    unsafe { std::slice::from_raw_parts(u_raw.as_ptr() as *const u16, eis * hs) };
                host_buf[dst_off..dst_off + eis * hs].copy_from_slice(g_slice);
                host_buf[dst_off + eis * hs..dst_off + expert_elems].copy_from_slice(u_slice);
            }
            let mut wg = writer_cell.borrow_mut();
            host_bf16_quantize_upload_cache(
                &host_buf,
                ne * 2 * eis,
                hs,
                expert_fmt,
                device,
                &fused_name,
                (*wg).as_deref_mut(),
            )?
        } else {
            // No gate_proj (relu² activation): load only up_proj per expert
            // Try bqnt per-expert concatenation first
            let first_up_name = [
                format!("{prefix}experts.0.up_proj.weight"),
                format!("{prefix}mlp.experts.0.up_proj.weight"),
            ]
            .into_iter()
            .find(|n| bqnt.map_or(false, |b| b.entry(n).is_some()) || st.tensor_data(n).is_ok());
            let bqnt_per_expert = first_up_name.as_ref().and_then(|_| bqnt).and_then(|b| {
                // Try to concatenate per-expert packed bytes from bqnt
                let first_name = [
                    format!("{prefix}experts.0.up_proj.weight"),
                    format!("{prefix}mlp.experts.0.up_proj.weight"),
                ]
                .into_iter()
                .find(|n| b.entry(n).is_some())?;
                let first_entry = b.entry(&first_name)?;
                let per_expert_bytes = first_entry.data_bytes as usize;
                let mut packed = vec![0u8; ne * per_expert_bytes];
                for e in 0..ne {
                    let name = first_name.replace(".0.", &format!(".{e}."));
                    let data = b.tensor_data(&name)?;
                    packed[e * per_expert_bytes..(e + 1) * per_expert_bytes].copy_from_slice(data);
                }
                let fmt = code_to_format(first_entry.format).and_then(|s| s.to_weight_format())?;
                let mut buf = DeviceBuffer::<u8>::alloc(device, packed.len()).ok()?;
                buf.copy_from_host(&packed).ok()?;
                Some(LinearWeight::Packed(PackedWeights {
                    data: buf,
                    format: fmt,
                    out_dim: ne * first_entry.out_features as usize,
                    in_dim: first_entry.in_features as usize,
                }))
            });
            if let Some(lw) = bqnt_per_expert {
                lw
            } else {
                let expert_elems = eis * hs;
                let mut host_buf = vec![0u16; ne * expert_elems];
                for e in 0..ne {
                    let up_name = [
                        format!("{prefix}experts.{e}.up_proj.weight"),
                        format!("{prefix}mlp.experts.{e}.up_proj.weight"),
                    ]
                    .into_iter()
                    .find(|n| st.tensor_data(n).is_ok())
                    .ok_or_else(|| {
                        ModelError::MissingWeight(format!("{prefix}experts.{e}.up_proj.weight"))
                    })?;
                    let u_raw = st
                        .tensor_data(&up_name)
                        .map_err(|_| ModelError::MissingWeight(up_name))?;
                    let u_slice = unsafe {
                        std::slice::from_raw_parts(u_raw.as_ptr() as *const u16, expert_elems)
                    };
                    let dst_off = e * expert_elems;
                    host_buf[dst_off..dst_off + expert_elems].copy_from_slice(u_slice);
                }
                // Store as expert_gate_up with size eis (not 2*eis) — dispatch must handle this
                let mut wg2 = writer_cell.borrow_mut();
                host_bf16_quantize_upload_cache(
                    &host_buf,
                    ne * eis,
                    hs,
                    expert_fmt,
                    device,
                    &fused_name,
                    (*wg2).as_deref_mut(),
                )?
            }
        };

        // --- Expert down ---
        let down_name = format!("{prefix}mlp.experts.down_proj");
        let bqnt_down = bqnt.and_then(|b| load_linear_weight_bqnt(b, &down_name, device).ok());
        let expert_down = if let Some(lw) = bqnt_down {
            lw
        } else if st.tensor_data(&down_name).is_ok() {
            if writer_cell.borrow().is_some() {
                let mut guard = writer_cell.borrow_mut();
                load_linear_weight_cached(
                    st,
                    &down_name,
                    device,
                    ne * hs,
                    eis,
                    wq,
                    guard.as_mut().unwrap(),
                )?
            } else {
                load_linear_weight(st, &down_name, device, ne * hs, eis, wq)?
            }
        } else {
            // Try bqnt per-expert concatenation for down_proj
            let bqnt_per_expert_down = bqnt.and_then(|b| {
                let first_name = [
                    format!("{prefix}mlp.experts.0.down_proj.weight"),
                    format!("{prefix}experts.0.down_proj.weight"),
                ]
                .into_iter()
                .find(|n| b.entry(n).is_some())?;
                let first_entry = b.entry(&first_name)?;
                let per_expert_bytes = first_entry.data_bytes as usize;
                let mut packed = vec![0u8; ne * per_expert_bytes];
                for e in 0..ne {
                    let name = first_name.replace(".0.", &format!(".{e}."));
                    let data = b.tensor_data(&name)?;
                    packed[e * per_expert_bytes..(e + 1) * per_expert_bytes].copy_from_slice(data);
                }
                let fmt = code_to_format(first_entry.format).and_then(|s| s.to_weight_format())?;
                let mut buf = DeviceBuffer::<u8>::alloc(device, packed.len()).ok()?;
                buf.copy_from_host(&packed).ok()?;
                Some(LinearWeight::Packed(PackedWeights {
                    data: buf,
                    format: fmt,
                    out_dim: ne * first_entry.out_features as usize,
                    in_dim: first_entry.in_features as usize,
                }))
            });
            if let Some(lw) = bqnt_per_expert_down {
                lw
            } else {
                let expert_elems_d = hs * eis;
                let mut host_buf_d = vec![0u16; ne * expert_elems_d];
                for e in 0..ne {
                    let dp = [
                        format!("{prefix}mlp.experts.{e}.down_proj.weight"),
                        format!("{prefix}experts.{e}.down_proj.weight"),
                        format!("{prefix}block_sparse_moe.experts.{e}.w2.weight"),
                    ]
                    .into_iter()
                    .find(|n| st.tensor_data(n).is_ok())
                    .ok_or_else(|| {
                        ModelError::MissingWeight(format!(
                            "{prefix}experts.{e}.down_proj (or variants)"
                        ))
                    })?;
                    let d_raw = st
                        .tensor_data(&dp)
                        .map_err(|_| ModelError::MissingWeight(dp))?;
                    let d_slice = unsafe {
                        std::slice::from_raw_parts(d_raw.as_ptr() as *const u16, expert_elems_d)
                    };
                    let dst_off = e * expert_elems_d;
                    host_buf_d[dst_off..dst_off + expert_elems_d].copy_from_slice(d_slice);
                }
                let mut wg3 = writer_cell.borrow_mut();
                host_bf16_quantize_upload_cache(
                    &host_buf_d,
                    ne * hs,
                    eis,
                    expert_fmt,
                    device,
                    &down_name,
                    (*wg3).as_deref_mut(),
                )?
            }
        };

        (expert_gate_up, expert_down)
    }; // end else (skip_experts)

    // Shared expert (optional)
    let shared_expert = if *num_shared > 0 {
        let sis = *shared_intermediate_size;
        let sis = if sis == 0 { eis } else { sis };
        // Try multiple naming patterns for shared expert weights
        let se_up_name = find_weight_name(
            st,
            bqnt,
            &[
                format!("{prefix}mlp.shared_expert.up_proj.weight"),
                format!("{prefix}shared_experts.up_proj.weight"),
            ],
        )?;
        let se_down_name = find_weight_name(
            st,
            bqnt,
            &[
                format!("{prefix}mlp.shared_expert.down_proj.weight"),
                format!("{prefix}shared_experts.down_proj.weight"),
            ],
        )?;
        let se_gate_name = find_weight_name(
            st,
            bqnt,
            &[
                format!("{prefix}mlp.shared_expert.gate_proj.weight"),
                format!("{prefix}shared_experts.gate_proj.weight"),
            ],
        );
        let gate_proj = if let Ok(name) = se_gate_name {
            load_lw(&name, sis, hs)?
        } else {
            // No gate_proj (relu² models) — allocate dummy
            LinearWeight::Bf16(DeviceBuffer::<u16>::alloc(device, 0)?)
        };
        Some(DenseFfnWeights {
            gate_proj,
            up_proj: load_lw(&se_up_name, sis, hs)?,
            down_proj: load_lw(&se_down_name, hs, sis)?,
        })
    } else {
        None
    };

    // Shared expert gate (optional)
    let shared_gate_name = format!("{prefix}mlp.shared_expert_gate.weight");
    let shared_expert_gate = if st.tensor_data(&shared_gate_name).is_ok() {
        Some(load_weight_bf16(st, &shared_gate_name, device, hs)?)
    } else {
        None
    };

    // Score correction bias (Nemotron): added to scores before top-k selection
    let bias_name = find_weight_name(
        st,
        bqnt,
        &[
            format!("{prefix}gate.e_score_correction_bias"),
            format!("{prefix}mlp.gate.e_score_correction_bias"),
        ],
    );
    let score_correction_bias = if let Ok(name) = bias_name {
        let raw = st
            .tensor_data(&name)
            .map_err(|_| ModelError::MissingWeight(name.clone()))?;
        // f32 tensor: 4 bytes per element
        let data: Vec<f32> =
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, ne) }.to_vec();
        Some(data)
    } else {
        None
    };

    let score_correction_bias_gpu = if let Some(ref bias) = score_correction_bias {
        let mut buf = DeviceBuffer::<f32>::alloc(device, ne)?;
        buf.copy_from_host(bias)?;
        Some(buf)
    } else {
        None
    };

    // Use actual expert in_dim from packed weights when available and non-zero.
    // Fall back to moe_latent_size (if set in config) or hs.
    let gate_up_in_dim = expert_gate_up.in_dim()
        .filter(|&d| d > 0)
        .unwrap_or_else(|| config.moe_latent_size.unwrap_or(hs));

    // fc1_latent_proj / fc2_latent_proj: optional projections between hidden↔latent space.
    // Present on models with moe_latent_size (e.g. Nemotron-H: fc1=[1024,4096], fc2=[4096,1024]).
    let fc1_latent_proj = load_latent_proj(prefix, "fc1_latent_proj", device, bqnt, st, wq)?;
    let fc2_latent_proj = load_latent_proj(prefix, "fc2_latent_proj", device, bqnt, st, wq)?;

    Ok(MoeWeights {
        gate,
        expert_gate_up,
        expert_down,
        shared_expert,
        shared_expert_gate,
        num_experts: ne,
        expert_intermediate_size: eis,
        has_gate_proj,
        gate_up_in_dim,
        fc1_latent_proj,
        fc2_latent_proj,
        score_correction_bias,
        score_correction_bias_gpu,
    })
}

/// Load a tensor as f32. For f32 on disk: reinterpret. For bf16: convert on CPU.
/// Only used for the few tensors that need f32 on the GPU (A_log, output_norm).
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

// D2D memcpy: copy `count` u16 elements from src at offset src_off to dst at offset dst_off
pub unsafe fn d2d_copy_u16(
    dst: &mut DeviceBuffer<u16>,
    dst_off: usize,
    src: &DeviceBuffer<u16>,
    src_off: usize,
    count: usize,
    stream: &Stream,
) -> HipResult<()> {
    unsafe {
        let dst_ptr = dst.as_mut_ptr().add(dst_off) as *mut std::ffi::c_void;
        let src_ptr = src.as_ptr().add(src_off) as *const std::ffi::c_void;
        braidinfer_hip::error::check(ffi::hipMemcpyAsync(
            dst_ptr,
            src_ptr,
            count * 2,
            ffi::hipMemcpyDeviceToDevice,
            stream.raw(),
        ))
    }
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

/// Distribute expert weights from single-GPU MoeWeights across multiple GPUs (round-robin).
/// Expert `e` goes to GPU `e % num_devices`.
/// Gate, shared expert, and bias remain in the original MoeWeights on GPU 0.
/// `start_gpu`: first GPU to receive experts (0 for kbk, 1 for gpu-dispatch).
pub fn distribute_moe_weights_from_ref(
    moe: &MoeWeights,
    num_devices: usize,
    hs: usize,
    start_gpu: usize,
) -> Result<DistributedMoeWeights, ModelError> {
    use braidinfer_hip::device::DeviceGuard;

    if start_gpu >= num_devices {
        return Err(ModelError::InvalidConfig(format!(
            "start_gpu {start_gpu} must be less than num_devices {num_devices}"
        )));
    }

    let ne = moe.num_experts;
    let eis = moe.expert_intermediate_size;

    // Compute byte strides. Use actual in_dim from weight (may be moe_latent_size, not hs).
    let gate_up_in_dim = moe.expert_gate_up.in_dim().unwrap_or(hs);
    let gate_up_rows_per_expert = if moe.has_gate_proj { 2 * eis } else { eis };
    let gate_up_expert_stride = moe
        .expert_gate_up
        .row_byte_offset_dim(gate_up_rows_per_expert, gate_up_in_dim);
    let down_expert_stride = moe.expert_down.row_byte_offset_dim(gate_up_in_dim, eis);
    let gate_up_row_stride = moe.expert_gate_up.row_byte_offset_dim(1, gate_up_in_dim);

    // Round-robin across GPUs start_gpu..num_devices-1.
    let worker_count = num_devices - start_gpu;
    let mut expert_device = vec![0usize; ne];
    let mut counts = vec![0usize; num_devices];
    for e in 0..ne {
        let gpu = start_gpu + (e % worker_count);
        expert_device[e] = gpu;
        counts[gpu] += 1;
    }

    // Allocate per-GPU buffers and build slot maps
    let mut expert_buffers = Vec::with_capacity(num_devices);
    for gpu in 0..num_devices {
        let device = DeviceId(gpu as u32);
        // DeviceGuard restores the caller's device at end of each iteration.
        let _guard = DeviceGuard::switch_to(device)?;

        let n = counts[gpu];
        let mut slot_map = vec![None; ne];
        let mut slot = 0;
        for e in 0..ne {
            if expert_device[e] == gpu {
                slot_map[e] = Some(slot);
                slot += 1;
            }
        }

        if gpu == 0 {
            // GPU 0: use original packed buffer directly (no extra allocation).
            // slot_map maps global expert_id → global expert_id (identity for GPU 0 experts).
            let mut slot_map_identity = vec![None; ne];
            for e in 0..ne {
                if expert_device[e] == 0 {
                    slot_map_identity[e] = Some(e); // global index = slot (original layout)
                }
            }
            expert_buffers.push(GpuExpertBuffer {
                device,
                gate_up: DeviceBuffer::<u8>::alloc(device, 0)?, // placeholder — dispatch uses moe.expert_gate_up
                down: DeviceBuffer::<u8>::alloc(device, 0)?,
                local_expert_count: n,
                slot_map: slot_map_identity,
            });
        } else {
            expert_buffers.push(GpuExpertBuffer {
                device,
                gate_up: DeviceBuffer::<u8>::alloc(device, n * gate_up_expert_stride)?,
                down: DeviceBuffer::<u8>::alloc(device, n * down_expert_stride)?,
                local_expert_count: n,
                slot_map,
            });
        }
    }

    // Copy expert weights from GPU 0's packed buffer to per-GPU buffers
    let src_gate_up = moe.expert_gate_up.raw_data_ptr();
    let src_down = moe.expert_down.raw_data_ptr();

    // Host-staged copy: GPU 0 → host → target GPU.
    // hipMemcpyPeer uses SDMA which PERMISSION_FAULTs on RDNA3 PCIe.
    let max_expert_bytes = gate_up_expert_stride.max(down_expert_stride);
    let mut host_buf = vec![0u8; max_expert_bytes];

    for e in 0..ne {
        let gpu = expert_device[e];
        if gpu == 0 {
            continue;
        } // GPU 0 uses original packed buffer

        let local_slot = expert_buffers[gpu].slot_map[e].unwrap();

        // gate_up: GPU 0 → host → target GPU. Scoped DeviceGuards switch the
        // current device for the duration of each memcpy and restore the
        // caller's device when they drop.
        let src_offset = e * gate_up_expert_stride;
        let dst_offset = local_slot * gate_up_expert_stride;
        {
            let _g0 = DeviceGuard::switch_to(DeviceId(0))?;
            braidinfer_hip::memory::memcpy_d2h(
                &mut host_buf,
                unsafe { src_gate_up.add(src_offset) },
                gate_up_expert_stride,
            )?;
        }
        {
            let _gd = DeviceGuard::switch_to(DeviceId(gpu as u32))?;
            braidinfer_hip::memory::memcpy_h2d(
                unsafe { expert_buffers[gpu].gate_up.as_write_ptr().add(dst_offset) },
                &host_buf,
                gate_up_expert_stride,
            )?;
        }

        // down: GPU 0 → host → target GPU
        let src_offset = e * down_expert_stride;
        let dst_offset = local_slot * down_expert_stride;
        {
            let _g0 = DeviceGuard::switch_to(DeviceId(0))?;
            braidinfer_hip::memory::memcpy_d2h(
                &mut host_buf,
                unsafe { src_down.add(src_offset) },
                down_expert_stride,
            )?;
        }
        {
            let _gd = DeviceGuard::switch_to(DeviceId(gpu as u32))?;
            braidinfer_hip::memory::memcpy_h2d(
                unsafe { expert_buffers[gpu].down.as_write_ptr().add(dst_offset) },
                &host_buf,
                down_expert_stride,
            )?;
        }
    }


    Ok(DistributedMoeWeights {
        expert_buffers,
        expert_device,
        has_gate_proj: moe.has_gate_proj,
        num_experts: ne,
        expert_intermediate_size: eis,
        gate_up_in_dim,
        gate_up_expert_stride,
        down_expert_stride,
        gate_up_row_stride,
        weight_format: moe.expert_gate_up.weight_format(),
        gpu0_gate_up_base: src_gate_up,
        gpu0_down_base: src_down,
    })
}

/// Load expert weights directly from bqnt to per-GPU buffers.
/// For models too large for single GPU (e.g. 122B).
/// `start_gpu`: first GPU to receive experts (0 for kbk, 1 for gpu-dispatch).
pub fn distribute_moe_weights_from_bqnt(
    moe: &MoeWeights,
    bqnt: &crate::bqnt::MmapBqnt,
    layer_idx: usize,
    prefix: &str,
    num_devices: usize,
    _hs: usize,
    start_gpu: usize,
) -> Result<DistributedMoeWeights, ModelError> {
    use braidinfer_hip::device::DeviceGuard;

    if start_gpu >= num_devices {
        return Err(ModelError::InvalidConfig(format!(
            "start_gpu {start_gpu} must be less than num_devices {num_devices}"
        )));
    }

    let ne = moe.num_experts;
    let eis = moe.expert_intermediate_size;
    let has_gate_proj = moe.has_gate_proj;

    // Detect ffn key: Qwen uses "mlp.", Nemotron uses "mixer."
    let ffn_key = if bqnt
        .entry(&format!(
            "{prefix}layers.{layer_idx}.mixer.experts.gate_up_proj"
        ))
        .is_some()
        || bqnt
            .entry(&format!(
                "{prefix}layers.{layer_idx}.mixer.experts.0.up_proj.weight"
            ))
            .is_some()
        || bqnt
            .entry(&format!(
                "{prefix}layers.{layer_idx}.mixer.experts.0.gate_proj.weight"
            ))
            .is_some()
    {
        "mixer"
    } else {
        "mlp"
    };

    // Check for fused gate_up_proj tensor
    let fused_name = format!("{prefix}layers.{layer_idx}.{ffn_key}.experts.gate_up_proj");
    let has_fused = bqnt.entry(&fused_name).is_some();

    // Determine weight format from bqnt entry
    let weight_format = {
        let probe_name = if has_fused {
            fused_name.clone()
        } else {
            let first_up = format!("{prefix}layers.{layer_idx}.{ffn_key}.experts.0.up_proj.weight");
            let first_gate =
                format!("{prefix}layers.{layer_idx}.{ffn_key}.experts.0.gate_proj.weight");
            if bqnt.entry(&first_gate).is_some() {
                first_gate
            } else {
                first_up
            }
        };
        let entry = bqnt
            .entry(&probe_name)
            .ok_or_else(|| ModelError::MissingWeight(probe_name.clone()))?;
        crate::bqnt::code_to_format(entry.format).and_then(|s| s.to_weight_format()).ok_or_else(|| {
            ModelError::MissingWeight(format!(
                "{probe_name}: not a linear bqnt format code {}",
                entry.format
            ))
        })?
    };

    // Get byte sizes per expert from bqnt entries, and the actual in_dim of gate/up weights.
    let (gu_bytes_per_expert, down_bytes_per_expert, gate_up_in_dim) = if has_fused {
        let entry = bqnt.entry(&fused_name).unwrap();
        let gu_total = entry.data_bytes as usize;
        let in_dim = entry.in_features as usize;
        let down_name = format!("{prefix}layers.{layer_idx}.{ffn_key}.experts.down_proj");
        let down_entry = bqnt
            .entry(&down_name)
            .ok_or_else(|| ModelError::MissingWeight(down_name))?;
        (gu_total / ne, down_entry.data_bytes as usize / ne, in_dim)
    } else {
        let first_up = format!("{prefix}layers.{layer_idx}.{ffn_key}.experts.0.up_proj.weight");
        let entry = bqnt
            .entry(&first_up)
            .ok_or_else(|| ModelError::MissingWeight(first_up))?;
        let up_bytes = entry.data_bytes as usize;
        let in_dim = entry.in_features as usize;
        let gu = if has_gate_proj {
            up_bytes * 2
        } else {
            up_bytes
        };
        let first_down = format!("{prefix}layers.{layer_idx}.{ffn_key}.experts.0.down_proj.weight");
        let down_entry = bqnt
            .entry(&first_down)
            .ok_or_else(|| ModelError::MissingWeight(first_down))?;
        (gu, down_entry.data_bytes as usize, in_dim)
    };

    // Use actual expert weight in_dim (may differ from hs when model uses an adapter projection).
    let gate_up_row_stride = match weight_format {
        crate::quant::WeightFormat::PcG32Q4 => (gate_up_in_dim + 31) / 32 * 20,
        crate::quant::WeightFormat::Rnf4G128 => (gate_up_in_dim + 127) / 128 * 132,
        crate::quant::WeightFormat::Bf16 => gate_up_in_dim * 2,
    };

    // Round-robin across GPUs start_gpu..num_devices-1.
    let worker_count = num_devices - start_gpu;
    let mut expert_device = vec![0usize; ne];
    let mut counts = vec![0usize; num_devices];
    for e in 0..ne {
        let gpu = start_gpu + (e % worker_count);
        expert_device[e] = gpu;
        counts[gpu] += 1;
    }

    // Diagnostic: log per-layer per-expert sizes on layer 0 only to give the
    // user a sense of allocation scale without spamming the output.
    if layer_idx == 0 {
        eprintln!(
            "  distribute layer 0: ne={} gu_bytes/expert={} down_bytes/expert={} gate_up_in_dim={} format={:?}",
            ne, gu_bytes_per_expert, down_bytes_per_expert, gate_up_in_dim, weight_format
        );
        let per_gpu_alloc_mb = (counts[0] as f64
            * (gu_bytes_per_expert + down_bytes_per_expert) as f64)
            / (1024.0 * 1024.0);
        eprintln!(
            "  per-GPU MoE alloc/layer ≈ {:.1} MB ({} experts × {:.1} MB/expert)",
            per_gpu_alloc_mb,
            counts[0],
            (gu_bytes_per_expert + down_bytes_per_expert) as f64 / (1024.0 * 1024.0)
        );
    }

    // Allocate per-GPU buffers
    let mut expert_buffers = Vec::with_capacity(num_devices);
    for gpu in 0..num_devices {
        let device = DeviceId(gpu as u32);
        let _guard = DeviceGuard::switch_to(device)?;
        let n = counts[gpu];
        let mut slot_map = vec![None; ne];
        let mut slot = 0;
        for e in 0..ne {
            if expert_device[e] == gpu {
                slot_map[e] = Some(slot);
                slot += 1;
            }
        }
        let gu_size = n * gu_bytes_per_expert;
        let down_size = n * down_bytes_per_expert;
        let gate_up_buf = DeviceBuffer::<u8>::alloc(device, gu_size).map_err(|e| {
            eprintln!(
                "  FAIL distribute layer {} GPU {}: gate_up alloc {} bytes ({:.1}MB) — {:?}",
                layer_idx, gpu, gu_size, gu_size as f64 / 1e6, e
            );
            e
        })?;
        let down_buf = DeviceBuffer::<u8>::alloc(device, down_size).map_err(|e| {
            eprintln!(
                "  FAIL distribute layer {} GPU {}: down alloc {} bytes ({:.1}MB) — {:?}",
                layer_idx, gpu, down_size, down_size as f64 / 1e6, e
            );
            e
        })?;
        expert_buffers.push(GpuExpertBuffer {
            device,
            gate_up: gate_up_buf,
            down: down_buf,
            local_expert_count: n,
            slot_map,
        });
    }

    // Load from bqnt host memory directly to per-GPU buffers
    if has_fused {
        let gu_data = bqnt
            .tensor_data(&fused_name)
            .ok_or_else(|| ModelError::MissingWeight(fused_name.clone()))?;
        let down_name = format!("{prefix}layers.{layer_idx}.{ffn_key}.experts.down_proj");
        let down_data = bqnt
            .tensor_data(&down_name)
            .ok_or_else(|| ModelError::MissingWeight(down_name))?;

        for e in 0..ne {
            let gpu = expert_device[e];
            let slot = expert_buffers[gpu].slot_map[e].unwrap();
            let _guard = DeviceGuard::switch_to(DeviceId(gpu as u32))?;

            let gu_src = &gu_data[e * gu_bytes_per_expert..(e + 1) * gu_bytes_per_expert];
            checked_hip_memcpy_h2d(
                unsafe {
                    expert_buffers[gpu]
                        .gate_up
                        .as_ptr()
                        .add(slot * gu_bytes_per_expert) as *mut _
                },
                gu_src.as_ptr() as *const _,
                gu_bytes_per_expert,
            )?;
            let d_src = &down_data[e * down_bytes_per_expert..(e + 1) * down_bytes_per_expert];
            checked_hip_memcpy_h2d(
                unsafe {
                    expert_buffers[gpu]
                        .down
                        .as_ptr()
                        .add(slot * down_bytes_per_expert) as *mut _
                },
                d_src.as_ptr() as *const _,
                down_bytes_per_expert,
            )?;
        }
    } else {
        for e in 0..ne {
            let gpu = expert_device[e];
            let slot = expert_buffers[gpu].slot_map[e].unwrap();
            let _guard = DeviceGuard::switch_to(DeviceId(gpu as u32))?;

            // gate_up
            if has_gate_proj {
                let gate_name =
                    format!("{prefix}layers.{layer_idx}.{ffn_key}.experts.{e}.gate_proj.weight");
                let up_name =
                    format!("{prefix}layers.{layer_idx}.{ffn_key}.experts.{e}.up_proj.weight");
                let g = bqnt
                    .tensor_data(&gate_name)
                    .ok_or_else(|| ModelError::MissingWeight(gate_name))?;
                let u = bqnt
                    .tensor_data(&up_name)
                    .ok_or_else(|| ModelError::MissingWeight(up_name))?;
                let dst = unsafe {
                    expert_buffers[gpu]
                        .gate_up
                        .as_ptr()
                        .add(slot * gu_bytes_per_expert)
                };
                checked_hip_memcpy_h2d(dst as *mut _, g.as_ptr() as *const _, g.len())?;
                checked_hip_memcpy_h2d(
                    unsafe { dst.add(g.len()) as *mut _ },
                    u.as_ptr() as *const _,
                    u.len(),
                )?;
            } else {
                let up_name =
                    format!("{prefix}layers.{layer_idx}.{ffn_key}.experts.{e}.up_proj.weight");
                let u = bqnt
                    .tensor_data(&up_name)
                    .ok_or_else(|| ModelError::MissingWeight(up_name))?;
                checked_hip_memcpy_h2d(
                    unsafe {
                        expert_buffers[gpu]
                            .gate_up
                            .as_ptr()
                            .add(slot * gu_bytes_per_expert) as *mut _
                    },
                    u.as_ptr() as *const _,
                    u.len(),
                )?;
            }

            // down
            let down_name =
                format!("{prefix}layers.{layer_idx}.{ffn_key}.experts.{e}.down_proj.weight");
            let d = bqnt
                .tensor_data(&down_name)
                .ok_or_else(|| ModelError::MissingWeight(down_name))?;
            checked_hip_memcpy_h2d(
                unsafe {
                    expert_buffers[gpu]
                        .down
                        .as_ptr()
                        .add(slot * down_bytes_per_expert) as *mut _
                },
                d.as_ptr() as *const _,
                d.len(),
            )?;
        }
    }

    eprintln!(
        "  Layer {layer_idx}: {ne} experts distributed ({} per GPU, {} GPUs)",
        ne / num_devices,
        num_devices
    );

    let gpu0_gu = expert_buffers[0].gate_up.as_ptr();
    let gpu0_d = expert_buffers[0].down.as_ptr();
    Ok(DistributedMoeWeights {
        expert_buffers,
        expert_device,
        has_gate_proj,
        num_experts: ne,
        expert_intermediate_size: eis,
        gate_up_in_dim,
        gate_up_expert_stride: gu_bytes_per_expert,
        down_expert_stride: down_bytes_per_expert,
        gate_up_row_stride,
        weight_format,
        gpu0_gate_up_base: gpu0_gu,
        gpu0_down_base: gpu0_d,
    })
}
