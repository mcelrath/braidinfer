//! Weight types, loading, and activation buffer allocation.
//! Extracted from model.rs for maintainability.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{ffi, HipResult};
use braidinfer_core::safetensors::SafeTensorSet;
use safetensors::Dtype;

use crate::config::*;
use crate::kernel::{
    ArgmaxKernel, CausalConv1dUpdateKernel, EmbeddingKernel, FfnFusedKernel, GdnGateKernel,
    GdnRecurrentStepV2Kernel, GqaAttentionKernel, LinearProjKernel, LmHeadKernel,
    MRoPEKernel, MoeGateKernel, OutputGateKernel, QkNormKernel, ResidualAddKernel,
    RmsNormGatedKernel, RmsNormKernel, SelectiveStateUpdateKernel, SiluMulKernel,
};
pub use crate::quant::{WeightFormat, PackedWeights, WeightQuantMode, LinearWeight, quantize_rnf4_g128, quantize_pc_g32_q4};

// ---- Layer weight structs ----

pub struct GdnLayerWeights {
    pub input_norm: DeviceBuffer<u16>,  // bf16: (1+w) pattern, zeros init
    pub w_qkv: LinearWeight,           // [6144, 1024]
    pub w_a: LinearWeight,             // [16, 1024]
    pub w_b: LinearWeight,             // [16, 1024]
    pub w_z: LinearWeight,             // [2048, 1024]
    pub conv1d_weight: DeviceBuffer<u16>, // bf16 [6144, 4] (kept for traced path)
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
    pub input_norm: DeviceBuffer<u16>,  // bf16: stays bf16
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
    pub input_norm: DeviceBuffer<u16>,      // bf16 rmsnorm weight [hidden_size]
    pub in_proj: LinearWeight,              // [hidden_size, in_proj_size]
    pub conv1d_weight: DeviceBuffer<u16>,   // bf16 [conv_dim, 1, conv_kernel] (depthwise)
    pub conv1d_bias: DeviceBuffer<f32>,     // f32 [conv_dim]
    pub dt_bias: DeviceBuffer<f32>,         // f32 [num_heads]
    pub a_log: DeviceBuffer<f32>,           // f32 [num_heads]
    pub d: DeviceBuffer<f32>,               // f32 [num_heads]
    pub norm_weight: DeviceBuffer<f32>,     // f32 rmsnorm_gated weight [intermediate]
    pub out_proj: LinearWeight,             // [intermediate, hidden_size]
}

/// Standalone MoE FFN layer (Nemotron-H 'E' layers) — just norm + MoE dispatch
pub struct MoeFfnLayerWeights {
    pub input_norm: DeviceBuffer<u16>,      // bf16 rmsnorm weight [hidden_size]
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
    pub gate: DeviceBuffer<u16>,                    // [num_experts, hidden_size] — router, MUST stay bf16
    pub expert_gate_up: LinearWeight,               // SwiGLU: [ne, 2*eis, hs] fused; relu²: [ne, eis, hs] (up only)
    pub expert_down: LinearWeight,                  // [num_experts, hidden_size, expert_is]
    pub shared_expert: Option<DenseFfnWeights>,      // always-on shared expert
    pub shared_expert_gate: Option<DeviceBuffer<u16>>, // [1, hidden_size] gate for shared expert
    pub has_gate_proj: bool,                          // false for relu² (Nemotron), true for SwiGLU
    pub score_correction_bias: Option<Vec<f32>>,      // [num_experts] f32, added to scores before top-k
    pub score_correction_bias_gpu: Option<DeviceBuffer<f32>>, // GPU copy of correction_bias
    pub num_experts: usize,
    pub expert_intermediate_size: usize,
}


/// Per-GPU expert weight buffer for distributed MoE.
pub struct GpuExpertBuffer {
    pub device: DeviceId,
    pub gate_up: DeviceBuffer<u8>,       // packed expert weights on this GPU
    pub down: DeviceBuffer<u8>,          // packed down_proj weights on this GPU
    pub local_expert_count: usize,       // how many experts on this GPU
    /// Maps global expert_id → local slot index (None if not on this GPU).
    /// Indexed by global expert_id, len = num_experts.
    pub slot_map: Vec<Option<usize>>,
}

/// Distributed expert weight buffers across GPUs.
/// Gate, shared expert, and metadata stay in MoeWeights on GPU 0.
/// This struct holds only the per-GPU expert copies.
pub struct DistributedMoeWeights {
    pub expert_buffers: Vec<GpuExpertBuffer>,       // [num_devices]
    pub expert_device: Vec<usize>,                  // [num_experts] → device index
    pub has_gate_proj: bool,
    pub num_experts: usize,
    pub expert_intermediate_size: usize,
    pub gate_up_expert_stride: usize,
    pub down_expert_stride: usize,
    pub gate_up_row_stride: usize,
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
    pub hidden: DeviceBuffer<f32>,   // [hidden_size]
    pub normed: DeviceBuffer<f32>,   // [hidden_size]
    // GDN temporaries
    pub qkv: DeviceBuffer<f32>,      // [6144]
    pub q_gdn: DeviceBuffer<f32>,    // [16*128] = [2048]
    pub k_gdn: DeviceBuffer<f32>,    // [16*128] = [2048]
    pub v_gdn: DeviceBuffer<f32>,    // [16*128] = [2048]
    pub a_proj: DeviceBuffer<f32>,   // [16]
    pub b_proj: DeviceBuffer<f32>,   // [16]
    pub z_proj: DeviceBuffer<f32>,   // [2048]
    pub gate_gdn: DeviceBuffer<f32>, // [16]
    pub recurrent_out: DeviceBuffer<f32>, // [2048]
    pub normed_gated: DeviceBuffer<f32>,  // [2048]
    pub out_proj: DeviceBuffer<f32>, // [1024]
    // Attention temporaries
    pub q_gate_attn: DeviceBuffer<f32>, // [4096] Q+gate
    pub q_attn: DeviceBuffer<f32>,      // [2048]
    pub gate_attn: DeviceBuffer<f32>,   // [2048]
    pub k_attn: DeviceBuffer<f32>,      // [512]
    pub v_attn: DeviceBuffer<f32>,      // [512]
    pub attn_out: DeviceBuffer<f32>,    // [2048]
    pub gated_out: DeviceBuffer<f32>,   // [2048]
    // FFN temporaries
    pub ffn_gate: DeviceBuffer<f32>,  // [3584]
    pub ffn_up: DeviceBuffer<f32>,    // [3584]
    pub ffn_act: DeviceBuffer<f32>,   // [3584]
    pub ffn_down: DeviceBuffer<f32>,  // [1024]
    // Shared
    pub residual: DeviceBuffer<f32>,  // [1024]
    // Final
    pub logits: DeviceBuffer<f32>,    // [vocab_size]
    // inv_freq and position_ids for mRoPE
    pub inv_freq: DeviceBuffer<f32>,  // [rope_dim/2]
    pub position_ids: DeviceBuffer<i32>, // [3]
    // conv states per GDN layer (allocated separately)
    // Pre-allocated GDN conv state temp buffers (reused each gdn_forward call)
    pub gdn_cs_q: DeviceBuffer<f32>,      // [nh*kd*(ck-1)]
    pub gdn_cs_k: DeviceBuffer<f32>,      // [nh*kd*(ck-1)]
    pub gdn_cs_v: DeviceBuffer<f32>,      // [nh*vd*(ck-1)]
    pub gdn_conv_out_q: DeviceBuffer<f32>, // [nh*kd]
    pub gdn_conv_out_k: DeviceBuffer<f32>, // [nh*kd]
    pub gdn_conv_out_v: DeviceBuffer<f32>, // [nh*vd]
    // MoE scratch buffers (pre-allocated to avoid hipMalloc in hot path)
    pub moe_scores: DeviceBuffer<f32>,        // [max_num_experts]
    pub moe_expert_ids: DeviceBuffer<i32>,    // [max_k] — GPU-side top-k output
    pub moe_expert_weights: DeviceBuffer<f32>, // [max_k] — GPU-side top-k weights
    pub moe_expert_gate: DeviceBuffer<f32>,   // [max_expert_intermediate_size]
    pub moe_expert_up: DeviceBuffer<f32>,     // [max_expert_intermediate_size]
    pub moe_expert_act: DeviceBuffer<f32>,    // [max_expert_intermediate_size]
    pub moe_expert_out: DeviceBuffer<f32>,    // [hidden_size]
    // Mamba2 scratch buffers
    pub mamba2_in_proj: DeviceBuffer<f32>,    // [in_proj_size] (gate + xBC + dt)
    pub mamba2_conv_out: DeviceBuffer<f32>,   // [conv_dim] (after conv1d + activation)
    pub mamba2_ssm_out: DeviceBuffer<f32>,    // [intermediate] (SSM output y)
    // GPU-resident argmax
    pub argmax_result: DeviceBuffer<i32>,    // [1] — single token ID
}

// ---- All kernels ----

pub struct AllKernels {
    pub rmsnorm: RmsNormKernel,
    pub linear_proj: LinearProjKernel,
    pub silu_mul: SiluMulKernel,
    pub residual_add: ResidualAddKernel,
    pub embedding: EmbeddingKernel,
    pub lm_head: LmHeadKernel,
    pub mrope: MRoPEKernel,
    pub gqa_attention: GqaAttentionKernel,
    pub gdn_recurrent_v2: GdnRecurrentStepV2Kernel,
    pub causal_conv1d: CausalConv1dUpdateKernel,
    pub qk_norm: QkNormKernel,
    pub rmsnorm_gated: RmsNormGatedKernel,
    pub output_gate: OutputGateKernel,
    pub gdn_gate: GdnGateKernel,
    pub ffn_fused: FfnFusedKernel,
    pub ssm_update: SelectiveStateUpdateKernel,
    pub argmax: ArgmaxKernel,
    pub moe_gate: MoeGateKernel,
}


impl AllKernels {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        Ok(AllKernels {
            rmsnorm: RmsNormKernel::load(device)?,
            linear_proj: LinearProjKernel::load(device)?,
            silu_mul: SiluMulKernel::load(device)?,
            residual_add: ResidualAddKernel::load(device)?,
            embedding: EmbeddingKernel::load(device)?,
            lm_head: LmHeadKernel::load(device)?,
            mrope: MRoPEKernel::load(device)?,
            gqa_attention: GqaAttentionKernel::load(device)?,
            gdn_recurrent_v2: GdnRecurrentStepV2Kernel::load(device)?,
            causal_conv1d: CausalConv1dUpdateKernel::load(device)?,
            qk_norm: QkNormKernel::load(device)?,
            rmsnorm_gated: RmsNormGatedKernel::load(device)?,
            output_gate: OutputGateKernel::load(device)?,
            gdn_gate: GdnGateKernel::load(device)?,
            ffn_fused: FfnFusedKernel::load(device)?,
            ssm_update: SelectiveStateUpdateKernel::load(device)?,
            argmax: ArgmaxKernel::load(device)?,
            moe_gate: MoeGateKernel::load(device)?,
        })
    }
}

// ---- Error type ----

#[derive(Debug)]
pub enum ModelError {
    Hip(braidinfer_hip::HipError),
    SafeTensors(braidinfer_core::safetensors::SafeTensorsError),
    MissingWeight(String),
    Io(std::io::Error),
}

impl From<braidinfer_hip::HipError> for ModelError {
    fn from(e: braidinfer_hip::HipError) -> Self { ModelError::Hip(e) }
}

impl From<braidinfer_core::safetensors::SafeTensorsError> for ModelError {
    fn from(e: braidinfer_core::safetensors::SafeTensorsError) -> Self { ModelError::SafeTensors(e) }
}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self { ModelError::Io(e) }
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Hip(e) => write!(f, "HIP error: {e:?}"),
            ModelError::SafeTensors(e) => write!(f, "SafeTensors error: {e}"),
            ModelError::MissingWeight(s) => write!(f, "Missing weight: {s}"),
            ModelError::Io(e) => write!(f, "IO error: {e}"),
        }
    }
}

impl std::error::Error for ModelError {}

// ---- Helper: load a tensor by name, convert to f32, upload to GPU ----

/// Try multiple name patterns, return first that exists in safetensors.
pub fn find_weight_name(st: &SafeTensorSet, candidates: &[String]) -> Result<String, ModelError> {
    for name in candidates {
        if st.tensor_data(name).is_ok() {
            return Ok(name.clone());
        }
    }
    Err(ModelError::MissingWeight(format!("none of {:?} found", candidates)))
}

/// Load a bf16 tensor, returning a typed DeviceBuffer<u16>.
/// The underlying data is copied directly from mmap — zero conversion.
pub fn load_weight_bf16(
    st: &SafeTensorSet,
    name: &str,
    device: DeviceId,
    expected_len: usize,
) -> Result<DeviceBuffer<u16>, ModelError> {
    let raw = st.tensor_data(name)
        .map_err(|_| ModelError::MissingWeight(name.to_string()))?;
    assert_eq!(raw.len(), expected_len * 2, "weight {name}: expected {} bytes, got {}", expected_len * 2, raw.len());
    let mut buf = DeviceBuffer::<u16>::alloc(device, expected_len)?;
    let data: &[u16] = unsafe {
        std::slice::from_raw_parts(raw.as_ptr() as *const u16, expected_len)
    };
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
    let raw = st.tensor_data(name)
        .map_err(|_| ModelError::MissingWeight(name.to_string()))?;
    assert_eq!(raw.len(), expected_len * 2, "weight {name}: expected {} bytes, got {}", expected_len * 2, raw.len());
    let bf16_data: &[u16] = unsafe {
        std::slice::from_raw_parts(raw.as_ptr() as *const u16, expected_len)
    };

    match format {
        WeightFormat::Bf16 => {
            let mut buf = DeviceBuffer::<u8>::alloc(device, expected_len * 2)?;
            buf.copy_from_host(raw)?;
            Ok(PackedWeights { data: buf, format, out_dim, in_dim })
        }
        WeightFormat::Rnf4G128 => {
            let packed = quantize_rnf4_g128(bf16_data, out_dim, in_dim);
            let mut buf = DeviceBuffer::<u8>::alloc(device, packed.len())?;
            buf.copy_from_host(&packed)?;
            Ok(PackedWeights { data: buf, format, out_dim, in_dim })
        }
        WeightFormat::PcG32Q4 => {
            let packed = quantize_pc_g32_q4(bf16_data, out_dim, in_dim);
            let mut buf = DeviceBuffer::<u8>::alloc(device, packed.len())?;
            buf.copy_from_host(&packed)?;
            Ok(PackedWeights { data: buf, format, out_dim, in_dim })
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
            if name.contains("mlp.") || name.contains("gate_proj") || name.contains("up_proj") || name.contains("down_proj") {
                // But NOT the MoE router gate
                if name.contains("mlp.gate.weight") || name.contains("block_sparse_moe.gate") || name.contains("mlp.router") {
                    WeightFormat::Bf16  // router stays bf16
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
            Ok(LinearWeight::Packed(PackedWeights { data: buf, format: fmt, out_dim, in_dim }))
        }
    }
}

// --- BQNT (pre-quantized) loading ---

use crate::bqnt::{MmapBqnt, code_to_format};

/// Load a linear weight directly from a pre-quantized .bqnt file.
/// Zero quantization cost — packed bytes go straight from mmap to GPU.
pub fn load_linear_weight_bqnt(
    bqnt: &MmapBqnt,
    name: &str,
    device: DeviceId,
) -> Result<LinearWeight, ModelError> {
    let entry = bqnt.entry(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    let format = code_to_format(entry.format)
        .ok_or_else(|| ModelError::MissingWeight(format!("{name}: unknown format code {}", entry.format)))?;
    let data = bqnt.tensor_data(name)
        .ok_or_else(|| ModelError::MissingWeight(format!("{name}: data out of bounds")))?;

    match format {
        WeightFormat::Bf16 => {
            let n_elements = entry.out_features as usize * entry.in_features as usize;
            let bf16_data: &[u16] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u16, n_elements)
            };
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
        assert_eq!(data.len(), expected_len * 2,
            "bqnt weight {name}: expected {} bytes, got {}", expected_len * 2, data.len());
        let bf16_data: &[u16] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u16, expected_len)
        };
        let mut buf = DeviceBuffer::<u16>::alloc(device, expected_len)?;
        buf.copy_from_host(bf16_data)?;
        Ok(buf)
    } else {
        Err(ModelError::MissingWeight(name.to_string()))
    }
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
    let FfnType::MoE { num_experts, expert_intermediate_size, num_shared, shared_intermediate_size, .. } = ffn_type
        else { unreachable!("load_moe_weights called on non-MoE layer") };
    let ne = *num_experts;
    let eis = *expert_intermediate_size;
    let hs = config.hidden_size;

    // Helper: try bqnt first, then safetensors
    let load_lw = |name: &str, out_dim: usize, in_dim: usize| -> Result<LinearWeight, ModelError> {
        if let Some(b) = bqnt {
            if let Ok(lw) = load_linear_weight_bqnt(b, name, device) {
                return Ok(lw);
            }
        }
        load_linear_weight(st, name, device, out_dim, in_dim, wq)
    };

    // Router gate: try mlp.gate, gate (Nemotron), block_sparse_moe.gate, mlp.router (always bf16)
    let gate_name = [
        format!("{prefix}mlp.gate.weight"),
        format!("{prefix}gate.weight"),
        format!("{prefix}block_sparse_moe.gate.weight"),
        format!("{prefix}mlp.router.weight"),
    ].into_iter().find(|n| st.tensor_data(n).is_ok())
        .ok_or_else(|| ModelError::MissingWeight(format!("{prefix}mlp.gate.weight (or variants)")))?;
    let gate = load_weight_bf16(st, &gate_name, device, ne * hs)?;

    // Detect whether experts have gate_proj (SwiGLU) or just up_proj (relu²)
    // Check per-expert gate_proj OR fused gate_up_proj (which implies SwiGLU)
    let fused_name_check = format!("{prefix}mlp.experts.gate_up_proj");
    let has_fused_gate_up = st.tensor_data(&fused_name_check).is_ok()
        || bqnt.map_or(false, |b| b.entry(&fused_name_check).is_some());
    let has_gate_proj = has_fused_gate_up || [
        format!("{prefix}mlp.experts.0.gate_proj.weight"),
        format!("{prefix}experts.0.gate_proj.weight"),
        format!("{prefix}block_sparse_moe.experts.0.w1.weight"),
    ].iter().any(|n| st.tensor_data(n).is_ok());

    let expert_fmt = if has_gate_proj {
        weight_format_for(&format!("{prefix}mlp.experts.0.gate_proj.weight"), wq)
    } else {
        // Try Nemotron naming: experts.0.up_proj.weight (under mixer. prefix)
        weight_format_for(&format!("{prefix}experts.0.up_proj.weight"), wq)
    };

    // Expert gate+up: try bqnt fused, then safetensors fused, else per-expert fuse on host
    let fused_name = format!("{prefix}mlp.experts.gate_up_proj");
    let bqnt_fused = bqnt.and_then(|b| load_linear_weight_bqnt(b, &fused_name, device).ok());
    let expert_gate_up = if let Some(lw) = bqnt_fused {
        lw
    } else if st.tensor_data(&fused_name).is_ok() {
        load_linear_weight(st, &fused_name, device, ne * 2 * eis, hs, wq)?
    } else if has_gate_proj {
        // SwiGLU: fuse gate_proj + up_proj per expert
        let expert_elems = 2 * eis * hs;
        let mut host_buf = vec![0u16; ne * expert_elems];
        for e in 0..ne {
            let (gp, up) = [
                (format!("{prefix}mlp.experts.{e}.gate_proj.weight"), format!("{prefix}mlp.experts.{e}.up_proj.weight")),
                (format!("{prefix}block_sparse_moe.experts.{e}.w1.weight"), format!("{prefix}block_sparse_moe.experts.{e}.w3.weight")),
            ].into_iter().find(|(g, _)| st.tensor_data(g).is_ok())
                .ok_or_else(|| ModelError::MissingWeight(format!("{prefix}experts.{e}.gate_proj.weight (or variants)")))?;
            let g_raw = st.tensor_data(&gp).map_err(|_| ModelError::MissingWeight(gp))?;
            let u_raw = st.tensor_data(&up).map_err(|_| ModelError::MissingWeight(up))?;
            let dst_off = e * expert_elems;
            let g_slice = unsafe { std::slice::from_raw_parts(g_raw.as_ptr() as *const u16, eis * hs) };
            let u_slice = unsafe { std::slice::from_raw_parts(u_raw.as_ptr() as *const u16, eis * hs) };
            host_buf[dst_off..dst_off + eis * hs].copy_from_slice(g_slice);
            host_buf[dst_off + eis * hs..dst_off + expert_elems].copy_from_slice(u_slice);
        }
        host_bf16_to_linear_weight(&host_buf, ne * 2 * eis, hs, expert_fmt, device)?
    } else {
        // No gate_proj (relu² activation): load only up_proj per expert
        // Try bqnt per-expert concatenation first
        let first_up_name = [
            format!("{prefix}experts.0.up_proj.weight"),
            format!("{prefix}mlp.experts.0.up_proj.weight"),
        ].into_iter().find(|n| bqnt.map_or(false, |b| b.entry(n).is_some()) || st.tensor_data(n).is_ok());
        let bqnt_per_expert = first_up_name.as_ref().and_then(|_| bqnt).and_then(|b| {
            // Try to concatenate per-expert packed bytes from bqnt
            let first_name = [
                format!("{prefix}experts.0.up_proj.weight"),
                format!("{prefix}mlp.experts.0.up_proj.weight"),
            ].into_iter().find(|n| b.entry(n).is_some())?;
            let first_entry = b.entry(&first_name)?;
            let per_expert_bytes = first_entry.data_bytes as usize;
            let mut packed = vec![0u8; ne * per_expert_bytes];
            for e in 0..ne {
                let name = first_name.replace(".0.", &format!(".{e}."));
                let data = b.tensor_data(&name)?;
                packed[e * per_expert_bytes..(e + 1) * per_expert_bytes].copy_from_slice(data);
            }
            let fmt = code_to_format(first_entry.format)?;
            let mut buf = DeviceBuffer::<u8>::alloc(device, packed.len()).ok()?;
            buf.copy_from_host(&packed).ok()?;
            Some(LinearWeight::Packed(PackedWeights {
                data: buf, format: fmt,
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
            ].into_iter().find(|n| st.tensor_data(n).is_ok())
                .ok_or_else(|| ModelError::MissingWeight(format!("{prefix}experts.{e}.up_proj.weight")))?;
            let u_raw = st.tensor_data(&up_name).map_err(|_| ModelError::MissingWeight(up_name))?;
            let u_slice = unsafe { std::slice::from_raw_parts(u_raw.as_ptr() as *const u16, expert_elems) };
            let dst_off = e * expert_elems;
            host_buf[dst_off..dst_off + expert_elems].copy_from_slice(u_slice);
        }
        // Store as expert_gate_up with size eis (not 2*eis) — dispatch must handle this
        host_bf16_to_linear_weight(&host_buf, ne * eis, hs, expert_fmt, device)?
    }};

    // Expert down: try bqnt fused, then safetensors fused, else per-expert load
    let down_name = format!("{prefix}mlp.experts.down_proj");
    let bqnt_down = bqnt.and_then(|b| load_linear_weight_bqnt(b, &down_name, device).ok());
    let expert_down = if let Some(lw) = bqnt_down {
        lw
    } else if st.tensor_data(&down_name).is_ok() {
        load_linear_weight(st, &down_name, device, ne * hs, eis, wq)?
    } else {
        // Try bqnt per-expert concatenation for down_proj
        let bqnt_per_expert_down = bqnt.and_then(|b| {
            let first_name = [
                format!("{prefix}mlp.experts.0.down_proj.weight"),
                format!("{prefix}experts.0.down_proj.weight"),
            ].into_iter().find(|n| b.entry(n).is_some())?;
            let first_entry = b.entry(&first_name)?;
            let per_expert_bytes = first_entry.data_bytes as usize;
            let mut packed = vec![0u8; ne * per_expert_bytes];
            for e in 0..ne {
                let name = first_name.replace(".0.", &format!(".{e}."));
                let data = b.tensor_data(&name)?;
                packed[e * per_expert_bytes..(e + 1) * per_expert_bytes].copy_from_slice(data);
            }
            let fmt = code_to_format(first_entry.format)?;
            let mut buf = DeviceBuffer::<u8>::alloc(device, packed.len()).ok()?;
            buf.copy_from_host(&packed).ok()?;
            Some(LinearWeight::Packed(PackedWeights {
                data: buf, format: fmt,
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
            ].into_iter().find(|n| st.tensor_data(n).is_ok())
                .ok_or_else(|| ModelError::MissingWeight(format!("{prefix}experts.{e}.down_proj (or variants)")))?;
            let d_raw = st.tensor_data(&dp).map_err(|_| ModelError::MissingWeight(dp))?;
            let d_slice = unsafe { std::slice::from_raw_parts(d_raw.as_ptr() as *const u16, expert_elems_d) };
            let dst_off = e * expert_elems_d;
            host_buf_d[dst_off..dst_off + expert_elems_d].copy_from_slice(d_slice);
        }
        host_bf16_to_linear_weight(&host_buf_d, ne * hs, eis, expert_fmt, device)?
    }};

    // Shared expert (optional)
    let shared_expert = if *num_shared > 0 {
        let sis = *shared_intermediate_size;
        let sis = if sis == 0 { eis } else { sis };
        // Try multiple naming patterns for shared expert weights
        let se_up_name = find_weight_name(st, &[
            format!("{prefix}mlp.shared_expert.up_proj.weight"),
            format!("{prefix}shared_experts.up_proj.weight"),
        ])?;
        let se_down_name = find_weight_name(st, &[
            format!("{prefix}mlp.shared_expert.down_proj.weight"),
            format!("{prefix}shared_experts.down_proj.weight"),
        ])?;
        let se_gate_name = find_weight_name(st, &[
            format!("{prefix}mlp.shared_expert.gate_proj.weight"),
            format!("{prefix}shared_experts.gate_proj.weight"),
        ]);
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
    } else { None };

    // Shared expert gate (optional)
    let shared_gate_name = format!("{prefix}mlp.shared_expert_gate.weight");
    let shared_expert_gate = if st.tensor_data(&shared_gate_name).is_ok() {
        Some(load_weight_bf16(st, &shared_gate_name, device, hs)?)
    } else { None };

    // Score correction bias (Nemotron): added to scores before top-k selection
    let bias_name = find_weight_name(st, &[
        format!("{prefix}gate.e_score_correction_bias"),
        format!("{prefix}mlp.gate.e_score_correction_bias"),
    ]);
    let score_correction_bias = if let Ok(name) = bias_name {
        let raw = st.tensor_data(&name).map_err(|_| ModelError::MissingWeight(name.clone()))?;
        // f32 tensor: 4 bytes per element
        let data: Vec<f32> = unsafe {
            std::slice::from_raw_parts(raw.as_ptr() as *const f32, ne)
        }.to_vec();
        Some(data)
    } else { None };

    let score_correction_bias_gpu = if let Some(ref bias) = score_correction_bias {
        let mut buf = DeviceBuffer::<f32>::alloc(device, ne)?;
        buf.copy_from_host(bias)?;
        Some(buf)
    } else { None };

    Ok(MoeWeights { gate, expert_gate_up, expert_down, shared_expert, shared_expert_gate,
        num_experts: ne, expert_intermediate_size: eis, has_gate_proj, score_correction_bias,
        score_correction_bias_gpu })
}

/// Load a tensor as f32. For f32 on disk: reinterpret. For bf16: convert on CPU.
/// Only used for the few tensors that need f32 on the GPU (A_log, output_norm).
pub fn load_weight_f32(
    st: &SafeTensorSet,
    name: &str,
    device: DeviceId,
    expected_len: usize,
) -> Result<DeviceBuffer<f32>, ModelError> {
    let info = st.tensor_info(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    let raw = st.tensor_data(name)
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

// D2D memcpy: copy `count` f32 elements from src at offset src_off to dst at offset dst_off
pub unsafe fn d2d_copy_f32(
    dst: &mut DeviceBuffer<f32>,
    dst_off: usize,
    src: &DeviceBuffer<f32>,
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
            count * 4,
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
pub fn distribute_moe_weights_from_ref(
    moe: &MoeWeights,
    num_devices: usize,
    hs: usize,
) -> Result<DistributedMoeWeights, ModelError> {
    use braidinfer_hip::device::Device;

    let ne = moe.num_experts;
    let eis = moe.expert_intermediate_size;

    // Compute byte strides
    let gate_up_rows_per_expert = if moe.has_gate_proj { 2 * eis } else { eis };
    let gate_up_expert_stride = moe.expert_gate_up.row_byte_offset_dim(gate_up_rows_per_expert, hs);
    let down_expert_stride = moe.expert_down.row_byte_offset_dim(hs, eis);
    let gate_up_row_stride = moe.expert_gate_up.row_byte_offset_dim(1, hs);

    // Count experts per GPU
    let mut expert_device = vec![0usize; ne];
    let mut counts = vec![0usize; num_devices];
    for e in 0..ne {
        let gpu = e % num_devices;
        expert_device[e] = gpu;
        counts[gpu] += 1;
    }

    // Allocate per-GPU buffers and build slot maps
    let mut expert_buffers = Vec::with_capacity(num_devices);
    for gpu in 0..num_devices {
        let device = DeviceId(gpu as u32);
        Device::set_current(device)?;

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

    for e in 0..ne {
        let gpu = expert_device[e];
        if gpu == 0 { continue; } // GPU 0 uses original packed buffer

        let local_slot = expert_buffers[gpu].slot_map[e].unwrap();
        let dst_device = DeviceId(gpu as u32);

        // gate_up: P2P copy from GPU 0 to target GPU
        let src_offset = e * gate_up_expert_stride;
        let dst_offset = local_slot * gate_up_expert_stride;
        unsafe {
            braidinfer_hip::ffi::hipMemcpyPeer(
                expert_buffers[gpu].gate_up.as_ptr().add(dst_offset) as *mut std::ffi::c_void,
                dst_device.0 as i32,
                src_gate_up.add(src_offset) as *const std::ffi::c_void,
                0, // src device = GPU 0
                gate_up_expert_stride,
            );
        }

        // down: P2P copy
        let src_offset = e * down_expert_stride;
        let dst_offset = local_slot * down_expert_stride;
        unsafe {
            braidinfer_hip::ffi::hipMemcpyPeer(
                expert_buffers[gpu].down.as_ptr().add(dst_offset) as *mut std::ffi::c_void,
                dst_device.0 as i32,
                src_down.add(src_offset) as *const std::ffi::c_void,
                0,
                down_expert_stride,
            );
        }
    }

    // Restore GPU 0 context
    Device::set_current(DeviceId(0))?;

    Ok(DistributedMoeWeights {
        expert_buffers,
        expert_device,
        has_gate_proj: moe.has_gate_proj,
        num_experts: ne,
        expert_intermediate_size: eis,
        gate_up_expert_stride,
        down_expert_stride,
        gate_up_row_stride,
        gpu0_gate_up_base: src_gate_up,
        gpu0_down_base: src_down,
    })
}

