use std::path::Path;

use serde::Deserialize;

use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{ffi, HipResult};
use braidinfer_core::safetensors::SafeTensorSet;
use safetensors::Dtype;

use crate::kernel::{
    CausalConv1dUpdateKernel, EmbeddingKernel, FfnFusedKernel, GdnGateKernel,
    GdnRecurrentStepV2Kernel, GqaAttentionKernel, LinearProjKernel, LmHeadKernel,
    MRoPEKernel, OutputGateKernel, QkNormKernel, ResidualAddKernel, RmsNormGatedKernel,
    RmsNormKernel,
};
use crate::megakernel::{MegakernelProgram, PrefillBuffers, CHUNK_TOKENS};
use crate::paged_kv::{self, PageAllocator, RecurrentCheckpointPool, SequenceState};

// ---- Model config ----

#[derive(Debug, Clone)]
pub enum RecurrentLayerKind {
    Gdn {
        num_heads: usize,
        key_value_dim: usize,
        conv_dim: usize,
        kernel_size: usize,
    },
    Mamba2 {
        state_dim: usize,
        num_heads: usize,
        head_dim: usize,
        conv_kernel: usize,
    },
    None,
}

#[derive(Debug, Clone)]
pub enum RopeType {
    Standard { rotary_dim: usize },
    MRope { sections: [usize; 3] },
}

pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    // mrope sections (pairs)
    pub mrope_section: [usize; 3],
    // GDN config
    pub linear_num_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    // layer types: true = full_attention, false = linear_attention (GDN)
    pub layer_is_attention: Vec<bool>,
    pub max_seq_len: usize,
    // Extended config
    pub recurrent_kind: RecurrentLayerKind,
    pub rope_type: RopeType,
    pub has_qk_norm: bool,
    pub attention_layer_indices: Vec<usize>,
    pub model_type: String,
}

// Serde structs for config.json parsing

#[derive(Deserialize)]
struct RopeParameters {
    #[serde(default)]
    mrope_section: Option<[usize; 3]>,
    rope_theta: Option<f64>,
    partial_rotary_factor: Option<f64>,
}

#[derive(Deserialize)]
struct TextConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: Option<f64>,
    layer_norm_epsilon: Option<f64>,
    max_position_embeddings: Option<usize>,
    rope_parameters: Option<RopeParameters>,
    layer_types: Option<Vec<String>>,
    linear_num_key_heads: Option<usize>,
    linear_key_head_dim: Option<usize>,
    linear_value_head_dim: Option<usize>,
    linear_conv_kernel_dim: Option<usize>,
    model_type: Option<String>,
}

#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    text_config: Option<TextConfig>,
    // Nemotron-style: top-level fields
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    intermediate_size: Option<usize>,
    vocab_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    partial_rotary_factor: Option<f64>,
    hybrid_override_pattern: Option<String>,
    ssm_state_size: Option<usize>,
    mamba_num_heads: Option<usize>,
    mamba_head_dim: Option<usize>,
    conv_kernel: Option<usize>,
    max_position_embeddings: Option<usize>,
}

impl ModelConfig {
    pub fn qwen35_0_8b() -> Self {
        // From config.json: layer_types = ["linear_attention"]*3 + ["full_attention"] repeated 6x
        // → 24 layers total:  0,1,2 = GDN; 3 = attn; 4,5,6 = GDN; 7 = attn; ...
        let mut layer_is_attention = vec![false; 24];
        let attention_layer_indices: Vec<usize> = vec![3, 7, 11, 15, 19, 23];
        for &i in &attention_layer_indices {
            layer_is_attention[i] = true;
        }
        ModelConfig {
            hidden_size: 1024,
            num_layers: 24,
            intermediate_size: 3584,
            vocab_size: 248320,
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 256,
            rope_dim: 64,
            rope_theta: 10_000_000.0,
            rms_norm_eps: 1e-6,
            mrope_section: [11, 11, 10],
            linear_num_heads: 16,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            layer_is_attention,
            max_seq_len: 2048, // fallback for tests; load() reads from config.json
            recurrent_kind: RecurrentLayerKind::Gdn {
                num_heads: 16,
                key_value_dim: 128,
                conv_dim: 6144,
                kernel_size: 4,
            },
            rope_type: RopeType::MRope { sections: [11, 11, 10] },
            has_qk_norm: true,
            attention_layer_indices,
            model_type: "qwen3_5".to_string(),
        }
    }

    pub fn from_config_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let raw: RawConfig = serde_json::from_str(&data)?;

        match raw.model_type.as_str() {
            "qwen3_5" => Self::from_qwen35_config(raw),
            "nemotron_h" => Self::from_nemotron_config(raw),
            other => Err(format!("Unknown model_type: {other}").into()),
        }
    }

    fn from_qwen35_config(raw: RawConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let tc = raw.text_config.ok_or("qwen3_5 config missing text_config")?;

        let num_layers = tc.num_hidden_layers;
        let layer_types = tc.layer_types.unwrap_or_else(|| vec!["linear_attention".to_string(); num_layers]);

        let attention_layer_indices: Vec<usize> = layer_types.iter().enumerate()
            .filter(|(_, t)| t.as_str() == "full_attention")
            .map(|(i, _)| i)
            .collect();

        let mut layer_is_attention = vec![false; num_layers];
        for &i in &attention_layer_indices {
            layer_is_attention[i] = true;
        }

        let rope_params = tc.rope_parameters.unwrap_or(RopeParameters {
            mrope_section: None,
            rope_theta: None,
            partial_rotary_factor: None,
        });

        let rope_theta = rope_params.rope_theta.unwrap_or(10_000_000.0) as f32;
        let partial_rotary_factor = rope_params.partial_rotary_factor.unwrap_or(0.25);
        let rope_dim = ((tc.head_dim as f64) * partial_rotary_factor) as usize;
        let mrope_section = rope_params.mrope_section.unwrap_or([0, 0, 0]);

        let rope_type = if let Some(sec) = rope_params.mrope_section {
            RopeType::MRope { sections: sec }
        } else {
            RopeType::Standard { rotary_dim: rope_dim }
        };

        let linear_num_heads = tc.linear_num_key_heads.unwrap_or(16);
        let linear_key_head_dim = tc.linear_key_head_dim.unwrap_or(128);
        let linear_value_head_dim = tc.linear_value_head_dim.unwrap_or(128);
        let linear_conv_kernel_dim = tc.linear_conv_kernel_dim.unwrap_or(4);
        let conv_dim = linear_num_heads * (linear_key_head_dim + linear_value_head_dim);

        Ok(ModelConfig {
            hidden_size: tc.hidden_size,
            num_layers,
            intermediate_size: tc.intermediate_size,
            vocab_size: tc.vocab_size,
            num_q_heads: tc.num_attention_heads,
            num_kv_heads: tc.num_key_value_heads,
            head_dim: tc.head_dim,
            rope_dim,
            rope_theta,
            rms_norm_eps: tc.rms_norm_eps.unwrap_or(1e-6) as f32,
            mrope_section,
            linear_num_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            layer_is_attention,
            max_seq_len: tc.max_position_embeddings.unwrap_or(2048),
            recurrent_kind: RecurrentLayerKind::Gdn {
                num_heads: linear_num_heads,
                key_value_dim: linear_key_head_dim,
                conv_dim,
                kernel_size: linear_conv_kernel_dim,
            },
            rope_type,
            has_qk_norm: true,
            attention_layer_indices,
            model_type: "qwen3_5".to_string(),
        })
    }

    fn from_nemotron_config(raw: RawConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let num_layers = raw.num_hidden_layers.ok_or("missing num_hidden_layers")?;
        let hidden_size = raw.hidden_size.ok_or("missing hidden_size")?;
        let head_dim = raw.head_dim.ok_or("missing head_dim")?;
        let num_q_heads = raw.num_attention_heads.ok_or("missing num_attention_heads")?;
        let num_kv_heads = raw.num_key_value_heads.ok_or("missing num_key_value_heads")?;
        let intermediate_size = raw.intermediate_size.ok_or("missing intermediate_size")?;
        let vocab_size = raw.vocab_size.ok_or("missing vocab_size")?;
        let rope_theta = raw.rope_theta.unwrap_or(10000.0) as f32;
        let partial_rotary_factor = raw.partial_rotary_factor.unwrap_or(1.0);
        let rope_dim = ((head_dim as f64) * partial_rotary_factor) as usize;
        let rms_norm_eps = raw.norm_eps.unwrap_or(1e-5) as f32;

        let pattern = raw.hybrid_override_pattern.unwrap_or_default();
        let attention_layer_indices: Vec<usize> = pattern.chars().enumerate()
            .filter(|(_, c)| *c == '*')
            .map(|(i, _)| i)
            .collect();
        let mut layer_is_attention = vec![false; num_layers];
        for &i in &attention_layer_indices {
            if i < num_layers {
                layer_is_attention[i] = true;
            }
        }

        let ssm_state_dim = raw.ssm_state_size.unwrap_or(128);
        let mamba_num_heads = raw.mamba_num_heads.unwrap_or(64);
        let mamba_head_dim = raw.mamba_head_dim.unwrap_or(64);
        let conv_kernel = raw.conv_kernel.unwrap_or(4);

        Ok(ModelConfig {
            hidden_size,
            num_layers,
            intermediate_size,
            vocab_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
            rope_theta,
            rms_norm_eps,
            mrope_section: [0, 0, 0],
            linear_num_heads: mamba_num_heads,
            linear_key_head_dim: mamba_head_dim,
            linear_value_head_dim: mamba_head_dim,
            linear_conv_kernel_dim: conv_kernel,
            layer_is_attention,
            max_seq_len: raw.max_position_embeddings.unwrap_or(2048),
            recurrent_kind: RecurrentLayerKind::Mamba2 {
                state_dim: ssm_state_dim,
                num_heads: mamba_num_heads,
                head_dim: mamba_head_dim,
                conv_kernel,
            },
            rope_type: RopeType::Standard { rotary_dim: rope_dim },
            has_qk_norm: false,
            attention_layer_indices,
            model_type: "nemotron_h".to_string(),
        })
    }

    pub fn chunk_kv_bytes(&self, chunk_tokens: usize) -> usize {
        let num_attn = self.num_attn_layers();
        // k and v, each: chunk_tokens * num_kv_heads * head_dim * 4 bytes (f32)
        2 * num_attn * chunk_tokens * self.num_kv_heads * self.head_dim * 4
    }

    pub fn recurrent_state_bytes_per_layer(&self) -> usize {
        match &self.recurrent_kind {
            RecurrentLayerKind::Gdn { num_heads, key_value_dim, .. } => {
                num_heads * key_value_dim * key_value_dim * 4
            }
            RecurrentLayerKind::Mamba2 { state_dim, num_heads, head_dim, .. } => {
                num_heads * head_dim * state_dim * 4
            }
            RecurrentLayerKind::None => 0,
        }
    }

    pub fn total_recurrent_checkpoint_bytes(&self) -> usize {
        self.num_recurrent_layers() * self.recurrent_state_bytes_per_layer()
    }

    pub fn num_attn_layers(&self) -> usize {
        self.layer_is_attention.iter().filter(|&&a| a).count()
    }

    pub fn num_recurrent_layers(&self) -> usize {
        self.layer_is_attention.iter().filter(|&&a| !a).count()
    }
}

// ---- Layer weight structs ----

pub struct GdnLayerWeights {
    pub input_norm: DeviceBuffer<u16>,  // bf16: (1+w) pattern, zeros init
    pub w_qkv: DeviceBuffer<u16>,      // bf16 [6144, 1024]
    pub w_a: DeviceBuffer<u16>,        // bf16 [16, 1024]
    pub w_b: DeviceBuffer<u16>,        // bf16 [16, 1024]
    pub w_z: DeviceBuffer<u16>,        // bf16 [2048, 1024]
    pub conv1d_weight: DeviceBuffer<u16>, // bf16 [6144, 4] (kept for traced path)
    pub conv1d_weight_q: DeviceBuffer<u16>, // bf16 [nh*kd, ck] pre-split Q slice
    pub conv1d_weight_k: DeviceBuffer<u16>, // bf16 [nh*kd, ck] pre-split K slice
    pub conv1d_weight_v: DeviceBuffer<u16>, // bf16 [nh*vd, ck] pre-split V slice
    pub a_log: DeviceBuffer<f32>,      // f32 (special: log space)
    pub dt_bias: DeviceBuffer<u16>,    // bf16 [16]
    pub output_norm: DeviceBuffer<f32>, // f32 [128] (QK-norm, (1+w) pattern)
    pub w_out: DeviceBuffer<u16>,      // bf16 [1024, 2048]
    pub post_norm: DeviceBuffer<u16>,  // bf16
    pub w_gate: DeviceBuffer<u16>,     // bf16 [3584, 1024]
    pub w_up: DeviceBuffer<u16>,       // bf16 [3584, 1024]
    pub w_down: DeviceBuffer<u16>,     // bf16 [1024, 3584]
}

pub struct AttentionLayerWeights {
    pub input_norm: DeviceBuffer<u16>,  // bf16: (1+w) pattern
    pub w_q_gate: DeviceBuffer<u16>,   // bf16 [4096, 1024]
    pub w_k: DeviceBuffer<u16>,        // bf16 [512, 1024]
    pub w_v: DeviceBuffer<u16>,        // bf16 [512, 1024]
    pub w_o: DeviceBuffer<u16>,        // bf16 [1024, 2048]
    pub q_norm: DeviceBuffer<u16>,     // bf16 [256] (QK-norm, (1+w) pattern)
    pub k_norm: DeviceBuffer<u16>,     // bf16 [256] (QK-norm, (1+w) pattern)
    pub post_norm: DeviceBuffer<u16>,  // bf16
    pub w_gate: DeviceBuffer<u16>,     // bf16
    pub w_up: DeviceBuffer<u16>,       // bf16
    pub w_down: DeviceBuffer<u16>,     // bf16
}

pub enum LayerWeights {
    Gdn(GdnLayerWeights),
    Attention(AttentionLayerWeights),
}

pub struct GdnState {
    pub recurrent: DeviceBuffer<f32>, // [16, 128, 128]
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
}

// SiluMulKernel needs to be imported (it's in kernel.rs but not listed in the use above)
use crate::kernel::SiluMulKernel;

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
        })
    }
}

// ---- Main model struct ----

pub struct Qwen35Model {
    pub(crate) config: ModelConfig,
    pub(crate) device: DeviceId,
    pub(crate) stream: Stream,
    kernels: AllKernels,
    pub(crate) embed_weight: DeviceBuffer<u16>,
    pub(crate) final_norm_weight: DeviceBuffer<u16>,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) activations: ActivationBuffers,
    pub(crate) gdn_conv_states: Vec<DeviceBuffer<f32>>, // [6144, 3] per GDN layer
    pub(crate) kv_caches: Vec<KvCache>,
    pub(crate) gdn_states: Vec<GdnState>,
    pub(crate) seq_len: u32,
    megakernel: Option<MegakernelProgram>,
    // Paged KV path (lazy-init)
    megakernel_paged: Option<MegakernelProgram>,
    page_allocator: Option<PageAllocator>,
    paged_seq: Option<SequenceState>,
    checkpoint_pool: Option<RecurrentCheckpointPool>,
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

/// Load a tensor as f32 from safetensors. For f32 on disk, zero-copy reinterpret.
/// For bf16 on disk, converts to f32 on the CPU.
fn load_weight(
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
        other => panic!("load_weight: unsupported dtype {other:?} for {name}"),
    };
    let mut buf = DeviceBuffer::<f32>::alloc(device, expected_len)?;
    buf.copy_from_host(&data)?;
    Ok(buf)
}

/// Load a tensor as bf16 (u16) from safetensors. For bf16 on disk, zero-copy
/// reinterpret of the mmap'd bytes — no allocation, direct hipMemcpy from mmap.
fn load_weight_bf16(
    st: &SafeTensorSet,
    name: &str,
    device: DeviceId,
    expected_len: usize,
) -> Result<DeviceBuffer<u16>, ModelError> {
    let raw = st.tensor_data(name)
        .map_err(|_| ModelError::MissingWeight(name.to_string()))?;
    assert_eq!(
        raw.len(),
        expected_len * 2,
        "weight {name}: expected {} bytes, got {}",
        expected_len * 2,
        raw.len()
    );
    let mut buf = DeviceBuffer::<u16>::alloc(device, expected_len)?;
    // raw is &[u8] from mmap, reinterpret as &[u16] for copy_from_host
    let data: &[u16] = unsafe {
        std::slice::from_raw_parts(raw.as_ptr() as *const u16, expected_len)
    };
    buf.copy_from_host(data)?;
    Ok(buf)
}

// D2D memcpy: copy `count` u16 elements from src at offset src_off to dst at offset dst_off
unsafe fn d2d_copy_u16(
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
unsafe fn d2d_copy_f32(
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

fn compute_inv_freq(rope_dim: usize, rope_theta: f32) -> Vec<f32> {
    let num_pairs = rope_dim / 2;
    (0..num_pairs)
        .map(|i| {
            let exp = 2.0 * i as f32 / rope_dim as f32;
            1.0 / rope_theta.powf(exp)
        })
        .collect()
}

// ---- Model impl ----

impl Qwen35Model {
    /// Default max_seq_len cap for flat KV cache (limits VRAM usage).
    /// Override with `load_with_max_seq_len`. Paged KV grows dynamically.
    const DEFAULT_MAX_SEQ_LEN: usize = 8192;

    pub fn load(model_dir: &Path, device: DeviceId) -> Result<Self, ModelError> {
        Self::load_with_max_seq_len(model_dir, device, None)
    }

    pub fn load_with_max_seq_len(model_dir: &Path, device: DeviceId, max_seq_len: Option<usize>) -> Result<Self, ModelError> {
        let config_path = model_dir.join("config.json");
        let mut config = if config_path.exists() {
            ModelConfig::from_config_json(&config_path)
                .map_err(|e| ModelError::MissingWeight(format!("config.json: {e}")))?
        } else {
            ModelConfig::qwen35_0_8b()
        };
        // Cap max_seq_len: model may claim 262144 but flat KV can't afford that.
        // User override takes priority, otherwise cap at DEFAULT_MAX_SEQ_LEN.
        config.max_seq_len = max_seq_len.unwrap_or(config.max_seq_len.min(Self::DEFAULT_MAX_SEQ_LEN));
        let st = SafeTensorSet::open_directory(model_dir)?;
        let prefix = "model.language_model.";

        let stream = Stream::new(device)?;
        let kernels = AllKernels::load(device)?;

        // Global weights (BF16)
        let embed_weight = load_weight_bf16(
            &st,
            &format!("{prefix}embed_tokens.weight"),
            device,
            config.vocab_size * config.hidden_size,
        )?;
        let final_norm_weight = load_weight_bf16(
            &st,
            &format!("{prefix}norm.weight"),
            device,
            config.hidden_size,
        )?;

        // Per-layer weights
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("{prefix}layers.{i}.");
            if config.layer_is_attention[i] {
                let w = AttentionLayerWeights {
                    input_norm: load_weight_bf16(&st, &format!("{p}input_layernorm.weight"), device, config.hidden_size)?,
                    w_q_gate: load_weight_bf16(&st, &format!("{p}self_attn.q_proj.weight"), device, 4096 * config.hidden_size)?,
                    w_k: load_weight_bf16(&st, &format!("{p}self_attn.k_proj.weight"), device, 512 * config.hidden_size)?,
                    w_v: load_weight_bf16(&st, &format!("{p}self_attn.v_proj.weight"), device, 512 * config.hidden_size)?,
                    w_o: load_weight_bf16(&st, &format!("{p}self_attn.o_proj.weight"), device, config.hidden_size * 2048)?,
                    q_norm: load_weight_bf16(&st, &format!("{p}self_attn.q_norm.weight"), device, config.head_dim)?,
                    k_norm: load_weight_bf16(&st, &format!("{p}self_attn.k_norm.weight"), device, config.head_dim)?,
                    post_norm: load_weight_bf16(&st, &format!("{p}post_attention_layernorm.weight"), device, config.hidden_size)?,
                    w_gate: load_weight_bf16(&st, &format!("{p}mlp.gate_proj.weight"), device, config.intermediate_size * config.hidden_size)?,
                    w_up: load_weight_bf16(&st, &format!("{p}mlp.up_proj.weight"), device, config.intermediate_size * config.hidden_size)?,
                    w_down: load_weight_bf16(&st, &format!("{p}mlp.down_proj.weight"), device, config.hidden_size * config.intermediate_size)?,
                };
                layers.push(LayerWeights::Attention(w));
            } else {
                let nh = config.linear_num_heads;
                let kd = config.linear_key_head_dim;
                let vd = config.linear_value_head_dim;
                let qkv_out = nh * kd + nh * kd + nh * vd; // 2048+2048+2048=6144
                let z_out = nh * vd; // 2048
                let ck = config.linear_conv_kernel_dim;
                let q_dim = nh * kd;
                let v_dim = nh * vd;
                let conv_total = qkv_out * ck;
                let conv_name = format!("{p}linear_attn.conv1d.weight");
                let conv_raw_bytes = st.tensor_data(&conv_name)
                    .map_err(|_| ModelError::MissingWeight(conv_name.clone()))?;
                assert_eq!(conv_raw_bytes.len(), conv_total * 2);
                let conv_raw: &[u16] = unsafe {
                    std::slice::from_raw_parts(conv_raw_bytes.as_ptr() as *const u16, conv_total)
                };
                let mut conv1d_weight_buf = DeviceBuffer::<u16>::alloc(device, conv_total)?;
                conv1d_weight_buf.copy_from_host(conv_raw)?;
                let mut conv_w_q_buf = DeviceBuffer::<u16>::alloc(device, q_dim * ck)?;
                let mut conv_w_k_buf = DeviceBuffer::<u16>::alloc(device, q_dim * ck)?;
                let mut conv_w_v_buf = DeviceBuffer::<u16>::alloc(device, v_dim * ck)?;
                conv_w_q_buf.copy_from_host(&conv_raw[..q_dim * ck])?;
                conv_w_k_buf.copy_from_host(&conv_raw[q_dim * ck..2 * q_dim * ck])?;
                conv_w_v_buf.copy_from_host(&conv_raw[2 * q_dim * ck..])?;
                let w = GdnLayerWeights {
                    input_norm: load_weight_bf16(&st, &format!("{p}input_layernorm.weight"), device, config.hidden_size)?,
                    w_qkv: load_weight_bf16(&st, &format!("{p}linear_attn.in_proj_qkv.weight"), device, qkv_out * config.hidden_size)?,
                    w_a: load_weight_bf16(&st, &format!("{p}linear_attn.in_proj_a.weight"), device, nh * config.hidden_size)?,
                    w_b: load_weight_bf16(&st, &format!("{p}linear_attn.in_proj_b.weight"), device, nh * config.hidden_size)?,
                    w_z: load_weight_bf16(&st, &format!("{p}linear_attn.in_proj_z.weight"), device, z_out * config.hidden_size)?,
                    conv1d_weight: conv1d_weight_buf,
                    conv1d_weight_q: conv_w_q_buf,
                    conv1d_weight_k: conv_w_k_buf,
                    conv1d_weight_v: conv_w_v_buf,
                    a_log: load_weight(&st, &format!("{p}linear_attn.A_log"), device, nh)?,  // f32
                    dt_bias: load_weight_bf16(&st, &format!("{p}linear_attn.dt_bias"), device, nh)?,
                    output_norm: load_weight(&st, &format!("{p}linear_attn.norm.weight"), device, kd)?,  // f32
                    w_out: load_weight_bf16(&st, &format!("{p}linear_attn.out_proj.weight"), device, config.hidden_size * z_out)?,
                    post_norm: load_weight_bf16(&st, &format!("{p}post_attention_layernorm.weight"), device, config.hidden_size)?,
                    w_gate: load_weight_bf16(&st, &format!("{p}mlp.gate_proj.weight"), device, config.intermediate_size * config.hidden_size)?,
                    w_up: load_weight_bf16(&st, &format!("{p}mlp.up_proj.weight"), device, config.intermediate_size * config.hidden_size)?,
                    w_down: load_weight_bf16(&st, &format!("{p}mlp.down_proj.weight"), device, config.hidden_size * config.intermediate_size)?,
                };
                layers.push(LayerWeights::Gdn(w));
            }
        }

        // GDN states: [nh * kd * vd] = [16*128*128] = 262144 per GDN layer
        let nh = config.linear_num_heads;
        let kd = config.linear_key_head_dim;
        let vd = config.linear_value_head_dim;
        let ck = config.linear_conv_kernel_dim;
        let qkv_out = nh * kd * 2 + nh * vd; // 6144

        let mut gdn_states = Vec::new();
        let mut gdn_conv_states = Vec::new();
        for i in 0..config.num_layers {
            if !config.layer_is_attention[i] {
                let mut recurrent = DeviceBuffer::<f32>::alloc(device, nh * kd * vd)?;
                // zero-init
                let zeros = vec![0.0f32; nh * kd * vd];
                recurrent.copy_from_host(&zeros)?;
                gdn_states.push(GdnState { recurrent });

                let mut conv_state = DeviceBuffer::<f32>::alloc(device, qkv_out * (ck - 1))?;
                let zeros = vec![0.0f32; qkv_out * (ck - 1)];
                conv_state.copy_from_host(&zeros)?;
                gdn_conv_states.push(conv_state);
            }
        }

        // KV caches
        let kv_size = config.max_seq_len * config.num_kv_heads * config.head_dim;
        let zeros_kv = vec![0.0f32; kv_size];
        let mut kv_caches = Vec::new();
        for i in 0..config.num_layers {
            if config.layer_is_attention[i] {
                let mut k = DeviceBuffer::<f32>::alloc(device, kv_size)?;
                let mut v = DeviceBuffer::<f32>::alloc(device, kv_size)?;
                k.copy_from_host(&zeros_kv)?;
                v.copy_from_host(&zeros_kv)?;
                kv_caches.push(KvCache { k, v });
            }
        }

        // inv_freq
        let inv_freq_data = compute_inv_freq(config.rope_dim, config.rope_theta);
        let mut inv_freq_buf = DeviceBuffer::<f32>::alloc(device, inv_freq_data.len())?;
        inv_freq_buf.copy_from_host(&inv_freq_data)?;

        let mut pos_buf = DeviceBuffer::<i32>::alloc(device, 3)?;
        pos_buf.copy_from_host(&[0i32, 0, 0])?;

        let hs = config.hidden_size;
        let is = config.intermediate_size;
        let vs = config.vocab_size;
        let nqh = config.num_q_heads;
        let hd = config.head_dim;
        let nkh = config.num_kv_heads;

        let activations = ActivationBuffers {
            hidden: DeviceBuffer::<f32>::alloc(device, hs)?,
            normed: DeviceBuffer::<f32>::alloc(device, hs)?,
            qkv: DeviceBuffer::<f32>::alloc(device, qkv_out)?,
            q_gdn: DeviceBuffer::<f32>::alloc(device, nh * kd)?,
            k_gdn: DeviceBuffer::<f32>::alloc(device, nh * kd)?,
            v_gdn: DeviceBuffer::<f32>::alloc(device, nh * vd)?,
            a_proj: DeviceBuffer::<f32>::alloc(device, nh)?,
            b_proj: DeviceBuffer::<f32>::alloc(device, nh)?,
            z_proj: DeviceBuffer::<f32>::alloc(device, nh * vd)?,
            gate_gdn: DeviceBuffer::<f32>::alloc(device, nh)?,
            recurrent_out: DeviceBuffer::<f32>::alloc(device, nh * vd)?,
            normed_gated: DeviceBuffer::<f32>::alloc(device, nh * vd)?,
            out_proj: DeviceBuffer::<f32>::alloc(device, hs)?,
            q_gate_attn: DeviceBuffer::<f32>::alloc(device, nqh * hd * 2)?,
            q_attn: DeviceBuffer::<f32>::alloc(device, nqh * hd)?,
            gate_attn: DeviceBuffer::<f32>::alloc(device, nqh * hd)?,
            k_attn: DeviceBuffer::<f32>::alloc(device, nkh * hd)?,
            v_attn: DeviceBuffer::<f32>::alloc(device, nkh * hd)?,
            attn_out: DeviceBuffer::<f32>::alloc(device, nqh * hd)?,
            gated_out: DeviceBuffer::<f32>::alloc(device, nqh * hd)?,
            ffn_gate: DeviceBuffer::<f32>::alloc(device, is)?,
            ffn_up: DeviceBuffer::<f32>::alloc(device, is)?,
            ffn_act: DeviceBuffer::<f32>::alloc(device, is)?,
            ffn_down: DeviceBuffer::<f32>::alloc(device, hs)?,
            residual: DeviceBuffer::<f32>::alloc(device, hs)?,
            logits: DeviceBuffer::<f32>::alloc(device, vs)?,
            inv_freq: inv_freq_buf,
            position_ids: pos_buf,
            gdn_cs_q: DeviceBuffer::<f32>::alloc(device, nh * kd * (ck - 1))?,
            gdn_cs_k: DeviceBuffer::<f32>::alloc(device, nh * kd * (ck - 1))?,
            gdn_cs_v: DeviceBuffer::<f32>::alloc(device, nh * vd * (ck - 1))?,
            gdn_conv_out_q: DeviceBuffer::<f32>::alloc(device, nh * kd)?,
            gdn_conv_out_k: DeviceBuffer::<f32>::alloc(device, nh * kd)?,
            gdn_conv_out_v: DeviceBuffer::<f32>::alloc(device, nh * vd)?,
        };

        Ok(Qwen35Model {
            config,
            device,
            stream,
            kernels,
            embed_weight,
            final_norm_weight,
            layers,
            activations,
            gdn_conv_states,
            kv_caches,
            gdn_states,
            seq_len: 0,
            megakernel: None,
            megakernel_paged: None,
            page_allocator: None,
            paged_seq: None,
            checkpoint_pool: None,
        })
    }

    fn gdn_forward(
        &mut self,
        layer_idx: usize,
        gdn_idx: usize,
    ) -> HipResult<()> {
        let cfg = &self.config;
        let hs = cfg.hidden_size as u32;
        let nh = cfg.linear_num_heads as u32;
        let kd = cfg.linear_key_head_dim as u32;
        let vd = cfg.linear_value_head_dim as u32;
        let ck = cfg.linear_conv_kernel_dim as u32;
        let eps = cfg.rms_norm_eps;

        let weights = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => panic!("expected GDN layer at {layer_idx}"),
        };

        // 1. RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed,
            &self.activations.hidden,
            &weights.input_norm,
            1,
            hs,
            eps,
            &self.stream,
        )?;

        // 2. Project QKV [6144]
        self.kernels.linear_proj.forward(
            &mut self.activations.qkv,
            &weights.w_qkv,
            &self.activations.normed,
            nh * kd * 2 + nh * vd,
            hs,
            &self.stream,
        )?;

        // 3. Project a [nh], b [nh], z [nh*vd]
        self.kernels.linear_proj.forward(
            &mut self.activations.a_proj,
            &weights.w_a,
            &self.activations.normed,
            nh,
            hs,
            &self.stream,
        )?;
        self.kernels.linear_proj.forward(
            &mut self.activations.b_proj,
            &weights.w_b,
            &self.activations.normed,
            nh,
            hs,
            &self.stream,
        )?;
        self.kernels.linear_proj.forward(
            &mut self.activations.z_proj,
            &weights.w_z,
            &self.activations.normed,
            nh * vd,
            hs,
            &self.stream,
        )?;

        // 4. Causal conv1d: split qkv into q/k/v, run 3 depthwise convs using pre-split weights
        // D2D copy: qkv[0..2048] → q_gdn, qkv[2048..4096] → k_gdn, qkv[4096..6144] → v_gdn
        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &self.activations.qkv, 0, nh as usize * kd as usize, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &self.activations.qkv, nh as usize * kd as usize, nh as usize * kd as usize, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &self.activations.qkv, nh as usize * kd as usize * 2, nh as usize * vd as usize, &self.stream)?;
        }

        let conv_q_out_len = nh as usize * kd as usize;
        let conv_k_out_len = nh as usize * kd as usize;
        let conv_v_out_len = nh as usize * vd as usize;
        let ck_usize = ck as usize;

        // Split conv state into q/k/v sub-states
        // gdn_conv_states[gdn_idx] is [6144, ck-1] = [6144 * (ck-1)].
        // Split into 3 sub-states: q=[2048,ck-1], k=[2048,ck-1], v=[2048,ck-1].
        let conv_state_q_len = conv_q_out_len * (ck_usize - 1);
        let conv_state_k_len = conv_k_out_len * (ck_usize - 1);
        let conv_state_v_len = conv_v_out_len * (ck_usize - 1);

        unsafe {
            d2d_copy_f32(&mut self.activations.gdn_cs_q, 0, &self.gdn_conv_states[gdn_idx], 0, conv_state_q_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.gdn_cs_k, 0, &self.gdn_conv_states[gdn_idx], conv_state_q_len, conv_state_k_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.gdn_cs_v, 0, &self.gdn_conv_states[gdn_idx], conv_state_q_len + conv_state_k_len, conv_state_v_len, &self.stream)?;
        }

        // Run 3 conv1d operations using pre-split weight buffers from the layer
        let (conv_w_q_ptr, conv_w_k_ptr, conv_w_v_ptr) = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => (
                &w.conv1d_weight_q as *const DeviceBuffer<u16>,
                &w.conv1d_weight_k as *const DeviceBuffer<u16>,
                &w.conv1d_weight_v as *const DeviceBuffer<u16>,
            ),
            _ => unreachable!(),
        };
        unsafe {
            self.kernels.causal_conv1d.forward(
                &mut self.activations.gdn_cs_q,
                &self.activations.q_gdn,
                &*conv_w_q_ptr,
                &mut self.activations.gdn_conv_out_q,
                conv_q_out_len as u32,
                ck,
                &self.stream,
            )?;
            self.kernels.causal_conv1d.forward(
                &mut self.activations.gdn_cs_k,
                &self.activations.k_gdn,
                &*conv_w_k_ptr,
                &mut self.activations.gdn_conv_out_k,
                conv_k_out_len as u32,
                ck,
                &self.stream,
            )?;
            self.kernels.causal_conv1d.forward(
                &mut self.activations.gdn_cs_v,
                &self.activations.v_gdn,
                &*conv_w_v_ptr,
                &mut self.activations.gdn_conv_out_v,
                conv_v_out_len as u32,
                ck,
                &self.stream,
            )?;
        }

        // Write back updated conv states
        unsafe {
            d2d_copy_f32(&mut self.gdn_conv_states[gdn_idx], 0, &self.activations.gdn_cs_q, 0, conv_state_q_len, &self.stream)?;
            d2d_copy_f32(&mut self.gdn_conv_states[gdn_idx], conv_state_q_len, &self.activations.gdn_cs_k, 0, conv_state_k_len, &self.stream)?;
            d2d_copy_f32(&mut self.gdn_conv_states[gdn_idx], conv_state_q_len + conv_state_k_len, &self.activations.gdn_cs_v, 0, conv_state_v_len, &self.stream)?;
        }

        // conv_out_q/k/v now hold the post-conv Q,K,V (with SiLU applied inside the kernel)
        // Copy them back to q_gdn, k_gdn, v_gdn
        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &self.activations.gdn_conv_out_q, 0, conv_q_out_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &self.activations.gdn_conv_out_k, 0, conv_k_out_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &self.activations.gdn_conv_out_v, 0, conv_v_out_len, &self.stream)?;
        }

        // 5. Compute GDN gate
        let weights_gdn = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => unreachable!(),
        };
        self.kernels.gdn_gate.forward(
            &mut self.activations.gate_gdn,
            &weights_gdn.a_log,
            &self.activations.a_proj,
            &weights_gdn.dt_bias,
            nh,
            &self.stream,
        )?;

        // 6. GDN recurrent step v2
        self.kernels.gdn_recurrent_v2.forward(
            &self.activations.q_gdn,
            &self.activations.k_gdn,
            &self.activations.v_gdn,
            &self.activations.gate_gdn,
            &self.activations.b_proj,
            &mut self.gdn_states[gdn_idx].recurrent,
            &mut self.activations.recurrent_out,
            nh,
            kd,
            vd,
            &self.stream,
        )?;

        // 7. RMSNorm gated (recurrent_out with z gate)
        let weights_gdn = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => unreachable!(),
        };
        self.kernels.rmsnorm_gated.forward(
            &mut self.activations.normed_gated,
            &self.activations.recurrent_out,
            &self.activations.z_proj,
            &weights_gdn.output_norm,
            nh,
            vd,
            eps,
            &self.stream,
        )?;

        // 8. Output projection [1024, 2048]
        let weights_gdn = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => w,
            _ => unreachable!(),
        };
        self.kernels.linear_proj.forward(
            &mut self.activations.out_proj,
            &weights_gdn.w_out,
            &self.activations.normed_gated,
            hs,
            nh * vd,
            &self.stream,
        )?;

        // 9. Residual add: hidden = out_proj + hidden
        // Copy hidden to residual first, then add
        unsafe {
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?;
        }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.out_proj,
            &self.activations.residual,
            hs,
            &self.stream,
        )?;

        // 10. FFN — extract raw pointers to avoid borrow conflict with &mut self
        let (post_norm_p, w_gate_p, w_up_p, w_down_p) = match &self.layers[layer_idx] {
            LayerWeights::Gdn(w) => (
                &w.post_norm as *const DeviceBuffer<u16>,
                &w.w_gate as *const DeviceBuffer<u16>,
                &w.w_up as *const DeviceBuffer<u16>,
                &w.w_down as *const DeviceBuffer<u16>,
            ),
            _ => unreachable!(),
        };
        unsafe { self.ffn_forward(&*post_norm_p, &*w_gate_p, &*w_up_p, &*w_down_p) }
    }

    fn ffn_forward(
        &mut self,
        post_norm: &DeviceBuffer<u16>,
        w_gate: &DeviceBuffer<u16>,
        w_up: &DeviceBuffer<u16>,
        w_down: &DeviceBuffer<u16>,
    ) -> HipResult<()> {
        let hs = self.config.hidden_size as u32;
        let is = self.config.intermediate_size as u32;
        let eps = self.config.rms_norm_eps;

        self.kernels.ffn_fused.forward_gate_up(
            &mut self.activations.ffn_act,
            &self.activations.hidden,
            post_norm,
            w_gate,
            w_up,
            hs,
            is,
            eps,
            &self.stream,
        )?;

        // Save residual
        unsafe {
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?;
        }
        self.kernels.ffn_fused.forward_down_residual(
            &mut self.activations.hidden,
            &self.activations.residual,
            w_down,
            &self.activations.ffn_act,
            hs,
            is,
            &self.stream,
        )
    }

    fn attention_forward(
        &mut self,
        layer_idx: usize,
        kv_cache_idx: usize,
        position: u32,
    ) -> HipResult<()> {
        let cfg = &self.config;
        let hs = cfg.hidden_size as u32;
        let nqh = cfg.num_q_heads as u32;
        let nkh = cfg.num_kv_heads as u32;
        let hd = cfg.head_dim as u32;
        let rd = cfg.rope_dim as u32;
        let s0 = cfg.mrope_section[0] as u32;
        let s1 = cfg.mrope_section[1] as u32;
        let s2 = cfg.mrope_section[2] as u32;
        let eps = cfg.rms_norm_eps;
        let max_sl = cfg.max_seq_len as u32;

        // 1. RMSNorm
        let input_norm = match &self.layers[layer_idx] {
            LayerWeights::Attention(w) => &w.input_norm as *const DeviceBuffer<u16>,
            _ => panic!("expected attention layer"),
        };
        unsafe {
            self.kernels.rmsnorm.forward(
                &mut self.activations.normed,
                &self.activations.hidden,
                &*input_norm,
                1,
                hs,
                eps,
                &self.stream,
            )?;
        }

        // 2. Project Q+gate [4096], K [512], V [512]
        let (w_q_gate, w_k, w_v, w_o_ptr, q_norm_w, k_norm_w, post_norm_w, w_gate_w, w_up_w, w_down_w) =
            match &self.layers[layer_idx] {
                LayerWeights::Attention(w) => (
                    &w.w_q_gate as *const DeviceBuffer<u16>,
                    &w.w_k as *const DeviceBuffer<u16>,
                    &w.w_v as *const DeviceBuffer<u16>,
                    &w.w_o as *const DeviceBuffer<u16>,
                    &w.q_norm as *const DeviceBuffer<u16>,
                    &w.k_norm as *const DeviceBuffer<u16>,
                    &w.post_norm as *const DeviceBuffer<u16>,
                    &w.w_gate as *const DeviceBuffer<u16>,
                    &w.w_up as *const DeviceBuffer<u16>,
                    &w.w_down as *const DeviceBuffer<u16>,
                ),
                _ => unreachable!(),
            };

        unsafe {
            self.kernels.linear_proj.forward(
                &mut self.activations.q_gate_attn,
                &*w_q_gate,
                &self.activations.normed,
                nqh * hd * 2,
                hs,
                &self.stream,
            )?;
            self.kernels.linear_proj.forward(
                &mut self.activations.k_attn,
                &*w_k,
                &self.activations.normed,
                nkh * hd,
                hs,
                &self.stream,
            )?;
            self.kernels.linear_proj.forward(
                &mut self.activations.v_attn,
                &*w_v,
                &self.activations.normed,
                nkh * hd,
                hs,
                &self.stream,
            )?;
        }

        // 3. Split q_gate_attn: interleaved [q0_hd, gate0_hd, q1_hd, gate1_hd, ...]
        //    → q=[nqh*hd], gate=[nqh*hd]
        let q_size = nqh as usize * hd as usize;
        let hd_usize = hd as usize;
        unsafe {
            for h in 0..nqh as usize {
                let src_q = h * hd_usize * 2;
                let src_g = h * hd_usize * 2 + hd_usize;
                let dst = h * hd_usize;
                d2d_copy_f32(&mut self.activations.q_attn, dst, &self.activations.q_gate_attn, src_q, hd_usize, &self.stream)?;
                d2d_copy_f32(&mut self.activations.gate_attn, dst, &self.activations.q_gate_attn, src_g, hd_usize, &self.stream)?;
            }
        }

        // 4. QK norm (in-place on q_attn, k_attn)
        unsafe {
            self.kernels.qk_norm.forward(
                &mut self.activations.q_attn,
                &mut self.activations.k_attn,
                &*q_norm_w,
                &*k_norm_w,
                nqh,
                nkh,
                hd,
                eps,
                &self.stream,
            )?;
        }

        // 5. Update position IDs and apply mRoPE
        let pos_data = [position as i32, position as i32, position as i32];
        self.activations.position_ids.copy_from_host(&pos_data)?;

        self.kernels.mrope.forward(
            &mut self.activations.q_attn,
            &mut self.activations.k_attn,
            &self.activations.inv_freq,
            &self.activations.position_ids,
            nqh,
            nkh,
            hd,
            rd,
            s0,
            s1,
            s2,
            &self.stream,
        )?;

        // 6. Write K,V to cache at position `position` ([H,T,D] layout)
        let max_sl = self.config.max_seq_len;
        for h in 0..nkh as usize {
            let src_off = h * hd as usize;
            let dst_off = h * max_sl * hd as usize + position as usize * hd as usize;
            unsafe {
                d2d_copy_f32(&mut self.kv_caches[kv_cache_idx].k, dst_off, &self.activations.k_attn, src_off, hd as usize, &self.stream)?;
                d2d_copy_f32(&mut self.kv_caches[kv_cache_idx].v, dst_off, &self.activations.v_attn, src_off, hd as usize, &self.stream)?;
            }
        }

        // 7. GQA attention
        let seq_len = position + 1;
        self.kernels.gqa_attention.forward(
            &mut self.activations.attn_out,
            &self.activations.q_attn,
            &self.kv_caches[kv_cache_idx].k,
            &self.kv_caches[kv_cache_idx].v,
            nqh,
            nkh,
            hd,
            seq_len,
            max_sl as u32,
            &self.stream,
        )?;

        // 8. Output gate: gated_out = attn_out * sigmoid(gate)
        self.kernels.output_gate.forward(
            &mut self.activations.gated_out,
            &self.activations.attn_out,
            &self.activations.gate_attn,
            nqh * hd,
            &self.stream,
        )?;

        // 9. Output projection [1024, 2048]
        unsafe {
            self.kernels.linear_proj.forward(
                &mut self.activations.out_proj,
                &*w_o_ptr,
                &self.activations.gated_out,
                hs,
                nqh * hd,
                &self.stream,
            )?;
        }

        // 10. Residual add
        unsafe {
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?;
        }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.out_proj,
            &self.activations.residual,
            hs,
            &self.stream,
        )?;

        // 11. FFN
        unsafe {
            self.ffn_forward(&*post_norm_w, &*w_gate_w, &*w_up_w, &*w_down_w)
        }
    }

    pub fn config(&self) -> &ModelConfig { &self.config }
    pub fn stream(&self) -> &Stream { &self.stream }
    pub fn vocab_size(&self) -> usize { self.config.vocab_size }

    pub fn set_position(&mut self, position: u32) -> HipResult<()> {
        let pos_data = [position as i32, position as i32, position as i32];
        self.activations.position_ids.copy_from_host(&pos_data)
    }

    pub fn read_logits(&self) -> Result<Vec<f32>, ModelError> {
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;
        Ok(logits)
    }

    /// Run a single decode step. Returns logits [vocab_size].
    pub fn decode_step(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        // Lazy-init megakernel on first call
        if self.megakernel.is_none() {
            self.megakernel = Some(MegakernelProgram::compile(self)?);
        }
        let mk = self.megakernel.as_mut().unwrap();
        mk.update_step(token_id, position, &self.stream)?;
        mk.execute(&self.stream)?;

        self.stream.synchronize()?;

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;

        self.seq_len = position + 1;
        Ok(logits)
    }

    /// Run a single decode step using the paged KV cache path.
    /// Returns logits [vocab_size].
    pub fn decode_step_paged(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        // Lazy-init: compile paged megakernel
        if self.megakernel_paged.is_none() {
            let mut mk = MegakernelProgram::compile_paged(self)?;
            // max_chunks: enough for max_seq_len tokens in CHUNK_TOKENS-sized chunks
            let max_chunks = (self.config.max_seq_len + CHUNK_TOKENS - 1) / CHUNK_TOKENS;
            mk.init_paged_buffers(max_chunks)?;
            self.megakernel_paged = Some(mk);
        }

        // Lazy-init: PageAllocator and SequenceState
        if self.page_allocator.is_none() {
            let max_chunks = (self.config.max_seq_len + CHUNK_TOKENS - 1) / CHUNK_TOKENS;
            self.page_allocator = Some(PageAllocator::new(
                self.device,
                &self.config,
                CHUNK_TOKENS,
                max_chunks as u32,
            )?);
            let seq = SequenceState::new(CHUNK_TOKENS as u32);
            self.paged_seq = Some(seq);
        }

        // append_token: allocates new chunk if needed + increments chunk len.
        // Must happen BEFORE update_step_paged so current_chunk_offset() is correct.
        {
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            seq_mut.append_token(alloc_mut)?;
        }

        let stream = &self.stream;
        let mk = self.megakernel_paged.as_mut().unwrap();
        let seq = self.paged_seq.as_ref().unwrap();
        let allocator = self.page_allocator.as_ref().unwrap();

        mk.update_step_paged(token_id, position, seq, allocator, stream)?;
        mk.execute(stream)?;
        stream.synchronize()?;

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;
        Ok(logits)
    }

    /// Save the current GDN recurrent states into a checkpoint pool slot.
    /// Lazy-initializes the pool on first call. Returns the slot index.
    pub fn save_recurrent_checkpoint(&mut self) -> Result<u32, ModelError> {
        if self.checkpoint_pool.is_none() {
            self.checkpoint_pool = Some(RecurrentCheckpointPool::new(
                self.device,
                &self.config,
                4,
            )?);
        }
        let recurrent_bufs: Vec<&DeviceBuffer<f32>> = self.gdn_states.iter().map(|s| &s.recurrent).collect();
        let pool = self.checkpoint_pool.as_mut().unwrap();
        let slot = paged_kv::save_checkpoint(pool, &recurrent_bufs, self.stream.raw())?;
        Ok(slot)
    }

    /// Process a sequence of tokens (prefill). Returns logits for the last token.
    /// Saves GDN checkpoints at each 64-token chunk boundary.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::MissingWeight("empty token sequence".into()));
        }
        let mut pos = 0u32;
        for chunk in tokens.chunks(CHUNK_TOKENS) {
            let mut bufs = PrefillBuffers::alloc(self.device, &self.config, chunk.len())?;
            let program = MegakernelProgram::compile_prefill(self, chunk, pos, &mut bufs)?;
            program.execute(&self.stream)?;
            self.stream.synchronize()?;
            pos += chunk.len() as u32;
            if pos < tokens.len() as u32 {
                let _slot = self.save_recurrent_checkpoint()?;
            }
        }
        self.read_logits()
    }

    /// Read all GDN recurrent state to host (for testing).
    pub fn read_gdn_state(&self) -> Result<Vec<Vec<f32>>, ModelError> {
        self.stream.synchronize()?;
        let mut result = Vec::with_capacity(self.gdn_states.len());
        for state in &self.gdn_states {
            let n = state.recurrent.len();
            let mut buf = vec![0.0f32; n];
            state.recurrent.copy_to_host(&mut buf)?;
            result.push(buf);
        }
        Ok(result)
    }

    /// Restore GDN recurrent states from a previously saved checkpoint slot.
    pub fn restore_recurrent_checkpoint(&mut self, slot: u32) -> Result<(), ModelError> {
        let pool = self.checkpoint_pool.as_ref()
            .ok_or_else(|| ModelError::MissingWeight("checkpoint_pool not initialized".into()))?;
        let mut recurrent_bufs: Vec<&mut DeviceBuffer<f32>> = self.gdn_states.iter_mut().map(|s| &mut s.recurrent).collect();
        let stream_raw = self.stream.raw();
        paged_kv::restore_checkpoint(pool, slot, &mut recurrent_bufs, stream_raw)?;
        self.stream.synchronize()?;
        Ok(())
    }

    fn read_hidden(&self) -> Result<Vec<f32>, ModelError> {
        self.stream.synchronize()?;
        let mut buf = vec![0.0f32; self.config.hidden_size];
        self.activations.hidden.copy_to_host(&mut buf)?;
        Ok(buf)
    }

    pub fn decode_step_traced(&mut self, token_id: u32, position: u32) -> Result<(Vec<f32>, Vec<(String, Vec<f32>)>), ModelError> {
        let hs = self.config.hidden_size as u32;
        let vs = self.config.vocab_size as u32;
        let mut traces: Vec<(String, Vec<f32>)> = Vec::new();

        self.kernels.embedding.forward(
            &mut self.activations.hidden, &self.embed_weight,
            token_id as i32, hs, &self.stream,
        )?;
        traces.push(("embed".into(), self.read_hidden()?));

        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        for i in 0..self.config.num_layers {
            if self.config.layer_is_attention[i] {
                self.attention_forward(i, kv_idx, position)?;
                kv_idx += 1;
            } else {
                self.gdn_forward(i, gdn_idx)?;
                gdn_idx += 1;
            }
            traces.push((format!("layer_{i}"), self.read_hidden()?));
        }

        unsafe { d2d_copy_f32(&mut self.activations.normed, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?; }
        self.kernels.rmsnorm.forward(
            &mut self.activations.hidden, &self.activations.normed,
            &self.final_norm_weight, 1, hs, self.config.rms_norm_eps, &self.stream,
        )?;
        traces.push(("final_norm".into(), self.read_hidden()?));

        self.kernels.lm_head.forward(
            &mut self.activations.logits, &self.embed_weight,
            &self.activations.hidden, vs, hs, &self.stream,
        )?;
        self.stream.synchronize()?;
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;
        self.seq_len = position + 1;
        Ok((logits, traces))
    }

    fn read_buf(&self, buf: &DeviceBuffer<f32>) -> Result<Vec<f32>, ModelError> {
        self.stream.synchronize()?;
        let mut v = vec![0.0f32; buf.len()];
        buf.copy_to_host(&mut v)?;
        Ok(v)
    }

    pub fn gdn_layer0_trace(&mut self, token_id: u32) -> Result<Vec<(String, Vec<f32>)>, ModelError> {
        let hs = self.config.hidden_size as u32;
        let nh = self.config.linear_num_heads as u32;
        let kd = self.config.linear_key_head_dim as u32;
        let vd = self.config.linear_value_head_dim as u32;
        let ck = self.config.linear_conv_kernel_dim as u32;
        let eps = self.config.rms_norm_eps;
        let mut traces: Vec<(String, Vec<f32>)> = Vec::new();

        // Embedding
        self.kernels.embedding.forward(
            &mut self.activations.hidden, &self.embed_weight,
            token_id as i32, hs, &self.stream,
        )?;
        traces.push(("embed".into(), self.read_hidden()?));

        let weights = match &self.layers[0] {
            LayerWeights::Gdn(w) => w as *const GdnLayerWeights,
            _ => panic!("layer 0 not GDN"),
        };
        let w = unsafe { &*weights };

        // RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed, &self.activations.hidden,
            &w.input_norm, 1, hs, eps, &self.stream,
        )?;
        traces.push(("normed".into(), self.read_buf(&self.activations.normed)?));

        // QKV projection
        self.kernels.linear_proj.forward(
            &mut self.activations.qkv, &w.w_qkv, &self.activations.normed,
            nh * kd * 2 + nh * vd, hs, &self.stream,
        )?;
        traces.push(("qkv_pre_conv".into(), self.read_buf(&self.activations.qkv)?));

        // a, b, z projections
        self.kernels.linear_proj.forward(
            &mut self.activations.a_proj, &w.w_a, &self.activations.normed,
            nh, hs, &self.stream,
        )?;
        self.kernels.linear_proj.forward(
            &mut self.activations.b_proj, &w.w_b, &self.activations.normed,
            nh, hs, &self.stream,
        )?;
        self.kernels.linear_proj.forward(
            &mut self.activations.z_proj, &w.w_z, &self.activations.normed,
            nh * vd, hs, &self.stream,
        )?;
        traces.push(("a_proj".into(), self.read_buf(&self.activations.a_proj)?));
        traces.push(("b_proj".into(), self.read_buf(&self.activations.b_proj)?));
        traces.push(("z_proj".into(), self.read_buf(&self.activations.z_proj)?));

        // Conv1d: split qkv, run 3 separate convs, reassemble
        let conv_q_len = (nh * kd) as usize;
        let conv_k_len = (nh * kd) as usize;
        let conv_v_len = (nh * vd) as usize;
        let ck_usize = ck as usize;

        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &self.activations.qkv, 0, conv_q_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &self.activations.qkv, conv_q_len, conv_k_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &self.activations.qkv, conv_q_len + conv_k_len, conv_v_len, &self.stream)?;
        }

        let mut conv_w_q = DeviceBuffer::<u16>::alloc(self.device, conv_q_len * ck_usize)?;
        let mut conv_w_k = DeviceBuffer::<u16>::alloc(self.device, conv_k_len * ck_usize)?;
        let mut conv_w_v = DeviceBuffer::<u16>::alloc(self.device, conv_v_len * ck_usize)?;
        unsafe {
            d2d_copy_u16(&mut conv_w_q, 0, &w.conv1d_weight, 0, conv_q_len * ck_usize, &self.stream)?;
            d2d_copy_u16(&mut conv_w_k, 0, &w.conv1d_weight, conv_q_len * ck_usize, conv_k_len * ck_usize, &self.stream)?;
            d2d_copy_u16(&mut conv_w_v, 0, &w.conv1d_weight, (conv_q_len + conv_k_len) * ck_usize, conv_v_len * ck_usize, &self.stream)?;
        }

        let conv_state_q_len = conv_q_len * (ck_usize - 1);
        let conv_state_k_len = conv_k_len * (ck_usize - 1);
        let conv_state_v_len = conv_v_len * (ck_usize - 1);

        let mut cs_q = DeviceBuffer::<f32>::alloc(self.device, conv_state_q_len)?;
        let mut cs_k = DeviceBuffer::<f32>::alloc(self.device, conv_state_k_len)?;
        let mut cs_v = DeviceBuffer::<f32>::alloc(self.device, conv_state_v_len)?;
        unsafe {
            d2d_copy_f32(&mut cs_q, 0, &self.gdn_conv_states[0], 0, conv_state_q_len, &self.stream)?;
            d2d_copy_f32(&mut cs_k, 0, &self.gdn_conv_states[0], conv_state_q_len, conv_state_k_len, &self.stream)?;
            d2d_copy_f32(&mut cs_v, 0, &self.gdn_conv_states[0], conv_state_q_len + conv_state_k_len, conv_state_v_len, &self.stream)?;
        }

        let mut conv_out_q = DeviceBuffer::<f32>::alloc(self.device, conv_q_len)?;
        let mut conv_out_k = DeviceBuffer::<f32>::alloc(self.device, conv_k_len)?;
        let mut conv_out_v = DeviceBuffer::<f32>::alloc(self.device, conv_v_len)?;

        self.kernels.causal_conv1d.forward(&mut cs_q, &self.activations.q_gdn, &conv_w_q, &mut conv_out_q, conv_q_len as u32, ck, &self.stream)?;
        self.kernels.causal_conv1d.forward(&mut cs_k, &self.activations.k_gdn, &conv_w_k, &mut conv_out_k, conv_k_len as u32, ck, &self.stream)?;
        self.kernels.causal_conv1d.forward(&mut cs_v, &self.activations.v_gdn, &conv_w_v, &mut conv_out_v, conv_v_len as u32, ck, &self.stream)?;

        traces.push(("conv_out_q".into(), self.read_buf(&conv_out_q)?));
        traces.push(("conv_out_k".into(), self.read_buf(&conv_out_k)?));
        traces.push(("conv_out_v".into(), self.read_buf(&conv_out_v)?));

        // Copy conv outputs to q/k/v
        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &conv_out_q, 0, conv_q_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &conv_out_k, 0, conv_k_len, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &conv_out_v, 0, conv_v_len, &self.stream)?;
        }

        // Gate
        self.kernels.gdn_gate.forward(
            &mut self.activations.gate_gdn, &w.a_log, &self.activations.a_proj,
            &w.dt_bias, nh, &self.stream,
        )?;
        traces.push(("gate".into(), self.read_buf(&self.activations.gate_gdn)?));

        // Recurrent
        self.kernels.gdn_recurrent_v2.forward(
            &self.activations.q_gdn, &self.activations.k_gdn, &self.activations.v_gdn,
            &self.activations.gate_gdn, &self.activations.b_proj,
            &mut self.gdn_states[0].recurrent, &mut self.activations.recurrent_out,
            nh, kd, vd, &self.stream,
        )?;
        traces.push(("recurrent_out".into(), self.read_buf(&self.activations.recurrent_out)?));

        // RMSNormGated
        self.kernels.rmsnorm_gated.forward(
            &mut self.activations.normed_gated, &self.activations.recurrent_out,
            &self.activations.z_proj, &w.output_norm, nh, vd, eps, &self.stream,
        )?;
        traces.push(("normed_gated".into(), self.read_buf(&self.activations.normed_gated)?));

        // out_proj
        self.kernels.linear_proj.forward(
            &mut self.activations.out_proj, &w.w_out, &self.activations.normed_gated,
            hs, nh * vd, &self.stream,
        )?;
        traces.push(("out_proj".into(), self.read_buf(&self.activations.out_proj)?));

        // Residual
        unsafe { d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize, &self.stream)?; }
        self.kernels.residual_add.forward(
            &mut self.activations.hidden, &self.activations.out_proj,
            &self.activations.residual, hs, &self.stream,
        )?;
        traces.push(("after_residual".into(), self.read_hidden()?));

        Ok(traces)
    }

    pub fn reset_state(&mut self) -> Result<(), ModelError> {
        let nh = self.config.linear_num_heads;
        let kd = self.config.linear_key_head_dim;
        let vd = self.config.linear_value_head_dim;
        let ck = self.config.linear_conv_kernel_dim;
        let qkv_out = nh * kd * 2 + nh * vd;

        for state in &mut self.gdn_states {
            let zeros = vec![0.0f32; nh * kd * vd];
            state.recurrent.copy_from_host(&zeros)?;
        }
        for conv_state in &mut self.gdn_conv_states {
            let zeros = vec![0.0f32; qkv_out * (ck - 1)];
            conv_state.copy_from_host(&zeros)?;
        }
        let kv_size = self.config.max_seq_len * self.config.num_kv_heads * self.config.head_dim;
        let zeros_kv = vec![0.0f32; kv_size];
        for cache in &mut self.kv_caches {
            cache.k.copy_from_host(&zeros_kv)?;
            cache.v.copy_from_host(&zeros_kv)?;
        }
        self.seq_len = 0;
        Ok(())
    }
}
