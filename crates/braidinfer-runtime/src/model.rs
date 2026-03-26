use std::path::Path;

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

#[derive(Debug, Clone, PartialEq)]
pub enum LayerType {
    Attention,
    Gdn,
    Mamba2,
    LfmConv,
}

#[derive(Debug, Clone)]
pub enum GateType {
    Softmax,
    NormTopK { routed_scaling_factor: f32 },
}

#[derive(Debug, Clone)]
pub enum FfnType {
    Dense,
    MoE {
        num_experts: usize,
        num_active: usize,
        num_shared: usize,
        expert_intermediate_size: usize,
        shared_intermediate_size: usize,
        gate_type: GateType,
    },
}

#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub layer_type: LayerType,
    pub ffn_type: FfnType,
}

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
    pub linear_num_heads: usize,       // num_key_heads for GDN
    pub linear_num_value_heads: usize,  // may differ from linear_num_heads (e.g. 4B: 32 vs 16)
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    // Per-layer config: layer type + FFN type
    pub layers: Vec<LayerConfig>,
    // Legacy compat (derived from layers)
    pub layer_is_attention: Vec<bool>,
    pub max_seq_len: usize,
    // MoE config (global defaults, may be overridden per-layer)
    pub num_experts: usize,
    pub num_active_experts: usize,
    pub num_shared_experts: usize,
    pub expert_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    // Extended config
    pub recurrent_kind: RecurrentLayerKind,
    pub rope_type: RopeType,
    pub has_qk_norm: bool,
    pub has_output_gate: bool,  // Qwen3.5 interleaves Q+gate; others don't
    pub attention_layer_indices: Vec<usize>,
    pub model_type: String,
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    pub fn qwen35_0_8b() -> Self {
        let attention_layer_indices: Vec<usize> = vec![3, 7, 11, 15, 19, 23];
        let ffn = FfnType::Dense;
        let layers: Vec<LayerConfig> = (0..24).map(|i| LayerConfig {
            layer_type: if attention_layer_indices.contains(&i) { LayerType::Attention } else { LayerType::Gdn },
            ffn_type: ffn.clone(),
        }).collect();
        let layer_is_attention: Vec<bool> = layers.iter().map(|l| l.layer_type == LayerType::Attention).collect();
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
            linear_num_value_heads: 16,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            layers,
            layer_is_attention,
            max_seq_len: 2048,
            num_experts: 0,
            num_active_experts: 0,
            num_shared_experts: 0,
            expert_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            recurrent_kind: RecurrentLayerKind::Gdn {
                num_heads: 16,
                key_value_dim: 128,
                conv_dim: 6144,
                kernel_size: 4,
            },
            rope_type: RopeType::MRope { sections: [11, 11, 10] },
            has_qk_norm: false,
            has_output_gate: false,
            attention_layer_indices,
            model_type: "qwen3_5".to_string(),
            tie_word_embeddings: true, // 0.8B fallback for tests
        }
    }

    pub fn from_config_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&data)?;

        // Flatten: text_config fields override top-level
        let tc = v.get("text_config");
        let get = |key: &str| -> Option<&serde_json::Value> {
            tc.and_then(|t| t.get(key)).or_else(|| v.get(key))
        };
        let get_usize = |key: &str| -> Option<usize> {
            get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
        };
        let get_f64 = |key: &str| -> Option<f64> {
            get(key).and_then(|v| v.as_f64())
        };
        let get_bool = |key: &str| -> Option<bool> {
            get(key).and_then(|v| v.as_bool())
        };
        let get_str = |key: &str| -> Option<String> {
            get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
        };

        let model_type = get_str("model_type").unwrap_or_default();
        let hidden_size = get_usize("hidden_size")
            .or(get_usize("n_embd"))
            .or(get_usize("d_model"))
            .ok_or("missing hidden_size")?;
        let num_layers = get_usize("num_hidden_layers")
            .or(get_usize("n_layer"))
            .or_else(|| get("layers_block_type").and_then(|v| v.as_array()).map(|a| a.len()))
            .or_else(|| get_str("hybrid_override_pattern").map(|p| p.len()))
            .ok_or("missing num_hidden_layers")?;
        let vocab_size = get_usize("vocab_size")
            .or(get_usize("n_vocab"))
            .ok_or("missing vocab_size")?;
        let num_q_heads = get_usize("num_attention_heads")
            .or(get_usize("n_head"))
            .unwrap_or(1);  // pure recurrent models (falcon_mamba) have no attention heads
        let num_kv_heads = get_usize("num_key_value_heads").unwrap_or(num_q_heads);
        let head_dim = get_usize("head_dim").unwrap_or(hidden_size / num_q_heads);
        let intermediate_size = get_usize("intermediate_size").unwrap_or(0);
        let rms_norm_eps = get_f64("rms_norm_eps").or(get_f64("norm_eps")).or(get_f64("layer_norm_epsilon")).unwrap_or(1e-5) as f32;
        let rope_theta = get_f64("rope_theta")
            .or_else(|| get("rope_parameters").and_then(|rp| rp.get("rope_theta")).and_then(|v| v.as_f64()))
            .unwrap_or(10000.0) as f32;
        let partial_rotary_factor = get_f64("partial_rotary_factor")
            .or_else(|| get("rope_parameters").and_then(|rp| rp.get("partial_rotary_factor")).and_then(|v| v.as_f64()))
            .unwrap_or(1.0);
        let rope_dim = ((head_dim as f64) * partial_rotary_factor) as usize;
        let max_seq_len = get_usize("max_position_embeddings").unwrap_or(2048);
        let tie_word_embeddings = get_bool("tie_word_embeddings").unwrap_or(true);

        // mRoPE
        let mrope_section = get("rope_parameters")
            .and_then(|rp| rp.get("mrope_section"))
            .and_then(|v| v.as_array())
            .and_then(|a| if a.len() == 3 {
                Some([a[0].as_u64()? as usize, a[1].as_u64()? as usize, a[2].as_u64()? as usize])
            } else { None })
            .unwrap_or([0, 0, 0]);
        let rope_type = if mrope_section != [0, 0, 0] {
            RopeType::MRope { sections: mrope_section }
        } else {
            RopeType::Standard { rotary_dim: rope_dim }
        };

        // GDN / recurrent config
        let linear_num_heads = get_usize("linear_num_key_heads").unwrap_or(0);
        let linear_num_value_heads = get_usize("linear_num_value_heads").unwrap_or(linear_num_heads);
        let linear_key_head_dim = get_usize("linear_key_head_dim").unwrap_or(128);
        let linear_value_head_dim = get_usize("linear_value_head_dim").unwrap_or(128);
        let linear_conv_kernel_dim = get_usize("linear_conv_kernel_dim")
            .or(get_usize("conv_kernel")).unwrap_or(4);

        // Mamba2 config
        let ssm_state_size = get_usize("ssm_state_size");
        let mamba_num_heads = get_usize("mamba_num_heads");
        let mamba_head_dim = get_usize("mamba_head_dim");

        let recurrent_kind = if linear_num_heads > 0 {
            let conv_dim = linear_num_heads * (linear_key_head_dim + linear_value_head_dim);
            RecurrentLayerKind::Gdn {
                num_heads: linear_num_heads, key_value_dim: linear_key_head_dim,
                conv_dim, kernel_size: linear_conv_kernel_dim,
            }
        } else if let (Some(sd), Some(nh), Some(hd)) = (ssm_state_size, mamba_num_heads, mamba_head_dim) {
            RecurrentLayerKind::Mamba2 {
                state_dim: sd, num_heads: nh, head_dim: hd,
                conv_kernel: linear_conv_kernel_dim,
            }
        } else {
            RecurrentLayerKind::None
        };

        // MoE config
        let gate_type = if get_bool("norm_topk_prob").unwrap_or(false) {
            GateType::NormTopK { routed_scaling_factor: get_f64("routed_scaling_factor").unwrap_or(1.0) as f32 }
        } else { GateType::Softmax };
        let num_experts = get_usize("num_experts")
            .or(get_usize("n_routed_experts"))
            .or(get_usize("num_local_experts"))
            .unwrap_or(0);
        let num_active_experts = get_usize("num_experts_per_tok")
            .or(get_usize("num_selected_experts"))
            .unwrap_or(0);
        let num_shared_experts = get_usize("n_shared_experts").unwrap_or(0);
        let expert_intermediate_size = get_usize("moe_intermediate_size").unwrap_or(intermediate_size);
        let shared_expert_intermediate_size = get_usize("moe_shared_expert_intermediate_size")
            .or(get_usize("shared_expert_intermediate_size"))
            .unwrap_or(0);

        let moe_ffn = if num_experts > 0 {
            FfnType::MoE {
                num_experts, num_active: num_active_experts, num_shared: num_shared_experts,
                expert_intermediate_size, shared_intermediate_size: shared_expert_intermediate_size,
                gate_type: gate_type.clone(),
            }
        } else { FfnType::Dense };
        let dense_ffn = FfnType::Dense;

        // Layer pattern detection
        let layers: Vec<LayerConfig> = if let Some(lt) = get("layer_types").and_then(|v| v.as_array()) {
            lt.iter().map(|t| {
                let ts = t.as_str().unwrap_or("");
                let layer_type = match ts {
                    "full_attention" | "attention" => LayerType::Attention,
                    "linear_attention" => LayerType::Gdn,
                    "conv" => LayerType::LfmConv,
                    "sliding_attention" => LayerType::Attention,
                    _ => LayerType::Attention,
                };
                let ffn = if num_experts > 0 { moe_ffn.clone() } else { dense_ffn.clone() };
                LayerConfig { layer_type, ffn_type: ffn }
            }).collect()
        } else if let Some(pattern) = get_str("hybrid_override_pattern") {
            pattern.chars().map(|c| match c {
                'M' => LayerConfig { layer_type: LayerType::Mamba2, ffn_type: dense_ffn.clone() },
                'E' => LayerConfig { layer_type: LayerType::Mamba2, ffn_type: moe_ffn.clone() },
                '*' => LayerConfig { layer_type: LayerType::Attention, ffn_type: dense_ffn.clone() },
                _ => LayerConfig { layer_type: LayerType::Attention, ffn_type: dense_ffn.clone() },
            }).collect()
        } else if let Some(lbt) = get("layers_block_type").and_then(|v| v.as_array()) {
            lbt.iter().map(|t| {
                let ts = t.as_str().unwrap_or("");
                match ts {
                    "mamba" => LayerConfig { layer_type: LayerType::Mamba2, ffn_type: dense_ffn.clone() },
                    "moe" => LayerConfig { layer_type: LayerType::Mamba2, ffn_type: moe_ffn.clone() },
                    "attention" => LayerConfig { layer_type: LayerType::Attention, ffn_type: dense_ffn.clone() },
                    _ => LayerConfig { layer_type: LayerType::Attention, ffn_type: dense_ffn.clone() },
                }
            }).collect()
        } else if let Some(interval) = get_usize("full_attention_interval") {
            (0..num_layers).map(|i| {
                let lt = if (i + 1) % interval == 0 { LayerType::Attention } else { LayerType::Gdn };
                let ffn = if num_experts > 0 { moe_ffn.clone() } else { dense_ffn.clone() };
                LayerConfig { layer_type: lt, ffn_type: ffn }
            }).collect()
        } else {
            // Default: all attention, MoE if experts detected
            let ffn = if num_experts > 0 { moe_ffn.clone() } else { dense_ffn.clone() };
            (0..num_layers).map(|_| LayerConfig { layer_type: LayerType::Attention, ffn_type: ffn.clone() }).collect()
        };

        let layer_is_attention: Vec<bool> = layers.iter().map(|l| l.layer_type == LayerType::Attention).collect();
        let attention_layer_indices: Vec<usize> = layers.iter().enumerate()
            .filter(|(_, l)| l.layer_type == LayerType::Attention)
            .map(|(i, _)| i).collect();

        let intermediate_size = if intermediate_size == 0 { expert_intermediate_size } else { intermediate_size };

        Ok(ModelConfig {
            hidden_size, num_layers, intermediate_size, vocab_size,
            num_q_heads, num_kv_heads, head_dim, rope_dim, rope_theta, rms_norm_eps,
            mrope_section,
            linear_num_heads, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim, linear_conv_kernel_dim,
            layers, layer_is_attention, max_seq_len,
            num_experts, num_active_experts, num_shared_experts,
            expert_intermediate_size, shared_expert_intermediate_size,
            recurrent_kind, rope_type,
            has_qk_norm: false, has_output_gate: false, // auto-detected from tensor names at load time
            attention_layer_indices, model_type, tie_word_embeddings,
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

/// Combined layer: recurrent/attention weights + FFN weights (dense or MoE)
pub struct FullLayerWeights {
    pub layer: LayerWeights,
    pub ffn: FfnWeights,
}

/// Dense FFN weights (gate_proj + up_proj + down_proj)
pub struct DenseFfnWeights {
    pub gate_proj: DeviceBuffer<u16>,
    pub up_proj: DeviceBuffer<u16>,
    pub down_proj: DeviceBuffer<u16>,
}

/// MoE FFN weights for one layer
pub struct MoeWeights {
    pub gate: DeviceBuffer<u16>,                    // [num_experts, hidden_size] — router
    pub expert_gate_up: DeviceBuffer<u16>,           // [num_experts, 2*expert_is, hidden_size] fused
    pub expert_down: DeviceBuffer<u16>,              // [num_experts, hidden_size, expert_is]
    pub shared_expert: Option<DenseFfnWeights>,      // always-on shared expert
    pub shared_expert_gate: Option<DeviceBuffer<u16>>, // [1, hidden_size] gate for shared expert
    pub num_experts: usize,
    pub expert_intermediate_size: usize,
}

/// Per-layer FFN weights: either dense or MoE
pub enum FfnWeights {
    Dense(DenseFfnWeights),
    MoE(MoeWeights),
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

pub struct Model {
    pub(crate) config: ModelConfig,
    pub(crate) device: DeviceId,
    pub(crate) stream: Stream,
    kernels: AllKernels,
    pub(crate) embed_weight: DeviceBuffer<u16>,
    pub(crate) lm_head_weight: DeviceBuffer<u16>,  // separate from embed when tie_word_embeddings=false
    pub(crate) final_norm_weight: DeviceBuffer<u16>,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) moe_weights: Vec<Option<MoeWeights>>,  // per-layer MoE FFN (None for dense FFN layers)
    pub(crate) activations: ActivationBuffers,
    pub(crate) gdn_conv_states: Vec<DeviceBuffer<f32>>, // [6144, 3] per GDN layer
    pub(crate) kv_caches: Vec<KvCache>,
    pub(crate) gdn_states: Vec<GdnState>,
    pub(crate) seq_len: u32,
    megakernel: Option<MegakernelProgram>,
    // Paged KV path (lazy-init)
    megakernel_paged: Option<MegakernelProgram>,
    page_allocator: Option<PageAllocator>,
    quant_allocator: Option<PageAllocator>,
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

/// Load a tensor's raw bytes from safetensors directly to GPU. Zero-copy from mmap.
/// Returns DeviceBuffer<u8> containing the on-disk representation.
#[allow(dead_code)]
fn load_tensor_raw(
    st: &SafeTensorSet,
    name: &str,
    device: DeviceId,
) -> Result<DeviceBuffer<u8>, ModelError> {
    let raw = st.tensor_data(name)
        .map_err(|_| ModelError::MissingWeight(name.to_string()))?;
    let mut buf = DeviceBuffer::<u8>::alloc(device, raw.len())?;
    buf.copy_from_host(raw)?;
    Ok(buf)
}

/// Load a bf16 tensor, returning a typed DeviceBuffer<u16>.
/// The underlying data is copied directly from mmap — zero conversion.
fn load_weight_bf16(
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

/// Load a tensor as f32. For f32 on disk: reinterpret. For bf16: convert on CPU.
/// Only used for the few tensors that need f32 on the GPU (A_log, output_norm).
fn load_weight_f32(
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

impl Model {
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

        // Pin mmap'd shard regions so hipMemcpy can DMA directly (avoids bounce buffer).
        // Costs ~300ms upfront to fault in pages, but saves ~500ms on weight copies.
        // Some models have mmap regions that fail hipHostRegister (non-page-aligned etc.);
        // track which succeeded so we only unregister those.
        let shard_ptrs: Vec<(*mut std::ffi::c_void, usize)> = st.shard_mmaps()
            .map(|m| (m.as_ptr() as *mut std::ffi::c_void, m.len()))
            .collect();
        let mut pinned: Vec<*mut std::ffi::c_void> = Vec::new();
        for &(ptr, len) in &shard_ptrs {
            let rc = unsafe { ffi::hipHostRegister(ptr, len, 0) };
            if rc == 0 {
                pinned.push(ptr);
            }
        }

        // Discover tensor name prefix by finding "layers.0." in tensor names.
        // Prefer prefixes containing "model" to avoid matching MTP/draft heads.
        let prefix = {
            let names = st.tensor_names();
            let candidates: Vec<&str> = names.iter()
                .filter(|n| n.contains("layers.0."))
                .map(|n| &n[..n.find("layers.0.").unwrap()])
                .collect();
            let prefix = candidates.iter()
                .find(|p| p.contains("model"))
                .or_else(|| candidates.iter().find(|p| !p.contains("mtp")))
                .or(candidates.first())
                .ok_or_else(|| ModelError::MissingWeight("no layers.0. tensor found".into()))?;
            prefix.to_string()
        };

        let stream = Stream::new(device)?;
        let kernels = AllKernels::load(device)?;

        // Discover model features from tensor names
        let names = st.tensor_names();
        let has_qk_norm = names.iter().any(|n| n.contains("q_norm.weight"));
        config.has_qk_norm = has_qk_norm;

        // Detect gated Q: Qwen3.5 packs Q+gate in q_proj [nqh*hd*2, hidden].
        // Standard models have q_proj [nqh*hd, hidden].
        let first_attn_idx = config.layers.iter().position(|l| l.layer_type == LayerType::Attention);
        let has_output_gate = if let Some(ai) = first_attn_idx {
            let q_name = format!("{prefix}layers.{ai}.self_attn.q_proj.weight");
            if let Ok(raw) = st.tensor_data(&q_name) {
                let expected_gated = config.num_q_heads * config.head_dim * 2 * config.hidden_size * 2; // bf16
                raw.len() == expected_gated
            } else { false }
        } else { false };
        config.has_output_gate = has_output_gate;
        let embed_name = names.iter()
            .find(|n| n.starts_with(&prefix) && (n.contains("embed_tokens.weight") || n.contains("tok_embeddings.weight") || n.ends_with("wte.weight")))
            .or_else(|| names.iter().find(|n| n.contains("embed_tokens.weight") || n.contains("tok_embeddings.weight") || n.ends_with("wte.weight")))
            .ok_or_else(|| ModelError::MissingWeight("embedding tensor not found".into()))?
            .to_string();
        let norm_name = names.iter()
            .find(|n| n.starts_with(&prefix) && (n.ends_with("norm.weight") || n.ends_with("ln_f.weight")) && !n.contains("layers."))
            .or_else(|| names.iter().find(|n| (n.contains("norm.weight") || n.contains("ln_f.weight")) && !n.contains("layers.") && !n.contains("visual") && !n.contains("mtp")))
            .ok_or_else(|| ModelError::MissingWeight("final norm tensor not found".into()))?
            .to_string();

        let embed_weight = load_weight_bf16(&st, &embed_name, device, config.vocab_size * config.hidden_size)?;
        let lm_head_weight = if config.tie_word_embeddings {
            // Weight-tied: reuse embed_weight pointer (allocate a dummy — the megakernel uses embed_weight)
            DeviceBuffer::<u16>::alloc(device, 0)?  // placeholder, megakernel will use embed_weight
        } else {
            let lm_head_name = names.iter()
                .find(|n| n.contains("lm_head.weight"))
                .ok_or_else(|| ModelError::MissingWeight("lm_head.weight not found".into()))?
                .to_string();
            load_weight_bf16(&st, &lm_head_name, device, config.vocab_size * config.hidden_size)?
        };
        let final_norm_weight = load_weight_bf16(&st, &norm_name, device, config.hidden_size)?;

        // Per-layer weights
        let mut layers = Vec::with_capacity(config.num_layers);
        let mut moe_weights_vec: Vec<Option<MoeWeights>> = (0..config.num_layers).map(|_| None).collect();
        for i in 0..config.num_layers {
            let p = format!("{prefix}layers.{i}.");
            let is_moe = matches!(config.layers[i].ffn_type, FfnType::MoE { .. });
            if config.layer_is_attention[i] {
                let w = AttentionLayerWeights {
                    input_norm: load_weight_bf16(&st, &format!("{p}input_layernorm.weight"), device, config.hidden_size)?,
                    w_q_gate: {
                        let q_mult = if has_output_gate { 2 } else { 1 };
                        load_weight_bf16(&st, &format!("{p}self_attn.q_proj.weight"), device, config.num_q_heads * config.head_dim * q_mult * config.hidden_size)?
                    },
                    w_k: load_weight_bf16(&st, &format!("{p}self_attn.k_proj.weight"), device, config.num_kv_heads * config.head_dim * config.hidden_size)?,
                    w_v: load_weight_bf16(&st, &format!("{p}self_attn.v_proj.weight"), device, config.num_kv_heads * config.head_dim * config.hidden_size)?,
                    w_o: load_weight_bf16(&st, &format!("{p}self_attn.o_proj.weight"), device, config.hidden_size * config.num_q_heads * config.head_dim)?,
                    q_norm: if has_qk_norm {
                        let name = format!("{p}self_attn.q_norm.weight");
                        let raw = st.tensor_data(&name).map_err(|_| ModelError::MissingWeight(name.clone()))?;
                        load_weight_bf16(&st, &name, device, raw.len() / 2)?
                    } else { DeviceBuffer::<u16>::alloc(device, 0)? },
                    k_norm: if has_qk_norm {
                        let name = format!("{p}self_attn.k_norm.weight");
                        let raw = st.tensor_data(&name).map_err(|_| ModelError::MissingWeight(name.clone()))?;
                        load_weight_bf16(&st, &name, device, raw.len() / 2)?
                    } else { DeviceBuffer::<u16>::alloc(device, 0)? },
                    post_norm: load_weight_bf16(&st, &format!("{p}post_attention_layernorm.weight"), device, config.hidden_size)?,
                    w_gate: if !is_moe {
                        load_weight_bf16(&st, &format!("{p}mlp.gate_proj.weight"), device, config.intermediate_size * config.hidden_size)?
                    } else { DeviceBuffer::<u16>::alloc(device, 0)? },
                    w_up: if !is_moe {
                        load_weight_bf16(&st, &format!("{p}mlp.up_proj.weight"), device, config.intermediate_size * config.hidden_size)?
                    } else { DeviceBuffer::<u16>::alloc(device, 0)? },
                    w_down: if !is_moe {
                        load_weight_bf16(&st, &format!("{p}mlp.down_proj.weight"), device, config.hidden_size * config.intermediate_size)?
                    } else { DeviceBuffer::<u16>::alloc(device, 0)? },
                };
                layers.push(LayerWeights::Attention(w));

                // Load MoE weights if this layer uses MoE FFN
                if is_moe {
                    if let FfnType::MoE { num_experts, expert_intermediate_size, num_shared, shared_intermediate_size, .. } = &config.layers[i].ffn_type {
                        let ne = *num_experts;
                        let eis = *expert_intermediate_size;
                        let hs = config.hidden_size;

                        // Router gate: try mlp.gate, block_sparse_moe.gate, mlp.router
                        let gate_name = [
                            format!("{p}mlp.gate.weight"),
                            format!("{p}block_sparse_moe.gate.weight"),
                            format!("{p}mlp.router.weight"),
                        ].into_iter().find(|n| st.tensor_data(n).is_ok())
                            .ok_or_else(|| ModelError::MissingWeight(format!("{p}mlp.gate.weight (or variants)")))?;
                        let gate = load_weight_bf16(&st, &gate_name, device, ne * hs)?;

                        // Expert weights: try fused, then per-expert with multiple name patterns
                        let fused_name = format!("{p}mlp.experts.gate_up_proj");
                        let expert_gate_up = if st.tensor_data(&fused_name).is_ok() {
                            load_weight_bf16(&st, &fused_name, device, ne * 2 * eis * hs)?
                        } else {
                            // Per-expert: try gate_proj/up_proj (Qwen) or w1/w3 (Mixtral)
                            // Also try mlp.experts.N or block_sparse_moe.experts.N
                            let mut buf = DeviceBuffer::<u16>::alloc(device, ne * 2 * eis * hs)?;
                            for e in 0..ne {
                                let (gp, up) = [
                                    (format!("{p}mlp.experts.{e}.gate_proj.weight"), format!("{p}mlp.experts.{e}.up_proj.weight")),
                                    (format!("{p}block_sparse_moe.experts.{e}.w1.weight"), format!("{p}block_sparse_moe.experts.{e}.w3.weight")),
                                ].into_iter().find(|(g, _)| st.tensor_data(g).is_ok())
                                    .ok_or_else(|| ModelError::MissingWeight(format!("{p}mlp.experts.{e}.gate_proj.weight (or variants)")))?;
                                let g_raw = st.tensor_data(&gp).map_err(|_| ModelError::MissingWeight(gp))?;
                                let u_raw = st.tensor_data(&up).map_err(|_| ModelError::MissingWeight(up))?;
                                // gate_proj and up_proj are each [expert_is, hidden_size]
                                // Fuse into [2*expert_is, hidden_size] per expert
                                let offset = e * 2 * eis * hs * 2; // bytes
                                unsafe {
                                    let dst = (buf.as_mut_ptr() as *mut u8).add(offset);
                                    std::ptr::copy_nonoverlapping(g_raw.as_ptr(), dst, eis * hs * 2);
                                    std::ptr::copy_nonoverlapping(u_raw.as_ptr(), dst.add(eis * hs * 2), eis * hs * 2);
                                }
                            }
                            buf
                        };

                        let down_name = format!("{p}mlp.experts.down_proj");
                        let expert_down = if st.tensor_data(&down_name).is_ok() {
                            load_weight_bf16(&st, &down_name, device, ne * hs * eis)?
                        } else {
                            let mut buf = DeviceBuffer::<u16>::alloc(device, ne * hs * eis)?;
                            for e in 0..ne {
                                let dp = [
                                    format!("{p}mlp.experts.{e}.down_proj.weight"),
                                    format!("{p}block_sparse_moe.experts.{e}.w2.weight"),
                                ].into_iter().find(|n| st.tensor_data(n).is_ok())
                                    .ok_or_else(|| ModelError::MissingWeight(format!("{p}experts.{e}.down_proj (or variants)")))?;
                                let d_raw = st.tensor_data(&dp).map_err(|_| ModelError::MissingWeight(dp))?;
                                let offset = e * hs * eis * 2;
                                unsafe {
                                    let dst = (buf.as_mut_ptr() as *mut u8).add(offset);
                                    std::ptr::copy_nonoverlapping(d_raw.as_ptr(), dst, hs * eis * 2);
                                }
                            }
                            buf
                        };

                        // Shared expert
                        let shared_expert = if *num_shared > 0 {
                            let sis = *shared_intermediate_size;
                            let sis = if sis == 0 { eis } else { sis };
                            Some(DenseFfnWeights {
                                gate_proj: load_weight_bf16(&st, &format!("{p}mlp.shared_expert.gate_proj.weight"), device, sis * hs)?,
                                up_proj: load_weight_bf16(&st, &format!("{p}mlp.shared_expert.up_proj.weight"), device, sis * hs)?,
                                down_proj: load_weight_bf16(&st, &format!("{p}mlp.shared_expert.down_proj.weight"), device, hs * sis)?,
                            })
                        } else { None };

                        // Shared expert gate (optional, e.g. Qwen3.5-122B has shared_expert_gate)
                        let shared_gate_name = format!("{p}mlp.shared_expert_gate.weight");
                        let shared_expert_gate = if st.tensor_data(&shared_gate_name).is_ok() {
                            Some(load_weight_bf16(&st, &shared_gate_name, device, hs)?)
                        } else { None };

                        moe_weights_vec[i] = Some(MoeWeights {
                            gate,
                            expert_gate_up,
                            expert_down,
                            shared_expert,
                            shared_expert_gate,
                            num_experts: ne,
                            expert_intermediate_size: eis,
                        });
                    }
                }
            } else {
                let nh = config.linear_num_heads;
                let nvh = config.linear_num_value_heads;
                let kd = config.linear_key_head_dim;
                let vd = config.linear_value_head_dim;
                let qkv_out = nh * kd + nh * kd + nvh * vd;
                let z_out = nvh * vd;
                let ck = config.linear_conv_kernel_dim;
                let q_dim = nh * kd;
                let v_dim = nvh * vd;
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
                    w_a: load_weight_bf16(&st, &format!("{p}linear_attn.in_proj_a.weight"), device, nvh * config.hidden_size)?,
                    w_b: load_weight_bf16(&st, &format!("{p}linear_attn.in_proj_b.weight"), device, nvh * config.hidden_size)?,
                    w_z: load_weight_bf16(&st, &format!("{p}linear_attn.in_proj_z.weight"), device, z_out * config.hidden_size)?,
                    conv1d_weight: conv1d_weight_buf,
                    conv1d_weight_q: conv_w_q_buf,
                    conv1d_weight_k: conv_w_k_buf,
                    conv1d_weight_v: conv_w_v_buf,
                    a_log: load_weight_f32(&st, &format!("{p}linear_attn.A_log"), device, nvh)?,  // f32
                    dt_bias: load_weight_bf16(&st, &format!("{p}linear_attn.dt_bias"), device, nvh)?,
                    output_norm: load_weight_f32(&st, &format!("{p}linear_attn.norm.weight"), device, kd)?,  // f32
                    w_out: load_weight_bf16(&st, &format!("{p}linear_attn.out_proj.weight"), device, config.hidden_size * z_out)?,
                    post_norm: load_weight_bf16(&st, &format!("{p}post_attention_layernorm.weight"), device, config.hidden_size)?,
                    w_gate: if !is_moe {
                        load_weight_bf16(&st, &format!("{p}mlp.gate_proj.weight"), device, config.intermediate_size * config.hidden_size)?
                    } else { DeviceBuffer::<u16>::alloc(device, 0)? },
                    w_up: if !is_moe {
                        load_weight_bf16(&st, &format!("{p}mlp.up_proj.weight"), device, config.intermediate_size * config.hidden_size)?
                    } else { DeviceBuffer::<u16>::alloc(device, 0)? },
                    w_down: if !is_moe {
                        load_weight_bf16(&st, &format!("{p}mlp.down_proj.weight"), device, config.hidden_size * config.intermediate_size)?
                    } else { DeviceBuffer::<u16>::alloc(device, 0)? },
                };
                layers.push(LayerWeights::Gdn(w));

                // Load MoE weights for GDN layers with MoE FFN (e.g. Qwen3.5-122B)
                if is_moe {
                    if let FfnType::MoE { num_experts, expert_intermediate_size, num_shared, shared_intermediate_size, .. } = &config.layers[i].ffn_type {
                        let ne = *num_experts;
                        let eis = *expert_intermediate_size;
                        let hs = config.hidden_size;

                        let gate_name = [
                            format!("{p}mlp.gate.weight"),
                            format!("{p}block_sparse_moe.gate.weight"),
                            format!("{p}mlp.router.weight"),
                        ].into_iter().find(|n| st.tensor_data(n).is_ok())
                            .ok_or_else(|| ModelError::MissingWeight(format!("{p}mlp.gate.weight (GDN MoE)")))?;
                        let gate = load_weight_bf16(&st, &gate_name, device, ne * hs)?;

                        let fused_name = format!("{p}mlp.experts.gate_up_proj");
                        let expert_gate_up = if st.tensor_data(&fused_name).is_ok() {
                            load_weight_bf16(&st, &fused_name, device, ne * 2 * eis * hs)?
                        } else {
                            let mut buf = DeviceBuffer::<u16>::alloc(device, ne * 2 * eis * hs)?;
                            for e in 0..ne {
                                let (gp, up) = [
                                    (format!("{p}mlp.experts.{e}.gate_proj.weight"), format!("{p}mlp.experts.{e}.up_proj.weight")),
                                    (format!("{p}block_sparse_moe.experts.{e}.w1.weight"), format!("{p}block_sparse_moe.experts.{e}.w3.weight")),
                                ].into_iter().find(|(g, _)| st.tensor_data(g).is_ok())
                                    .ok_or_else(|| ModelError::MissingWeight(format!("{p}experts.{e}.gate_proj (GDN MoE)")))?;
                                let g_raw = st.tensor_data(&gp).map_err(|_| ModelError::MissingWeight(gp))?;
                                let u_raw = st.tensor_data(&up).map_err(|_| ModelError::MissingWeight(up))?;
                                let offset = e * 2 * eis * hs * 2;
                                unsafe {
                                    let dst = (buf.as_mut_ptr() as *mut u8).add(offset);
                                    std::ptr::copy_nonoverlapping(g_raw.as_ptr(), dst, eis * hs * 2);
                                    std::ptr::copy_nonoverlapping(u_raw.as_ptr(), dst.add(eis * hs * 2), eis * hs * 2);
                                }
                            }
                            buf
                        };

                        let down_name = format!("{p}mlp.experts.down_proj");
                        let expert_down = if st.tensor_data(&down_name).is_ok() {
                            load_weight_bf16(&st, &down_name, device, ne * hs * eis)?
                        } else {
                            let mut buf = DeviceBuffer::<u16>::alloc(device, ne * hs * eis)?;
                            for e in 0..ne {
                                let dp = [
                                    format!("{p}mlp.experts.{e}.down_proj.weight"),
                                    format!("{p}block_sparse_moe.experts.{e}.w2.weight"),
                                ].into_iter().find(|n| st.tensor_data(n).is_ok())
                                    .ok_or_else(|| ModelError::MissingWeight(format!("{p}experts.{e}.down_proj (or variants)")))?;
                                let d_raw = st.tensor_data(&dp).map_err(|_| ModelError::MissingWeight(dp))?;
                                let offset = e * hs * eis * 2;
                                unsafe {
                                    let dst = (buf.as_mut_ptr() as *mut u8).add(offset);
                                    std::ptr::copy_nonoverlapping(d_raw.as_ptr(), dst, hs * eis * 2);
                                }
                            }
                            buf
                        };

                        let shared_expert = if *num_shared > 0 {
                            let sis = *shared_intermediate_size;
                            let sis = if sis == 0 { eis } else { sis };
                            Some(DenseFfnWeights {
                                gate_proj: load_weight_bf16(&st, &format!("{p}mlp.shared_expert.gate_proj.weight"), device, sis * hs)?,
                                up_proj: load_weight_bf16(&st, &format!("{p}mlp.shared_expert.up_proj.weight"), device, sis * hs)?,
                                down_proj: load_weight_bf16(&st, &format!("{p}mlp.shared_expert.down_proj.weight"), device, hs * sis)?,
                            })
                        } else { None };

                        let shared_gate_name = format!("{p}mlp.shared_expert_gate.weight");
                        let shared_expert_gate = if st.tensor_data(&shared_gate_name).is_ok() {
                            Some(load_weight_bf16(&st, &shared_gate_name, device, hs)?)
                        } else { None };

                        moe_weights_vec[i] = Some(MoeWeights {
                            gate, expert_gate_up, expert_down,
                            shared_expert, shared_expert_gate,
                            num_experts: ne, expert_intermediate_size: eis,
                        });
                    }
                }
            }
        }

        // Unpin mmap'd regions now that all weights are on GPU
        for ptr in &pinned {
            unsafe { ffi::hipHostUnregister(*ptr) };
        }

        // GDN states: [nh * kd * vd] per GDN layer
        let nh = config.linear_num_heads;
        let nvh = config.linear_num_value_heads;
        let kd = config.linear_key_head_dim;
        let vd = config.linear_value_head_dim;
        let ck = config.linear_conv_kernel_dim;
        let qkv_out = nh * kd * 2 + nvh * vd;

        let mut gdn_states = Vec::new();
        let mut gdn_conv_states = Vec::new();
        for i in 0..config.num_layers {
            if !config.layer_is_attention[i] {
                let mut recurrent = DeviceBuffer::<f32>::alloc(device, nvh * kd * vd)?;
                let zeros = vec![0.0f32; nvh * kd * vd];
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
            v_gdn: DeviceBuffer::<f32>::alloc(device, nvh * vd)?,
            a_proj: DeviceBuffer::<f32>::alloc(device, nvh)?,
            b_proj: DeviceBuffer::<f32>::alloc(device, nvh)?,
            z_proj: DeviceBuffer::<f32>::alloc(device, nvh * vd)?,
            gate_gdn: DeviceBuffer::<f32>::alloc(device, nvh)?,
            recurrent_out: DeviceBuffer::<f32>::alloc(device, nvh * vd)?,
            normed_gated: DeviceBuffer::<f32>::alloc(device, nvh * vd)?,
            out_proj: DeviceBuffer::<f32>::alloc(device, hs)?,
            q_gate_attn: DeviceBuffer::<f32>::alloc(device, nqh * hd * if config.has_output_gate { 2 } else { 1 })?,
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
            gdn_cs_v: DeviceBuffer::<f32>::alloc(device, nvh * vd * (ck - 1))?,
            gdn_conv_out_q: DeviceBuffer::<f32>::alloc(device, nh * kd)?,
            gdn_conv_out_k: DeviceBuffer::<f32>::alloc(device, nh * kd)?,
            gdn_conv_out_v: DeviceBuffer::<f32>::alloc(device, nvh * vd)?,
        };

        Ok(Model {
            config,
            device,
            stream,
            kernels,
            embed_weight,
            lm_head_weight,
            final_norm_weight,
            layers,
            moe_weights: moe_weights_vec,
            activations,
            gdn_conv_states,
            kv_caches,
            gdn_states,
            seq_len: 0,
            megakernel: None,
            megakernel_paged: None,
            page_allocator: None,
            quant_allocator: None,
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
        let nvh = cfg.linear_num_value_heads as u32;
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
            nh * kd * 2 + nvh * vd,
            hs,
            &self.stream,
        )?;

        // 3. Project a [nvh], b [nvh], z [nvh*vd]
        self.kernels.linear_proj.forward(
            &mut self.activations.a_proj,
            &weights.w_a,
            &self.activations.normed,
            nvh,
            hs,
            &self.stream,
        )?;
        self.kernels.linear_proj.forward(
            &mut self.activations.b_proj,
            &weights.w_b,
            &self.activations.normed,
            nvh,
            hs,
            &self.stream,
        )?;
        self.kernels.linear_proj.forward(
            &mut self.activations.z_proj,
            &weights.w_z,
            &self.activations.normed,
            nvh * vd,
            hs,
            &self.stream,
        )?;

        // 4. Causal conv1d: split qkv into q/k/v
        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &self.activations.qkv, 0, nh as usize * kd as usize, &self.stream)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &self.activations.qkv, nh as usize * kd as usize, nh as usize * kd as usize, &self.stream)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &self.activations.qkv, nh as usize * kd as usize * 2, nvh as usize * vd as usize, &self.stream)?;
        }

        let conv_q_out_len = nh as usize * kd as usize;
        let conv_k_out_len = nh as usize * kd as usize;
        let conv_v_out_len = nvh as usize * vd as usize;
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
            nvh,
            &self.stream,
        )?;

        // 6. GDN recurrent step v2 (nvh heads, GQA group = nvh/nh)
        let gqa_group = nvh / nh;
        self.kernels.gdn_recurrent_v2.forward(
            &self.activations.q_gdn,
            &self.activations.k_gdn,
            &self.activations.v_gdn,
            &self.activations.gate_gdn,
            &self.activations.b_proj,
            &mut self.gdn_states[gdn_idx].recurrent,
            &mut self.activations.recurrent_out,
            nvh,
            kd,
            vd,
            gqa_group,
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
        Ok(())
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

    /// MoE FFN forward: route to top-k experts, run expert FFNs, combine.
    /// Uses individual kernel launches (no megakernel).
    fn moe_ffn_forward(&mut self, layer_idx: usize) -> HipResult<()> {
        let moe = self.moe_weights[layer_idx].as_ref()
            .expect("moe_ffn_forward called on non-MoE layer");
        let hs = self.config.hidden_size;
        let eis = moe.expert_intermediate_size;
        let ne = moe.num_experts;
        let eps = self.config.rms_norm_eps;

        // Get post_norm weight from the layer
        let post_norm = match &self.layers[layer_idx] {
            LayerWeights::Attention(w) => &w.post_norm as *const DeviceBuffer<u16>,
            LayerWeights::Gdn(w) => &w.post_norm as *const DeviceBuffer<u16>,
        };

        // 1. RMSNorm(hidden) → normed
        unsafe {
            self.kernels.rmsnorm.forward(
                &mut self.activations.normed,
                &self.activations.hidden,
                &*post_norm,
                1, hs as u32, eps, &self.stream,
            )?;
        }

        // Save residual
        unsafe {
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs, &self.stream)?;
        }

        // 2. Gate projection: normed → scores[num_experts]
        // We need a temporary buffer for scores
        let mut scores_buf = DeviceBuffer::<f32>::alloc(self.device, ne)?;
        self.kernels.linear_proj.forward(
            &mut scores_buf,
            &moe.gate,
            &self.activations.normed,
            ne as u32, hs as u32, &self.stream,
        )?;
        self.stream.synchronize()?;

        // 3. Read scores to CPU and do top-k selection
        let mut scores = vec![0.0f32; ne];
        scores_buf.copy_to_host(&mut scores)?;

        let k = match &self.config.layers[layer_idx].ffn_type {
            FfnType::MoE { num_active, .. } => *num_active,
            _ => unreachable!(),
        };

        // Top-k selection
        let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let topk: Vec<(usize, f32)> = indexed[..k].to_vec();

        // Softmax over selected
        let max_s = topk.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = topk.iter().map(|(_, s)| (s - max_s).exp()).sum();
        let weights: Vec<f32> = topk.iter().map(|(_, s)| (s - max_s).exp() / exp_sum).collect();

        let scaling = match &self.config.layers[layer_idx].ffn_type {
            FfnType::MoE { gate_type: GateType::NormTopK { routed_scaling_factor }, .. } => *routed_scaling_factor,
            _ => 1.0,
        };

        // 4. Zero output buffer (reuse ffn_down for accumulation)
        let zeros = vec![0.0f32; hs];
        self.activations.ffn_down.copy_from_host(&zeros)?;

        // 5. For each selected expert: run FFN and accumulate
        let mut expert_scratch_gate = DeviceBuffer::<f32>::alloc(self.device, eis)?;
        let mut expert_scratch_up = DeviceBuffer::<f32>::alloc(self.device, eis)?;
        let mut expert_scratch_act = DeviceBuffer::<f32>::alloc(self.device, eis)?;
        let mut expert_output = DeviceBuffer::<f32>::alloc(self.device, hs)?;

        for (j, &(expert_id, _)) in topk.iter().enumerate() {
            let w = weights[j] * scaling;

            // Expert weight pointers into contiguous buffer
            let gate_w_ptr = unsafe { moe.expert_gate_up.as_ptr().add(expert_id * 2 * eis * hs) };
            let up_w_ptr = unsafe { moe.expert_gate_up.as_ptr().add(expert_id * 2 * eis * hs + eis * hs) };
            let down_w_ptr = unsafe { moe.expert_down.as_ptr().add(expert_id * hs * eis) };

            // Gate projection: bf16_weights[eis, hs] × f32_normed[hs] → f32_gate[eis]
            self.kernels.linear_proj.forward_ptr(
                expert_scratch_gate.as_mut_ptr(),
                gate_w_ptr,
                self.activations.normed.as_ptr(),
                eis as u32, hs as u32, &self.stream,
            )?;

            // Up projection
            self.kernels.linear_proj.forward_ptr(
                expert_scratch_up.as_mut_ptr(),
                up_w_ptr,
                self.activations.normed.as_ptr(),
                eis as u32, hs as u32, &self.stream,
            )?;

            // SiLU(gate) * up → scratch_act
            self.kernels.silu_mul.forward(
                &mut expert_scratch_act,
                &expert_scratch_gate,
                &expert_scratch_up,
                eis as u32, &self.stream,
            )?;

            // Down projection: bf16_weights[hs, eis] × f32_act[eis] → f32_output[hs]
            self.kernels.linear_proj.forward_ptr(
                expert_output.as_mut_ptr(),
                down_w_ptr,
                expert_scratch_act.as_ptr(),
                hs as u32, eis as u32, &self.stream,
            )?;

            // Weighted accumulate: ffn_down[h] += w * expert_output[h]
            // Use a simple kernel or CPU-side scaling
            // For now: scale expert_output by w, then add to ffn_down
            // We can do this with residual_add if we pre-scale
            // Actually, just accumulate on host for v1 correctness
            self.stream.synchronize()?;
            let mut exp_out = vec![0.0f32; hs];
            expert_output.copy_to_host(&mut exp_out)?;
            let mut accum = vec![0.0f32; hs];
            self.activations.ffn_down.copy_to_host(&mut accum)?;
            for h in 0..hs {
                accum[h] += w * exp_out[h];
            }
            self.activations.ffn_down.copy_from_host(&accum)?;
        }

        // 6. Residual add: hidden = residual + ffn_down
        self.kernels.residual_add.forward(
            &mut self.activations.hidden,
            &self.activations.residual,
            &self.activations.ffn_down,
            hs as u32, &self.stream,
        )?;

        Ok(())
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
        let _max_sl = cfg.max_seq_len as u32;

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
        let (w_q_gate, w_k, w_v, w_o_ptr, q_norm_w, k_norm_w) =
            match &self.layers[layer_idx] {
                LayerWeights::Attention(w) => (
                    &w.w_q_gate as *const DeviceBuffer<u16>,
                    &w.w_k as *const DeviceBuffer<u16>,
                    &w.w_v as *const DeviceBuffer<u16>,
                    &w.w_o as *const DeviceBuffer<u16>,
                    &w.q_norm as *const DeviceBuffer<u16>,
                    &w.k_norm as *const DeviceBuffer<u16>,
                ),
                _ => unreachable!(),
            };

        let q_mult = if cfg.has_output_gate { 2u32 } else { 1 };
        unsafe {
            self.kernels.linear_proj.forward(
                &mut self.activations.q_gate_attn,
                &*w_q_gate,
                &self.activations.normed,
                nqh * hd * q_mult,
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


        // 3. Split q_gate_attn → q, gate (gated) or just copy (non-gated)
        let hd_usize = hd as usize;
        if cfg.has_output_gate {
            unsafe {
                for h in 0..nqh as usize {
                    let src_q = h * hd_usize * 2;
                    let src_g = h * hd_usize * 2 + hd_usize;
                    let dst = h * hd_usize;
                    d2d_copy_f32(&mut self.activations.q_attn, dst, &self.activations.q_gate_attn, src_q, hd_usize, &self.stream)?;
                    d2d_copy_f32(&mut self.activations.gate_attn, dst, &self.activations.q_gate_attn, src_g, hd_usize, &self.stream)?;
                }
            }
        } else {
            // Non-gated: q_gate_attn IS q_attn, just copy
            let total = nqh as usize * hd_usize;
            unsafe {
                d2d_copy_f32(&mut self.activations.q_attn, 0, &self.activations.q_gate_attn, 0, total, &self.stream)?;
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


        // 8. Output gate (Qwen3.5 only) or pass-through
        let final_attn = if cfg.has_output_gate {
            self.kernels.output_gate.forward(
                &mut self.activations.gated_out,
                &self.activations.attn_out,
                &self.activations.gate_attn,
                nqh * hd,
                &self.stream,
            )?;
            &self.activations.gated_out as *const DeviceBuffer<f32>
        } else {
            &self.activations.attn_out as *const DeviceBuffer<f32>
        };


        // 9. Output projection
        unsafe {
            self.kernels.linear_proj.forward(
                &mut self.activations.out_proj,
                &*w_o_ptr,
                &*final_attn,
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

        Ok(())
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
        let has_moe = self.config.layers.iter().any(|l| matches!(l.ffn_type, FfnType::MoE { .. }));
        if has_moe {
            return self.decode_step_moe(token_id, position);
        }

        // Dense models: use megakernel
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

    /// MoE decode step: kernel-by-kernel execution with MoE FFN dispatch.
    fn decode_step_moe(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        let hs = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps;

        // Set position_ids for mRoPE/RoPE
        let pos_data = [position as i32, position as i32, position as i32];
        self.activations.position_ids.copy_from_host(&pos_data)?;

        // Embedding
        self.kernels.embedding.forward(
            &mut self.activations.hidden,
            &self.embed_weight,
            token_id as i32, hs, &self.stream,
        )?;

        // Process each layer
        let mut gdn_idx = 0usize;
        let mut kv_idx = 0usize;
        for layer_i in 0..self.config.num_layers {
            match self.config.layers[layer_i].layer_type {
                LayerType::Attention => {
                    self.attention_forward(layer_i, kv_idx, position)?;
                    kv_idx += 1;
                }
                LayerType::Gdn => {
                    self.gdn_forward(layer_i, gdn_idx)?;
                    gdn_idx += 1;
                }
                _ => panic!("unsupported layer type for MoE decode"),
            }

            // FFN: dense or MoE
            if matches!(self.config.layers[layer_i].ffn_type, FfnType::MoE { .. }) {
                self.moe_ffn_forward(layer_i)?;
            } else {
                let (post_norm, w_gate, w_up, w_down) = match &self.layers[layer_i] {
                    LayerWeights::Attention(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
                    LayerWeights::Gdn(w) => (&w.post_norm, &w.w_gate, &w.w_up, &w.w_down),
                };
                let post_norm = post_norm as *const DeviceBuffer<u16>;
                let w_gate = w_gate as *const DeviceBuffer<u16>;
                let w_up = w_up as *const DeviceBuffer<u16>;
                let w_down = w_down as *const DeviceBuffer<u16>;
                unsafe { self.ffn_forward(&*post_norm, &*w_gate, &*w_up, &*w_down)?; }
            }
        }

        // Final RMSNorm
        self.kernels.rmsnorm.forward(
            &mut self.activations.normed,
            &self.activations.hidden,
            &self.final_norm_weight,
            1, hs, eps, &self.stream,
        )?;

        // LM head
        let lm_head_w = if self.config.tie_word_embeddings {
            &self.embed_weight
        } else {
            &self.lm_head_weight
        };
        self.kernels.linear_proj.forward(
            &mut self.activations.logits,
            lm_head_w,
            &self.activations.normed,
            self.config.vocab_size as u32, hs, &self.stream,
        )?;

        self.stream.synchronize()?;

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;

        self.seq_len = position + 1;
        Ok(logits)
    }

    /// Run a single decode step using the paged KV cache path.
    /// Returns logits [vocab_size].
    pub fn decode_step_paged(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        self.decode_step_paged_inner(token_id, position, false)
    }

    /// Run a single decode step with quantized KV cache (4-bit residual_pc).
    /// Sealed chunks are quantized to int4; active chunk stays f32.
    pub fn decode_step_paged_quantized(&mut self, token_id: u32, position: u32) -> Result<Vec<f32>, ModelError> {
        self.decode_step_paged_inner(token_id, position, true)
    }

    fn decode_step_paged_inner(&mut self, token_id: u32, position: u32, quantized: bool) -> Result<Vec<f32>, ModelError> {
        let max_chunks = (self.config.max_seq_len + CHUNK_TOKENS - 1) / CHUNK_TOKENS;

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
            assert_eq!(mk.quantized_kv, quantized,
                "cannot mix decode_step_paged and decode_step_paged_quantized on the same model");
        }

        // Lazy-init: f32 PageAllocator (staging) and SequenceState
        if self.page_allocator.is_none() {
            self.page_allocator = Some(PageAllocator::new(
                self.device, &self.config, CHUNK_TOKENS, max_chunks as u32,
            )?);
            self.paged_seq = Some(SequenceState::new(CHUNK_TOKENS as u32));
        }

        // Lazy-init: quantized PageAllocator
        if quantized && self.quant_allocator.is_none() {
            self.quant_allocator = Some(PageAllocator::new_quantized(
                self.device, &self.config, CHUNK_TOKENS, max_chunks as u32,
            )?);
        }

        // append_token
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

        // Post-step: handle chunk seal + quantization
        {
            let mk = self.megakernel_paged.as_mut().unwrap();
            let seq_mut = self.paged_seq.as_mut().unwrap();
            let alloc_mut = self.page_allocator.as_mut().unwrap();
            let q_alloc = self.quant_allocator.as_mut();
            mk.post_step_paged(position, seq_mut, alloc_mut, q_alloc, &self.config, &self.stream)?;
        }

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

        // MoE models can't use megakernel prefill (FFN weights not compiled in).
        // Fall back to sequential decode_step_moe.
        let has_moe = self.config.layers.iter().any(|l| matches!(l.ffn_type, FfnType::MoE { .. }));
        if has_moe {
            let mut logits = vec![];
            for (i, &tok) in tokens.iter().enumerate() {
                logits = self.decode_step_moe(tok, i as u32)?;
            }
            return Ok(logits);
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

        let nvh_traced = self.config.linear_num_value_heads as u32;
        let gqa_traced = nvh_traced / nh;

        // QKV projection
        self.kernels.linear_proj.forward(
            &mut self.activations.qkv, &w.w_qkv, &self.activations.normed,
            nh * kd * 2 + nvh_traced * vd, hs, &self.stream,
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
            nvh_traced, kd, vd, gqa_traced, &self.stream,
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
        let nvh_r = self.config.linear_num_value_heads;
        let qkv_out = nh * kd * 2 + nvh_r * vd;

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
        // Free quantized KV slots back to pool
        if let (Some(seq), Some(q_alloc)) = (self.paged_seq.as_mut(), self.quant_allocator.as_mut()) {
            seq.free_quant_slots(q_alloc);
        }
        Ok(())
    }
}
