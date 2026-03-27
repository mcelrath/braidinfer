//! Model configuration types and parsing.
//! Parsed from HuggingFace config.json — zero per-model classes.

use std::path::Path;
use crate::quant::WeightQuantMode;

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
        num_heads: usize,        // key heads
        num_value_heads: usize,  // value heads (may differ from key heads)
        key_dim: usize,
        value_dim: usize,
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
    pub mrope_section: [usize; 3],  // TODO(kvn.1): redundant with rope_type.sections, unify during split
    // GDN config
    pub linear_num_heads: usize,       // num_key_heads for GDN
    pub linear_num_value_heads: usize,  // may differ from linear_num_heads (e.g. 4B: 32 vs 16)
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    // Per-layer config: layer type + FFN type
    pub layers: Vec<LayerConfig>,
    // Legacy compat (derived from layers)
    pub layer_is_attention: Vec<bool>,  // TODO(kvn.1): derive from layers vec during model.rs split
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
    pub rms_norm_one_plus_w: bool, // true: (1+w)*x (Qwen3.5), false: w*x (Llama, OLMoE)
    pub attention_layer_indices: Vec<usize>,
    pub model_type: String,
    pub tie_word_embeddings: bool,
    pub weight_quant: WeightQuantMode,
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
                num_value_heads: 16,
                key_dim: 128,
                value_dim: 128,
                conv_dim: 6144,
                kernel_size: 4,
            },
            rope_type: RopeType::MRope { sections: [11, 11, 10] },
            has_qk_norm: false,
            has_output_gate: false,
            rms_norm_one_plus_w: true,
            attention_layer_indices,
            model_type: "qwen3_5".to_string(),
            tie_word_embeddings: true, // 0.8B fallback for tests
            weight_quant: WeightQuantMode::Bf16,
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
            .or(get_usize("n_embed"))  // bloom
            .or(get_usize("d_model"))
            .or(get_usize("model_dim"))  // openelm
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
        // Default true matches HF convention (most models tie). Models with separate lm_head
        // that omit this field will use embedding weights — load path should verify at runtime.
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
            let conv_dim = 2 * linear_num_heads * linear_key_head_dim + linear_num_value_heads * linear_value_head_dim;
            RecurrentLayerKind::Gdn {
                num_heads: linear_num_heads, num_value_heads: linear_num_value_heads,
                key_dim: linear_key_head_dim, value_dim: linear_value_head_dim,
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
            rms_norm_one_plus_w: model_type.starts_with("qwen3_5"),
            attention_layer_indices, model_type, tie_word_embeddings,
            weight_quant: WeightQuantMode::Bf16,
        })
    }


    pub fn chunk_kv_bytes(&self, chunk_tokens: usize) -> usize {
        let num_attn = self.num_attn_layers();
        // k and v, each: chunk_tokens * num_kv_heads * head_dim * 4 bytes (f32)
        2 * num_attn * chunk_tokens * self.num_kv_heads * self.head_dim * 4
    }

    pub fn recurrent_state_bytes_per_layer(&self) -> usize {
        match &self.recurrent_kind {
            RecurrentLayerKind::Gdn { num_value_heads, key_dim, value_dim, .. } => {
                num_value_heads * key_dim * value_dim * 4
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
