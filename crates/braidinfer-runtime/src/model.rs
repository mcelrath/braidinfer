use std::path::Path;

use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::{ffi, HipResult};
use braidinfer_core::safetensors::SafeTensorSet;

use crate::kernel::{
    CausalConv1dUpdateKernel, EmbeddingKernel, FfnFusedKernel, GdnGateKernel,
    GdnRecurrentStepV2Kernel, GqaAttentionKernel, LinearProjKernel, LmHeadKernel,
    MRoPEKernel, OutputGateKernel, QkNormKernel, ResidualAddKernel, RmsNormGatedKernel,
    RmsNormKernel,
};

// ---- Model config ----

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
}

impl ModelConfig {
    pub fn qwen35_0_8b() -> Self {
        // From config.json: layer_types = ["linear_attention"]*3 + ["full_attention"] repeated 6x
        // → 24 layers total:  0,1,2 = GDN; 3 = attn; 4,5,6 = GDN; 7 = attn; ...
        let mut layer_is_attention = vec![false; 24];
        for &i in &[3usize, 7, 11, 15, 19, 23] {
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
            max_seq_len: 2048,
        }
    }
}

// ---- Layer weight structs ----

pub struct GdnLayerWeights {
    pub input_norm: DeviceBuffer<u16>,  // bf16: (1+w) pattern, zeros init
    pub w_qkv: DeviceBuffer<u16>,      // bf16 [6144, 1024]
    pub w_a: DeviceBuffer<u16>,        // bf16 [16, 1024]
    pub w_b: DeviceBuffer<u16>,        // bf16 [16, 1024]
    pub w_z: DeviceBuffer<u16>,        // bf16 [2048, 1024]
    pub conv1d_weight: DeviceBuffer<u16>, // bf16 [6144, 4]
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
    pub k: DeviceBuffer<f32>, // [max_seq_len, num_kv_heads, head_dim]
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

fn load_weight(
    st: &SafeTensorSet,
    name: &str,
    device: DeviceId,
    expected_len: usize,
) -> Result<DeviceBuffer<f32>, ModelError> {
    let data = st
        .tensor_as_f32(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    assert_eq!(
        data.len(),
        expected_len,
        "weight {name}: expected {expected_len} elements, got {}",
        data.len()
    );
    let mut buf = DeviceBuffer::<f32>::alloc(device, expected_len)?;
    buf.copy_from_host(&data)?;
    Ok(buf)
}

fn load_weight_bf16(
    st: &SafeTensorSet,
    name: &str,
    device: DeviceId,
    expected_len: usize,
) -> Result<DeviceBuffer<u16>, ModelError> {
    let data = st
        .tensor_as_u16(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    assert_eq!(
        data.len(),
        expected_len,
        "weight {name}: expected {expected_len} elements, got {}",
        data.len()
    );
    let mut buf = DeviceBuffer::<u16>::alloc(device, expected_len)?;
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
) -> HipResult<()> {
    unsafe {
        let dst_ptr = dst.as_mut_ptr().add(dst_off) as *mut std::ffi::c_void;
        let src_ptr = src.as_ptr().add(src_off) as *const std::ffi::c_void;
        braidinfer_hip::error::check(ffi::hipMemcpy(
            dst_ptr,
            src_ptr,
            count * 2,
            ffi::hipMemcpyDeviceToDevice,
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
) -> HipResult<()> {
    unsafe {
        let dst_ptr = dst.as_mut_ptr().add(dst_off) as *mut std::ffi::c_void;
        let src_ptr = src.as_ptr().add(src_off) as *const std::ffi::c_void;
        braidinfer_hip::error::check(ffi::hipMemcpy(
            dst_ptr,
            src_ptr,
            count * 4,
            ffi::hipMemcpyDeviceToDevice,
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
    pub fn load(model_dir: &Path, device: DeviceId) -> Result<Self, ModelError> {
        let config = ModelConfig::qwen35_0_8b();
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
                let w = GdnLayerWeights {
                    input_norm: load_weight_bf16(&st, &format!("{p}input_layernorm.weight"), device, config.hidden_size)?,
                    w_qkv: load_weight_bf16(&st, &format!("{p}linear_attn.in_proj_qkv.weight"), device, qkv_out * config.hidden_size)?,
                    w_a: load_weight_bf16(&st, &format!("{p}linear_attn.in_proj_a.weight"), device, nh * config.hidden_size)?,
                    w_b: load_weight_bf16(&st, &format!("{p}linear_attn.in_proj_b.weight"), device, nh * config.hidden_size)?,
                    w_z: load_weight_bf16(&st, &format!("{p}linear_attn.in_proj_z.weight"), device, z_out * config.hidden_size)?,
                    conv1d_weight: load_weight_bf16(&st, &format!("{p}linear_attn.conv1d.weight"), device, qkv_out * config.linear_conv_kernel_dim)?,
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

        // 4. Causal conv1d update on QKV (in-place: qkv → qkv)
        // conv state is [qkv_out, ck-1]. We need a scratch buffer for output.
        // Since causal_conv1d writes to output, we'll use q_gdn temporarily as scratch
        // then copy back. But q_gdn is nh*kd=2048, while qkv=6144. We need a separate buf.
        // We'll use a trick: project into qkv (already done), run conv, store back to qkv.
        // CausalConv1dUpdateKernel signature: state[conv_dim, ks-1], input[conv_dim], weight[conv_dim,ks], output[conv_dim]
        // We need output to be same as input (update in-place). But kernel takes separate in/out.
        // Allocate a temp conv output the first time... but we don't have one in activations.
        // Solution: we'll run qkv as input and write output to q_gdn+k_gdn+v_gdn temporarily.
        // q_gdn=2048, k_gdn=2048, v_gdn=2048 → total 6144 = qkv_out. We can do 3 separate conv calls.
        // Each conv call covers 2048 elements. But conv operates on all 6144 at once.
        // Alternative: use normed buffer [1024] as temp, won't fit.
        // Best approach: do 3 separate conv passes covering different channels.
        // Actually CausalConv1d is depthwise: each channel independent. We can split.

        // Split qkv into q_gdn, k_gdn, v_gdn via D2D copy, then run 3 conv passes.
        // But the weights are [6144, ck] — we'd need sliced weights too.
        // Simplest: add a conv_out activation buffer. For now, reuse residual (hidden_size=1024 < 6144).
        // We'll use recurrent_out as temp for first 2048, and normed_gated for second 2048,
        // and z_proj for the last 2048 (z_proj will be overwritten after we're done with it).
        // Actually z_proj is already computed and needed later.
        //
        // The cleanest solution: run conv on q_gdn (nh*kd=2048), k_gdn (nh*kd=2048), v_gdn (nh*vd=2048)
        // separately using the corresponding slices of qkv and conv1d_weight.
        // We need device-side pointer arithmetic for weight slices.
        // For now, copy qkv→{q_gdn,k_gdn,v_gdn} then conv, write back to qkv (not needed if we keep split).

        // D2D copy: qkv[0..2048] → q_gdn, qkv[2048..4096] → k_gdn, qkv[4096..6144] → v_gdn
        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &self.activations.qkv, 0, nh as usize * kd as usize)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &self.activations.qkv, nh as usize * kd as usize, nh as usize * kd as usize)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &self.activations.qkv, nh as usize * kd as usize * 2, nh as usize * vd as usize)?;
        }

        // Now run causal_conv1d on each segment separately.
        // We need subviews of conv1d_weight too. We do D2D to create temp weight slices.
        // Use recurrent_out[0..2048], normed_gated[0..2048], out_proj[0..2048] as conv outputs.
        // (recurrent_out = nh*vd=2048, normed_gated=2048, out_proj=hs=1024 — out_proj too small for v_gdn!)
        // Just use the same buffers since sizes match: q_gdn=2048, k_gdn=2048, v_gdn=2048.
        // Run conv with q_gdn as input and recurrent_out as output (both 2048), etc.
        // Then copy recurrent_out → q_gdn, etc.

        // For weight slicing, we need to create sub-weight DeviceBuffers.
        // This is getting complex without sub-buffer support.
        // Alternative simpler approach: allocate 3 temporary weight buffers in load() and pre-split them.
        // For now, use a workaround: run 3 separate CausalConv1d operations using pointer-offset weight views
        // via unsafe raw pointer approach within the D2D copy helper.
        //
        // Since we don't have sub-buffer views, let's take the simplest possible approach:
        // Store conv weights pre-split per GDN layer. But we didn't do that in load().
        //
        // WORKAROUND: Allocate temp weight slices on-the-fly using DeviceBuffer::alloc + D2D copy.
        // This is slow (allocation per step) but correct for an initial working implementation.

        let conv_q_out_len = nh as usize * kd as usize;
        let conv_k_out_len = nh as usize * kd as usize;
        let conv_v_out_len = nh as usize * vd as usize;
        let ck_usize = ck as usize;

        let mut conv_w_q = DeviceBuffer::<u16>::alloc(self.device, conv_q_out_len * ck_usize)?;
        let mut conv_w_k = DeviceBuffer::<u16>::alloc(self.device, conv_k_out_len * ck_usize)?;
        let mut conv_w_v = DeviceBuffer::<u16>::alloc(self.device, conv_v_out_len * ck_usize)?;

        // NOTE: conv1d_weight layout is [6144, ck] row-major. Each row is one channel's kernel.
        // We need rows 0..2048 for q, 2048..4096 for k, 4096..6144 for v.
        unsafe {
            let weights_gdn = match &self.layers[layer_idx] {
                LayerWeights::Gdn(w) => w,
                _ => unreachable!(),
            };
            d2d_copy_u16(&mut conv_w_q, 0, &weights_gdn.conv1d_weight, 0, conv_q_out_len * ck_usize)?;
            d2d_copy_u16(&mut conv_w_k, 0, &weights_gdn.conv1d_weight, conv_q_out_len * ck_usize, conv_k_out_len * ck_usize)?;
            d2d_copy_u16(&mut conv_w_v, 0, &weights_gdn.conv1d_weight, (conv_q_out_len + conv_k_out_len) * ck_usize, conv_v_out_len * ck_usize)?;
        }

        // For the conv state, we need sub-views too.
        // gdn_conv_states[gdn_idx] is [6144, ck-1] = [6144 * (ck-1)].
        // Split into 3 sub-states: q=[2048,ck-1], k=[2048,ck-1], v=[2048,ck-1].
        let conv_state_q_len = conv_q_out_len * (ck_usize - 1);
        let conv_state_k_len = conv_k_out_len * (ck_usize - 1);
        let conv_state_v_len = conv_v_out_len * (ck_usize - 1);

        let mut cs_q = DeviceBuffer::<f32>::alloc(self.device, conv_state_q_len)?;
        let mut cs_k = DeviceBuffer::<f32>::alloc(self.device, conv_state_k_len)?;
        let mut cs_v = DeviceBuffer::<f32>::alloc(self.device, conv_state_v_len)?;
        unsafe {
            d2d_copy_f32(&mut cs_q, 0, &self.gdn_conv_states[gdn_idx], 0, conv_state_q_len)?;
            d2d_copy_f32(&mut cs_k, 0, &self.gdn_conv_states[gdn_idx], conv_state_q_len, conv_state_k_len)?;
            d2d_copy_f32(&mut cs_v, 0, &self.gdn_conv_states[gdn_idx], conv_state_q_len + conv_state_k_len, conv_state_v_len)?;
        }

        // Run 3 conv1d operations
        let mut conv_out_q = DeviceBuffer::<f32>::alloc(self.device, conv_q_out_len)?;
        let mut conv_out_k = DeviceBuffer::<f32>::alloc(self.device, conv_k_out_len)?;
        let mut conv_out_v = DeviceBuffer::<f32>::alloc(self.device, conv_v_out_len)?;

        self.kernels.causal_conv1d.forward(
            &mut cs_q,
            &self.activations.q_gdn,
            &conv_w_q,
            &mut conv_out_q,
            conv_q_out_len as u32,
            ck,
            &self.stream,
        )?;
        self.kernels.causal_conv1d.forward(
            &mut cs_k,
            &self.activations.k_gdn,
            &conv_w_k,
            &mut conv_out_k,
            conv_k_out_len as u32,
            ck,
            &self.stream,
        )?;
        self.kernels.causal_conv1d.forward(
            &mut cs_v,
            &self.activations.v_gdn,
            &conv_w_v,
            &mut conv_out_v,
            conv_v_out_len as u32,
            ck,
            &self.stream,
        )?;

        // Write back updated conv states
        unsafe {
            d2d_copy_f32(&mut self.gdn_conv_states[gdn_idx], 0, &cs_q, 0, conv_state_q_len)?;
            d2d_copy_f32(&mut self.gdn_conv_states[gdn_idx], conv_state_q_len, &cs_k, 0, conv_state_k_len)?;
            d2d_copy_f32(&mut self.gdn_conv_states[gdn_idx], conv_state_q_len + conv_state_k_len, &cs_v, 0, conv_state_v_len)?;
        }

        // conv_out_q/k/v now hold the post-conv Q,K,V (with SiLU applied inside the kernel)
        // Copy them back to q_gdn, k_gdn, v_gdn
        unsafe {
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &conv_out_q, 0, conv_q_out_len)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &conv_out_k, 0, conv_k_out_len)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &conv_out_v, 0, conv_v_out_len)?;
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
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize)?;
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
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize)?;
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
                d2d_copy_f32(&mut self.activations.q_attn, dst, &self.activations.q_gate_attn, src_q, hd_usize)?;
                d2d_copy_f32(&mut self.activations.gate_attn, dst, &self.activations.q_gate_attn, src_g, hd_usize)?;
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

        // 6. Write K,V to cache at position `position`
        let kv_stride = nkh as usize * hd as usize; // per position
        let pos_off = position as usize * kv_stride;
        unsafe {
            d2d_copy_f32(&mut self.kv_caches[kv_cache_idx].k, pos_off, &self.activations.k_attn, 0, kv_stride)?;
            d2d_copy_f32(&mut self.kv_caches[kv_cache_idx].v, pos_off, &self.activations.v_attn, 0, kv_stride)?;
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
            max_sl,
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
            d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize)?;
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
        let hs = self.config.hidden_size as u32;
        let vs = self.config.vocab_size as u32;

        // 1. Embedding lookup
        self.kernels.embedding.forward(
            &mut self.activations.hidden,
            &self.embed_weight,
            token_id as i32,
            hs,
            &self.stream,
        )?;

        // 2. Run each layer
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
        }

        // 3. Final RMSNorm
        // Need to copy hidden to normed, then norm into hidden
        unsafe {
            d2d_copy_f32(&mut self.activations.normed, 0, &self.activations.hidden, 0, hs as usize)?;
        }
        self.kernels.rmsnorm.forward(
            &mut self.activations.hidden,
            &self.activations.normed,
            &self.final_norm_weight,
            1,
            hs,
            self.config.rms_norm_eps,
            &self.stream,
        )?;

        // 4. LM head (tied to embed_weight)
        self.kernels.lm_head.forward(
            &mut self.activations.logits,
            &self.embed_weight,
            &self.activations.hidden,
            vs,
            hs,
            &self.stream,
        )?;

        // 5. Sync and copy logits to host
        self.stream.synchronize()?;

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.activations.logits.copy_to_host(&mut logits)?;

        self.seq_len = position + 1;
        Ok(logits)
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

        unsafe { d2d_copy_f32(&mut self.activations.normed, 0, &self.activations.hidden, 0, hs as usize)?; }
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
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &self.activations.qkv, 0, conv_q_len)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &self.activations.qkv, conv_q_len, conv_k_len)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &self.activations.qkv, conv_q_len + conv_k_len, conv_v_len)?;
        }

        let mut conv_w_q = DeviceBuffer::<u16>::alloc(self.device, conv_q_len * ck_usize)?;
        let mut conv_w_k = DeviceBuffer::<u16>::alloc(self.device, conv_k_len * ck_usize)?;
        let mut conv_w_v = DeviceBuffer::<u16>::alloc(self.device, conv_v_len * ck_usize)?;
        unsafe {
            d2d_copy_u16(&mut conv_w_q, 0, &w.conv1d_weight, 0, conv_q_len * ck_usize)?;
            d2d_copy_u16(&mut conv_w_k, 0, &w.conv1d_weight, conv_q_len * ck_usize, conv_k_len * ck_usize)?;
            d2d_copy_u16(&mut conv_w_v, 0, &w.conv1d_weight, (conv_q_len + conv_k_len) * ck_usize, conv_v_len * ck_usize)?;
        }

        let conv_state_q_len = conv_q_len * (ck_usize - 1);
        let conv_state_k_len = conv_k_len * (ck_usize - 1);
        let conv_state_v_len = conv_v_len * (ck_usize - 1);

        let mut cs_q = DeviceBuffer::<f32>::alloc(self.device, conv_state_q_len)?;
        let mut cs_k = DeviceBuffer::<f32>::alloc(self.device, conv_state_k_len)?;
        let mut cs_v = DeviceBuffer::<f32>::alloc(self.device, conv_state_v_len)?;
        unsafe {
            d2d_copy_f32(&mut cs_q, 0, &self.gdn_conv_states[0], 0, conv_state_q_len)?;
            d2d_copy_f32(&mut cs_k, 0, &self.gdn_conv_states[0], conv_state_q_len, conv_state_k_len)?;
            d2d_copy_f32(&mut cs_v, 0, &self.gdn_conv_states[0], conv_state_q_len + conv_state_k_len, conv_state_v_len)?;
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
            d2d_copy_f32(&mut self.activations.q_gdn, 0, &conv_out_q, 0, conv_q_len)?;
            d2d_copy_f32(&mut self.activations.k_gdn, 0, &conv_out_k, 0, conv_k_len)?;
            d2d_copy_f32(&mut self.activations.v_gdn, 0, &conv_out_v, 0, conv_v_len)?;
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
        unsafe { d2d_copy_f32(&mut self.activations.residual, 0, &self.activations.hidden, 0, hs as usize)?; }
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
