use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::HipResult;
use std::ffi::c_void;
use std::path::PathBuf;

pub(crate) fn kernel_dir() -> PathBuf {
    PathBuf::from(env!("BRAIDINFER_KERNEL_DIR"))
}

// GdnRecurrentStepKernel v1 removed — replaced by GdnRecurrentStepV2Kernel (with gqa_group param)

pub struct RmsNormKernel {
    module: Module,
    device: DeviceId,
}

impl RmsNormKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("rmsnorm.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        num_rows: u32,
        hidden_size: u32,
        eps: f32,
        one_plus_w: bool,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(input.device(), self.device);
        assert_eq!(output.device(), self.device);
        assert_eq!(weight.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("rmsnorm_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut w_ptr: *const c_void = weight.as_ptr().cast();
        let mut hs = hidden_size as i32;
        let mut ep = eps;
        let mut opw = if one_plus_w { 1i32 } else { 0i32 };

        let mut args: [*mut c_void; 6] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(hs).cast(),
            std::ptr::addr_of_mut!(ep).cast(),
            std::ptr::addr_of_mut!(opw).cast(),
        ];

        let block_size = 256u32.min(hidden_size);
        func.launch(
            (num_rows, 1, 1),
            (block_size, 1, 1),
            block_size * 4,
            stream,
            &mut args,
        )
    }
}

pub struct LinearProjKernel {
    module: Module,
    device: DeviceId,
}

impl LinearProjKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("linear_proj.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        input: &DeviceBuffer<f32>,
        out_dim: u32,
        in_dim: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(input.device(), self.device);
        assert_eq!(output.device(), self.device);
        assert_eq!(weight.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("linear_proj_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut w_ptr: *const c_void = weight.as_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut od = out_dim as i32;
        let mut id = in_dim as i32;

        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(od).cast(),
            std::ptr::addr_of_mut!(id).cast(),
        ];

        let block_size = 256u32.min(in_dim);
        func.launch(
            (out_dim, 1, 1),
            (block_size, 1, 1),
            256 * 4,
            stream,
            &mut args,
        )
    }

    /// Raw pointer version for MoE expert sub-buffer access.
    pub fn forward_ptr(
        &self,
        output: *mut f32,
        weight: *const u16,
        input: *const f32,
        out_dim: u32,
        in_dim: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = self.module.get_function("linear_proj_f32")?;
        let mut out_ptr: *mut c_void = output.cast();
        let mut w_ptr: *const c_void = weight.cast();
        let mut in_ptr: *const c_void = input.cast();
        let mut od = out_dim as i32;
        let mut id = in_dim as i32;
        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(od).cast(),
            std::ptr::addr_of_mut!(id).cast(),
        ];
        let block_size = 256u32.min(in_dim);
        func.launch(
            (out_dim, 1, 1),
            (block_size, 1, 1),
            256 * 4,
            stream,
            &mut args,
        )
    }

    /// Forward pass with PackedWeights (dispatches by format).
    pub fn forward_packed(
        &self,
        output: &mut DeviceBuffer<f32>,
        weight: &crate::model::PackedWeights,
        input: &DeviceBuffer<f32>,
        stream: &Stream,
    ) -> HipResult<()> {
        let out_dim = weight.out_dim as u32;
        let in_dim = weight.in_dim as u32;

        let func_name = match weight.format {
            crate::model::WeightFormat::Bf16 => "linear_proj_f32",
            crate::model::WeightFormat::Rnf4G128 => "linear_proj_rnf4_g128",
            crate::model::WeightFormat::PcG32Q4 => "linear_proj_pcg32_q4",
        };
        let func = self.module.get_function(func_name)?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut w_ptr: *const c_void = weight.data.as_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut od = out_dim as i32;
        let mut id = in_dim as i32;

        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(od).cast(),
            std::ptr::addr_of_mut!(id).cast(),
        ];

        let block_size = 256u32;
        func.launch(
            (out_dim, 1, 1),
            (block_size, 1, 1),
            256 * 4,
            stream,
            &mut args,
        )
    }

    /// Raw pointer version for quantized weights with byte offset (MoE experts).
    pub fn forward_packed_ptr(
        &self,
        output: *mut f32,
        weight: *const u8,
        input: *const f32,
        out_dim: u32,
        in_dim: u32,
        func_name: &str,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = self.module.get_function(func_name)?;
        let mut out_ptr: *mut c_void = output.cast();
        let mut w_ptr: *const c_void = weight.cast();
        let mut in_ptr: *const c_void = input.cast();
        let mut od = out_dim as i32;
        let mut id = in_dim as i32;
        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(od).cast(),
            std::ptr::addr_of_mut!(id).cast(),
        ];
        let block_size = 256u32;
        func.launch(
            (out_dim, 1, 1),
            (block_size, 1, 1),
            256 * 4,
            stream,
            &mut args,
        )
    }
}

pub struct SiluMulKernel {
    module: Module,
    device: DeviceId,
}

impl SiluMulKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("silu_mul.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        size: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(gate.device(), self.device);
        assert_eq!(up.device(), self.device);
        assert_eq!(output.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("silu_mul_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut gate_ptr: *const c_void = gate.as_ptr().cast();
        let mut up_ptr: *const c_void = up.as_ptr().cast();
        let mut sz = size as i32;

        let mut args: [*mut c_void; 4] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(gate_ptr).cast(),
            std::ptr::addr_of_mut!(up_ptr).cast(),
            std::ptr::addr_of_mut!(sz).cast(),
        ];

        let block_size = 256u32;
        let grid_size = (size + block_size - 1) / block_size;
        func.launch(
            (grid_size, 1, 1),
            (block_size, 1, 1),
            0,
            stream,
            &mut args,
        )
    }
}

pub struct ResidualAddKernel {
    module: Module,
    device: DeviceId,
}

impl ResidualAddKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("residual_add.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        size: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(x.device(), self.device);
        assert_eq!(residual.device(), self.device);
        assert_eq!(output.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("residual_add_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut x_ptr: *const c_void = x.as_ptr().cast();
        let mut r_ptr: *const c_void = residual.as_ptr().cast();
        let mut sz = size as i32;

        let mut args: [*mut c_void; 4] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(x_ptr).cast(),
            std::ptr::addr_of_mut!(r_ptr).cast(),
            std::ptr::addr_of_mut!(sz).cast(),
        ];

        let block_size = 256u32;
        let grid_size = (size + block_size - 1) / block_size;
        func.launch(
            (grid_size, 1, 1),
            (block_size, 1, 1),
            0,
            stream,
            &mut args,
        )
    }

    /// GPU-side weighted accumulate: output[i] += weight * input[i]
    pub fn weighted_accumulate(
        &self,
        output: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        weight: f32,
        size: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = self.module.get_function("weighted_accumulate_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut w = weight;
        let mut sz = size as i32;

        let mut args: [*mut c_void; 4] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(w).cast(),
            std::ptr::addr_of_mut!(sz).cast(),
        ];

        let block_size = 256u32;
        let grid_size = (size + block_size - 1) / block_size;
        func.launch(
            (grid_size, 1, 1),
            (block_size, 1, 1),
            0,
            stream,
            &mut args,
        )
    }
}

pub struct EmbeddingKernel {
    module: Module,
    device: DeviceId,
}

impl EmbeddingKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("embedding.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        embed_table: &DeviceBuffer<u16>,
        token_id: i32,
        hidden_size: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(embed_table.device(), self.device);
        assert_eq!(output.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("embedding_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut table_ptr: *const c_void = embed_table.as_ptr().cast();
        let mut tok = token_id;
        let mut hs = hidden_size as i32;

        let mut args: [*mut c_void; 4] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(table_ptr).cast(),
            std::ptr::addr_of_mut!(tok).cast(),
            std::ptr::addr_of_mut!(hs).cast(),
        ];

        let block_size = 256u32.min(hidden_size);
        func.launch(
            (1, 1, 1),
            (block_size, 1, 1),
            0,
            stream,
            &mut args,
        )
    }
}

pub struct LmHeadKernel {
    module: Module,
    device: DeviceId,
}

impl LmHeadKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("lm_head.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        input: &DeviceBuffer<f32>,
        vocab_size: u32,
        hidden_size: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(input.device(), self.device);
        assert_eq!(output.device(), self.device);
        assert_eq!(weight.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("lm_head_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut w_ptr: *const c_void = weight.as_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut vs = vocab_size as i32;
        let mut hs = hidden_size as i32;

        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(vs).cast(),
            std::ptr::addr_of_mut!(hs).cast(),
        ];

        let block_size = 256u32.min(hidden_size);
        func.launch(
            (vocab_size, 1, 1),
            (block_size, 1, 1),
            256 * 4,
            stream,
            &mut args,
        )
    }
}

pub struct MRoPEKernel {
    module: Module,
    device: DeviceId,
}

impl MRoPEKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("mrope.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    /// Apply mRoPE in-place to Q and K tensors.
    ///
    /// q:            [num_q_heads, head_dim]
    /// k:            [num_kv_heads, head_dim]
    /// inv_freq:     [rope_dim/2]
    /// position_ids: [3] — temporal, height, width (all same for text-only)
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: &mut DeviceBuffer<f32>,
        k: &mut DeviceBuffer<f32>,
        inv_freq: &DeviceBuffer<f32>,
        position_ids: &DeviceBuffer<i32>,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rope_dim: u32,
        section0_pairs: u32,
        section1_pairs: u32,
        section2_pairs: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(q.device(), self.device);
        assert_eq!(k.device(), self.device);
        assert_eq!(inv_freq.device(), self.device);
        assert_eq!(position_ids.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("mrope_f32")?;

        let mut q_ptr: *mut c_void = q.as_mut_ptr().cast();
        let mut k_ptr: *mut c_void = k.as_mut_ptr().cast();
        let mut inv_ptr: *const c_void = inv_freq.as_ptr().cast();
        let mut pos_ptr: *const c_void = position_ids.as_ptr().cast();
        let mut nqh = num_q_heads as i32;
        let mut nkh = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut rd = rope_dim as i32;
        let mut s0 = section0_pairs as i32;
        let mut s1 = section1_pairs as i32;
        let mut s2 = section2_pairs as i32;

        let mut args: [*mut c_void; 11] = [
            std::ptr::addr_of_mut!(q_ptr).cast(),
            std::ptr::addr_of_mut!(k_ptr).cast(),
            std::ptr::addr_of_mut!(inv_ptr).cast(),
            std::ptr::addr_of_mut!(pos_ptr).cast(),
            std::ptr::addr_of_mut!(nqh).cast(),
            std::ptr::addr_of_mut!(nkh).cast(),
            std::ptr::addr_of_mut!(hd).cast(),
            std::ptr::addr_of_mut!(rd).cast(),
            std::ptr::addr_of_mut!(s0).cast(),
            std::ptr::addr_of_mut!(s1).cast(),
            std::ptr::addr_of_mut!(s2).cast(),
        ];

        let total_heads = num_q_heads + num_kv_heads;
        let total_pairs = rope_dim / 2;
        let block_size = 32u32.max(total_pairs).next_power_of_two().min(256);

        func.launch(
            (total_heads, 1, 1),
            (block_size, 1, 1),
            0,
            stream,
            &mut args,
        )
    }
}

pub struct FfnFusedKernel {
    module: Module,
    device: DeviceId,
}

impl FfnFusedKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("ffn_fused.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    /// Sub-kernel 1: RMSNorm → gate_proj + up_proj → silu(gate)*up → scratch
    ///
    /// input:      [hidden_size]
    /// rms_weight: [hidden_size]
    /// w_gate:     [intermediate_size, hidden_size]
    /// w_up:       [intermediate_size, hidden_size]
    /// scratch:    [intermediate_size]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_gate_up(
        &self,
        scratch: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        rms_weight: &DeviceBuffer<u16>,
        w_gate: &DeviceBuffer<u16>,
        w_up: &DeviceBuffer<u16>,
        hidden_size: u32,
        intermediate_size: u32,
        eps: f32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(input.device(), self.device);
        assert_eq!(rms_weight.device(), self.device);
        assert_eq!(w_gate.device(), self.device);
        assert_eq!(w_up.device(), self.device);
        assert_eq!(scratch.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("ffn_fused_gate_up_f32")?;

        let mut scratch_ptr: *mut c_void = scratch.as_mut_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut rw_ptr: *const c_void = rms_weight.as_ptr().cast();
        let mut wg_ptr: *const c_void = w_gate.as_ptr().cast();
        let mut wu_ptr: *const c_void = w_up.as_ptr().cast();
        let mut hs = hidden_size as i32;
        let mut is_ = intermediate_size as i32;
        let mut ep = eps;

        let mut args: [*mut c_void; 8] = [
            std::ptr::addr_of_mut!(scratch_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(rw_ptr).cast(),
            std::ptr::addr_of_mut!(wg_ptr).cast(),
            std::ptr::addr_of_mut!(wu_ptr).cast(),
            std::ptr::addr_of_mut!(hs).cast(),
            std::ptr::addr_of_mut!(is_).cast(),
            std::ptr::addr_of_mut!(ep).cast(),
        ];

        let block_size = 256u32.min(hidden_size);
        func.launch(
            (intermediate_size, 1, 1),
            (block_size, 1, 1),
            block_size * 4,
            stream,
            &mut args,
        )
    }

    /// Sub-kernel 2: down_proj(scratch) + residual(input) → output
    ///
    /// output:  [hidden_size]
    /// input:   [hidden_size] — original input (residual)
    /// w_down:  [hidden_size, intermediate_size]
    /// scratch: [intermediate_size]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_down_residual(
        &self,
        output: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        w_down: &DeviceBuffer<u16>,
        scratch: &DeviceBuffer<f32>,
        hidden_size: u32,
        intermediate_size: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(output.device(), self.device);
        assert_eq!(input.device(), self.device);
        assert_eq!(w_down.device(), self.device);
        assert_eq!(scratch.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("ffn_fused_down_residual_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut wd_ptr: *const c_void = w_down.as_ptr().cast();
        let mut sc_ptr: *const c_void = scratch.as_ptr().cast();
        let mut hs = hidden_size as i32;
        let mut is_ = intermediate_size as i32;

        let mut args: [*mut c_void; 6] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(wd_ptr).cast(),
            std::ptr::addr_of_mut!(sc_ptr).cast(),
            std::ptr::addr_of_mut!(hs).cast(),
            std::ptr::addr_of_mut!(is_).cast(),
        ];

        let block_size = 256u32.min(intermediate_size);
        func.launch(
            (hidden_size, 1, 1),
            (block_size, 1, 1),
            block_size * 4,
            stream,
            &mut args,
        )
    }
}

pub struct GqaAttentionKernel {
    module: Module,
    device: DeviceId,
}

impl GqaAttentionKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("gqa_attention.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    /// Single-token decode GQA attention.
    ///
    /// output:  [num_q_heads, head_dim]
    /// q:       [num_q_heads, head_dim]
    /// k_cache: [num_kv_heads, max_seq_len, head_dim]
    /// v_cache: [num_kv_heads, max_seq_len, head_dim]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        k_cache: &DeviceBuffer<f32>,
        v_cache: &DeviceBuffer<f32>,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        max_seq_len: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(output.device(), self.device);
        assert_eq!(q.device(), self.device);
        assert_eq!(k_cache.device(), self.device);
        assert_eq!(v_cache.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("gqa_attention_decode_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut q_ptr: *const c_void = q.as_ptr().cast();
        let mut k_ptr: *const c_void = k_cache.as_ptr().cast();
        let mut v_ptr: *const c_void = v_cache.as_ptr().cast();
        let mut nqh = num_q_heads as i32;
        let mut nkh = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sl = seq_len as i32;
        let mut msl = max_seq_len as i32;

        let mut args: [*mut c_void; 9] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(q_ptr).cast(),
            std::ptr::addr_of_mut!(k_ptr).cast(),
            std::ptr::addr_of_mut!(v_ptr).cast(),
            std::ptr::addr_of_mut!(nqh).cast(),
            std::ptr::addr_of_mut!(nkh).cast(),
            std::ptr::addr_of_mut!(hd).cast(),
            std::ptr::addr_of_mut!(sl).cast(),
            std::ptr::addr_of_mut!(msl).cast(),
        ];

        let block_size = head_dim.min(256);
        let shared_bytes = block_size * 4;

        func.launch(
            (num_q_heads, 1, 1),
            (block_size, 1, 1),
            shared_bytes,
            stream,
            &mut args,
        )
    }
}

pub struct GdnLayerFusedKernel {
    module: Module,
    device: DeviceId,
}

impl GdnLayerFusedKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("gdn_layer_fused.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    /// Fused GDN decode layer: RMSNorm + Q/K/V/g/b projections + recurrent step + output proj + residual.
    ///
    /// scratch: [2*num_heads*key_dim + num_heads*value_dim + 2*num_heads + num_heads*value_dim] floats
    /// state:   [num_heads * key_dim * value_dim] persistent recurrent state (updated in-place)
    /// output:  [hidden_size]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        scratch: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        rms_weight: &DeviceBuffer<f32>,
        w_q: &DeviceBuffer<f32>,
        w_k: &DeviceBuffer<f32>,
        w_v: &DeviceBuffer<f32>,
        w_g: &DeviceBuffer<f32>,
        w_b: &DeviceBuffer<f32>,
        w_o: &DeviceBuffer<f32>,
        state: &mut DeviceBuffer<f32>,
        hidden_size: u32,
        num_heads: u32,
        key_dim: u32,
        value_dim: u32,
        eps: f32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(output.device(), self.device);
        assert_eq!(scratch.device(), self.device);
        assert_eq!(input.device(), self.device);
        assert_eq!(rms_weight.device(), self.device);
        assert_eq!(w_q.device(), self.device);
        assert_eq!(w_k.device(), self.device);
        assert_eq!(w_v.device(), self.device);
        assert_eq!(w_g.device(), self.device);
        assert_eq!(w_b.device(), self.device);
        assert_eq!(w_o.device(), self.device);
        assert_eq!(state.device(), self.device);
        assert_eq!(stream.device(), self.device);

        // --- Sub-kernel 1: RMSNorm + projections ---
        {
            let func = self.module.get_function("gdn_fused_proj_f32")?;

            let mut scratch_ptr: *mut c_void = scratch.as_mut_ptr().cast();
            let mut input_ptr: *const c_void = input.as_ptr().cast();
            let mut rms_ptr: *const c_void = rms_weight.as_ptr().cast();
            let mut wq_ptr: *const c_void = w_q.as_ptr().cast();
            let mut wk_ptr: *const c_void = w_k.as_ptr().cast();
            let mut wv_ptr: *const c_void = w_v.as_ptr().cast();
            let mut wg_ptr: *const c_void = w_g.as_ptr().cast();
            let mut wb_ptr: *const c_void = w_b.as_ptr().cast();
            let mut hs = hidden_size as i32;
            let mut nh = num_heads as i32;
            let mut kd = key_dim as i32;
            let mut vd = value_dim as i32;
            let mut ep = eps;

            let mut args: [*mut c_void; 13] = [
                std::ptr::addr_of_mut!(scratch_ptr).cast(),
                std::ptr::addr_of_mut!(input_ptr).cast(),
                std::ptr::addr_of_mut!(rms_ptr).cast(),
                std::ptr::addr_of_mut!(wq_ptr).cast(),
                std::ptr::addr_of_mut!(wk_ptr).cast(),
                std::ptr::addr_of_mut!(wv_ptr).cast(),
                std::ptr::addr_of_mut!(wg_ptr).cast(),
                std::ptr::addr_of_mut!(wb_ptr).cast(),
                std::ptr::addr_of_mut!(hs).cast(),
                std::ptr::addr_of_mut!(nh).cast(),
                std::ptr::addr_of_mut!(kd).cast(),
                std::ptr::addr_of_mut!(vd).cast(),
                std::ptr::addr_of_mut!(ep).cast(),
            ];

            let nk = num_heads * key_dim;
            let nv = num_heads * value_dim;
            let grid = 2 * nk + nv + 2 * num_heads;
            let block_size = 256u32.min(hidden_size);
            let shared_bytes = block_size * 4;
            func.launch((grid, 1, 1), (block_size, 1, 1), shared_bytes, stream, &mut args)?;
        }

        // --- Sub-kernel 2: recurrent step ---
        {
            let func = self.module.get_function("gdn_fused_recurrent_f32")?;

            let mut scratch_ptr: *mut c_void = scratch.as_mut_ptr().cast();
            let mut state_ptr: *mut c_void = state.as_mut_ptr().cast();
            let mut nh = num_heads as i32;
            let mut kd = key_dim as i32;
            let mut vd = value_dim as i32;

            let mut args: [*mut c_void; 5] = [
                std::ptr::addr_of_mut!(scratch_ptr).cast(),
                std::ptr::addr_of_mut!(state_ptr).cast(),
                std::ptr::addr_of_mut!(nh).cast(),
                std::ptr::addr_of_mut!(kd).cast(),
                std::ptr::addr_of_mut!(vd).cast(),
            ];

            func.launch((num_heads, 1, 1), (256, 1, 1), 0, stream, &mut args)?;
        }

        // --- Sub-kernel 3: output projection + residual ---
        {
            let func = self.module.get_function("gdn_fused_output_f32")?;

            let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
            let mut scratch_ptr: *const c_void = scratch.as_ptr().cast();
            let mut input_ptr: *const c_void = input.as_ptr().cast();
            let mut wo_ptr: *const c_void = w_o.as_ptr().cast();
            let mut hs = hidden_size as i32;
            let mut nh = num_heads as i32;
            let mut kd = key_dim as i32;
            let mut vd = value_dim as i32;

            let mut args: [*mut c_void; 8] = [
                std::ptr::addr_of_mut!(out_ptr).cast(),
                std::ptr::addr_of_mut!(scratch_ptr).cast(),
                std::ptr::addr_of_mut!(input_ptr).cast(),
                std::ptr::addr_of_mut!(wo_ptr).cast(),
                std::ptr::addr_of_mut!(hs).cast(),
                std::ptr::addr_of_mut!(nh).cast(),
                std::ptr::addr_of_mut!(kd).cast(),
                std::ptr::addr_of_mut!(vd).cast(),
            ];

            let block_size = 256u32.min(hidden_size);
            let shared_bytes = block_size * 4;
            func.launch((hidden_size, 1, 1), (block_size, 1, 1), shared_bytes, stream, &mut args)?;
        }

        Ok(())
    }
}

pub struct CausalConv1dUpdateKernel {
    pub(crate) module: Module,
    device: DeviceId,
}

impl CausalConv1dUpdateKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("causal_conv1d_update.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    /// Decode-step causal 1D convolution update.
    ///
    /// state:  [conv_dim, kernel_size-1]  updated in-place
    /// input:  [conv_dim]
    /// weight: [conv_dim, kernel_size]
    /// output: [conv_dim]
    pub fn forward(
        &self,
        state: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        output: &mut DeviceBuffer<f32>,
        conv_dim: u32,
        kernel_size: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(state.device(), self.device);
        assert_eq!(input.device(), self.device);
        assert_eq!(weight.device(), self.device);
        assert_eq!(output.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("causal_conv1d_update_f32")?;

        let mut state_ptr: *mut c_void = state.as_mut_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut w_ptr: *const c_void = weight.as_ptr().cast();
        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut cd = conv_dim as i32;
        let mut ks = kernel_size as i32;

        let mut args: [*mut c_void; 6] = [
            std::ptr::addr_of_mut!(state_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(cd).cast(),
            std::ptr::addr_of_mut!(ks).cast(),
        ];

        let block_size = 256u32;
        let grid_size = (conv_dim + block_size - 1) / block_size;
        func.launch((grid_size, 1, 1), (block_size, 1, 1), 0, stream, &mut args)
    }

    /// Variant with per-channel bias (for Mamba2 conv1d).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_bias(
        &self,
        state: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        bias: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        conv_dim: u32,
        kernel_size: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = self.module.get_function("causal_conv1d_update_bias_f32")?;

        let mut state_ptr: *mut c_void = state.as_mut_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut w_ptr: *const c_void = weight.as_ptr().cast();
        let mut bias_ptr: *const c_void = bias.as_ptr().cast();
        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut cd = conv_dim as i32;
        let mut ks = kernel_size as i32;

        let mut args: [*mut c_void; 7] = [
            std::ptr::addr_of_mut!(state_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(bias_ptr).cast(),
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(cd).cast(),
            std::ptr::addr_of_mut!(ks).cast(),
        ];

        let block_size = 256u32;
        let grid_size = (conv_dim + block_size - 1) / block_size;
        func.launch((grid_size, 1, 1), (block_size, 1, 1), 0, stream, &mut args)
    }
}

pub struct QkNormKernel {
    module: Module,
    device: DeviceId,
}

impl QkNormKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("qk_norm.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    /// Per-head RMSNorm on Q and K in-place.
    ///
    /// q:        [num_q_heads, head_dim]
    /// k:        [num_kv_heads, head_dim]
    /// q_weight: [head_dim]
    /// k_weight: [head_dim]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: &mut DeviceBuffer<f32>,
        k: &mut DeviceBuffer<f32>,
        q_weight: &DeviceBuffer<u16>,
        k_weight: &DeviceBuffer<u16>,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        eps: f32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(q.device(), self.device);
        assert_eq!(k.device(), self.device);
        assert_eq!(q_weight.device(), self.device);
        assert_eq!(k_weight.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("qk_norm_f32")?;

        let mut q_ptr: *mut c_void = q.as_mut_ptr().cast();
        let mut k_ptr: *mut c_void = k.as_mut_ptr().cast();
        let mut qw_ptr: *const c_void = q_weight.as_ptr().cast();
        let mut kw_ptr: *const c_void = k_weight.as_ptr().cast();
        let mut nqh = num_q_heads as i32;
        let mut nkh = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;

        let mut args: [*mut c_void; 8] = [
            std::ptr::addr_of_mut!(q_ptr).cast(),
            std::ptr::addr_of_mut!(k_ptr).cast(),
            std::ptr::addr_of_mut!(qw_ptr).cast(),
            std::ptr::addr_of_mut!(kw_ptr).cast(),
            std::ptr::addr_of_mut!(nqh).cast(),
            std::ptr::addr_of_mut!(nkh).cast(),
            std::ptr::addr_of_mut!(hd).cast(),
            std::ptr::addr_of_mut!(ep).cast(),
        ];

        let total_heads = num_q_heads + num_kv_heads;
        let block_size = 256u32.min(head_dim);
        func.launch(
            (total_heads, 1, 1),
            (block_size, 1, 1),
            block_size * 4,
            stream,
            &mut args,
        )
    }
}

pub struct RmsNormGatedKernel {
    module: Module,
    device: DeviceId,
}

impl RmsNormGatedKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("rmsnorm_gated.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    /// Gated RMSNorm: output = rms_norm(x) * silu(z)
    ///
    /// x, z:   [num_heads, value_dim]
    /// weight: [value_dim]
    /// output: [num_heads, value_dim]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        z: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        num_heads: u32,
        value_dim: u32,
        eps: f32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(output.device(), self.device);
        assert_eq!(x.device(), self.device);
        assert_eq!(z.device(), self.device);
        assert_eq!(weight.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("rmsnorm_gated_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut x_ptr: *const c_void = x.as_ptr().cast();
        let mut z_ptr: *const c_void = z.as_ptr().cast();
        let mut w_ptr: *const c_void = weight.as_ptr().cast();
        let mut nh = num_heads as i32;
        let mut vd = value_dim as i32;
        let mut ep = eps;

        let mut args: [*mut c_void; 7] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(x_ptr).cast(),
            std::ptr::addr_of_mut!(z_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(nh).cast(),
            std::ptr::addr_of_mut!(vd).cast(),
            std::ptr::addr_of_mut!(ep).cast(),
        ];

        let block_size = 256u32.min(value_dim);
        func.launch(
            (num_heads, 1, 1),
            (block_size, 1, 1),
            block_size * 4,
            stream,
            &mut args,
        )
    }
}

pub struct OutputGateKernel {
    module: Module,
    device: DeviceId,
}

impl OutputGateKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("output_gate.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    /// output = attn_output * sigmoid(gate)
    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        attn_output: &DeviceBuffer<f32>,
        gate: &DeviceBuffer<f32>,
        size: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(output.device(), self.device);
        assert_eq!(attn_output.device(), self.device);
        assert_eq!(gate.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("output_gate_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut attn_ptr: *const c_void = attn_output.as_ptr().cast();
        let mut gate_ptr: *const c_void = gate.as_ptr().cast();
        let mut sz = size as i32;

        let mut args: [*mut c_void; 4] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(attn_ptr).cast(),
            std::ptr::addr_of_mut!(gate_ptr).cast(),
            std::ptr::addr_of_mut!(sz).cast(),
        ];

        let block_size = 256u32;
        let grid_size = (size + block_size - 1) / block_size;
        func.launch((grid_size, 1, 1), (block_size, 1, 1), 0, stream, &mut args)
    }
}

pub struct GdnGateKernel {
    module: Module,
    device: DeviceId,
}

impl GdnGateKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("gdn_gate.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    /// Compute GDN decay gate: gate[h] = exp(-exp(A_log[h]) * softplus(a[h] + dt_bias[h]))
    ///
    /// Output is in (0, 1) — a proper decay factor.
    pub fn forward(
        &self,
        gate: &mut DeviceBuffer<f32>,
        a_log: &DeviceBuffer<f32>,
        a: &DeviceBuffer<f32>,
        dt_bias: &DeviceBuffer<u16>,
        num_heads: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(gate.device(), self.device);
        assert_eq!(a_log.device(), self.device);
        assert_eq!(a.device(), self.device);
        assert_eq!(dt_bias.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("gdn_gate_f32")?;

        let mut gate_ptr: *mut c_void = gate.as_mut_ptr().cast();
        let mut alog_ptr: *const c_void = a_log.as_ptr().cast();
        let mut a_ptr: *const c_void = a.as_ptr().cast();
        let mut dt_ptr: *const c_void = dt_bias.as_ptr().cast();
        let mut nh = num_heads as i32;

        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(gate_ptr).cast(),
            std::ptr::addr_of_mut!(alog_ptr).cast(),
            std::ptr::addr_of_mut!(a_ptr).cast(),
            std::ptr::addr_of_mut!(dt_ptr).cast(),
            std::ptr::addr_of_mut!(nh).cast(),
        ];

        let block_size = 256u32;
        let grid_size = (num_heads + block_size - 1) / block_size;
        func.launch((grid_size, 1, 1), (block_size, 1, 1), 0, stream, &mut args)
    }
}

pub struct GdnRecurrentStepV2Kernel {
    module: Module,
    device: DeviceId,
}

impl GdnRecurrentStepV2Kernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("gdn_recurrent_step_v2.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    /// GDN recurrent step v2: pre-computed decay gate, L2-normalized Q and K.
    ///
    /// q, k:   [num_heads, key_dim]
    /// v:      [num_heads, value_dim]
    /// gate:   [num_heads] — pre-computed decay factors in (0,1) from GdnGateKernel
    /// b:      [num_heads] — beta logits (sigmoid applied inside)
    /// state:  [num_heads, key_dim, value_dim]  updated in-place
    /// output: [num_heads, value_dim]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: &DeviceBuffer<f32>,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
        gate: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        state: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        num_heads: u32,
        key_dim: u32,
        value_dim: u32,
        gqa_group: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(q.device(), self.device);
        assert_eq!(k.device(), self.device);
        assert_eq!(v.device(), self.device);
        assert_eq!(gate.device(), self.device);
        assert_eq!(b.device(), self.device);
        assert_eq!(state.device(), self.device);
        assert_eq!(output.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("gdn_recurrent_step_v2_f32")?;

        let mut q_ptr: *const c_void = q.as_ptr().cast();
        let mut k_ptr: *const c_void = k.as_ptr().cast();
        let mut v_ptr: *const c_void = v.as_ptr().cast();
        let mut gate_ptr: *const c_void = gate.as_ptr().cast();
        let mut b_ptr: *const c_void = b.as_ptr().cast();
        let mut state_ptr: *mut c_void = state.as_mut_ptr().cast();
        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut kd = key_dim as i32;
        let mut vd = value_dim as i32;
        let mut gqa = gqa_group as i32;

        let mut args: [*mut c_void; 10] = [
            std::ptr::addr_of_mut!(q_ptr).cast(),
            std::ptr::addr_of_mut!(k_ptr).cast(),
            std::ptr::addr_of_mut!(v_ptr).cast(),
            std::ptr::addr_of_mut!(gate_ptr).cast(),
            std::ptr::addr_of_mut!(b_ptr).cast(),
            std::ptr::addr_of_mut!(state_ptr).cast(),
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(kd).cast(),
            std::ptr::addr_of_mut!(vd).cast(),
            std::ptr::addr_of_mut!(gqa).cast(),
        ];

        // Shared memory: 2 * block_size floats for q/k norm reductions
        let block_size = 256u32;
        let shared_bytes = block_size * 4 * 2;
        func.launch(
            (num_heads, 1, 1),
            (block_size, 1, 1),
            shared_bytes,
            stream,
            &mut args,
        )
    }
}

pub struct AttnLayerFusedKernel {
    module: Module,
    device: DeviceId,
}

impl AttnLayerFusedKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("attn_layer_fused.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    /// Fused attention decode layer:
    ///   RMSNorm → QKV proj → mRoPE → KV-cache write → GQA attn → output proj → residual
    ///
    /// scratch:      [5120 f32] temporary workspace (q/k/v/attn_out)
    /// input:        [hidden_size]
    /// rms_weight:   [hidden_size]
    /// w_q:          [num_q_heads*head_dim, hidden_size]
    /// w_k:          [num_kv_heads*head_dim, hidden_size]
    /// w_v:          [num_kv_heads*head_dim, hidden_size]
    /// w_o:          [hidden_size, num_q_heads*head_dim]
    /// inv_freq:     [rope_dim/2]
    /// position_ids: [3]
    /// k_cache:      [max_seq_len, num_kv_heads, head_dim]
    /// v_cache:      [max_seq_len, num_kv_heads, head_dim]
    /// output:       [hidden_size]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        output: &mut DeviceBuffer<f32>,
        scratch: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        rms_weight: &DeviceBuffer<f32>,
        w_q: &DeviceBuffer<f32>,
        w_k: &DeviceBuffer<f32>,
        w_v: &DeviceBuffer<f32>,
        w_o: &DeviceBuffer<f32>,
        inv_freq: &DeviceBuffer<f32>,
        position_ids: &DeviceBuffer<i32>,
        k_cache: &mut DeviceBuffer<f32>,
        v_cache: &mut DeviceBuffer<f32>,
        hidden_size: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rope_dim: u32,
        section0_pairs: u32,
        section1_pairs: u32,
        section2_pairs: u32,
        seq_pos: u32,
        seq_len: u32,
        max_seq_len: u32,
        eps: f32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(output.device(), self.device);
        assert_eq!(scratch.device(), self.device);
        assert_eq!(input.device(), self.device);
        assert_eq!(rms_weight.device(), self.device);
        assert_eq!(w_q.device(), self.device);
        assert_eq!(w_k.device(), self.device);
        assert_eq!(w_v.device(), self.device);
        assert_eq!(w_o.device(), self.device);
        assert_eq!(inv_freq.device(), self.device);
        assert_eq!(position_ids.device(), self.device);
        assert_eq!(k_cache.device(), self.device);
        assert_eq!(v_cache.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let q_out_dim  = num_q_heads  * head_dim;
        let kv_out_dim = num_kv_heads * head_dim;

        // Sub-kernel 1: RMSNorm + QKV projections
        {
            let func = self.module.get_function("attn_fused_proj_f32")?;

            let mut scratch_ptr: *mut c_void   = scratch.as_mut_ptr().cast();
            let mut in_ptr: *const c_void       = input.as_ptr().cast();
            let mut rms_ptr: *const c_void      = rms_weight.as_ptr().cast();
            let mut wq_ptr: *const c_void       = w_q.as_ptr().cast();
            let mut wk_ptr: *const c_void       = w_k.as_ptr().cast();
            let mut wv_ptr: *const c_void       = w_v.as_ptr().cast();
            let mut hs  = hidden_size  as i32;
            let mut nqh = num_q_heads  as i32;
            let mut nkh = num_kv_heads as i32;
            let mut hd  = head_dim     as i32;
            let mut ep  = eps;

            let mut args: [*mut c_void; 11] = [
                std::ptr::addr_of_mut!(scratch_ptr).cast(),
                std::ptr::addr_of_mut!(in_ptr).cast(),
                std::ptr::addr_of_mut!(rms_ptr).cast(),
                std::ptr::addr_of_mut!(wq_ptr).cast(),
                std::ptr::addr_of_mut!(wk_ptr).cast(),
                std::ptr::addr_of_mut!(wv_ptr).cast(),
                std::ptr::addr_of_mut!(hs).cast(),
                std::ptr::addr_of_mut!(nqh).cast(),
                std::ptr::addr_of_mut!(nkh).cast(),
                std::ptr::addr_of_mut!(hd).cast(),
                std::ptr::addr_of_mut!(ep).cast(),
            ];

            let grid_size  = q_out_dim + kv_out_dim * 2;
            let block_size = 256u32.min(hidden_size);
            func.launch(
                (grid_size, 1, 1),
                (block_size, 1, 1),
                block_size * 4,
                stream,
                &mut args,
            )?;
        }

        // Sub-kernel 2: mRoPE on Q,K + KV cache write
        {
            let func = self.module.get_function("attn_fused_rope_cache_f32")?;

            let mut scratch_ptr: *mut c_void   = scratch.as_mut_ptr().cast();
            let mut kc_ptr: *mut c_void         = k_cache.as_mut_ptr().cast();
            let mut vc_ptr: *mut c_void         = v_cache.as_mut_ptr().cast();
            let mut inv_ptr: *const c_void      = inv_freq.as_ptr().cast();
            let mut pos_ptr: *const c_void      = position_ids.as_ptr().cast();
            let mut nqh = num_q_heads   as i32;
            let mut nkh = num_kv_heads  as i32;
            let mut hd  = head_dim      as i32;
            let mut rd  = rope_dim      as i32;
            let mut s0  = section0_pairs as i32;
            let mut s1  = section1_pairs as i32;
            let mut s2  = section2_pairs as i32;
            let mut sp  = seq_pos        as i32;
            let mut msl = max_seq_len    as i32;

            let mut args: [*mut c_void; 14] = [
                std::ptr::addr_of_mut!(scratch_ptr).cast(),
                std::ptr::addr_of_mut!(kc_ptr).cast(),
                std::ptr::addr_of_mut!(vc_ptr).cast(),
                std::ptr::addr_of_mut!(inv_ptr).cast(),
                std::ptr::addr_of_mut!(pos_ptr).cast(),
                std::ptr::addr_of_mut!(nqh).cast(),
                std::ptr::addr_of_mut!(nkh).cast(),
                std::ptr::addr_of_mut!(hd).cast(),
                std::ptr::addr_of_mut!(rd).cast(),
                std::ptr::addr_of_mut!(s0).cast(),
                std::ptr::addr_of_mut!(s1).cast(),
                std::ptr::addr_of_mut!(s2).cast(),
                std::ptr::addr_of_mut!(sp).cast(),
                std::ptr::addr_of_mut!(msl).cast(),
            ];

            let total_heads = num_q_heads + num_kv_heads;
            let total_pairs = rope_dim / 2;
            // block must cover both rope pairs and head_dim for cache write
            let block_size = 32u32.max(total_pairs).next_power_of_two().max(head_dim.min(256)).min(256);
            func.launch(
                (total_heads, 1, 1),
                (block_size, 1, 1),
                0,
                stream,
                &mut args,
            )?;
        }

        // Sub-kernel 3: GQA decode attention
        {
            let func = self.module.get_function("attn_fused_attention_f32")?;

            let mut scratch_ptr: *mut c_void   = scratch.as_mut_ptr().cast();
            let mut kc_ptr: *const c_void       = k_cache.as_ptr().cast();
            let mut vc_ptr: *const c_void       = v_cache.as_ptr().cast();
            let mut nqh = num_q_heads  as i32;
            let mut nkh = num_kv_heads as i32;
            let mut hd  = head_dim     as i32;
            let mut sl  = seq_len      as i32;
            let mut msl = max_seq_len  as i32;

            let mut args: [*mut c_void; 8] = [
                std::ptr::addr_of_mut!(scratch_ptr).cast(),
                std::ptr::addr_of_mut!(kc_ptr).cast(),
                std::ptr::addr_of_mut!(vc_ptr).cast(),
                std::ptr::addr_of_mut!(nqh).cast(),
                std::ptr::addr_of_mut!(nkh).cast(),
                std::ptr::addr_of_mut!(hd).cast(),
                std::ptr::addr_of_mut!(sl).cast(),
                std::ptr::addr_of_mut!(msl).cast(),
            ];

            let block_size   = head_dim.min(256);
            let shared_bytes = block_size * 4;
            func.launch(
                (num_q_heads, 1, 1),
                (block_size, 1, 1),
                shared_bytes,
                stream,
                &mut args,
            )?;
        }

        // Sub-kernel 4: Output projection + residual
        {
            let func = self.module.get_function("attn_fused_output_f32")?;

            let mut out_ptr: *mut c_void       = output.as_mut_ptr().cast();
            let mut scratch_ptr: *const c_void = scratch.as_ptr().cast();
            let mut in_ptr: *const c_void       = input.as_ptr().cast();
            let mut wo_ptr: *const c_void       = w_o.as_ptr().cast();
            let mut hs  = hidden_size  as i32;
            let mut nqh = num_q_heads  as i32;
            let mut nkh = num_kv_heads as i32;
            let mut hd  = head_dim     as i32;

            let mut args: [*mut c_void; 8] = [
                std::ptr::addr_of_mut!(out_ptr).cast(),
                std::ptr::addr_of_mut!(scratch_ptr).cast(),
                std::ptr::addr_of_mut!(in_ptr).cast(),
                std::ptr::addr_of_mut!(wo_ptr).cast(),
                std::ptr::addr_of_mut!(hs).cast(),
                std::ptr::addr_of_mut!(nqh).cast(),
                std::ptr::addr_of_mut!(nkh).cast(),
                std::ptr::addr_of_mut!(hd).cast(),
            ];

            let block_size = 256u32.min(q_out_dim);
            func.launch(
                (hidden_size, 1, 1),
                (block_size, 1, 1),
                256 * 4,
                stream,
                &mut args,
            )?;
        }

        Ok(())
    }
}

/// Mamba2 selective state update kernel.
pub struct SelectiveStateUpdateKernel {
    pub(crate) module: Module,
    _device: DeviceId,
}

impl SelectiveStateUpdateKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("selective_state_update.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, _device: device })
    }

    /// Mamba2 SSM recurrence: updates ssm_state in-place and writes output.
    ///
    /// ssm_state: [num_heads, head_dim, state_size] f32 — updated in-place
    /// x:         [num_heads * head_dim] f32
    /// dt:        [num_heads] f32 (before softplus)
    /// dt_bias:   [num_heads] f32
    /// a_log:     [num_heads] f32
    /// b:         [n_groups * state_size] f32
    /// c:         [n_groups * state_size] f32
    /// d:         [num_heads] f32
    /// output:    [num_heads * head_dim] f32
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ssm_state: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        dt: &DeviceBuffer<f32>,
        dt_bias: &DeviceBuffer<f32>,
        a_log: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &DeviceBuffer<f32>,
        d_param: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        num_heads: u32,
        head_dim: u32,
        state_size: u32,
        n_groups: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = self.module.get_function("selective_state_update_f32")?;

        let mut state_ptr: *mut c_void = ssm_state.as_mut_ptr().cast();
        let mut x_ptr: *const c_void = x.as_ptr().cast();
        let mut dt_ptr: *const c_void = dt.as_ptr().cast();
        let mut dt_bias_ptr: *const c_void = dt_bias.as_ptr().cast();
        let mut a_log_ptr: *const c_void = a_log.as_ptr().cast();
        let mut b_ptr: *const c_void = b.as_ptr().cast();
        let mut c_ptr: *const c_void = c.as_ptr().cast();
        let mut d_ptr: *const c_void = d_param.as_ptr().cast();
        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut ss = state_size as i32;
        let mut ng = n_groups as i32;

        let mut args: [*mut c_void; 13] = [
            std::ptr::addr_of_mut!(state_ptr).cast(),
            std::ptr::addr_of_mut!(x_ptr).cast(),
            std::ptr::addr_of_mut!(dt_ptr).cast(),
            std::ptr::addr_of_mut!(dt_bias_ptr).cast(),
            std::ptr::addr_of_mut!(a_log_ptr).cast(),
            std::ptr::addr_of_mut!(b_ptr).cast(),
            std::ptr::addr_of_mut!(c_ptr).cast(),
            std::ptr::addr_of_mut!(d_ptr).cast(),
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(nh).cast(),
            std::ptr::addr_of_mut!(hd).cast(),
            std::ptr::addr_of_mut!(ss).cast(),
            std::ptr::addr_of_mut!(ng).cast(),
        ];

        func.launch(
            (num_heads, 1, 1),
            (256, 1, 1),
            0,
            stream,
            &mut args,
        )
    }
}
