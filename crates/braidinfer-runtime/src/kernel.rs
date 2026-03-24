use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::HipResult;
use std::ffi::c_void;
use std::path::PathBuf;

fn kernel_dir() -> PathBuf {
    PathBuf::from(env!("BRAIDINFER_KERNEL_DIR"))
}

pub struct GdnRecurrentStepKernel {
    module: Module,
    device: DeviceId,
}

impl GdnRecurrentStepKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("gdn_recurrent_step.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, device })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: &DeviceBuffer<f32>,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
        g: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        state: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        num_heads: u32,
        key_dim: u32,
        value_dim: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(q.device(), self.device);
        assert_eq!(k.device(), self.device);
        assert_eq!(v.device(), self.device);
        assert_eq!(g.device(), self.device);
        assert_eq!(b.device(), self.device);
        assert_eq!(state.device(), self.device);
        assert_eq!(output.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("gdn_recurrent_step_f32")?;

        let mut q_ptr: *const c_void = q.as_ptr().cast();
        let mut k_ptr: *const c_void = k.as_ptr().cast();
        let mut v_ptr: *const c_void = v.as_ptr().cast();
        let mut g_ptr: *const c_void = g.as_ptr().cast();
        let mut b_ptr: *const c_void = b.as_ptr().cast();
        let mut state_ptr: *mut c_void = state.as_mut_ptr().cast();
        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut kd = key_dim as i32;
        let mut vd = value_dim as i32;

        let mut args: [*mut c_void; 9] = [
            std::ptr::addr_of_mut!(q_ptr).cast(),
            std::ptr::addr_of_mut!(k_ptr).cast(),
            std::ptr::addr_of_mut!(v_ptr).cast(),
            std::ptr::addr_of_mut!(g_ptr).cast(),
            std::ptr::addr_of_mut!(b_ptr).cast(),
            std::ptr::addr_of_mut!(state_ptr).cast(),
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(kd).cast(),
            std::ptr::addr_of_mut!(vd).cast(),
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
        weight: &DeviceBuffer<f32>,
        num_rows: u32,
        hidden_size: u32,
        eps: f32,
        stream: &Stream,
    ) -> HipResult<()> {
        assert_eq!(input.device(), self.device);
        assert_eq!(output.device(), self.device);
        assert_eq!(weight.device(), self.device);
        assert_eq!(stream.device(), self.device);

        let func = self.module.get_function("rmsnorm_f32")?;

        // kernel_params are pointers-to-arg-values. The GPU reads through these
        // indirections; the const→mut cast at the FFI boundary is required by
        // hipModuleLaunchKernel's signature but does not cause writes.
        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut w_ptr: *const c_void = weight.as_ptr().cast();
        let mut hs = hidden_size as i32;
        let mut ep = eps;

        let mut args: [*mut c_void; 5] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(w_ptr).cast(),
            std::ptr::addr_of_mut!(hs).cast(),
            std::ptr::addr_of_mut!(ep).cast(),
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
        weight: &DeviceBuffer<f32>,
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
        embed_table: &DeviceBuffer<f32>,
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
        weight: &DeviceBuffer<f32>,
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
        rms_weight: &DeviceBuffer<f32>,
        w_gate: &DeviceBuffer<f32>,
        w_up: &DeviceBuffer<f32>,
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
        w_down: &DeviceBuffer<f32>,
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
    /// k_cache: [max_seq_len, num_kv_heads, head_dim]
    /// v_cache: [max_seq_len, num_kv_heads, head_dim]
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
