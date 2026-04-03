use std::path::PathBuf;

pub(crate) fn kernel_dir() -> PathBuf {
    PathBuf::from(env!("BRAIDINFER_KERNEL_DIR"))
}

mod kernel_matmul;
mod kernel_norm;
mod kernel_fused;

pub use kernel_matmul::*;
pub use kernel_norm::*;
pub use kernel_fused::*;

use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use braidinfer_hip::HipResult;
use std::ffi::c_void;


/// GPU-resident argmax: returns index of max element without transferring logits to CPU.
pub struct ArgmaxKernel {
    module: Module,
    _device: DeviceId,
}

impl ArgmaxKernel {
    pub fn load(device: DeviceId) -> HipResult<Self> {
        let path = kernel_dir().join("argmax.hsaco");
        let module = Module::load(device, &path)?;
        Ok(Self { module, _device: device })
    }

    /// Find argmax of input[0..size], write result index to output[0].
    /// Returns the index as u32 after synchronizing.
    pub fn forward(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<i32>,
        size: u32,
        stream: &Stream,
    ) -> HipResult<u32> {
        let func = self.module.get_function("argmax_f32")?;

        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut sz = size as i32;

        let mut args: [*mut c_void; 3] = [
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(sz).cast(),
        ];

        let block_size = 256u32;
        let shared_mem = block_size * 4 * 2; // float + int per thread
        func.launch((1, 1, 1), (block_size, 1, 1), shared_mem, stream, &mut args)?;
        stream.synchronize()?;

        let mut result = [0i32];
        output.copy_to_host(&mut result)?;
        Ok(result[0] as u32)
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

    /// ReLU²: output[i] = max(0, input[i])²
    /// For Nemotron-H MoE (relu2 activation, no gate_proj).
    pub fn relu_squared(
        &self,
        output: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        size: u32,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = self.module.get_function("relu_squared_f32")?;

        let mut out_ptr: *mut c_void = output.as_mut_ptr().cast();
        let mut in_ptr: *const c_void = input.as_ptr().cast();
        let mut sz = size as i32;

        let mut args: [*mut c_void; 3] = [
            std::ptr::addr_of_mut!(out_ptr).cast(),
            std::ptr::addr_of_mut!(in_ptr).cast(),
            std::ptr::addr_of_mut!(sz).cast(),
        ];

        let block_size = 256u32;
        let grid_size = (size + block_size - 1) / block_size;
        func.launch((grid_size, 1, 1), (block_size, 1, 1), 0, stream, &mut args)
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

