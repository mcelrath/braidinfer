use braidinfer_core::types::DeviceId;
use braidinfer_hip::HipResult;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::module::Module;
use braidinfer_hip::stream::Stream;
use std::ffi::c_void;

use super::kernel_dir;

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
        self.forward_ptr(
            q.as_mut_ptr(), k.as_mut_ptr(),
            q_weight.as_ptr(), k_weight.as_ptr(),
            num_q_heads, num_kv_heads, head_dim, eps, stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_ptr(
        &self,
        q: *mut f32,
        k: *mut f32,
        q_weight: *const u16,
        k_weight: *const u16,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        eps: f32,
        stream: &Stream,
    ) -> HipResult<()> {
        let func = self.module.get_function("qk_norm_f32")?;
        let mut qp = q as *mut std::ffi::c_void;
        let mut kp = k as *mut std::ffi::c_void;
        let mut qwp = q_weight as *const std::ffi::c_void;
        let mut kwp = k_weight as *const std::ffi::c_void;
        let mut nqh = num_q_heads as i32;
        let mut nkh = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;
        let mut args: [*mut std::ffi::c_void; 8] = [
            std::ptr::addr_of_mut!(qp).cast(),
            std::ptr::addr_of_mut!(kp).cast(),
            std::ptr::addr_of_mut!(qwp).cast(),
            std::ptr::addr_of_mut!(kwp).cast(),
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
    pub(crate) module: Module,
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
