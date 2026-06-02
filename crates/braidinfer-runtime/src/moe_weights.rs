//! MoE weight loading and distribution across GPUs.
//! Extracted from weights.rs for diff-locality: MoE-loading changes no longer
//! force review of ActivationBuffers field additions and vice-versa.

use braidinfer_core::safetensors::SafeTensorSet;
use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::DeviceBuffer;
use braidinfer_hip::ffi;

use crate::bqnt::{MmapBqnt, code_to_format};
use crate::config::*;
use crate::weights::{
    ModelError,
    MoeWeights, GpuExpertBuffer, DistributedMoeWeights, DenseFfnWeights,
    LinearWeight, PackedWeights, WeightFormat, WeightQuantMode,
    load_weight_bf16, load_linear_weight, load_linear_weight_cached,
    load_linear_weight_bqnt, host_bf16_to_linear_weight,
    host_bf16_quantize_upload_cache, find_weight_name, weight_format_for,
};

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
    // bd 4ayf.12: (arena_base_ptr, data_start) — Packed latent_proj becomes a non-owning arena
    // view (no copy) when present.
    arena: Option<(*const u8, u64)>,
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
                // bd 4ayf.12: arena view (no copy) when present; else alloc + copy.
                let data_buf = if let Some((arena_base, data_start)) = arena {
                    let off = (entry.data_offset - data_start) as usize;
                    unsafe { DeviceBuffer::<u8>::view(device, arena_base.add(off), data.len()) }
                } else {
                    let mut buf = DeviceBuffer::<u8>::alloc(device, data.len())?;
                    buf.copy_from_host(data)?;
                    buf
                };
                return Ok(Some(LinearWeight::Packed(PackedWeights { data: data_buf, format: fmt, out_dim, in_dim })));
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

/// Concatenate per-expert packed bytes from bqnt into a single fused LinearWeight.
/// `first_name` must be the name of expert 0's weight (e.g. "...experts.0.up_proj.weight").
/// All expert names are derived by replacing ".0." with ".{e}.".
/// Returns None if any expert's data is missing or format is non-linear.
fn concat_per_expert_bqnt(
    b: &MmapBqnt,
    first_name: &str,
    ne: usize,
    device: DeviceId,
) -> Option<LinearWeight> {
    let first_entry = b.entry(first_name)?;
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
            if let Ok(lw) = load_linear_weight_bqnt(b, name, device, None) {
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
    // Probe both bqnt (name existence only) and safetensors — gate weight stays bf16.
    let gate_name = [
        format!("{prefix}mlp.gate.weight"),
        format!("{prefix}gate.weight"),
        format!("{prefix}block_sparse_moe.gate.weight"),
        format!("{prefix}mlp.router.weight"),
    ]
    .into_iter()
    .find(|n| bqnt.map(|b| b.entry(n).is_some()).unwrap_or(false) || st.tensor_data(n).is_ok())
    .ok_or_else(|| ModelError::MissingWeight(format!("{prefix}mlp.gate.weight (or variants)")))?;
    // bd-2kgw fix: the router gate IS stored in the bqnt (bf16), but this previously loaded it
    // ONLY from safetensors (st). For a self-contained .bqnt (empty st) that is a guaranteed
    // spurious MissingWeight even though gate_name was just found in the bqnt. Load from the
    // bqnt first (load_weight_bf16_bqnt), fall back to st. (Every other MoE weight already goes
    // bqnt-first via load_lw / load_linear_weight_bqnt; the gate was the lone st-only loader.)
    let gate = match bqnt {
        Some(b) => crate::weights::load_weight_bf16_bqnt(b, &gate_name, device, ne * hs, None)
            .or_else(|_| load_weight_bf16(st, &gate_name, device, ne * hs))?,
        None => load_weight_bf16(st, &gate_name, device, ne * hs)?,
    };

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
        .any(|n| st.tensor_data(n).is_ok() || bqnt.map_or(false, |b| b.entry(n).is_some()));

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
        let bqnt_fused = bqnt.and_then(|b| load_linear_weight_bqnt(b, &fused_name, device, None).ok());
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
                let first_name = [
                    format!("{prefix}experts.0.up_proj.weight"),
                    format!("{prefix}mlp.experts.0.up_proj.weight"),
                ]
                .into_iter()
                .find(|n| b.entry(n).is_some())?;
                concat_per_expert_bqnt(b, &first_name, ne, device)
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
        let bqnt_down = bqnt.and_then(|b| load_linear_weight_bqnt(b, &down_name, device, None).ok());
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
                concat_per_expert_bqnt(b, &first_name, ne, device)
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

    // Shared expert gate (optional). bd-2kgw: presence + load must be bqnt-aware — previously
    // decided presence via st.tensor_data ONLY, so for a self-contained .bqnt (empty st) it was
    // silently None even when the gate IS in the bqnt (another 4ayf empty-st gap, like the router
    // gate above). Probe bqnt OR st for presence; load bqnt-first with st fallback.
    let shared_gate_name = format!("{prefix}mlp.shared_expert_gate.weight");
    let shared_gate_present = bqnt.map_or(false, |b| b.entry(&shared_gate_name).is_some())
        || st.tensor_data(&shared_gate_name).is_ok();
    let shared_expert_gate = if shared_gate_present {
        Some(match bqnt {
            Some(b) => crate::weights::load_weight_bf16_bqnt(b, &shared_gate_name, device, hs, None)
                .or_else(|_| load_weight_bf16(st, &shared_gate_name, device, hs))?,
            None => load_weight_bf16(st, &shared_gate_name, device, hs)?,
        })
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
        // bd 4ayf A3.2.3b (+v1 backward-compat fix): bias raw bytes bqnt-first ONLY if present
        // at the expected f32 size (ne*4); a v1 bqnt may store it bf16/differently, so a
        // size-mismatch falls back to st rather than misreading the bytes as f32.
        let raw = match bqnt
            .and_then(|b| b.tensor_data(&name))
            .filter(|d| d.len() == ne * 4)
        {
            Some(d) => d,
            None => st
                .tensor_data(&name)
                .map_err(|_| ModelError::MissingWeight(name.clone()))?,
        };
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
    // bd 4ayf.12: MoE models skip the arena (experts are fused, can't be views — see model_load
    // gate), so latent_proj loads per-tensor (arena=None).
    let fc1_latent_proj = load_latent_proj(prefix, "fc1_latent_proj", device, bqnt, st, wq, None)?;
    let fc2_latent_proj = load_latent_proj(prefix, "fc2_latent_proj", device, bqnt, st, wq, None)?;

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
