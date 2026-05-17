/// MTYPE audit: dump (memory_type, alloc_flags) for every cross-agent or
/// reused buffer in the multi-GPU decode path.  Per GFX1100_ARCH.md §5.5
/// Rule 5 — `mem_type=2 alloc_flags=0x0` == cached device buffer == L2-stale
/// candidate.  `0x3` = UC.  `1` = host.
///
/// Activated by `BRAIDINFER_MTYPE_AUDIT=1` at model-load time.
pub fn dump(model: &crate::model::Model) {
    eprintln!("=== MTYPE audit (5ax-decode) ===");
    eprintln!("Legend: mem_type 1=Host 2=Device | alloc_flags 0x0=cached 0x1=fine-grained 0x3=UC");
    let dev = |b: &braidinfer_hip::DeviceBuffer<f32>, name: &str| {
        match b.pointer_attributes() {
            Ok((t, f)) => eprintln!("  {name:46} mem_type={t} alloc_flags=0x{f:x}"),
            Err(e) => eprintln!("  {name:46} ERR {e:?}"),
        }
    };
    let host = |b: &braidinfer_hip::MappedHostBuffer<f32>, name: &str| {
        match b.pointer_attributes() {
            Ok((t, f)) => eprintln!("  {name:46} mem_type={t} alloc_flags=0x{f:x}"),
            Err(e) => eprintln!("  {name:46} ERR {e:?}"),
        }
    };

    eprintln!("-- activations (GPU 0) --");
    dev(&model.activations.hidden, "activations.hidden");
    dev(&model.activations.normed, "activations.normed");
    host(&model.activations.normed_stage, "activations.normed_stage");
    dev(&model.activations.q_attn, "activations.q_attn");
    dev(&model.activations.k_attn, "activations.k_attn");
    dev(&model.activations.v_attn, "activations.v_attn");
    dev(&model.activations.gate_attn, "activations.gate_attn");
    dev(&model.activations.attn_out, "activations.attn_out");
    dev(&model.activations.gated_out, "activations.gated_out");
    dev(&model.activations.residual, "activations.residual");

    if let Some(legacy) = model.legacy_kv_caches.as_ref() {
        eprintln!("-- legacy_kv_caches (GPU 0, prefill K/V) --");
        for (i, kv) in legacy.iter().enumerate() {
            dev(&kv.k, &format!("legacy_kv_caches[{i}].k"));
            if i == 0 { dev(&kv.v, &format!("legacy_kv_caches[{i}].v")); }
            if i >= 2 { eprintln!("  ... ({} total layers)", legacy.len()); break; }
        }
    }

    if let Some(mgpu) = model.multi_gpu.as_ref() {
        for (gpu_i, w) in mgpu.workers.iter().enumerate() {
            eprintln!("-- worker[{gpu_i}] (device {}) --", w.device.0);
            if let Some(b) = w.attn_normed.as_ref() { dev(b, &format!("workers[{gpu_i}].attn_normed")); }
            if let Some(b) = w.attn_q_gate.as_ref() { dev(b, &format!("workers[{gpu_i}].attn_q_gate")); }
            if let Some(b) = w.attn_k.as_ref()      { dev(b, &format!("workers[{gpu_i}].attn_k")); }
            if let Some(b) = w.attn_v.as_ref()      { dev(b, &format!("workers[{gpu_i}].attn_v")); }
            if let Some(b) = w.attn_gate.as_ref()   { dev(b, &format!("workers[{gpu_i}].attn_gate")); }
            if let Some(b) = w.attn_out.as_ref()    { host(b, &format!("workers[{gpu_i}].attn_out")); }
            for (i, kv) in w.attn_kv_caches.iter().enumerate() {
                if i < 2 {
                    dev(&kv.k, &format!("workers[{gpu_i}].attn_kv_caches[{i}].k"));
                    dev(&kv.v, &format!("workers[{gpu_i}].attn_kv_caches[{i}].v"));
                }
            }
            if w.attn_kv_caches.len() > 2 {
                eprintln!("  ... ({} attn_kv_cache layers)", w.attn_kv_caches.len());
            }
        }
    }

    if let Some(p2p) = model.moe_p2p.as_ref() {
        eprintln!("-- moe_p2p --");
        host(&p2p.output_slots, "moe_p2p.output_slots");
        host(&p2p.activation_staging, "moe_p2p.activation_staging");
    }
    eprintln!("=== end MTYPE audit ===");
}
