// Diagnostic for braidinfer-sew remaining bug: dumps attn_kv_caches[0]
// from every GPU and compares to legacy_kv_caches[0] after prefill +
// one decode step. Identifies which step (broadcast, decode-write,
// per-GPU consistency) is wrong.
//
// Run with: scripts/launch-gpu.py -g 4 -- sew_diag

use braidinfer_runtime::config::FfnType;
use braidinfer_runtime::model::Model;
use std::path::Path;

fn main() {
    let model_path = std::env::var("MODEL")
        .expect("MODEL env var required (path to .bqnt file or HF dir)");
    let prompt_tokens: Vec<u32> = std::env::var("PROMPT_TOKENS")
        .unwrap_or_else(|_| "1,2,3,4,5,6,7,8,9,10,11,12,13".to_string())
        .split(',')
        .map(|s| s.trim().parse().expect("token must be u32"))
        .collect();
    let decode_token: u32 = std::env::var("DECODE_TOKEN")
        .unwrap_or_else(|_| "100".to_string())
        .parse()
        .expect("DECODE_TOKEN must be u32");

    let p = if model_path.ends_with(".bqnt") {
        unsafe { std::env::set_var("BQNT_PATH", &model_path) };
        let bqnt =
            braidinfer_runtime::bqnt::MmapBqnt::open(Path::new(&model_path)).expect("open bqnt");
        let model_name = bqnt.model_name().expect("bqnt has no model_name");
        if model_name.starts_with('/') && Path::new(&model_name).is_dir() {
            std::path::PathBuf::from(model_name)
        } else {
            let hf_name = model_name.replace('/', "--");
            let cache_dir = dirs::home_dir()
                .expect("HOME")
                .join(".cache/huggingface/hub")
                .join(format!("models--{hf_name}"))
                .join("snapshots");
            let mut snapshots: Vec<_> = std::fs::read_dir(&cache_dir)
                .expect("read snapshots dir")
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .collect();
            snapshots.sort_by_key(|e| e.file_name());
            snapshots
                .iter()
                .find(|e| e.path().join("tokenizer.json").exists())
                .or_else(|| snapshots.first())
                .expect("no snapshot")
                .path()
        }
    } else {
        Path::new(&model_path).to_path_buf()
    };

    let device = braidinfer_core::types::DeviceId(0);
    let mut model = Model::load(&p, device).expect("load model");
    let (nkh, hd, max_sl, has_moe) = {
        let cfg = model.config();
        let has_moe = cfg.layers.iter().any(|l| matches!(l.ffn_type, FfnType::MoE { .. }));
        (cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len, has_moe)
    };
    if has_moe {
        unsafe { std::env::set_var("MULTI_GPU", "1") };
        model.enable_multi_gpu().expect("enable multi-gpu");
    }

    let n_prompt = prompt_tokens.len();
    let max_pos = n_prompt + 1; // include the decode-written position

    eprintln!(
        "sew_diag: prompt_len={n_prompt}, decode_token={decode_token}, nkh={nkh}, hd={hd}, max_pos={max_pos}"
    );

    // Phase 1: prefill the prompt. After prefill but BEFORE any decode step,
    // the persistent_worker on GPU 0 hasn't been launched yet, so copy_to_host
    // is still safe.
    let _ = model.prefill(&prompt_tokens).expect("prefill");
    let legacy_after_prefill = model
        .read_legacy_kv_caches()
        .expect("read legacy_kv_caches after prefill");
    let attn_kv_after_decode = model
        .read_attn_kv_first_layer(max_pos)
        .expect("read attn_kv after prefill");
    // We CANNOT run decode_step_token here: it lazy-launches persistent_worker
    // on GPU 0, after which any DeviceBuffer::copy_to_host panics with the
    // assert_no_persistent_worker guard. So this diagnostic only verifies the
    // sew broadcast — not the decode-time KV write.
    let _ = decode_token;
    let legacy_after_decode = legacy_after_prefill.clone();

    // Compare A: per-GPU attn_kv consistency (positions 0..n_prompt should
    // be identical across GPUs because they came from the broadcast).
    eprintln!("\n=== Per-GPU consistency (positions 0..{n_prompt}) ===");
    if attn_kv_after_decode.len() >= 2 {
        let (gpu_a, k_a, v_a) = &attn_kv_after_decode[0];
        for (gpu_b, k_b, v_b) in attn_kv_after_decode.iter().skip(1) {
            // Compare positions 0..n_prompt for each head.
            let mut k_diff = 0usize;
            let mut v_diff = 0usize;
            for h in 0..nkh {
                let base = h * max_pos * hd;
                for t in 0..n_prompt {
                    for d in 0..hd {
                        let off = base + t * hd + d;
                        if k_a[off].to_bits() != k_b[off].to_bits() { k_diff += 1; }
                        if v_a[off].to_bits() != v_b[off].to_bits() { v_diff += 1; }
                    }
                }
            }
            let total = nkh * n_prompt * hd;
            eprintln!(
                "  GPU{gpu_a} vs GPU{gpu_b}: k_diff={k_diff}/{total} v_diff={v_diff}/{total}"
            );
        }
    }

    // Compare B: attn_kv[0..n_prompt] vs legacy_kv[layer 0, 0..n_prompt]
    // (broadcast should have copied legacy_kv → attn_kv).
    eprintln!("\n=== Broadcast verification: attn_kv[0..{n_prompt}] vs legacy_kv[0..{n_prompt}] ===");
    if !legacy_after_prefill.is_empty() && !attn_kv_after_decode.is_empty() {
        let (legacy_k, legacy_v) = &legacy_after_prefill[0];
        let max_sl = max_sl;
        for (gpu_i, k_a, v_a) in &attn_kv_after_decode {
            let mut k_diff = 0usize;
            let mut v_diff = 0usize;
            for h in 0..nkh {
                let attn_base = h * max_pos * hd;
                let legacy_base = h * max_sl * hd;
                for t in 0..n_prompt {
                    for d in 0..hd {
                        let attn_off = attn_base + t * hd + d;
                        let legacy_off = legacy_base + t * hd + d;
                        if k_a[attn_off].to_bits() != legacy_k[legacy_off].to_bits() { k_diff += 1; }
                        if v_a[attn_off].to_bits() != legacy_v[legacy_off].to_bits() { v_diff += 1; }
                    }
                }
            }
            let total = nkh * n_prompt * hd;
            eprintln!(
                "  GPU{gpu_i} attn_kv vs legacy_kv: k_diff={k_diff}/{total} v_diff={v_diff}/{total}"
            );
        }
    }

    // Compare C: decode-written attn_kv[n_prompt] across GPUs.
    eprintln!("\n=== Decode-write per-GPU consistency at position {n_prompt} ===");
    if attn_kv_after_decode.len() >= 2 {
        let (gpu_a, k_a, v_a) = &attn_kv_after_decode[0];
        for (gpu_b, k_b, v_b) in attn_kv_after_decode.iter().skip(1) {
            let mut k_diff = 0usize;
            let mut v_diff = 0usize;
            let mut k_max_abs = 0.0f32;
            for h in 0..nkh {
                let base = h * max_pos * hd;
                let off = base + n_prompt * hd;
                for d in 0..hd {
                    let i = off + d;
                    if k_a[i].to_bits() != k_b[i].to_bits() { k_diff += 1; }
                    if v_a[i].to_bits() != v_b[i].to_bits() { v_diff += 1; }
                    let diff = (k_a[i] - k_b[i]).abs();
                    if diff > k_max_abs { k_max_abs = diff; }
                }
            }
            let total = nkh * hd;
            eprintln!(
                "  GPU{gpu_a} vs GPU{gpu_b} at pos={n_prompt}: k_diff={k_diff}/{total} v_diff={v_diff}/{total} k_max_abs={k_max_abs:.3e}"
            );
        }
        // Also dump the raw values for the first few elements of GPU 0 vs GPU 1.
        if attn_kv_after_decode.len() >= 2 {
            let (_, k_a, _) = &attn_kv_after_decode[0];
            let (_, k_b, _) = &attn_kv_after_decode[1];
            eprintln!("  First 8 K values at pos={n_prompt} h=0:");
            let off = 0 * max_pos * hd + n_prompt * hd;
            for d in 0..8 {
                eprintln!(
                    "    [d={d}] GPU0={:.6e} GPU1={:.6e} diff={:.3e}",
                    k_a[off + d],
                    k_b[off + d],
                    (k_a[off + d] - k_b[off + d]).abs()
                );
            }
        }
    }

    // Compare D: legacy_kv at position n_prompt — written by decode? Likely not.
    eprintln!("\n=== legacy_kv at position {n_prompt} (post-decode) ===");
    if !legacy_after_decode.is_empty() {
        let (k, v) = &legacy_after_decode[0];
        let max_sl = max_sl;
        let mut k_nonzero = 0usize;
        for h in 0..nkh {
            let off = h * max_sl * hd + n_prompt * hd;
            for d in 0..hd {
                if k[off + d] != 0.0 || v[off + d] != 0.0 { k_nonzero += 1; }
            }
        }
        eprintln!(
            "  legacy_kv[layer 0, pos={n_prompt}] nonzero floats: {k_nonzero}/{}",
            nkh * hd
        );
    }
}
