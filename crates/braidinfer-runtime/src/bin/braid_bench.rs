use braidinfer_core::types::DeviceId;
use braidinfer_runtime::config::FfnType;
use braidinfer_runtime::model::Model;
use std::path::Path;
use std::time::Instant;

const DEFAULT_MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn vram_free_per_gpu() -> Vec<usize> {
    let mut count: i32 = 0;
    unsafe { braidinfer_hip::ffi::hipGetDeviceCount(&mut count) };
    (0..count)
        .map(|i| {
            unsafe { braidinfer_hip::ffi::hipSetDevice(i) };
            let mut free: usize = 0;
            let mut total: usize = 0;
            unsafe { braidinfer_hip::ffi::hipMemGetInfo(&mut free, &mut total) };
            free
        })
        .collect()
}

fn resolve_hf_dir(bqnt_path: &str) -> Option<String> {
    let bqnt = braidinfer_runtime::bqnt::MmapBqnt::open(std::path::Path::new(bqnt_path)).ok()?;
    let model_name = bqnt.model_name()?;
    if model_name.starts_with('/') {
        let p = std::path::Path::new(&model_name);
        if p.is_dir() {
            return Some(model_name);
        }
    }
    let hf_name = model_name.replace('/', "--");
    let cache_dir = dirs::home_dir()?
        .join(".cache/huggingface/hub")
        .join(format!("models--{hf_name}"))
        .join("snapshots");
    let mut snapshots: Vec<_> = std::fs::read_dir(&cache_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    snapshots.sort_by_key(|e| e.file_name());
    snapshots
        .iter()
        .find(|e| e.path().join("tokenizer.json").exists())
        .or_else(|| snapshots.first())
        .map(|e| e.path().to_string_lossy().to_string())
}

fn load_model(model_dir: &Path, multi_gpu: bool) -> Model {
    let device = DeviceId(0);
    let max_seq_len: Option<usize> = std::env::var("MAX_SEQ_LEN")
        .ok()
        .and_then(|v| v.parse().ok());
    let mut model = Model::load_with_max_seq_len(model_dir, device, max_seq_len)
        .expect("load model");
    if multi_gpu {
        model.enable_multi_gpu().expect("enable multi-GPU");
    }
    model
}

fn percentile(sorted: &[u128], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (pct / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)] as f64 / 1_000_000.0
}

fn bench_decode(model: &mut Model, warmup: usize, runs: usize) {
    println!("=== Decode benchmark (warmup={warmup} runs={runs}) ===");
    let token_id = 9906u32; // "Hello"

    // Warmup: advance position to warmup, let the model run without timing
    for p in 0..warmup as u32 {
        model.decode_step_token(token_id, p).expect("decode");
    }

    // Sync before timing: ensure all warmup GPU work is complete before t0.
    model.stream().synchronize().expect("stream sync");

    // Timed: run `runs` consecutive decode steps and time each
    let mut pos = warmup as u32;
    let mut times_ns = Vec::with_capacity(runs);
    for _ in 0..runs {
        model.stream().synchronize().expect("stream sync");
        let t0 = Instant::now();
        model.decode_step_token(token_id, pos).expect("decode");
        times_ns.push(t0.elapsed().as_nanos());
        pos += 1;
    }

    times_ns.sort_unstable();
    let avg_ms = times_ns.iter().sum::<u128>() as f64 / runs as f64 / 1_000_000.0;
    let p10 = percentile(&times_ns, 10.0);
    let p50 = percentile(&times_ns, 50.0);
    let p90 = percentile(&times_ns, 90.0);
    println!(
        "  positions {}-{}  avg={avg_ms:.2}ms  p10={p10:.2}ms  p50={p50:.2}ms  p90={p90:.2}ms  {:.1} tok/s",
        warmup, warmup + runs - 1, 1000.0 / avg_ms,
    );
}

fn bench_prefill(model: &mut Model, token_counts: &[usize]) {
    // MUST be called before bench_decode (before persistent worker starts) so reset_state works.
    // Calls model.prefill() which uses batched megakernel compile_prefill path.
    println!("=== Prefill benchmark (batched megakernel) ===");
    let token_id = 9906u32;

    // Warmup
    let warmup_tokens: Vec<u32> = vec![token_id; 8];
    model.reset_state().expect("reset");
    model.prefill(&warmup_tokens).expect("prefill warmup");

    for &n in token_counts {
        let tokens: Vec<u32> = vec![token_id; n];
        model.reset_state().expect("reset");
        model.stream().synchronize().expect("stream sync");
        let t0 = Instant::now();
        model.prefill(&tokens).expect("prefill");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let toks_per_sec = n as f64 / (elapsed_ms / 1000.0);
        println!("  n={n:4}  {elapsed_ms:7.1}ms  {toks_per_sec:6.1} tok/s");
    }
    model.reset_state().expect("reset");
}

fn bench_coherence(model: &mut Model, prompt_len: usize) {
    // MUST be called before bench_decode (before persistent worker starts) so reset_state works.
    // Validates batched prefill logits == sequential decode logits at the last token position.
    println!("=== Coherence test (batched prefill vs sequential decode) ===");
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 9906 + (i % 100)).collect();

    let top10 = |logits: &[f32]| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(10);
        idx
    };

    // Sequential reference: single-token prefill per step. Uses the full MoE dispatch path
    // (including P2P workers for multi-GPU), so the reference is correct for all model types.
    // decode_step_paged is NOT used because for multi-GPU MoE models it skips expert computation
    // (experts are lite-loaded on GPU 0 when MULTI_GPU=1).
    // Compute fingerprint of last-token logits: sum of finite values + count of NaN/Inf.
    // ULP-level diffs between runs will produce different sums.
    let fingerprint = |logits: &[f32]| -> (f64, usize, usize, f32, f32) {
        let mut sum = 0.0f64;
        let mut nans = 0;
        let mut infs = 0;
        let mut max = f32::NEG_INFINITY;
        let mut min = f32::INFINITY;
        for &v in logits {
            if v.is_nan() { nans += 1; continue; }
            if v.is_infinite() { infs += 1; continue; }
            sum += v as f64;
            if v > max { max = v; }
            if v < min { min = v; }
        }
        (sum, nans, infs, min, max)
    };

    // Per-step fingerprint to find FIRST DIVERGING STEP between two consecutive runs.
    // ALSO dumps GDN state (recurrent + conv) after step 0 to compare across runs.
    // If logits are bit-exact at step 0 but state bytes differ → confirms FP non-associativity
    // in GDN state writes (multi-block-per-head). If state bytes ARE bit-exact → step 1's reads
    // produce different output despite same input (mysterious, deeper investigation needed).
    let collect_step_logits = |m: &mut Model, prompt: &[u32]| -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<u8>, Vec<(Vec<f32>, Vec<f32>)>, Vec<Vec<f32>>, Vec<[u32; 9]>) {
        m.reset_state().expect("reset");
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(prompt.len());
        let mut state_after_step0: Vec<Vec<f32>> = Vec::new();
        let mut conv_state_after_step0: Vec<Vec<f32>> = Vec::new();
        let mut kv_after_step0: Vec<u8> = Vec::new();
        let mut legacy_kv_after_step0: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        let mut k_trace_after_step0: Vec<Vec<f32>> = Vec::new();
        let mut mrope_dump_after_step0: Vec<[u32; 9]> = Vec::new();
        for (i, &tok) in prompt.iter().enumerate() {
            let l = m.prefill(&[tok]).expect("seq prefill (per-step)");
            if i == 0 {
                state_after_step0 = m.read_gdn_state().expect("read gdn state");
                conv_state_after_step0 = m.read_gdn_conv_state().expect("read gdn conv state");
                kv_after_step0 = m.read_kv_chunk_slot0().expect("read kv chunk slot 0");
                legacy_kv_after_step0 = m.read_legacy_kv_caches().expect("read legacy kv caches");
                k_trace_after_step0 = m.read_k_trace_phases().expect("read k trace");
                mrope_dump_after_step0 = m.read_mrope_dump().expect("read mrope dump");
            }
            out.push(l);
        }
        (out, state_after_step0, conv_state_after_step0, kv_after_step0, legacy_kv_after_step0, k_trace_after_step0, mrope_dump_after_step0)
    };
    let (run1_steps, run1_gdn_state, run1_conv_state, run1_kv, run1_legacy_kv, run1_k_trace, run1_mrope_dump) = collect_step_logits(model, &prompt);
    let (run2_steps, run2_gdn_state, run2_conv_state, run2_kv, run2_legacy_kv, run2_k_trace, run2_mrope_dump) = collect_step_logits(model, &prompt);

    // Compare GDN recurrent state bytes after step 0
    println!("  GDN state comparison after step 0 (run1 vs run2):");
    let mut total_recur_diff = 0usize;
    let mut total_recur_floats = 0usize;
    let mut max_recur_abs = 0.0f32;
    let mut total_run1_recur_sum = 0.0f64;
    for (i, (l1, l2)) in run1_gdn_state.iter().zip(run2_gdn_state.iter()).enumerate() {
        let n = l1.len();
        let bit_diff = l1.iter().zip(l2.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let max_abs = l1.iter().zip(l2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let l1_abs_sum: f64 = l1.iter().map(|&v| v.abs() as f64).sum();
        let l1_max = l1.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        total_recur_diff += bit_diff;
        total_recur_floats += n;
        total_run1_recur_sum += l1_abs_sum;
        if max_abs > max_recur_abs { max_recur_abs = max_abs; }
        if i < 3 || bit_diff > 0 {
            println!("    GDN_RECUR layer {i}: bit_diff={bit_diff}/{n} max_abs_diff={max_abs:.3e} run1_abs_sum={l1_abs_sum:.3e} run1_max={l1_max:.3e}");
        }
    }
    println!("    GDN_RECUR total: {total_recur_diff}/{total_recur_floats} max_abs_diff={max_recur_abs:.3e} run1_total_abs_sum={total_run1_recur_sum:.3e}");

    // Compare GDN conv1d state bytes after step 0
    let mut total_conv_diff = 0usize;
    let mut total_conv_floats = 0usize;
    let mut max_conv_abs = 0.0f32;
    for (i, (l1, l2)) in run1_conv_state.iter().zip(run2_conv_state.iter()).enumerate() {
        let n = l1.len();
        let bit_diff = l1.iter().zip(l2.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let max_abs = l1.iter().zip(l2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        total_conv_diff += bit_diff;
        total_conv_floats += n;
        if max_abs > max_conv_abs { max_conv_abs = max_abs; }
        if i < 3 || bit_diff > 0 {
            println!("    GDN_CONV layer {i}: bit_diff={bit_diff}/{n} max_abs={max_abs:.3e}");
        }
    }
    println!("    GDN_CONV total: {total_conv_diff}/{total_conv_floats} max_abs={max_conv_abs:.3e}");

    // Compare KV chunk slot 0 bytes after step 0
    let kv_byte_diff = run1_kv.iter().zip(run2_kv.iter()).filter(|(a, b)| a != b).count();
    let kv_total = run1_kv.len();
    let run1_kv_nonzero = run1_kv.iter().filter(|&&b| b != 0).count();
    println!("    KV_CHUNK_SLOT0: byte_diff={kv_byte_diff}/{kv_total} run1_nonzero_bytes={run1_kv_nonzero}");

    // Compare legacy_kv_caches contents per layer after step 0 (multi-GPU MoE path uses these).
    // If K/V bit-exact across runs but logits at step 1 diverge → divergence is in step 1 read.
    // If K/V differ between runs → step 0's K/V write is non-deterministic.
    if !run1_legacy_kv.is_empty() {
        let mut total_k_diff = 0usize;
        let mut total_v_diff = 0usize;
        let mut total_floats = 0usize;
        let mut max_k_abs = 0.0f32;
        let mut max_v_abs = 0.0f32;
        let mut total_run1_k_abs_sum = 0.0f64;
        let mut first_diverge_layer: Option<usize> = None;
        for (i, ((k1, v1), (k2, v2))) in run1_legacy_kv.iter().zip(run2_legacy_kv.iter()).enumerate() {
            let k_diff = k1.iter().zip(k2.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let v_diff = v1.iter().zip(v2.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let k_max = k1.iter().zip(k2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let v_max = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let k1_abs_sum: f64 = k1.iter().map(|&v| v.abs() as f64).sum();
            total_k_diff += k_diff;
            total_v_diff += v_diff;
            total_floats += k1.len();
            total_run1_k_abs_sum += k1_abs_sum;
            if k_max > max_k_abs { max_k_abs = k_max; }
            if v_max > max_v_abs { max_v_abs = v_max; }
            if (k_diff > 0 || v_diff > 0) && first_diverge_layer.is_none() {
                first_diverge_layer = Some(i);
            }
            if i < 3 || k_diff > 0 || v_diff > 0 {
                println!("    LEGACY_KV layer {i}: k_diff={k_diff}/{} v_diff={v_diff}/{} max_k_abs={k_max:.3e} max_v_abs={v_max:.3e} run1_k_abs_sum={k1_abs_sum:.3e}",
                    k1.len(), v1.len());
            }
        }
        println!("    LEGACY_KV total: k_diff={total_k_diff}/{total_floats} v_diff={total_v_diff}/{total_floats} max_k_abs={max_k_abs:.3e} max_v_abs={max_v_abs:.3e} run1_k_total_abs_sum={total_run1_k_abs_sum:.3e}");
        if let Some(l) = first_diverge_layer {
            println!("    LEGACY_KV first divergent attn layer: {l}");
            // Detailed per-head + first-divergent-offsets for layer 0.
            // K layout: [nkh, max_seq_len, head_dim]. For seq_len=1, position 0 is first.
            // We need nkh and head_dim from the model config. Use the cache size to back them out.
            let (k1, _) = &run1_legacy_kv[l];
            let (k1_full, _) = &run1_legacy_kv[l];
            let (k2_full, _) = &run2_legacy_kv[l];
            // Heuristic: hd is the smallest power of 2 dividing 128 evenly that matches typical sizes.
            // For qwen35 family hd=128. If k_diff is exactly 128, that's one head_dim's worth.
            let total_floats = k1_full.len();
            // Try common values of head_dim. If 128 floats differ and divergence is contiguous,
            // we should see it at one (head_idx, position_idx) location.
            // Try hd=128 (most common); compute nkh, max_seq_len from total_floats / hd / max_seq_len.
            // We don't know max_seq_len directly; assume max_seq_len = total_floats / (nkh * hd).
            // For Qwen3 35B-A3B: nkh=16, max_seq_len=2048, hd=128 → 16*2048*128 = 4194304 ✓
            // Use actual config dimensions instead of guessing.
            let hd_guess = model.config.head_dim;
            let nkh_guess = model.config.num_kv_heads;
            let max_sl_guess = total_floats / (nkh_guess * hd_guess);
            if max_sl_guess * nkh_guess * hd_guess == total_floats {
                println!("    LEGACY_KV layer {l} per-head breakdown (nkh={nkh_guess}, hd={hd_guess}, max_sl={max_sl_guess}, rope_dim={}):", model.config.rope_dim);
                let head_stride = max_sl_guess * hd_guess;
                for h in 0..nkh_guess {
                    let head_diff: usize = (0..head_stride).filter(|&i| {
                        let off = h * head_stride + i;
                        k1_full[off].to_bits() != k2_full[off].to_bits()
                    }).count();
                    if head_diff > 0 {
                        // For each token position in this head, count the diffs.
                        let mut pos_diffs: Vec<(usize, usize)> = Vec::new();
                        for t in 0..max_sl_guess {
                            let pd: usize = (0..hd_guess).filter(|&d| {
                                let off = h * head_stride + t * hd_guess + d;
                                k1_full[off].to_bits() != k2_full[off].to_bits()
                            }).count();
                            if pd > 0 { pos_diffs.push((t, pd)); }
                        }
                        let pos_summary: String = pos_diffs.iter().take(5).map(|(t, n)| format!("t={}:{}", t, n)).collect::<Vec<_>>().join(", ");
                        println!("      head {h}: k_diff={head_diff} positions=[{pos_summary}]");
                    }
                }
                // Dump first 4 (offset, run1, run2) tuples for head with diverge.
                let mut first_tuples = Vec::new();
                for off in 0..total_floats.min(2_000_000) {
                    if k1_full[off].to_bits() != k2_full[off].to_bits() {
                        let h = off / head_stride;
                        let rem = off % head_stride;
                        let t = rem / hd_guess;
                        let d = rem % hd_guess;
                        first_tuples.push((h, t, d, k1_full[off], k2_full[off]));
                        if first_tuples.len() >= 8 { break; }
                    }
                }
                if !first_tuples.is_empty() {
                    println!("    LEGACY_KV layer {l} first 8 divergent (h,t,d) → (run1, run2):");
                    for (h, t, d, v1, v2) in &first_tuples {
                        println!("      [{h:2}, {t:2}, {d:3}] = ({v1:.6e}, {v2:.6e})  diff={:.3e}", v1 - v2);
                    }
                }
            }
        }
    }
    // 5ax K-trace per-phase comparison after step 0 (first attention layer).
    // Phases:
    //   0 = pre-LINEAR_PROJ K (normed input) — diverges → RMSNorm or earlier path is non-det
    //   1 = post-LINEAR_PROJ K — diverges only here → LINEAR_PROJ K is non-det
    //   2 = post-QK_NORM K — diverges only here → QK_NORM is non-det
    //   (post-MROPE K = legacy_kv_caches[0][0..nkh*hd], already shown above)
    if !run1_k_trace.is_empty() && run1_k_trace.len() == run2_k_trace.len() {
        let phase_names = ["pre-LINEAR_PROJ (normed)", "post-LINEAR_PROJ K", "post-QK_NORM K"];
        println!("  K-TRACE per-phase comparison (first attention layer, step 0):");
        for (p, (a, b)) in run1_k_trace.iter().zip(run2_k_trace.iter()).enumerate() {
            let n = a.len();
            let bit_diff = a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            let max_abs = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            let abs_sum: f64 = a.iter().map(|&v| v.abs() as f64).sum();
            let label = phase_names.get(p).copied().unwrap_or("?");
            println!("    phase {p} ({label}): bit_diff={bit_diff}/{n} max_abs={max_abs:.3e} run1_abs_sum={abs_sum:.3e}");
        }
    }

    // 5ax MROPE in-kernel dump comparison (first attention layer, K heads, token 0).
    // Per (k_head, pair): [pair, pos, theta, cos, sin, x0, x1, out0, out1]
    if !run1_mrope_dump.is_empty() && run1_mrope_dump.len() == run2_mrope_dump.len() {
        let n = run1_mrope_dump.len();
        let total_pairs = model.config.rope_dim / 2;
        let nkh = model.config.num_kv_heads;
        let mut field_diff = [0usize; 9];
        let mut first_diff_entry: Option<usize> = None;
        for (e, (a, b)) in run1_mrope_dump.iter().zip(run2_mrope_dump.iter()).enumerate() {
            for f in 0..9 {
                if a[f] != b[f] {
                    field_diff[f] += 1;
                    if first_diff_entry.is_none() {
                        first_diff_entry = Some(e);
                    }
                }
            }
        }
        let labels = ["pair", "pos", "theta", "cos", "sin", "x0", "x1", "out0", "out1"];
        println!("  MROPE-DUMP comparison (first attention layer, K heads, token 0, n={n} entries={nkh}×{total_pairs}):");
        for f in 0..9 {
            println!("    field {} ({}): bit_diff={}/{}", f, labels[f], field_diff[f], n);
        }
        // Print first 8 entries for run1 (raw values) so we can SEE what cos/sin/pos actually are.
        println!("  MROPE-DUMP run1 first 8 entries (h, pair, pos, theta, cos, sin, x0, x1, out0, out1):");
        for e in 0..n.min(8) {
            let h = e / total_pairs;
            let p = e % total_pairs;
            let r = &run1_mrope_dump[e];
            let theta = f32::from_bits(r[2]);
            let cos = f32::from_bits(r[3]);
            let sin = f32::from_bits(r[4]);
            let x0 = f32::from_bits(r[5]);
            let x1 = f32::from_bits(r[6]);
            let o0 = f32::from_bits(r[7]);
            let o1 = f32::from_bits(r[8]);
            println!("    [h={h}, p={p}] pair={} pos={} θ={theta:.3e} cos={cos:.6} sin={sin:.6} x0={x0:.4e} x1={x1:.4e} out0={o0:.4e} out1={o1:.4e}", r[0], r[1] as i32);
        }
        // If field_diff is non-zero anywhere, print the FIRST divergent entry side-by-side.
        if let Some(e) = first_diff_entry {
            let h = e / total_pairs;
            let p = e % total_pairs;
            let a = &run1_mrope_dump[e];
            let b = &run2_mrope_dump[e];
            println!("  MROPE-DUMP first divergent entry [h={h}, p={p}]:");
            for f in 0..9 {
                let av = if f < 2 { a[f] as i64 } else { f32::from_bits(a[f]).to_bits() as i64 };
                let bv = if f < 2 { b[f] as i64 } else { f32::from_bits(b[f]).to_bits() as i64 };
                let af = f32::from_bits(a[f]);
                let bf = f32::from_bits(b[f]);
                let mark = if a[f] != b[f] { "*" } else { " " };
                if f < 2 {
                    println!("    {} {}: run1={} run2={}", mark, labels[f], a[f] as i32, b[f] as i32);
                } else {
                    println!("    {} {}: run1={af:.6e} (0x{:08x}) run2={bf:.6e} (0x{:08x})", mark, labels[f], a[f], b[f]);
                }
                let _ = (av, bv);
            }
        }
    }

    println!("  per-step seq->seq comparison:");
    for (i, (l1, l2)) in run1_steps.iter().zip(run2_steps.iter()).enumerate() {
        let s1: f64 = l1.iter().filter(|v| v.is_finite()).map(|&v| v as f64).sum();
        let s2: f64 = l2.iter().filter(|v| v.is_finite()).map(|&v| v as f64).sum();
        let n_bit_diff = l1.iter().zip(l2.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let max_abs = l1.iter().zip(l2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("    step {i}: bit_diff={n_bit_diff}/{} max_abs={max_abs:.3e} sum1={s1:.4e} sum2={s2:.4e}", l1.len());
    }

    // Per-instruction dump comparison.
    // Run two consecutive prefill streams with enable_dump active, compare per-op outputs.
    // Find FIRST divergent op = the kernel op that introduces the non-determinism.
    if std::env::var("BZ0_DUMP").is_ok() {
        println!("  per-instruction dump comparison (BZ0_DUMP=1):");
        // Need megakernel_paged created first. Run one decode_step to lazy-init it.
        model.reset_state().expect("reset");
        let _ = model.prefill(&[prompt[0]]).expect("prime megakernel_paged");
        // Now enable dump.
        const MAX_SLOTS: i32 = 4096;
        let collect_dump = |m: &mut Model, prompt: &[u32]| -> Vec<(u32, u32, Vec<f32>)> {
            m.reset_state().expect("reset");
            m.enable_paged_dump(MAX_SLOTS).expect("enable dump");
            for &tok in prompt.iter() {
                let _ = m.prefill(&[tok]).expect("prefill (with dump)");
            }
            m.read_paged_dump().expect("read dump")
        };
        // Probe with enough tokens to expose step 1+ divergence (per the optimization
        // matrix in bd, divergence at step 1+ at -O3 default).
        let probe_prompt: Vec<u32> = prompt.iter().take(2).copied().collect();
        let dump1 = collect_dump(model, &probe_prompt);
        let dump2 = collect_dump(model, &probe_prompt);
        println!("    dump1 slots: {}, dump2 slots: {}", dump1.len(), dump2.len());
        let max_print = std::cmp::min(dump1.len(), dump2.len());
        let mut divergent_count = 0usize;
        let mut first_5: Vec<usize> = Vec::new();
        for slot_idx in 0..max_print {
            let (op1, idx1, data1) = &dump1[slot_idx];
            let (op2, idx2, data2) = &dump2[slot_idx];
            if op1 != op2 || idx1 != idx2 { continue; }
            let bit_diff = data1.iter().zip(data2.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            if bit_diff > 0 {
                divergent_count += 1;
                if first_5.len() < 5 {
                    first_5.push(slot_idx);
                }
            }
        }
        println!("    {} divergent ops out of {} compared. First 5 divergent slots: {:?}",
                 divergent_count, max_print, first_5);
        if let Some(&div_slot) = first_5.first() {
            let lo = div_slot.saturating_sub(15);
            let hi = (div_slot + 5).min(max_print);
            println!("    Context around first divergence (wider):");
            for s in lo..hi {
                let (op1, idx1, data1) = &dump1[s];
                let (_op2, _idx2, data2) = &dump2[s];
                let bit_diff = data1.iter().zip(data2.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                let max_abs = data1.iter().zip(data2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                println!(
                    "      slot {s}: op={} (op_id={op1}) inst_idx={idx1} size={} bit_diff={bit_diff} max_abs={max_abs:.3e}",
                    Model::opcode_name(*op1), data1.len()
                );
            }
        }
    }
    let seq_logits = run1_steps.last().unwrap().clone();
    let seq_logits2 = run2_steps.last().unwrap().clone();
    let seq_top = top10(&seq_logits);
    let fp1 = fingerprint(&seq_logits);
    let seq_top2 = top10(&seq_logits2);
    let fp2 = fingerprint(&seq_logits2);
    let same = seq_top.iter().zip(seq_top2.iter()).filter(|(a, b)| a == b).count();

    // Bit-exact comparison of logits arrays.
    let bit_exact = seq_logits.iter().zip(seq_logits2.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
    let n_diff = seq_logits.iter().zip(seq_logits2.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let max_abs_diff = seq_logits.iter().zip(seq_logits2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "  seq->seq same-process determinism: {same}/10 top, bit-exact={bit_exact}, n_diff_logits={n_diff}/{}, max_abs_diff={max_abs_diff:.3e}",
        seq_logits.len()
    );
    println!("    fp1: sum={:.6e} nans={} infs={} range=[{:.3e}, {:.3e}]", fp1.0, fp1.1, fp1.2, fp1.3, fp1.4);
    println!("    fp2: sum={:.6e} nans={} infs={} range=[{:.3e}, {:.3e}]", fp2.0, fp2.1, fp2.2, fp2.3, fp2.4);
    if same < 10 {
        println!("    seq run1: {:?}", &seq_top[..5.min(seq_top.len())]);
        println!("    seq run2: {:?}", &seq_top2[..5.min(seq_top2.len())]);
    }

    // Batched prefill comparison.
    model.reset_state().expect("reset");
    let prefill_logits = model.prefill(&prompt).expect("batched prefill");
    let prefill_top = top10(&prefill_logits);

    let matches = seq_top.iter().zip(prefill_top.iter()).filter(|(a, b)| a == b).count();
    let pass = matches >= 8; // allow ≥8/10 match (floating-point ordering may differ slightly)
    println!("  top-10 match: {matches}/10  [{}]", if pass { "PASS" } else { "FAIL" });
    println!("  seq top-5:     {:?}", &seq_top[..5.min(seq_top.len())]);
    println!("  prefill top-5: {:?}", &prefill_top[..5.min(prefill_top.len())]);

    model.reset_state().expect("reset");
}

/// Multi-GPU coherence test: verifies multi-GPU P2P prefill logits match single-GPU sequential reference.
///
/// Requires a MoE model (e.g. qwen35_35b_a3b.q4.bqnt) and >= 2 GPUs.
/// Loads the model twice: single-GPU for reference, multi-GPU for test.
/// Both runs use the sequential paged decode path (no persistent worker), so reset_state
/// works correctly and hipMemcpy is available throughout.
///
/// Skipped automatically if the model has no MoE layers or only 1 GPU is present.
fn bench_coherence_multi_gpu(model_dir: &Path, prompt_len: usize) {
    let top10 = |logits: &[f32]| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| {
            logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(10);
        idx
    };

    let gpu_count = vram_free_per_gpu().len();
    if gpu_count < 2 {
        println!("=== Multi-GPU coherence test: SKIPPED ({gpu_count} GPU(s) available, need >=2) ===");
        return;
    }

    println!("=== Multi-GPU coherence test ({gpu_count} GPUs, prompt_len={prompt_len}) ===");

    let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 9906 + (i % 100)).collect();

    // Single-GPU sequential reference: load without multi-GPU enabled.
    let orig_multi_gpu = std::env::var("MULTI_GPU").ok();
    unsafe { std::env::remove_var("MULTI_GPU") };

    let mut ref_model = load_model(model_dir, false);

    let model_has_moe = ref_model.config().layers.iter().any(|l| matches!(l.ffn_type, FfnType::MoE { .. }));
    if !model_has_moe {
        println!("  SKIPPED: model has no MoE layers (multi-GPU path is MoE-only)");
        if let Some(v) = orig_multi_gpu {
            unsafe { std::env::set_var("MULTI_GPU", v) };
        }
        return;
    }

    // Use flat-KV prefill (not paged decode) as reference so both paths use the same
    // KV cache layout. paged vs flat KV disagreement is a separate issue.
    ref_model.reset_state().expect("reset");
    let ref_logits = ref_model.prefill(&prompt).expect("ref prefill");
    let ref_top = top10(&ref_logits);
    ref_model.reset_state().expect("reset ref");
    drop(ref_model); // Free VRAM before loading multi-GPU model

    // Multi-GPU model: reload with enable_multi_gpu.
    unsafe { std::env::set_var("MULTI_GPU", "1") };
    let mut mg_model = load_model(model_dir, true);

    // prefill() on a multi-GPU MoE model takes the sequential paged path:
    // persistent_workers is None and has_moe=true so prefill_batched is skipped.
    // decode_step_paged never launches the persistent worker, so reset_state works.
    mg_model.reset_state().expect("reset mg");
    let mg_logits = mg_model.prefill(&prompt).expect("multi-GPU prefill");
    let mg_top = top10(&mg_logits);
    mg_model.reset_state().expect("reset mg after");
    drop(mg_model);

    if orig_multi_gpu.is_none() {
        unsafe { std::env::remove_var("MULTI_GPU") };
    } else if let Some(v) = orig_multi_gpu {
        unsafe { std::env::set_var("MULTI_GPU", v) };
    }

    let matches = ref_top.iter().filter(|t| mg_top.contains(t)).count();
    let pass = matches >= 8;
    println!("  top-10 overlap: {matches}/10  [{}]", if pass { "PASS" } else { "FAIL" });
    println!("  ref top-5:      {:?}", &ref_top[..5.min(ref_top.len())]);
    println!("  mg  top-5:      {:?}", &mg_top[..5.min(mg_top.len())]);
    if !pass {
        eprintln!("  ERROR: multi-GPU coherence FAILED ({matches}/10 top tokens match)");
    }
}

fn main() {
    let (model_path, bqnt_override) = match std::env::var("MODEL").ok() {
        Some(ref p) if p.ends_with(".bqnt") => {
            let hf_dir = resolve_hf_dir(p).unwrap_or_else(|| {
                eprintln!("Could not resolve HF cache dir for {p}");
                std::process::exit(1);
            });
            (hf_dir, Some(p.clone()))
        }
        Some(p) => (p, None),
        None => {
            let from_bqnt = std::env::var("BQNT_PATH")
                .ok()
                .and_then(|p| resolve_hf_dir(&p));
            (from_bqnt.unwrap_or_else(|| DEFAULT_MODEL_DIR.to_string()), None)
        }
    };
    if let Some(ref bqnt_path) = bqnt_override {
        unsafe { std::env::set_var("BQNT_PATH", bqnt_path) };
    }

    let model_dir = Path::new(&model_path);
    if !model_dir.exists() {
        eprintln!("Model not found at {model_path}");
        std::process::exit(1);
    }

    // Auto-detect MoE, MULTI_GPU, PERSISTENT (same logic as generate binary)
    let config_path = model_dir.join("config.json");
    let has_moe = config_path
        .exists()
        .then(|| braidinfer_runtime::config::ModelConfig::from_config_json(&config_path).ok())
        .flatten()
        .map_or(false, |c| c.layers.iter().any(|l| matches!(l.ffn_type, FfnType::MoE { .. })));

    let bqnt_size_bytes: u64 = std::env::var("BQNT_PATH")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            model_dir.file_name().map(|n| {
                model_dir
                    .parent()
                    .unwrap_or(model_dir)
                    .join(format!("{}.q4.bqnt", n.to_string_lossy()))
            })
        })
        .and_then(|p| std::fs::metadata(&p).ok())
        .map(|m| m.len())
        .unwrap_or(0);

    let free_per_gpu = vram_free_per_gpu();
    let single_gpu_vram = free_per_gpu.first().copied().unwrap_or(0);
    let multi_gpu = std::env::var("MULTI_GPU").is_ok()
        || (bqnt_size_bytes > 0 && bqnt_size_bytes as usize > single_gpu_vram * 85 / 100);
    if multi_gpu && std::env::var("MULTI_GPU").is_err() {
        eprintln!(
            "Auto: MULTI_GPU (model {:.1}GB > single-GPU {:.1}GB free)",
            bqnt_size_bytes as f64 / 1e9,
            single_gpu_vram as f64 / 1e9,
        );
        unsafe { std::env::set_var("MULTI_GPU", "1") };
    }

    let persistent = std::env::var("PERSISTENT").as_deref() == Ok("1") || multi_gpu || !has_moe;
    if persistent && std::env::var("PERSISTENT").is_err() {
        let reason = if multi_gpu { "required for multi-GPU" } else { "non-MoE model" };
        eprintln!("Auto: PERSISTENT ({reason})");
        unsafe { std::env::set_var("PERSISTENT", "1") };
    }

    // Multi-GPU coherence test runs before the main model is loaded to avoid holding
    // two large model instances in VRAM simultaneously.
    // Skip if model is too large to fit two copies (bqnt_size_bytes > 40% of total free VRAM).
    let total_free_vram: usize = vram_free_per_gpu().iter().sum();
    if bqnt_size_bytes as usize > total_free_vram * 2 / 5 {
        println!("=== Multi-GPU coherence test: SKIPPED (model {:.1}GB > 40% of {:.1}GB total free VRAM) ===",
            bqnt_size_bytes as f64 / 1e9, total_free_vram as f64 / 1e9);
    } else {
        bench_coherence_multi_gpu(model_dir, 8);
    }

    let mut model = load_model(model_dir, multi_gpu);
    eprintln!("Model loaded: {model_path}");

    let warmup: usize = std::env::var("BENCH_WARMUP").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let runs: usize = std::env::var("BENCH_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);

    // Coherence and prefill FIRST: must run before persistent worker starts.
    // Print bench header with config info
    let gpu_count = {
        let mut count: i32 = 0;
        unsafe { braidinfer_hip::ffi::hipGetDeviceCount(&mut count) };
        count
    };
    if multi_gpu {
        println!("multi-GPU: {gpu_count} GPUs");
    }

    // Skip coherence and prefill for very large models where VRAM is too tight for extra buffers.
    // Use post-load free VRAM to handle cases where bqnt file size is unavailable (bqnt_size_bytes=0).
    let post_load_free: usize = vram_free_per_gpu().iter().sum();
    let vram_used_pct = if total_free_vram > 0 { bqnt_size_bytes as usize * 100 / total_free_vram } else { 0 };
    // Skip if: bqnt says >80% used, OR post-load free < 20% of total (catches bqnt_size_bytes=0 case)
    let vram_too_tight = vram_used_pct > 80 || (total_free_vram > 0 && post_load_free * 100 / total_free_vram < 20);
    if vram_too_tight {
        println!("=== Coherence test: SKIPPED (model uses ~{vram_used_pct}% of total VRAM, {:.1}GB free post-load) ===",
            post_load_free as f64 / 1e9);
        println!("=== Prefill benchmark: SKIPPED (model uses ~{vram_used_pct}% of total VRAM) ===");
    } else {
        bench_coherence(&mut model, 8);
        bench_prefill(&mut model, &[1, 8, 32, 64, 128, 256, 512]);
    }
    bench_decode(&mut model, warmup, runs);
}
