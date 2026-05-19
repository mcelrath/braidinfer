use braidinfer_core::types::DeviceId;
use braidinfer_runtime::cli::{
    apply_auto_modes, extract_cli_flags, resolve_model_arg, vram_usage_mb,
};
use braidinfer_runtime::generate::{TokenConfig, chat_generate, greedy_generate, load_tokenizer};
use braidinfer_runtime::model::Model;
use std::time::Instant;

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let _flags = extract_cli_flags(&mut args);

    // Match chat.rs convention: model argument can come from MODEL env OR
    // argv[1] when it points to an existing model path (.bqnt file or a
    // directory). Otherwise argv[1..] is the prompt. This makes
    //   MODEL=... generate "prompt"            (content_sweep style)
    //   generate models/foo.bqnt "prompt"      (chat.rs-style positional)
    //   generate "prompt"                       (default model fallback)
    // all work consistently.
    let env_model = std::env::var("MODEL").ok();
    let positional_model = if env_model.is_none() && args.len() > 1 {
        let candidate = std::path::Path::new(&args[1]);
        if candidate.exists() && (args[1].ends_with(".bqnt") || candidate.is_dir()) {
            Some(args.remove(1))
        } else {
            None
        }
    } else {
        None
    };

    let prompt = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "Hello, world!".to_string()
    };

    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    let raw_mode = std::env::var("RAW").is_ok();

    let resolved = resolve_model_arg(env_model.or(positional_model));
    let model_dir = resolved.model_dir.as_path();

    let tokenizer = load_tokenizer(model_dir).expect("load tokenizer");
    let token_config = TokenConfig::from_model_dir(model_dir, &tokenizer);
    let device = DeviceId(0);
    let max_seq_len: Option<usize> = std::env::var("MAX_SEQ_LEN")
        .ok()
        .and_then(|v| v.parse().ok());

    let (multi_gpu, _persistent) = apply_auto_modes(model_dir);

    let mut model =
        Model::load_with_max_seq_len(model_dir, device, max_seq_len).expect("load model");

    if multi_gpu {
        if let Err(e) = model.enable_multi_gpu() {
            let per_gpu_free_mb: Vec<f64> =
                braidinfer_runtime::cli::vram_free_per_gpu()
                    .iter()
                    .map(|&b| b as f64 / (1024.0 * 1024.0))
                    .collect();
            eprintln!("ERROR: enable_multi_gpu failed: {e:?}");
            eprintln!("  Per-GPU free VRAM (MB): {:?}", per_gpu_free_mb);
            eprintln!("  Hints:");
            eprintln!("    - increase GPU count (try -g 4 or -g 8)");
            eprintln!("    - reduce MAX_SEQ_LEN (e.g. MAX_SEQ_LEN=4096)");
            eprintln!("    - use a smaller quant (e.g. .q4.bqnt instead of .q8.bqnt)");
            eprintln!("    - HipError(2) = hipErrorOutOfMemory: not enough VRAM for distributed weights + KV caches + scratch");
            std::process::exit(1);
        }
    }

    let (vram_used, vram_total) = vram_usage_mb();
    eprintln!("VRAM after load: {:.0}/{:.0} MB", vram_used, vram_total);
    eprintln!("max_seq_len: {}", model.config().max_seq_len);
    eprintln!("stop tokens: {:?}", token_config.eos_token_ids);

    // bd 4e2m / udi #3203: warmup-discard. Run a 4-token sacrificial decode to
    // detect Sig A queue-init NaN (cold-start KFD per-PASID first-queue init
    // produces ~40% NaN-corrupted first-decode). If detected, drop+reload model
    // and retry; max BRAIDINFER_WARMUP_RETRIES retries before exiting with error.
    // Skip via BRAIDINFER_WARMUP_SKIP=1.
    let max_warmup_retries: usize = std::env::var("BRAIDINFER_WARMUP_RETRIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let warmup_skip = std::env::var("BRAIDINFER_WARMUP_SKIP").is_ok();
    let warmup_mode = std::env::var("BRAIDINFER_WARMUP_MODE").unwrap_or_default();
    let warmup_start = Instant::now();
    let mut warmup_attempts: usize = 0;
    if !warmup_skip {
        loop {
            warmup_attempts += 1;
            let mailbox_fallback_to_decode = std::cell::Cell::new(false);
            // bd 4e2m / udi #3257: mailbox-only is DEFAULT as of this commit
            // (~320ms cold-start vs ~680ms full-decode). Opt out via
            // BRAIDINFER_WARMUP_MODE=full-decode. Single-GPU mode auto-
            // falls back to full-decode warmup via "single-gpu-fallback"
            // sentinel (no race in single-GPU; spawning a worker before
            // prefill would also break the lazy paged-KV init).
            let use_mailbox = warmup_mode != "full-decode";
            let dirty = if use_mailbox {
                match model.minimal_mailbox_warmup_no_prefill() {
                    Ok(diag) => {
                        eprintln!("warmup-mailbox-only attempt {warmup_attempts}: OK — {diag}");
                        false
                    }
                    Err(diag) if diag == "single-gpu-fallback" => {
                        eprintln!("warmup-mailbox-only attempt {warmup_attempts}: single-GPU mode — falling back to full-decode warmup");
                        mailbox_fallback_to_decode.set(true);
                        // Fall through to full-decode below via re-entry on next iteration
                        // — but we don't want to count this as a "dirty" attempt.
                        // Instead, run greedy_generate inline here.
                        let test = greedy_generate(&mut model, &tokenizer, &token_config, "Hello", 4);
                        match &test {
                            Ok(r) => {
                                let concat: String = r.text_pieces.iter().cloned().collect();
                                let bang_run = concat.chars().rev().take_while(|c| *c == '!').count();
                                bang_run >= 3
                            }
                            Err(_) => true,
                        }
                    }
                    Err(diag) => {
                        eprintln!("warmup-mailbox-only attempt {warmup_attempts}: FAIL — {diag}");
                        true
                    }
                }
            } else {
                let test = greedy_generate(&mut model, &tokenizer, &token_config, "Hello", 4);
                match &test {
                    Ok(r) => {
                        let concat: String = r.text_pieces.iter().cloned().collect();
                        // Sig A signature: NaN logits collapse to argmax of NaN array,
                        // producing repeated low-id token (often "!" in tokenizer).
                        let bang_run = concat.chars().rev().take_while(|c| *c == '!').count();
                        bang_run >= 3
                    }
                    Err(_) => true,
                }
            };
            if !dirty {
                eprintln!(
                    "warmup-discard: clean after {warmup_attempts} attempt(s) in {:.2}s",
                    warmup_start.elapsed().as_secs_f64()
                );
                break;
            }
            if warmup_attempts >= max_warmup_retries {
                eprintln!(
                    "ERROR: warmup-discard failed {} attempts ({:.2}s); exiting with NaN-detected code 100",
                    max_warmup_retries,
                    warmup_start.elapsed().as_secs_f64()
                );
                std::process::exit(100);
            }
            eprintln!(
                "warmup-discard NaN-detected, retry {}/{} ({:.2}s elapsed)",
                warmup_attempts, max_warmup_retries,
                warmup_start.elapsed().as_secs_f64()
            );
            // Drop + reload to recreate KFD queues with fresh PASID init.
            drop(model);
            model = Model::load_with_max_seq_len(model_dir, device, max_seq_len)
                .expect("reload model after warmup-discard");
            if multi_gpu {
                if let Err(e) = model.enable_multi_gpu() {
                    eprintln!("ERROR: enable_multi_gpu failed on warmup-discard reload: {e:?}");
                    std::process::exit(1);
                }
            }
        }
    }

    let start = Instant::now();
    let result = if raw_mode {
        greedy_generate(&mut model, &tokenizer, &token_config, &prompt, max_tokens)
    } else {
        chat_generate(
            &mut model,
            &tokenizer,
            &token_config,
            &prompt,
            None,
            max_tokens,
        )
    }
    .expect("generate");

    let elapsed = start.elapsed().as_secs_f64();
    let n_tokens = result.tokens.len();

    for piece in &result.text_pieces {
        print!("{}", piece);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    println!();

    let tok_per_sec = if elapsed > 0.0 {
        n_tokens as f64 / elapsed
    } else {
        0.0
    };
    eprintln!(
        "{} tokens in {:.3}s = {:.1} tok/s",
        n_tokens, elapsed, tok_per_sec
    );
}
