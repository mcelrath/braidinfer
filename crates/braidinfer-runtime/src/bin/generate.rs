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
        model.enable_multi_gpu().expect("enable multi-GPU");
    }

    let (vram_used, vram_total) = vram_usage_mb();
    eprintln!("VRAM after load: {:.0}/{:.0} MB", vram_used, vram_total);
    eprintln!("max_seq_len: {}", model.config().max_seq_len);
    eprintln!("stop tokens: {:?}", token_config.eos_token_ids);

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
