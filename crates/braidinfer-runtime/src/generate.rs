use std::path::Path;
use tokenizers::Tokenizer;

use crate::model::Model;
use crate::weights::ModelError;

pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

pub struct GenerateResult {
    pub tokens: Vec<u32>,
    pub text_pieces: Vec<String>,
}

/// Runtime-loaded token configuration from model files.
pub struct TokenConfig {
    pub im_start_id: Option<u32>,
    pub im_end_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    pub bos_token_id: Option<u32>,
    chat_template: Option<String>,
}

impl TokenConfig {
    /// Load token config from model directory. Reads tokenizer_config.json for
    /// eos_token, and chat_template.jinja for the chat template.
    /// Falls back gracefully if files are missing.
    pub fn from_model_dir(model_dir: &Path, tokenizer: &Tokenizer) -> Self {
        let im_start_id = tokenizer.token_to_id("<|im_start|>");
        let im_end_id = tokenizer.token_to_id("<|im_end|>");
        let endoftext_id = tokenizer.token_to_id("<|endoftext|>");

        // Collect all stop token IDs
        let mut eos_token_ids = Vec::new();
        if let Some(id) = im_end_id {
            eos_token_ids.push(id);
        }
        if let Some(id) = endoftext_id {
            eos_token_ids.push(id);
        }

        // Also check config.json text_config.eos_token_id
        if let Ok(data) = std::fs::read_to_string(model_dir.join("config.json")) {
            if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(val) = cfg.pointer("/text_config/eos_token_id") {
                    let ids: Vec<u32> = if let Some(n) = val.as_u64() {
                        vec![n as u32]
                    } else if let Some(arr) = val.as_array() {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    } else {
                        vec![]
                    };
                    for eos in ids {
                        if !eos_token_ids.contains(&eos) {
                            eos_token_ids.push(eos);
                        }
                    }
                }
            }
        }

        // Load chat template from jinja file
        let chat_template = std::fs::read_to_string(model_dir.join("chat_template.jinja"))
            .ok()
            .or_else(|| {
                // Fallback: check tokenizer_config.json chat_template field
                let data = std::fs::read_to_string(model_dir.join("tokenizer_config.json")).ok()?;
                let cfg: serde_json::Value = serde_json::from_str(&data).ok()?;
                cfg.get("chat_template")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            });

        // BOS token: check config.json bos_token_id
        let bos_token_id = std::fs::read_to_string(model_dir.join("config.json"))
            .ok()
            .and_then(|data| serde_json::from_str::<serde_json::Value>(&data).ok())
            .and_then(|cfg| {
                cfg.get("bos_token_id")
                    .and_then(|v| v.as_u64().map(|n| n as u32))
                    .or_else(|| {
                        cfg.pointer("/text_config/bos_token_id")
                            .and_then(|v| v.as_u64().map(|n| n as u32))
                    })
            });

        TokenConfig {
            im_start_id,
            im_end_id,
            eos_token_ids,
            bos_token_id,
            chat_template,
        }
    }

    /// bd 4ayf A3.2.3b: token config from the bqnt's embedded metadata (model_config for
    /// eos/bos, chat_template, tokenizer_config) — no HF dir needed. Mirrors from_model_dir.
    pub fn from_bqnt(bqnt: &crate::bqnt::MmapBqnt, tokenizer: &Tokenizer) -> Self {
        let meta: serde_json::Value = bqnt
            .metadata()
            .ok()
            .and_then(|m| serde_json::from_str(&m).ok())
            .unwrap_or(serde_json::Value::Null);
        let cfg = meta.get("model_config").cloned().unwrap_or(serde_json::Value::Null);

        let im_start_id = tokenizer.token_to_id("<|im_start|>");
        let im_end_id = tokenizer.token_to_id("<|im_end|>");
        let endoftext_id = tokenizer.token_to_id("<|endoftext|>");
        let mut eos_token_ids = Vec::new();
        if let Some(id) = im_end_id {
            eos_token_ids.push(id);
        }
        if let Some(id) = endoftext_id {
            eos_token_ids.push(id);
        }
        if let Some(val) = cfg
            .pointer("/text_config/eos_token_id")
            .or_else(|| cfg.get("eos_token_id"))
        {
            let ids: Vec<u32> = if let Some(n) = val.as_u64() {
                vec![n as u32]
            } else if let Some(arr) = val.as_array() {
                arr.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect()
            } else {
                vec![]
            };
            for eos in ids {
                if !eos_token_ids.contains(&eos) {
                    eos_token_ids.push(eos);
                }
            }
        }
        // chat_template: A2 stores it as a top-level metadata field; else tokenizer_config.
        let chat_template = meta
            .get("chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                meta.get("tokenizer_config")
                    .and_then(|tc| tc.get("chat_template"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            });
        let bos_token_id = cfg
            .get("bos_token_id")
            .and_then(|v| v.as_u64().map(|n| n as u32))
            .or_else(|| {
                cfg.pointer("/text_config/bos_token_id")
                    .and_then(|v| v.as_u64().map(|n| n as u32))
            });
        TokenConfig {
            im_start_id,
            im_end_id,
            eos_token_ids,
            bos_token_id,
            chat_template,
        }
    }

    pub fn is_stop_token(&self, token: u32) -> bool {
        self.eos_token_ids.contains(&token)
    }

    /// Whether a chat template is available for this model. Base (non-
    /// instruction-tuned) models return `None`; the `chat` binary surfaces
    /// this at startup rather than failing on first turn.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }
}

pub fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("failed to load tokenizer from {:?}: {}", tokenizer_path, e))?;
    Ok(tokenizer)
}

/// bd 4ayf A3.2.3b: load the tokenizer from the bqnt's embedded `tokenizer_json` metadata
/// (no HF dir). Returns an error if the bqnt predates the v2 self-contained format.
pub fn load_tokenizer_from_bqnt(
    bqnt: &crate::bqnt::MmapBqnt,
) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let meta = bqnt.metadata()?;
    let v: serde_json::Value = serde_json::from_str(&meta)?;
    let tj = v
        .get("tokenizer_json")
        .filter(|x| !x.is_null())
        .ok_or("bqnt has no embedded tokenizer_json (regenerate with the v2 quantizer)")?;
    let tj_str = serde_json::to_string(tj)?;
    Tokenizer::from_bytes(tj_str.as_bytes())
        .map_err(|e| format!("failed to parse embedded tokenizer: {e}").into())
}

/// bd 4ayf A3.2.3b: load the tokenizer + token config, preferring the bqnt's embedded copies
/// (self-contained, no HF dir) and falling back to the model_dir files. The 4 bins call this.
pub fn load_tokenizer_and_config(
    model_dir: &Path,
    bqnt_path: Option<&str>,
) -> Result<(Tokenizer, TokenConfig), Box<dyn std::error::Error>> {
    if let Some(bp) = bqnt_path {
        if let Ok(b) = crate::bqnt::MmapBqnt::open(Path::new(bp)) {
            if let Ok(tok) = load_tokenizer_from_bqnt(&b) {
                let tc = TokenConfig::from_bqnt(&b, &tok);
                return Ok((tok, tc));
            }
        }
    }
    let tok = load_tokenizer(model_dir)?;
    let tc = TokenConfig::from_model_dir(model_dir, &tok);
    Ok((tok, tc))
}

/// Apply chat template using the model's Jinja2 template.
pub fn apply_chat_template(
    tokenizer: &Tokenizer,
    token_config: &TokenConfig,
    messages: &[ChatMessage<'_>],
) -> Result<Vec<u32>, ModelError> {
    apply_chat_template_thinking(tokenizer, token_config, messages, false)
}

pub fn apply_chat_template_thinking(
    tokenizer: &Tokenizer,
    token_config: &TokenConfig,
    messages: &[ChatMessage<'_>],
    enable_thinking: bool,
) -> Result<Vec<u32>, ModelError> {
    let template_src = token_config
        .chat_template
        .as_deref()
        .ok_or_else(|| ModelError::MissingWeight("no chat_template found in model files".into()))?;

    let mut env = minijinja::Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_template("chat", template_src).map_err(|e| {
        ModelError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad chat template: {e}"),
        ))
    })?;

    let tmpl = env.get_template("chat").map_err(|e| {
        ModelError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })?;

    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
        .collect();

    let rendered = tmpl
        .render(minijinja::context! {
            messages => msgs,
            add_generation_prompt => true,
            enable_thinking => enable_thinking,
        })
        .map_err(|e| {
            ModelError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("template render: {e}"),
            ))
        })?;

    let encoding = tokenizer.encode(rendered.as_str(), false).map_err(|e| {
        ModelError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })?;
    Ok(encoding.get_ids().to_vec())
}

/// Generate from pre-tokenized prompt IDs (no chat template applied).
pub fn generate_from_ids(
    model: &mut Model,
    tokenizer: &Tokenizer,
    token_config: &TokenConfig,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> Result<GenerateResult, ModelError> {
    let n_prompt = prompt_ids.len();
    let pp_t0 = std::time::Instant::now();
    let last_logits = if n_prompt == 0 {
        return Ok(GenerateResult {
            tokens: vec![],
            text_pieces: vec![],
        });
    } else if n_prompt == 1 {
        model.decode_step(prompt_ids[0], 0)?
    } else {
        model.prefill(prompt_ids)?
    };
    // Prefill (prompt-processing) timing. pp tok/s = prompt tokens / prefill seconds.
    let pp_secs = pp_t0.elapsed().as_secs_f64();

    // Opt-in RT scheduling for the dispatch (= main) thread. Promoted AFTER
    // the first dispatch so the persistent worker launch + watchdog thread
    // spawn happen at SCHED_OTHER. If we promote earlier, the watchdog
    // (spawned by PersistentDispatch::add_device) inherits SCHED_FIFO via
    // PTHREAD_INHERIT_SCHED and the cooperative launch wedges (braidinfer-q0h).
    // Idempotent, no-op without BRAIDINFER_DISPATCH_RT=1. See README.md.
    if let Err(msg) = crate::persistent_dispatch::try_promote_dispatch_thread() {
        eprintln!("[braidinfer] dispatch RT promotion failed: {msg}");
    }

    // Debug: print top-5 logits and dump hidden state for first token
    if model.debug_nan {
        let mut indexed: Vec<(usize, f32)> = last_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("Top-5 logits:");
        for &(id, val) in &indexed[..5] {
            let tok = tokenizer.decode(&[id as u32], false).unwrap_or_default();
            eprintln!("  {val:.2}: {id} = {tok:?}");
        }
        // Dump hidden state to file for comparison with HF reference
        if let Ok(hidden) = model.read_hidden() {
            let max_abs = hidden.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let sum: f32 = hidden.iter().sum();
            let std = (hidden
                .iter()
                .map(|x| (x - sum / hidden.len() as f32).powi(2))
                .sum::<f32>()
                / hidden.len() as f32)
                .sqrt();
            eprintln!(
                "Final hidden: max_abs={max_abs:.4}, std={std:.4}, sum={sum:.2}, first10={:.4?}",
                &hidden[..10]
            );
        }
    }

    let mut all_tokens: Vec<u32> = prompt_ids.to_vec();
    let mut text_pieces: Vec<String> = Vec::new();
    let mut position = n_prompt as u32;

    // 5ax-decode probe: BRAIDINFER_LOGIT_TRACE=path logs per-step logit
    // hash + next-token to identify the first divergent decode step
    // across runs.
    let logit_trace_path = std::env::var("BRAIDINFER_LOGIT_TRACE").ok();
    let log_logits = |step: usize, tok: u32, logits: &[f32], trace: &Option<String>| {
        if let Some(p) = trace {
            let mut h: u64 = 0xcbf29ce484222325;
            for x in logits {
                h ^= (*x).to_bits() as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
                let _ = writeln!(f, "step={} tok={} logit_hash={:016x}", step, tok, h);
            }
        }
    };

    // First token from prefill/decode logits (already on CPU)
    let mut next_token = argmax(&last_logits);
    log_logits(0, next_token, &last_logits, &logit_trace_path);

    let tg_t0 = std::time::Instant::now();
    let mut tg_count = 0usize;
    for step in 0..max_tokens {
        if token_config.is_stop_token(next_token) {
            break;
        }
        all_tokens.push(next_token);

        let piece = tokenizer.decode(&[next_token], false).unwrap_or_default();
        text_pieces.push(piece);

        if logit_trace_path.is_some() {
            let logits = model.decode_step(next_token, position)?;
            next_token = argmax(&logits);
            log_logits(step + 1, next_token, &logits, &logit_trace_path);
        } else {
            next_token = model.decode_step_token(next_token, position)?;
        }
        position += 1;
        tg_count += 1;
    }
    // pp/tg split for the model sweep (parsed by scripts/sweep_all_models.py).
    // pp = prompt-processing (prefill) throughput; tg = token-generation (decode).
    let tg_secs = tg_t0.elapsed().as_secs_f64();
    let pp_toks = if n_prompt > 1 { n_prompt } else { 0 };
    eprintln!(
        "PPTG pp={:.1} tok/s ({} prompt tok / {:.3}s) tg={:.1} tok/s ({} tok / {:.3}s)",
        if pp_secs > 0.0 { pp_toks as f64 / pp_secs } else { 0.0 },
        pp_toks, pp_secs,
        if tg_secs > 0.0 { tg_count as f64 / tg_secs } else { 0.0 },
        tg_count, tg_secs,
    );

    Ok(GenerateResult {
        tokens: all_tokens[n_prompt..].to_vec(),
        text_pieces,
    })
}

/// Generate with chat template applied.
pub fn chat_generate(
    model: &mut Model,
    tokenizer: &Tokenizer,
    token_config: &TokenConfig,
    user_message: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
) -> Result<GenerateResult, ModelError> {
    let mut messages = Vec::new();
    if let Some(sys) = system_prompt {
        messages.push(ChatMessage {
            role: "system",
            content: sys,
        });
    }
    messages.push(ChatMessage {
        role: "user",
        content: user_message,
    });
    let prompt_ids = apply_chat_template(tokenizer, token_config, &messages)?;
    generate_from_ids(model, tokenizer, token_config, &prompt_ids, max_tokens)
}

/// Legacy: generate from raw text (no chat template).
pub fn greedy_generate(
    model: &mut Model,
    tokenizer: &Tokenizer,
    token_config: &TokenConfig,
    prompt: &str,
    max_tokens: usize,
) -> Result<GenerateResult, ModelError> {
    let encoding = tokenizer.encode(prompt, false).map_err(|e| {
        ModelError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })?;
    let mut prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    // Prepend BOS if the model has one and the tokenizer didn't add it
    if let Some(bos) = token_config.bos_token_id {
        if prompt_ids.first() != Some(&bos) {
            prompt_ids.insert(0, bos);
        }
    }
    generate_from_ids(model, tokenizer, token_config, &prompt_ids, max_tokens)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
