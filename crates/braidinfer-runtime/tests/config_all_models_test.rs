use braidinfer_runtime::model::ModelConfig;
use std::path::Path;

#[test]
fn test_parse_all_hf_models() {
    let hub = "/home/mcelrath/.cache/huggingface/hub/";
    let hub_path = Path::new(hub);
    if !hub_path.exists() { return; }

    // Universal parser — try ALL model types (skip non-decoder models)
    let skip_types = [
        "bert", "roberta", "t5", "nvembed", "llama_bidirec",  // encoder/embedding models
        "nemotron_parse",  // no hidden_size — uses nested text_encoder_config
    ];

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    for entry in std::fs::read_dir(hub_path).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("models--") { continue; }

        let snapshots = entry.path().join("snapshots");
        if !snapshots.exists() { continue; }
        let snap = match std::fs::read_dir(&snapshots).unwrap().next() {
            Some(Ok(s)) => s.path(),
            _ => continue,
        };
        let config_path = snap.join("config.json");
        if !config_path.exists() { continue; }

        let data = match std::fs::read_to_string(&config_path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let model_type: String = match serde_json::from_str::<serde_json::Value>(&data) {
            Ok(v) => v.get("model_type")
                .or_else(|| v.get("text_config").and_then(|tc| tc.get("model_type")))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            Err(_) => continue,
        };

        if skip_types.contains(&model_type.as_str()) {
            skipped += 1;
            continue;
        }

        let model_name = name.replace("models--", "").replace("--", "/");
        match ModelConfig::from_config_json(&config_path) {
            Ok(cfg) => {
                assert_eq!(cfg.layers.len(), cfg.num_layers, "{model_name}: layers.len mismatch");
                assert!(cfg.hidden_size > 0, "{model_name}: hidden_size=0");
                assert!(cfg.vocab_size > 0, "{model_name}: vocab_size=0");
                println!("OK  {model_type:20} {model_name:50} layers={} h={} experts={}",
                    cfg.num_layers, cfg.hidden_size, cfg.num_experts);
                passed += 1;
            }
            Err(e) => {
                println!("ERR {model_type:20} {model_name:50} {e}");
                failed += 1;
            }
        }
    }

    println!("\n=== Results: {passed} passed, {failed} failed, {skipped} skipped (unsupported type) ===");
    assert_eq!(failed, 0, "{failed} models failed to parse");
}
