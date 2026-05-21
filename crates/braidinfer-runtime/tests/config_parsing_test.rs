use braidinfer_runtime::config::{FfnType, LayerType, ModelConfig};
use std::path::Path;

fn parse_config(dir: &str) -> Option<ModelConfig> {
    let p = Path::new(dir);
    if !p.exists() {
        return None;
    }
    let config_path = p.join("config.json");
    if !config_path.exists() {
        return None;
    }
    Some(ModelConfig::from_config_json(&config_path).expect(&format!("parse {dir}")))
}

fn find_snapshot(model_name: &str) -> Option<String> {
    let hub = "/home/mcelrath/.cache/huggingface/hub";
    let model_dir = format!("{hub}/models--{model_name}");
    let snapshots = format!("{model_dir}/snapshots");
    let p = Path::new(&snapshots);
    if !p.exists() {
        return None;
    }
    std::fs::read_dir(p)
        .ok()?
        .next()?
        .ok()
        .map(|e| e.path().to_string_lossy().to_string())
}

#[test]
fn test_parse_qwen35_0_8b() {
    let dir = find_snapshot("Qwen--Qwen3.5-0.8B").expect("model not found");
    let cfg = parse_config(&dir).expect("config not found");
    assert_eq!(cfg.num_layers, 24);
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.layers.len(), 24);
    let gdn_count = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Gdn)
        .count();
    let attn_count = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Attention)
        .count();
    assert_eq!(gdn_count, 18);
    assert_eq!(attn_count, 6);
    assert!(matches!(cfg.layers[0].ffn_type, FfnType::Dense));
    assert_eq!(cfg.num_experts, 0);
    println!("Qwen3.5-0.8B: {gdn_count} GDN + {attn_count} Attn, dense FFN");
}

#[test]
fn test_parse_qwen35_122b() {
    let Some(dir) = find_snapshot("Qwen--Qwen3.5-122B-A10B") else {
        return;
    };
    let cfg = parse_config(&dir).expect("config not found");
    assert_eq!(cfg.num_layers, 48);
    let gdn = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Gdn)
        .count();
    let attn = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Attention)
        .count();
    assert_eq!(gdn, 36);
    assert_eq!(attn, 12);
    assert!(cfg.num_experts > 0, "should have MoE");
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.num_active_experts, 8);
    assert!(matches!(cfg.layers[0].ffn_type, FfnType::MoE { .. }));
    println!(
        "Qwen3.5-122B: {gdn} GDN + {attn} Attn, MoE {}/{}",
        cfg.num_active_experts, cfg.num_experts
    );
}

#[test]
fn test_parse_nemotron_cascade_30b() {
    let Some(dir) = find_snapshot("nvidia--Nemotron-Cascade-2-30B-A3B") else {
        return;
    };
    let cfg = parse_config(&dir).expect("config not found");
    assert_eq!(cfg.num_layers, 52);
    let mamba = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Mamba2)
        .count();
    let moe_ffn = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::MoeFfn)
        .count();
    let attn = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Attention)
        .count();
    assert_eq!(mamba, 23, "M layers = pure Mamba2 SSM");
    assert_eq!(moe_ffn, 23, "E layers = pure MoE FFN");
    assert_eq!(attn, 6, "* layers = attention");
    assert_eq!(mamba + moe_ffn + attn, 52);
    // M layers have no FFN, E layers have MoE FFN
    assert!(
        cfg.layers
            .iter()
            .filter(|l| l.layer_type == LayerType::Mamba2)
            .all(|l| matches!(l.ffn_type, FfnType::None))
    );
    assert!(
        cfg.layers
            .iter()
            .filter(|l| l.layer_type == LayerType::MoeFfn)
            .all(|l| matches!(l.ffn_type, FfnType::MoE { .. }))
    );
    println!("Nemotron-Cascade-30B: {mamba} Mamba2 + {moe_ffn} MoeFfn + {attn} Attn");
}

#[test]
fn test_parse_nemotron_120b() {
    let Some(dir) = find_snapshot("nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16") else {
        return;
    };
    let cfg = parse_config(&dir).expect("config not found");
    assert_eq!(cfg.num_layers, 88);
    let mamba = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Mamba2)
        .count();
    let moe_ffn = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::MoeFfn)
        .count();
    let attn = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Attention)
        .count();
    assert_eq!(mamba, 40, "M layers = pure Mamba2 SSM");
    assert_eq!(moe_ffn, 40, "E layers = pure MoE FFN");
    assert_eq!(attn, 8, "* layers = attention");
    println!("Nemotron-120B: {mamba} Mamba2 + {moe_ffn} MoeFfn + {attn} Attn");
}

#[test]
fn test_parse_devstral_123b() {
    let Some(dir) = find_snapshot("mistralai--Devstral-2-123B-Instruct-2512") else {
        return;
    };
    let cfg = parse_config(&dir).expect("config not found");
    assert_eq!(cfg.num_layers, 88);
    let attn = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Attention)
        .count();
    assert_eq!(attn, 88, "all layers should be attention");
    assert!(matches!(cfg.layers[0].ffn_type, FfnType::Dense));
    assert_eq!(cfg.num_experts, 0);
    println!("Devstral-123B: {attn} Attn, dense FFN");
}

#[test]
fn test_parse_qwen3_coder_next() {
    let Some(dir) = find_snapshot("Qwen--Qwen3-Coder-Next") else {
        return;
    };
    let cfg = parse_config(&dir).expect("config not found");
    assert_eq!(cfg.num_layers, 48);
    let gdn = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Gdn)
        .count();
    let attn = cfg
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Attention)
        .count();
    assert_eq!(gdn, 36);
    assert_eq!(attn, 12);
    assert!(cfg.num_experts > 0);
    assert_eq!(cfg.num_experts, 512);
    println!(
        "Qwen3-Coder-Next: {gdn} GDN + {attn} Attn, MoE {}/{}",
        cfg.num_active_experts, cfg.num_experts
    );
}
