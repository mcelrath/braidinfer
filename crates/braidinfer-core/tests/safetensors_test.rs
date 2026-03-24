use braidinfer_core::safetensors::SafeTensorSet;
use braidinfer_core::types::DType;
use std::path::Path;

const MODEL_DIR: &str = "/home/mcelrath/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

#[test]
fn test_load_qwen_model() {
    let dir = Path::new(MODEL_DIR);
    let set = SafeTensorSet::open_directory(dir).expect("failed to open model directory");

    let names = set.tensor_names();
    assert!(!names.is_empty(), "no tensors loaded");
    println!("total tensors: {}", names.len());

    let embed_name = "model.language_model.embed_tokens.weight";
    let info = set.tensor_info(embed_name).expect("embed_tokens.weight not found");
    assert_eq!(info.dtype, DType::BF16, "expected BF16");
    assert_eq!(info.shape, vec![248320, 1024], "unexpected shape");

    let f32_vals = set.tensor_as_f32(embed_name).expect("failed to convert to f32");
    assert_eq!(f32_vals.len(), 248320 * 1024);

    println!("first 10 embed_tokens.weight values:");
    for v in &f32_vals[..10] {
        println!("  {v}");
    }
}
