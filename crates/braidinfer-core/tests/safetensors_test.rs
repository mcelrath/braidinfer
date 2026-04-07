use braidinfer_core::safetensors::SafeTensorSet;
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
    let info = set
        .tensor_info(embed_name)
        .expect("embed_tokens.weight not found");
    assert_eq!(info.dtype, safetensors::Dtype::BF16, "expected BF16");
    assert_eq!(info.shape, vec![248320, 1024], "unexpected shape");

    let raw = set
        .tensor_data(embed_name)
        .expect("failed to get tensor data");
    assert_eq!(raw.len(), 248320 * 1024 * 2);

    println!("first 10 embed_tokens.weight values:");
    for chunk in raw[..20].chunks_exact(2) {
        let bits = u16::from_le_bytes(chunk.try_into().unwrap());
        let val = f32::from_bits((bits as u32) << 16);
        println!("  {val}");
    }
}
