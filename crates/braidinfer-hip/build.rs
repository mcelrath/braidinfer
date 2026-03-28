use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    let hipcc = format!("{rocm_path}/bin/hipcc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels");

    // Compile each .hip kernel to a code object (.hsaco) for runtime loading
    let kernels = ["rmsnorm", "linear_proj", "silu_mul", "residual_add", "embedding", "lm_head", "mrope", "gqa_attention", "ffn_fused", "gdn_layer_fused", "attn_layer_fused", "causal_conv1d_update", "qk_norm", "rmsnorm_gated", "output_gate", "gdn_gate", "gdn_recurrent_step_v2", "selective_state_update", "megakernel"];

    for kernel in &kernels {
        let src = kernel_dir.join(format!("{kernel}.hip"));
        let hsaco = out_dir.join(format!("{kernel}.hsaco"));

        let output = Command::new(&hipcc)
            .args([
                "--offload-arch=gfx1100",
                "--genco",
                "-O3",
                "-std=c++17",
                "-ffp-contract=fast",  // Aggressive FMA fusion for performance
                "-mwavefrontsize64", // Required for WMMA (V_WMMA_F32_16X16X16_F16)
                &format!("-I{}", kernel_dir.display()), // For opcodes.h
                "-o",
            ])
            .arg(&hsaco)
            .arg(&src)
            .output()
            .expect("failed to run hipcc");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("hipcc failed to compile {}: {stderr}", src.display());
        }

        println!("cargo:rerun-if-changed={}", src.display());
    }

    // Write kernel directory to a file that other crates can include
    let kernel_dir_file = out_dir.join("kernel_dir.txt");
    std::fs::write(&kernel_dir_file, out_dir.to_str().unwrap()).unwrap();

    // Also make it available to this crate
    println!(
        "cargo:rustc-env=BRAIDINFER_KERNEL_DIR={}",
        out_dir.display()
    );

    // DEP_BRAIDINFER_KERNELS_DIR will be available to dependent crates
    println!(
        "cargo:KERNEL_DIR={}",
        out_dir.display()
    );

    // Link to HIP runtime
    println!("cargo:rustc-link-search=native={rocm_path}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rerun-if-changed={}", kernel_dir.join("bf16_utils.h").display());
    println!("cargo:rerun-if-changed={}", kernel_dir.join("opcodes.h").display());
    println!("cargo:rerun-if-changed=build.rs");

    // Generate opcodes.rs from opcodes.h (single source of truth)
    let opcodes_h = std::fs::read_to_string(kernel_dir.join("opcodes.h"))
        .expect("failed to read opcodes.h");
    let mut opcodes_rs = String::from("// Auto-generated from kernels/opcodes.h — do not edit manually.\n\n");
    for line in opcodes_h.lines() {
        if let Some(rest) = line.strip_prefix("#define ") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() >= 2 {  // >= to handle trailing comments
                if let Ok(val) = parts[1].parse::<u32>() {
                    opcodes_rs.push_str(&format!(
                        "#[allow(dead_code)]\npub const {}: u32 = {val};\n",
                        parts[0]
                    ));
                }
            }
        }
    }
    std::fs::write(out_dir.join("opcodes.rs"), &opcodes_rs)
        .expect("failed to write opcodes.rs");
}
