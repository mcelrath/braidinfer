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

    // 77r.2.14: optional probe define passed to hipcc when the env var is set.
    // Adds a bf16 round-trip on op_linear_proj inputs to simulate the
    // precision loss of v_dot2_f32_bf16 adoption (kb 77r-2-13).
    println!("cargo:rerun-if-env-changed=BRAIDINFER_BF16_INPUT_PROBE");
    let bf16_probe = env::var("BRAIDINFER_BF16_INPUT_PROBE").is_ok();

    // 77r.2.15: optional define that wires __builtin_amdgcn_fdot2_f32_bf16
    // into op_linear_proj (kb 77r-2-13: 2.5x cyc/FMA at K>=1024). Coherence
    // gated by 77r.2.14 (12/12 byte-identical outputs across 3 models).
    println!("cargo:rerun-if-env-changed=BRAIDINFER_USE_DOT2");
    let use_dot2 = env::var("BRAIDINFER_USE_DOT2").is_ok();

    // 77r.2.8: optional per-CTA page rotation in op_attn_paged + _quant.
    // Each CTA starts at a different physical chunk index (rotation_offset =
    // blockIdx.x % num_chunks), wraps around. Logical-to-physical mapping
    // unchanged — only iteration order differs. CK measured +9% attention
    // bandwidth on similar hardware.
    println!("cargo:rerun-if-env-changed=BRAIDINFER_USE_PAGE_ROTATION");
    let use_page_rotation = env::var("BRAIDINFER_USE_PAGE_ROTATION").is_ok();

    // Compile each .hip kernel to a code object (.hsaco) for runtime loading
    let kernels = [
        "rmsnorm",
        "linear_proj",
        "silu_mul",
        "residual_add",
        "embedding",
        "lm_head",
        "mrope",
        "gqa_attention",
        "paged_attention",
        "ffn_fused",
        "gdn_layer_fused",
        "attn_layer_fused",
        "causal_conv1d_update",
        "qk_norm",
        "rmsnorm_gated",
        "output_gate",
        "gdn_gate",
        "gdn_recurrent_step_v2",
        "selective_state_update",
        "argmax",
        "moe_gate",
        "moe_prefill",
        "dot_sigmoid_scale_add",
        "megakernel",
        "peer_copy",
        "persistent_worker",
        "deinterleave",
        "sync_flag",
    ];

    for kernel in &kernels {
        let src = kernel_dir.join(format!("{kernel}.hip"));
        let hsaco = out_dir.join(format!("{kernel}.hsaco"));

        let mut hipcc_args: Vec<String> = vec![
            "--offload-arch=gfx1100".to_string(),
            "--genco".to_string(),
            "-O3".to_string(),
            "-std=c++17".to_string(),
            "-ffp-contract=fast".to_string(),
            "-mwavefrontsize64".to_string(),
            "-DHIP_API_PER_THREAD_DEFAULT_STREAM".to_string(),
            format!("-I{}", kernel_dir.display()),
        ];
        if bf16_probe {
            hipcc_args.push("-DBRAIDINFER_BF16_INPUT_PROBE".to_string());
        }
        if use_dot2 {
            hipcc_args.push("-DBRAIDINFER_USE_DOT2".to_string());
        }
        if use_page_rotation {
            hipcc_args.push("-DBRAIDINFER_USE_PAGE_ROTATION".to_string());
        }
        hipcc_args.push("-o".to_string());

        let output = Command::new(&hipcc)
            .args(&hipcc_args)
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

    // WMMA kernels require wave32 (-mno-wavefrontsize64), compiled separately.
    let wmma_kernels = [
        "wmma_gemm_bf16",
        "wmma_gemm_rnf4g128",
    ];

    for kernel in &wmma_kernels {
        let src = kernel_dir.join(format!("{kernel}.hip"));
        let hsaco = out_dir.join(format!("{kernel}.hsaco"));

        let output = Command::new(&hipcc)
            .args([
                "--offload-arch=gfx1100",
                "--genco",
                "-O3",
                "-std=c++17",
                "-ffp-contract=fast",
                "-mno-wavefrontsize64", // WMMA _w32 intrinsics require wave32
                "-DHIP_API_PER_THREAD_DEFAULT_STREAM",
                &format!("-I{}", kernel_dir.display()),
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

    // Track all .hip include files so changes trigger recompile
    for entry in std::fs::read_dir(&kernel_dir).expect("read kernel dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("hip")
            || path.extension().and_then(|e| e.to_str()) == Some("h")
        {
            println!("cargo:rerun-if-changed={}", path.display());
        }
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
    println!("cargo:KERNEL_DIR={}", out_dir.display());

    // Link to HIP runtime
    println!("cargo:rustc-link-search=native={rocm_path}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("bf16_utils.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("opcodes.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("megakernel_moe_dispatch.hip").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("moe_work_queue.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("moe_expert_ops.h").display()
    );
    println!("cargo:rerun-if-changed=build.rs");

    // Generate opcodes.rs from opcodes.h (single source of truth)
    let opcodes_h =
        std::fs::read_to_string(kernel_dir.join("opcodes.h")).expect("failed to read opcodes.h");
    let mut opcodes_rs =
        String::from("// Auto-generated from kernels/opcodes.h — do not edit manually.\n\n");
    for line in opcodes_h.lines() {
        if let Some(rest) = line.strip_prefix("#define ") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() >= 2 {
                // >= to handle trailing comments
                if let Ok(val) = parts[1].parse::<u32>() {
                    opcodes_rs.push_str(&format!(
                        "#[allow(dead_code)]\npub const {}: u32 = {val};\n",
                        parts[0]
                    ));
                }
            }
        }
    }
    std::fs::write(out_dir.join("opcodes.rs"), &opcodes_rs).expect("failed to write opcodes.rs");
}
