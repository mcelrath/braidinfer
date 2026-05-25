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

    // Build-time perf / feature flags. Documented in DIAGNOSTICS.md.
    // Production builds leave all of these unset.

    // 77r.2.15: __builtin_amdgcn_fdot2_f32_bf16 in op_linear_proj (kb 77r-2-13:
    // 2.5x cyc/FMA at K>=1024). Coherence validated 2026-03 across 12/12 models.
    println!("cargo:rerun-if-env-changed=BRAIDINFER_USE_DOT2");
    let use_dot2 = env::var("BRAIDINFER_USE_DOT2").is_ok();

    // 77r.2.4 (reframed): per-shape cache-hint tuning on op_attn_paged K/V loads.
    // Set BRAIDINFER_KV_LOAD_AUX={1,2,4,...} to replace plain global_loads with
    // raw_buffer_load using that aux byte (0x1=glc, 0x2=slc, 0x4=dlc).
    println!("cargo:rerun-if-env-changed=BRAIDINFER_KV_LOAD_AUX");
    let kv_load_aux = env::var("BRAIDINFER_KV_LOAD_AUX").ok();

    // braidinfer-xiu: per-op cycle profiling inside the persistent megakernel.
    // ~5% perf cost from atomic accumulators; diagnostic, not production.
    println!("cargo:rerun-if-env-changed=BRAIDINFER_OP_PROFILE");
    let op_profile = env::var("BRAIDINFER_OP_PROFILE").is_ok();

    // braidinfer-snl candidate fix (udi bridge #236): worker performs a volatile
    // non-posted PCIe read from out_p2p[0] before returning from
    // op_moe_ffn_remote. The PCIe §2.4 producer-consumer rule forces same-
    // requester same-target writes to drain before the read returns, so the
    // subsequent ack=seq is only visible to the host after output_slots have
    // landed in HBM. DO NOT REMOVE until braidinfer-snl is closed.
    println!("cargo:rerun-if-env-changed=BRAIDINFER_MOE_WORKER_READBACK_FENCE");
    let moe_worker_readback_fence = env::var("BRAIDINFER_MOE_WORKER_READBACK_FENCE").is_ok();
    println!("cargo:rerun-if-env-changed=BRAIDINFER_MOE_WORKER_DIAG");
    let moe_worker_diag = env::var("BRAIDINFER_MOE_WORKER_DIAG").is_ok();

    // Compile each .hip kernel to a code object (.hsaco) for runtime loading.
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
        // megakernel.hsaco contains BOTH megakernel_f32 and persistent_worker
        // entry points (zqw merge: kernels/megakernel.hip).
        "megakernel",
        "peer_copy",
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
        if use_dot2 {
            hipcc_args.push("-DBRAIDINFER_USE_DOT2".to_string());
        }
        if let Some(ref aux) = kv_load_aux {
            hipcc_args.push(format!("-DBRAIDINFER_KV_LOAD_AUX={aux}"));
        }
        if op_profile {
            hipcc_args.push("-DBRAIDINFER_OP_PROFILE".to_string());
        }
        if moe_worker_readback_fence {
            hipcc_args.push("-DBRAIDINFER_MOE_WORKER_READBACK_FENCE".to_string());
        }
        if moe_worker_diag {
            hipcc_args.push("-DBRAIDINFER_MOE_WORKER_DIAG".to_string());
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

    // Track all .hip and .h include files (recursively) so changes trigger
    // recompile. kernels/rdna3/*.h and other subdirectory headers must be
    // included here — a non-recursive walk would silently miss them.
    fn walk_for_rerun(dir: &std::path::Path) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk_for_rerun(&path);
            } else if matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("hip") | Some("h")
            ) {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    walk_for_rerun(&kernel_dir);

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
