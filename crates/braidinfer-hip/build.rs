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

    // 77r.2.4 (reframed): per-shape cache-hint tuning on op_attn_paged
    // K/V loads. Set BRAIDINFER_KV_LOAD_AUX={1,2,4,...} to replace plain
    // global_loads with raw_buffer_load using that aux byte (0x1=glc,
    // 0x2=slc, 0x4=dlc). Unset = production default (plain global_load).
    println!("cargo:rerun-if-env-changed=BRAIDINFER_KV_LOAD_AUX");
    let kv_load_aux = env::var("BRAIDINFER_KV_LOAD_AUX").ok();

    // braidinfer-xiu: per-op cycle profiling inside the persistent megakernel.
    // Set BRAIDINFER_OP_PROFILE=1 to enable. Adds ~5% perf cost from the
    // atomic accumulators; intended for diagnostics, not production.
    // See PLAN-op-profile.md and kernels/op_profile.h.
    println!("cargo:rerun-if-env-changed=BRAIDINFER_OP_PROFILE");
    let op_profile = env::var("BRAIDINFER_OP_PROFILE").is_ok();

    // braidinfer-pky.2 Phase 0b diagnostic: swap atomic_block_barrier for
    // cooperative_groups::grid_group::sync. Per kb
    // rdna3-grid-sync-vs-atomic-block-barrier-gfx1100 grid.sync is ~115x
    // slower per call but may sidestep the multi-GPU wedge documented in
    // kb rdna3-atomic-block-barrier-multi-gpu-fundamental-issue. Build
    // with BRAIDINFER_USE_GRID_SYNC=1 to enable; ships off by default
    // because of the perf cost on single-GPU paths.
    println!("cargo:rerun-if-env-changed=BRAIDINFER_USE_GRID_SYNC");
    let use_grid_sync = env::var("BRAIDINFER_USE_GRID_SYNC").is_ok();

    // exterior_algebra-zuk Phase 2 (2026-05-12): inline-GCN barrier variant
    // that omits s_waitcnt_vscnt null,0x0 to bypass the SYSTEM-scope vscnt
    // drain suspected of wedging the 4-GPU q8 MoE decode megakernel on
    // gfx1100 under PCIe pressure.  Set BRAIDINFER_BARRIER_V2=1 to route
    // all atomic_block_barrier() call sites to atomic_block_barrier_v2()
    // via the #define in kernels/rdna3_sync.h.
    println!("cargo:rerun-if-env-changed=BRAIDINFER_BARRIER_V2");
    let barrier_v2 = env::var("BRAIDINFER_BARRIER_V2").is_ok();

    // exterior_algebra-zuk Phase 3' (2026-05-12): ASM barrier variant that
    // fixes the GL1-cache-invalidation ordering in the spin loop.  The v1
    // atomic_block_barrier does: load -> s_waitcnt -> buffer_gl1_inv (stale
    // reads).  asm_block_barrier does: buffer_gl1_inv -> s_waitcnt -> load
    // (fresh L2 read every iteration).  Set BRAIDINFER_BARRIER_ASM=1 to
    // route all atomic_block_barrier() call sites to asm_block_barrier().
    println!("cargo:rerun-if-env-changed=BRAIDINFER_BARRIER_ASM");
    let barrier_asm = env::var("BRAIDINFER_BARRIER_ASM").is_ok();

    // exterior_algebra-zuk Phase 4' (2026-05-12): v4 barrier removes s_sleep 0
    // from the spin loop. Root cause: s_sleep on gfx1100 causes hardware
    // preemption — all blocks of a cooperative grid preempt simultaneously at
    // the same s_sleep; no last-arriver advances state->generation. v4 uses
    // s_nop 0x7f instead to keep the wave resident in the CU. Set
    // BRAIDINFER_BARRIER_V4=1 to route all atomic_block_barrier() call sites
    // to atomic_block_barrier_v4() via the #define in kernels/rdna3_sync.h.
    println!("cargo:rerun-if-env-changed=BRAIDINFER_BARRIER_V4");
    let barrier_v4 = env::var("BRAIDINFER_BARRIER_V4").is_ok();

    // pky.2 L2-bypass probe (2026-05-13): replace the worker inner-poll C-level
    // volatile load (queue->shutdown / queue->seq_num) with hand-coded
    // `global_load_b32 ... glc dlc` that bypasses BOTH L1 and L2 on RDNA3.
    // Tests whether L2 staleness on the host-mapped queue line is the wedge
    // mechanism that 963cd76 couldn't probe (no buffer_gl2_inv ISA).
    println!("cargo:rerun-if-env-changed=BRAIDINFER_POLL_LOAD_L2_BYPASS");
    let poll_load_l2_bypass = env::var("BRAIDINFER_POLL_LOAD_L2_BYPASS").is_ok();

    // pky.2 A1 probe (2026-05-13): replace inner-poll volatile load with
    // `global_atomic_or` with 0 — engages atomic-coherence FSM (separate
    // hardware path from regular load FSM on RDNA3).
    println!("cargo:rerun-if-env-changed=BRAIDINFER_POLL_ATOMIC_LOAD");
    let poll_atomic_load = env::var("BRAIDINFER_POLL_ATOMIC_LOAD").is_ok();

    // pky.2 A2 probe (2026-05-13): cache-line isolate seq_num in WorkerQueue.
    // Emits BOTH a hipcc define (for the C struct layout in worker_queue.h)
    // AND a cargo cfg flag (for the Rust mirror struct in persistent_dispatch.rs).
    // pky.2 A5 probe (2026-05-13): poll-load via global_load_b128 (16-byte fetch).
    println!("cargo:rerun-if-env-changed=BRAIDINFER_POLL_LOAD_WIDE");
    let poll_load_wide = env::var("BRAIDINFER_POLL_LOAD_WIDE").is_ok();

    // pky.2 A6 probe (2026-05-13): inject ~256 cycles of vector compute between
    // polls. Tests tight-cadence hypothesis.
    println!("cargo:rerun-if-env-changed=BRAIDINFER_POLL_INJECT_COMPUTE");
    let poll_inject_compute = env::var("BRAIDINFER_POLL_INJECT_COMPUTE").is_ok();

    // udi #161: at the poll site, store back the observed seq_num value into
    // progress_pc as 0xDEAD<low16> so the host can disambiguate "GPU reads
    // 0 forever" (coherence) vs "GPU reads fresh value but doesn't break"
    // (compiler/branch/SP) on wedge timeout.
    println!("cargo:rerun-if-env-changed=BRAIDINFER_DUMP_POLL_VALUE");
    let dump_poll_value = env::var("BRAIDINFER_DUMP_POLL_VALUE").is_ok();

    // braidinfer-snl Phase 2: dump output_slots state on entry to
    // OP_MOE_DISPATCH_POST. Used for the multi-GPU MoE non-determinism
    // investigation. See kernels/megakernel_moe_dispatch.hip.
    println!("cargo:rerun-if-env-changed=BRAIDINFER_DUMP_MOE_POST");
    let dump_moe_post = env::var("BRAIDINFER_DUMP_MOE_POST").is_ok();
    println!("cargo:rerun-if-env-changed=BRAIDINFER_MOE_POST_PREDELAY");
    let moe_post_predelay = env::var("BRAIDINFER_MOE_POST_PREDELAY").ok();
    println!("cargo:rerun-if-env-changed=BRAIDINFER_MOE_WORKER_DRAIN_VSCNT");
    let moe_worker_drain_vscnt = env::var("BRAIDINFER_MOE_WORKER_DRAIN_VSCNT").is_ok();
    println!("cargo:rerun-if-env-changed=BRAIDINFER_ACK_DRAIN_VSCNT");
    let ack_drain_vscnt = env::var("BRAIDINFER_ACK_DRAIN_VSCNT").is_ok();
    println!("cargo:rerun-if-env-changed=BRAIDINFER_MOE_WORKER_READBACK_FENCE");
    let moe_worker_readback_fence = env::var("BRAIDINFER_MOE_WORKER_READBACK_FENCE").is_ok();
    println!("cargo:rerun-if-env-changed=BRAIDINFER_DUMP_MOE_INPUT");
    let dump_moe_input = env::var("BRAIDINFER_DUMP_MOE_INPUT").is_ok();
    println!("cargo:rerun-if-env-changed=BRAIDINFER_MOE_WORKER_FENCE_SYSTEM");
    let moe_worker_fence_system = env::var("BRAIDINFER_MOE_WORKER_FENCE_SYSTEM").is_ok();

    println!("cargo:rerun-if-env-changed=BRAIDINFER_QUEUE_LINE_ISOLATE");
    let queue_line_isolate = env::var("BRAIDINFER_QUEUE_LINE_ISOLATE").is_ok();
    println!("cargo::rustc-check-cfg=cfg(queue_line_isolate)");
    if queue_line_isolate {
        println!("cargo:rustc-cfg=queue_line_isolate");
    }

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
        if bf16_probe {
            hipcc_args.push("-DBRAIDINFER_BF16_INPUT_PROBE".to_string());
        }
        if use_dot2 {
            hipcc_args.push("-DBRAIDINFER_USE_DOT2".to_string());
        }
        if let Some(ref aux) = kv_load_aux {
            hipcc_args.push(format!("-DBRAIDINFER_KV_LOAD_AUX={aux}"));
        }
        if op_profile {
            hipcc_args.push("-DBRAIDINFER_OP_PROFILE".to_string());
        }
        if use_grid_sync {
            hipcc_args.push("-DBRAIDINFER_USE_GRID_SYNC".to_string());
        }
        if barrier_v2 {
            hipcc_args.push("-DBRAIDINFER_BARRIER_V2".to_string());
        }
        if barrier_asm {
            hipcc_args.push("-DBRAIDINFER_BARRIER_ASM".to_string());
        }
        if barrier_v4 {
            hipcc_args.push("-DBRAIDINFER_BARRIER_V4".to_string());
        }
        if poll_load_l2_bypass {
            hipcc_args.push("-DBRAIDINFER_POLL_LOAD_L2_BYPASS".to_string());
        }
        if poll_atomic_load {
            hipcc_args.push("-DBRAIDINFER_POLL_ATOMIC_LOAD".to_string());
        }
        if poll_load_wide {
            hipcc_args.push("-DBRAIDINFER_POLL_LOAD_WIDE".to_string());
        }
        if poll_inject_compute {
            hipcc_args.push("-DBRAIDINFER_POLL_INJECT_COMPUTE".to_string());
        }
        if dump_poll_value {
            hipcc_args.push("-DBRAIDINFER_DUMP_POLL_VALUE".to_string());
        }
        if dump_moe_post {
            hipcc_args.push("-DBRAIDINFER_DUMP_MOE_POST".to_string());
        }
        if let Some(n) = moe_post_predelay.as_ref() {
            hipcc_args.push(format!("-DBRAIDINFER_MOE_POST_PREDELAY={n}"));
        }
        if moe_worker_drain_vscnt {
            hipcc_args.push("-DBRAIDINFER_MOE_WORKER_DRAIN_VSCNT".to_string());
        }
        if ack_drain_vscnt {
            hipcc_args.push("-DBRAIDINFER_ACK_DRAIN_VSCNT".to_string());
        }
        if moe_worker_readback_fence {
            hipcc_args.push("-DBRAIDINFER_MOE_WORKER_READBACK_FENCE".to_string());
        }
        if dump_moe_input {
            hipcc_args.push("-DBRAIDINFER_DUMP_MOE_INPUT".to_string());
        }
        if moe_worker_fence_system {
            hipcc_args.push("-DBRAIDINFER_MOE_WORKER_FENCE_SYSTEM".to_string());
        }
        if queue_line_isolate {
            hipcc_args.push("-DBRAIDINFER_QUEUE_LINE_ISOLATE".to_string());
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
