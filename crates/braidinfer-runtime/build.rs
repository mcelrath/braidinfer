fn main() {
    let kernel_dir = std::env::var("DEP_BRAIDINFER_KERNELS_KERNEL_DIR")
        .expect("DEP_BRAIDINFER_KERNELS_KERNEL_DIR not set — braidinfer-hip must be a dependency");
    println!("cargo:rustc-env=BRAIDINFER_KERNEL_DIR={kernel_dir}");

    // pky.2 A2 probe: mirror BRAIDINFER_QUEUE_LINE_ISOLATE into a Rust cfg
    // so WorkerQueueLayout's padding field matches the C-side struct.
    println!("cargo:rerun-if-env-changed=BRAIDINFER_QUEUE_LINE_ISOLATE");
    println!("cargo::rustc-check-cfg=cfg(queue_line_isolate)");
    if std::env::var("BRAIDINFER_QUEUE_LINE_ISOLATE").is_ok() {
        println!("cargo:rustc-cfg=queue_line_isolate");
    }
}
