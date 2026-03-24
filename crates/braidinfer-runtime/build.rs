fn main() {
    let kernel_dir =
        std::env::var("DEP_BRAIDINFER_KERNELS_KERNEL_DIR").expect(
            "DEP_BRAIDINFER_KERNELS_KERNEL_DIR not set — braidinfer-hip must be a dependency",
        );
    println!("cargo:rustc-env=BRAIDINFER_KERNEL_DIR={kernel_dir}");
}
