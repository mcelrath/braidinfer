//! GPU integration test for Phase B: promote/evict tier transitions.
//!
//! # What this tests
//!
//! 1. Allocate a HostPageAllocator + PageAllocator on a real device.
//! 2. Allocate two VRAM chunks with host backing via `alloc_with_host_backing`.
//! 3. Write a known byte pattern into chunk A's VRAM slot via hipMemcpy H2D.
//! 4. Evict chunk A to host via `evict_chunk_and_free` (single-call wrapper).
//! 5. Assert chunk A's VRAM slot is freed (free_count increases) and tier==HostPinned.
//! 6. Assert the host copy matches the original pattern.
//! 7. Promote chunk A back via `promote_chunk` + manual hipStreamSynchronize.
//! 8. Read back promoted VRAM data and assert byte-level data identity.
//!
//! # Run command (coordinator uses launch-gpu.py)
//!
//! ```
//! python3 scripts/launch-gpu.py --timeout 300 -- \
//!   cargo test -p braidinfer-runtime --test kv_tier_promote_evict_test \
//!   -- --nocapture
//! ```
//!
//! # GPU requirement
//!
//! Requires HIP device 0.  Skips gracefully if no GPU is available.

use braidinfer_core::types::DeviceId;
use braidinfer_hip::ffi;
use braidinfer_runtime::paged_kv::{
    ChunkTier, HostPageAllocator, PageAllocator, alloc_with_host_backing,
    evict_chunk_and_free, promote_chunk,
};
use braidinfer_runtime::config::ModelConfig;

/// Byte pattern used as KV data sentinel.
const PATTERN_BYTE: u8 = 0xCD;

#[test]
fn test_promote_evict_data_identity() {
    let device = DeviceId(0);

    // Skip rather than panic if no HIP GPU is available.
    let init_result = braidinfer_hip::error::check(unsafe { ffi::hipSetDevice(device.0 as i32) });
    if init_result.is_err() {
        eprintln!("kv_tier_promote_evict_test: no HIP device, skipping");
        return;
    }

    // Build HostPageAllocator + PageAllocator.
    // PageAllocator::new uses chunk_tokens to compute chunk_bytes from config.
    // We use chunk_tokens=1 to get a small allocation (fast test).
    let config = ModelConfig::qwen35_0_8b();
    let chunk_tokens: usize = 1;
    let vram_capacity: u32 = 4;
    let host_capacity: u32 = 4;

    let mut page_alloc = PageAllocator::new(device, &config, chunk_tokens, vram_capacity)
        .expect("PageAllocator::new failed");
    let alloc_chunk_bytes = page_alloc.chunk_bytes();

    let mut host_alloc = HostPageAllocator::new(alloc_chunk_bytes, host_capacity)
        .expect("HostPageAllocator::new returned None (hipHostMalloc failed)");

    // Create SDMA stream (same as persistent_dispatch::sdma_stream pattern).
    let mut sdma_stream: ffi::hipStream_t = std::ptr::null_mut();
    braidinfer_hip::error::check(unsafe { ffi::hipStreamCreate(&mut sdma_stream) })
        .expect("hipStreamCreate failed");

    // --- Allocate two chunks with host backing ---
    let handle_a = alloc_with_host_backing(&mut page_alloc, &mut host_alloc)
        .expect("alloc_with_host_backing chunk A failed");
    let handle_b = alloc_with_host_backing(&mut page_alloc, &mut host_alloc)
        .expect("alloc_with_host_backing chunk B failed");

    assert_eq!(handle_a.tier(), ChunkTier::Vram, "chunk A should start in Vram tier");
    assert_eq!(handle_b.tier(), ChunkTier::Vram, "chunk B should start in Vram tier");
    assert_ne!(handle_a.host_slot_index(), u32::MAX, "chunk A must have host backing");
    assert_ne!(handle_b.host_slot_index(), u32::MAX, "chunk B must have host backing");

    // --- Write known pattern into chunk A's VRAM slot ---
    let pattern: Vec<u8> = vec![PATTERN_BYTE; alloc_chunk_bytes];
    let a_vram_ptr = page_alloc
        .slot_ptr(handle_a.slot_index())
        .cast::<std::ffi::c_void>() as *mut std::ffi::c_void;
    braidinfer_hip::error::check(unsafe {
        ffi::hipMemcpy(
            a_vram_ptr,
            pattern.as_ptr().cast(),
            alloc_chunk_bytes,
            ffi::hipMemcpyHostToDevice,
        )
    })
    .expect("initial H2D write to chunk A failed");

    // Mark chunk A as sealed (len = chunk_tokens) — evict_chunk precondition.
    // chunk_tokens=1 so one increment suffices.
    for _ in 0..chunk_tokens {
        handle_a.increment_len();
    }
    assert_eq!(handle_a.len() as usize, chunk_tokens);

    // --- Evict chunk A to host ---
    // Record free count before evict.
    let free_before = page_alloc.free_count();

    // `evict_chunk_and_free` issues D2H + hipStreamSynchronize + frees VRAM + flips tier.
    evict_chunk_and_free(&handle_a, &mut page_alloc, &host_alloc, sdma_stream)
        .expect("evict_chunk_and_free failed");

    assert_eq!(handle_a.tier(), ChunkTier::HostPinned, "chunk A should be HostPinned after evict");
    assert_eq!(
        page_alloc.free_count(),
        free_before + 1,
        "VRAM free count should increase by 1 after evict"
    );

    // Verify host copy has the pattern.
    let host_data = unsafe {
        std::slice::from_raw_parts(handle_a.host_ptr(&host_alloc), alloc_chunk_bytes)
    };
    assert!(
        host_data.iter().all(|&b| b == PATTERN_BYTE),
        "host copy after evict does not match pattern (evict D2H data corruption)"
    );

    // --- Promote chunk A back to VRAM ---
    // promote_chunk: issues H2D async copy + updates vram_slot + flips tier.
    promote_chunk(&handle_a, &mut page_alloc, &host_alloc, sdma_stream)
        .expect("promote_chunk failed");

    // Caller must flush before using the slot (mirrors flush_tier_ops contract).
    braidinfer_hip::error::check(unsafe { ffi::hipStreamSynchronize(sdma_stream) })
        .expect("hipStreamSynchronize after promote failed");

    assert_eq!(handle_a.tier(), ChunkTier::Vram, "chunk A should be Vram after promote");

    // --- Read back promoted VRAM data and assert identity ---
    let mut readback = vec![0u8; alloc_chunk_bytes];
    let a_new_vram_ptr = page_alloc.slot_ptr(handle_a.slot_index());
    braidinfer_hip::error::check(unsafe {
        ffi::hipMemcpy(
            readback.as_mut_ptr().cast(),
            a_new_vram_ptr.cast(),
            alloc_chunk_bytes,
            ffi::hipMemcpyDeviceToHost,
        )
    })
    .expect("readback after promote failed");

    assert!(
        readback.iter().all(|&b| b == PATTERN_BYTE),
        "promote data identity failed: promoted VRAM != original pattern \
         (expected all 0x{:02X}, got first mismatch at byte {})",
        PATTERN_BYTE,
        readback.iter().position(|&b| b != PATTERN_BYTE).unwrap_or(0),
    );

    // Cleanup.
    braidinfer_hip::error::check(unsafe { ffi::hipStreamDestroy(sdma_stream) })
        .expect("hipStreamDestroy failed");

    // Note: page_alloc and host_alloc will be dropped here. In production they are
    // wrapped in ManuallyDrop (persistent worker teardown); in tests, the worker is
    // never started, so normal Drop is safe.
    println!(
        "PASS: promote/evict data identity verified ({} bytes per chunk, {} VRAM + {} host slots)",
        alloc_chunk_bytes, vram_capacity, host_capacity
    );
}
