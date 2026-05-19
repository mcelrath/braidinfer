// rdna3_mailbox.h — Mailbox payload load/store primitives for gfx1100.
//
// PURPOSE
// -------
// CPU-as-producer → GPU-as-consumer mailbox patterns (the persistent_worker
// pattern in this codebase): a host-mapped UC mailbox struct is declared
// `volatile WorkerQueue*` at the kernel signature, but its instruction
// payload (`queue->inst[]`) is typically accessed via a *cast* to compute a
// per-pc element address:
//
//     const u64* src = (const u64*)(queue->inst + pc * INST_SIZE_WORDS);
//
// That `(const u64*)` cast SILENTLY STRIPS volatile. ROCm clang then emits a
// plain L2-cached global_load_b64 for the descriptor read — no glc, no dlc.
// On a cold-start first-mailbox-transaction, the consumer worker GPU may
// hold stale data for that line and read garbage instruction words → NaN
// logits, "Sig A" cold-start race (see GFX1100_ARCH.md §11.19 "Cold-start mailbox
// visibility race" section, bd 4e2m).
//
// Adding glc+dlc via volatile-preserving casts is correct on first
// principles (forces L1+L2 invalidate-on-load), but EMPIRICALLY does NOT
// close the cold-start race on gfx1100 (Exp 1a: 16/30 vs 16/30 baseline;
// the stale layer is below L1/L2, in MES μC private cache or memory hub).
// The cure is warmup-discard (see generate.rs).
//
// However, the volatile-stripping cast is a real source-level bug — anyone
// writing similar mailbox-consumer code should use these primitives instead
// of raw pointer arithmetic + cast. They preserve volatile through the
// arithmetic and document the intent at point-of-use.
//
// USAGE
// -----
//     // Consumer (worker GPU reads from host-mapped mailbox payload):
//     u64 word = braidinfer::rdna3::mailbox_load_descriptor(
//         &queue->inst[pc * INST_SIZE_WORDS + threadIdx.x]
//     );
//
//     // Producer (shader writes to host-mapped mailbox payload — currently
//     // CPU-only in this codebase; this is forward-compat for any future
//     // GPU-as-producer pattern):
//     braidinfer::rdna3::mailbox_store_descriptor(
//         &queue->ack_payload[i], value
//     );
//
// Both expand to single global_load/store with glc+dlc bits set; zero
// runtime cost vs a hand-rolled volatile cast that did the same.
//
// REFS
// ----
// bd braidinfer-4e2m   (top issue: gfx1100 multi-GPU cold-start NaN race)
// bd braidinfer-tm5t   (mailbox warmup A2 experiment; 30/30 with prefill+warmup)
// bd braidinfer-upxd   (rc=134 shutdown abort, separate)
// linux-p2p 0012/0013  (kernel HDP flush patches, complementary)
// GFX1100_ARCH.md §11.19         (full mechanism analysis + falsified interventions)
//
// HW QUIRKS (recap, full detail in GFX1100_ARCH.md §11.19):
//   - gfx1100 has NO buffer_gl2_inv / buffer_invl2 in ISA (composable_kernel §5.3).
//     The only L2-control bit available is `dlc` on per-op loads.
//   - __hip_atomic_load(SYSTEM scope) HANGS on gfx1100 multi-GPU.
//     See rdna3_persistent.h:40-41. Signal LOADS use volatile reads only.
//   - host_uc_store_agent (rdna3_peer.h) is the canonical signal-write
//     primitive; AGENT scope (NOT SYSTEM) avoids s_waitcnt_vscnt traps.
//
#pragma once

#include <hip/hip_runtime.h>

namespace braidinfer { namespace rdna3 {

// Load one element of a host-mapped mailbox descriptor payload. The const
// volatile qualifier on the parameter prevents the volatile-stripping cast
// hazard at call sites: callers MUST pass a volatile pointer (which they
// already have from `volatile WorkerQueue* queue`). Returns the value via
// the volatile-preserving deref, which ROCm clang emits as a flat_load with
// glc+dlc bits set (L0/L1 + L2 invalidate-on-load).
//
// Note: glc+dlc is empirically necessary-but-not-sufficient for cold-start
// correctness on gfx1100 — see file header for the falsification record.
// This primitive is the SHADER-LEVEL hygiene. Warmup-discard is the cure.
template <typename T>
__device__ __forceinline__ T mailbox_load_descriptor(const volatile T* src) {
    return *src;
}

// Symmetric primitive for any future GPU-as-producer mailbox writes. The
// current persistent_worker pattern has CPU as producer (Rust write_volatile
// in persistent_dispatch.rs), so this is currently unused in tree. Kept for
// forward compatibility — if a future cross-GPU producer pattern emerges,
// using this primitive avoids the same cast-strip hazard on the write side.
template <typename T>
__device__ __forceinline__ void mailbox_store_descriptor(volatile T* dst, T value) {
    *dst = value;
}

}} // namespace braidinfer::rdna3
