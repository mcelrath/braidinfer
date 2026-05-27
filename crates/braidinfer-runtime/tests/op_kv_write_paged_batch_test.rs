//! bd srg6.4 Phase 3c: trace gate for OP_KV_WRITE_PAGED_BATCH.
//!
//! Verifies that the new batched paged KV writer produces byte-identical
//! chunk contents to the per-(token,head) OP_D2D_COPY emission path.
//!
//! Strategy (per srg6.4 plan, "ALTERNATIVE" branch):
//! - One MappedHostBuffer<f32> chunk pool (HOST-mapped, GPU-readable).
//! - Path A: emit one D2dCopyInst per (token,head) writing head_dim floats
//!   to the computed (chunk_idx, in_chunk, head) destination. Capture
//!   chunk bytes from host_ptr.
//! - Reset chunk pool to zero via host_ptr (CPU MMIO; safe under the
//!   running persistent worker — no HIP API involved).
//! - Path B: emit a single KvWritePagedBatchInst covering the same range.
//!   Capture chunk bytes.
//! - Assert byte-equal.
//!
//! Three parameter sets (nkh=2, head_dim=128, chunk_tokens=64):
//!   1. start_pos=0,  N=8    — single chunk
//!   2. start_pos=0,  N=128  — exactly two chunks
//!   3. start_pos=32, N=64   — mid-chunk start, spans chunk 0 + chunk 1
//!
//! All paths run on GPU 0 via PersistentDispatch::test_dispatch_batch_slice
//! (synchronous batched dispatch returning after worker ack).

use braidinfer_core::types::DeviceId;
use braidinfer_hip::memory::MappedHostBuffer;
use braidinfer_runtime::megakernel::srg6_4_test_api::{
    D2dCopyInst, Instruction, KvWritePagedBatchInst, OP_HALT, SHARED_LPROJ_TOTAL,
};
use braidinfer_runtime::persistent_dispatch::PersistentDispatch;
use braidinfer_runtime::watchdog::WatchdogThread;
use std::sync::Arc;

const NKH: usize = 2;
const HEAD_DIM: usize = 128;
const CHUNK_TOKENS: usize = 64;

fn run_case(
    dispatch: &mut PersistentDispatch,
    start_pos: u32,
    n_tokens: u32,
    case_name: &str,
) {
    let n = n_tokens as usize;
    let sp = start_pos as usize;
    let end_pos = sp + n;
    let num_chunks = (end_pos + CHUNK_TOKENS - 1) / CHUNK_TOKENS;
    // Floats per chunk (one layer only; layer_kv_offset=0).
    let chunk_floats = NKH * CHUNK_TOKENS * HEAD_DIM;
    let chunk_bytes = chunk_floats * std::mem::size_of::<f32>();

    // -- Allocations (host-mapped, coherent so GPU writes are immediately
    //    visible to host_ptr reads without a stream synchronize).
    let src_floats = n * NKH * HEAD_DIM;
    let mut src = MappedHostBuffer::<f32>::alloc_coherent(src_floats)
        .expect("alloc src");
    // Pool of all chunks back-to-back; per-chunk base pointer goes into
    // page_table[chunk_idx].
    let mut chunk_pool = MappedHostBuffer::<f32>::alloc_coherent(num_chunks * chunk_floats)
        .expect("alloc chunk_pool");
    let mut page_table = MappedHostBuffer::<u64>::alloc_coherent(num_chunks)
        .expect("alloc page_table");

    // Fill src with deterministic pattern.
    unsafe {
        let p = src.host_ptr();
        for t in 0..n {
            for h in 0..NKH {
                for d in 0..HEAD_DIM {
                    let v = (t * 1000 + h * 100 + d) as f32 + 0.5;
                    let idx = (t * NKH + h) * HEAD_DIM + d;
                    p.add(idx).write_volatile(v);
                }
            }
        }
    }
    // Populate page_table: each chunk's GPU-side base ptr.
    let dev_chunk_base = chunk_pool.as_ptr() as *const u8;
    unsafe {
        let pt = page_table.host_ptr();
        for ci in 0..num_chunks {
            let cb = dev_chunk_base.add(ci * chunk_bytes) as u64;
            pt.add(ci).write_volatile(cb);
        }
    }

    // --- Path A: emit one D2dCopyInst per (token, head) ---
    // Zero chunk pool (CPU).
    unsafe {
        let cp = chunk_pool.host_ptr();
        for i in 0..(num_chunks * chunk_floats) {
            cp.add(i).write_volatile(0.0);
        }
    }
    let mut prog_a: Vec<Instruction> = Vec::with_capacity(n * NKH + 1);
    for t in 0..n {
        let abs_pos = sp + t;
        let ci = abs_pos / CHUNK_TOKENS;
        let in_chunk = abs_pos % CHUNK_TOKENS;
        for h in 0..NKH {
            // dst = chunk_base[ci] + h * chunk_tokens * head_dim * 4 + in_chunk * head_dim * 4
            let dst_floats_off =
                h * CHUNK_TOKENS * HEAD_DIM + in_chunk * HEAD_DIM;
            let dst = unsafe {
                (chunk_pool.as_mut_ptr())
                    .add(ci * chunk_floats + dst_floats_off)
            };
            let src_off = (t * NKH + h) * HEAD_DIM;
            let src_ptr = unsafe { src.as_ptr().add(src_off) };
            let inst = D2dCopyInst::test_new(1, dst, src_ptr, HEAD_DIM as i32);
            prog_a.push(inst.into_inst());
        }
    }
    prog_a.push(Instruction::test_new(OP_HALT, 0));
    // dispatch_via_worker strips OP_HALT; we use the lower-level
    // test_dispatch_batch_slice and must strip ourselves.
    let halt_idx = prog_a.len() - 1;
    dispatch.test_dispatch_batch_slice(0, &prog_a[..halt_idx]);

    // Capture path-A output bytes.
    let total_bytes = num_chunks * chunk_bytes;
    let mut bytes_a: Vec<u8> = vec![0u8; total_bytes];
    unsafe {
        let src_b = chunk_pool.host_ptr() as *const u8;
        std::ptr::copy_nonoverlapping(src_b, bytes_a.as_mut_ptr(), total_bytes);
    }

    // --- Path B: single KvWritePagedBatchInst ---
    // Zero chunk pool again.
    unsafe {
        let cp = chunk_pool.host_ptr();
        for i in 0..(num_chunks * chunk_floats) {
            cp.add(i).write_volatile(0.0);
        }
    }
    let inst_b = KvWritePagedBatchInst::test_new(
        src.as_ptr(),
        page_table.as_ptr(),
        0, // layer_kv_offset
        start_pos,
        n_tokens,
        0,                // local_kv_head_start
        NKH as u16,       // local_nkh
        HEAD_DIM as u16,
        CHUNK_TOKENS as u16,
    );
    let prog_b = vec![inst_b.into_inst()];
    dispatch.test_dispatch_batch_slice(0, &prog_b);

    // Capture path-B output bytes.
    let mut bytes_b: Vec<u8> = vec![0u8; total_bytes];
    unsafe {
        let src_b = chunk_pool.host_ptr() as *const u8;
        std::ptr::copy_nonoverlapping(src_b, bytes_b.as_mut_ptr(), total_bytes);
    }

    // Compare.
    if bytes_a != bytes_b {
        let mut first_diff = None;
        for i in 0..total_bytes {
            if bytes_a[i] != bytes_b[i] {
                first_diff = Some(i);
                break;
            }
        }
        panic!(
            "[{}] start_pos={} N={} BYTES MISMATCH at offset {:?} \
             (chunk_bytes={}, num_chunks={})",
            case_name, start_pos, n_tokens, first_diff, chunk_bytes, num_chunks
        );
    }
    println!(
        "[{}] start_pos={} N={} OK ({} bytes across {} chunk(s))",
        case_name, start_pos, n_tokens, total_bytes, num_chunks
    );
}

#[test]
fn test_kv_write_paged_batch_matches_d2d_copy() {
    let device = DeviceId(0);
    let watchdog = Arc::new(WatchdogThread::spawn());
    let mut dispatch = PersistentDispatch::init_with_total(
        1,
        &[],
        SHARED_LPROJ_TOTAL,
        0,
        watchdog,
    )
    .expect("init dispatch");
    // GPU 0 worker is added lazily by Model in production. Here we add it
    // explicitly so test_dispatch_batch_slice has a worker on slot 0.
    dispatch
        .add_device(device, SHARED_LPROJ_TOTAL)
        .expect("add GPU 0 persistent worker");

    run_case(&mut dispatch, 0, 8, "case1_single_chunk_N8");
    run_case(&mut dispatch, 0, 128, "case2_two_chunks_N128");
    run_case(&mut dispatch, 32, 64, "case3_midchunk_start_N64");

    dispatch.shutdown();
}
