//! Per-opcode instruction structs for the megakernel.
//!
//! Each struct is `#[repr(C)]` with exactly `INST_SIZE` (18) u64-equivalent fields.
//! `words[0]` = `opcode_gridx: u64` encoding `opcode | (grid_x << 32)`.
//! All structs transmute to/from `Instruction` via `into_inst()`.
//!
//! Only the Rust encoding layer is changed here. The `Instruction { words: [u64; INST_SIZE] }`
//! storage type is preserved. GPU-side `.hip` files are NOT modified (Phase 2).

use super::{
    INST_SIZE, Instruction,
    OP_ATTN_PAGED, OP_ATTN_PAGED_Q, OP_ATTN_PREFILL, OP_BARRIER, OP_CONV1D, OP_D2D_COPY,
    OP_DEINTERLEAVE, OP_EMBEDDING, OP_GDN_GATE, OP_GDN_RECUR, OP_GQA_ATTN, OP_HALT,
    OP_MAMBA2_CONV1D, OP_MAMBA2_NORM_GATED,
    OP_MOE_FFN_REMOTE, OP_MOE_GATE, OP_MROPE, OP_OUTPUT_GATE, OP_QK_NORM,
    OP_RELU_SQ, OP_CONV1D_3X, OP_LINEAR_PROJ_2X, OP_RESIDUAL_ADD, OP_RMSNORM_GATE,
    OP_SCALE_ADD, OP_SIGMOID_WEIGHTED_ADD, OP_SILU_MUL, OP_SSM_UPDATE,
    OP_DOT_SIGMOID_SCALE_ADD,
};

// Compile-time size assertions: each struct must be exactly INST_SIZE * 8 bytes.
const _INST_BYTES: usize = INST_SIZE * 8;

macro_rules! assert_inst_size {
    ($t:ty) => {
        const _: () = assert!(
            std::mem::size_of::<$t>() == _INST_BYTES,
            concat!(stringify!($t), " size mismatch")
        );
    };
}

/// Helper: encode opcode + grid_x into words[0].
#[inline(always)]
pub(crate) fn make_opcode_gridx(opcode: u32, grid_x: u32) -> u64 {
    opcode as u64 | ((grid_x as u64) << 32)
}

macro_rules! impl_inst {
    ($t:ty) => {
        impl $t {
            #[allow(dead_code)]
            pub(crate) fn into_inst(self) -> Instruction {
                unsafe { std::mem::transmute(self) }
            }
        }
        // SAFETY: *Inst structs hold raw `*mut f32` / `*const f32` device pointers
        // that are valid in exactly one HIP context — the device on which the
        // megakernel program will execute. The structs are constructed by the
        // host (compiler-side) and consumed by the GPU kernel; once handed to
        // the kernel via the host-mapped WorkerQueue mailbox or as a device-
        // program upload, the host MUST NOT dereference them (HIP context
        // ordering would be violated). Sending an Instruction across threads
        // is safe because:
        //   (a) the host never derefs the pointers — they're opaque u64s in
        //       the on-wire instruction layout.
        //   (b) the receiving thread that hands them to the GPU does so under
        //       the same DeviceGuard ownership discipline as the constructor.
        //   (c) the GPU side has its own per-CU access; thread-of-origin on
        //       the host is irrelevant.
        // This Send impl is required because Rust auto-derives !Send for any
        // struct holding raw pointers. The actual safety invariant is the
        // device-context discipline above, not pointer aliasing — there is no
        // aliasing because the host never touches the data once dispatched.
        unsafe impl Send for $t {}
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_NOP / dump header (opcode=0)
// words[1] = dump_buf_ptr, words[2] = max_slots (i32), words[3] = dump_counter_ptr
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct NopInst {
    pub opcode_gridx: u64,
    pub dump_buf: *const u8,
    pub max_slots: u64,
    pub dump_counter: *const i32,
    pub _pad: [u64; 15],
}
assert_inst_size!(NopInst);
impl_inst!(NopInst);

// ─────────────────────────────────────────────────────────────────────────────
// OP_RMSNORM / OP_RMSNORM_WX (opcodes 1, 27)
// words[1]=output, [2]=input, [3]=weight, [4]=dim(i32), [5]=eps(f32 bits)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct RmsNormInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub input: *const f32,
    pub weight: *const u16, // bf16 weight
    pub dim: u64,           // i32 zero-extended
    pub eps_bits: u64,      // f32::to_bits() as u64
    // bd braidinfer-sm16 sentinel: when sentinel_ptr != 0, kernel writes
    // sentinel_seq to *(u32*)sentinel_ptr after the agent-scope fence at
    // op_rmsnorm_wx exit. Consumer (op_d2d_copy with matching sentinel
    // fields) spin-waits on this value.
    pub sentinel_ptr: u64,
    pub sentinel_seq: u64,
    pub _pad: [u64; 11],
}
assert_inst_size!(RmsNormInst);
impl_inst!(RmsNormInst);

impl RmsNormInst {
    pub(crate) fn new(opcode: u32, grid_x: u32, output: *mut f32, input: *const f32, weight: *const u16, dim: i32, eps: f32) -> Self {
        RmsNormInst {
            opcode_gridx: make_opcode_gridx(opcode, grid_x),
            output,
            input,
            weight,
            dim: dim as u64,
            eps_bits: eps.to_bits() as u64,
            sentinel_ptr: 0,
            sentinel_seq: 0,
            _pad: [0; 11],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_LINEAR_PROJ / OP_LINEAR_PROJ_RNF4 / OP_LINEAR_PROJ_PCG32 (opcodes 2, 25, 26)
// words[1]=output, [2]=weight, [3]=input, [4]=out_dim(i32), [5]=in_dim(i32), [6]=batch(i32)
// Note: batch defaults to 0 (kernel treats 0 as batch=1 for decode path).
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct LinearProjInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub weight: *const u8, // bf16 or packed u8
    pub input: *const f32,
    pub out_dim: u64,      // i32 zero-extended
    pub in_dim: u64,       // i32 zero-extended
    pub batch: u64,        // i32, 0 = single token
    pub _pad: [u64; 12],
}
assert_inst_size!(LinearProjInst);
impl_inst!(LinearProjInst);

impl LinearProjInst {
    pub(crate) fn new(opcode: u32, grid_x: u32, output: *mut f32, weight: *const u8, input: *const f32, out_dim: i32, in_dim: i32, batch: i32) -> Self {
        LinearProjInst {
            opcode_gridx: make_opcode_gridx(opcode, grid_x),
            output,
            weight,
            input,
            out_dim: out_dim as u64,
            in_dim: in_dim as u64,
            batch: batch as u64,
            _pad: [0; 12],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_CONV1D (opcode 3)
// words[1]=state(w), [2]=input, [3]=weight, [4]=output(w), [5]=dim(i32), [6]=kernel_size(i32)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct Conv1dInst {
    pub opcode_gridx: u64,
    pub state: *mut f32,
    pub input: *const f32,
    pub weight: *const u16, // bf16
    pub output: *mut f32,
    pub dim: u64,
    pub kernel_size: u64,
    pub _pad: [u64; 12],
}
assert_inst_size!(Conv1dInst);
impl_inst!(Conv1dInst);

impl Conv1dInst {
    pub(crate) fn new(grid_x: u32, state: *mut f32, input: *const f32, weight: *const u16, output: *mut f32, dim: i32, kernel_size: i32) -> Self {
        Conv1dInst {
            opcode_gridx: make_opcode_gridx(OP_CONV1D, grid_x),
            state,
            input,
            weight,
            output,
            dim: dim as u64,
            kernel_size: kernel_size as u64,
            _pad: [0; 12],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_GDN_GATE (opcode 4)
// words[1]=output(w), [2]=a_proj, [3]=a_log, [4]=dt_bias, [5]=num_heads(i32)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct GdnGateInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub a_proj: *const f32,
    pub a_log: *const f32,
    pub dt_bias: *const u16, // bf16
    pub num_heads: u64,
    pub _pad: [u64; 13],
}
assert_inst_size!(GdnGateInst);
impl_inst!(GdnGateInst);

impl GdnGateInst {
    pub(crate) fn new(grid_x: u32, output: *mut f32, a_proj: *const f32, a_log: *const f32, dt_bias: *const u16, num_heads: i32) -> Self {
        GdnGateInst {
            opcode_gridx: make_opcode_gridx(OP_GDN_GATE, grid_x),
            output,
            a_proj,
            a_log,
            dt_bias,
            num_heads: num_heads as u64,
            _pad: [0; 13],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_GDN_RECUR (opcode 5)
// words[1]=q, [2]=k, [3]=v, [4]=gate, [5]=b_proj, [6]=state(w), [7]=out(w),
//        [8]=kd(i32), [9]=vd(i32), [10]=gqa_group(i32)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct GdnRecurInst {
    pub opcode_gridx: u64,
    pub q: *const f32,
    pub k: *const f32,
    pub v: *const f32,
    pub gate: *const f32,
    pub b_proj: *const f32,
    pub state: *mut f32,
    pub output: *mut f32,
    pub kd: u64,
    pub vd: u64,
    pub gqa_group: u64,
    pub num_heads: u64,
    pub _pad: [u64; 7],
}
assert_inst_size!(GdnRecurInst);
impl_inst!(GdnRecurInst);

impl GdnRecurInst {
    pub(crate) fn new(grid_x: u32, num_heads: u32, q: *const f32, k: *const f32, v: *const f32, gate: *const f32, b_proj: *const f32, state: *mut f32, output: *mut f32, kd: i32, vd: i32, gqa_group: i32) -> Self {
        GdnRecurInst {
            opcode_gridx: make_opcode_gridx(OP_GDN_RECUR, grid_x),
            q, k, v, gate, b_proj, state, output,
            kd: kd as u64,
            vd: vd as u64,
            gqa_group: gqa_group as u64,
            num_heads: num_heads as u64,
            _pad: [0; 7],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_RMSNORM_GATE (opcode 6)
// words[1]=output(w), [2]=x, [3]=z, [4]=weight, [5]=num_heads, [6]=vd, [7]=eps
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct RmsNormGateInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub x: *const f32,
    pub z: *const f32,
    pub weight: *const f32, // f32 (GDN output_norm uses f32)
    pub num_heads: u64,
    pub vd: u64,
    pub eps_bits: u64,
    pub _pad: [u64; 11],
}
assert_inst_size!(RmsNormGateInst);
impl_inst!(RmsNormGateInst);

impl RmsNormGateInst {
    pub(crate) fn new(grid_x: u32, output: *mut f32, x: *const f32, z: *const f32, weight: *const f32, num_heads: i32, vd: i32, eps: f32) -> Self {
        RmsNormGateInst {
            opcode_gridx: make_opcode_gridx(OP_RMSNORM_GATE, grid_x),
            output, x, z, weight,
            num_heads: num_heads as u64,
            vd: vd as u64,
            eps_bits: eps.to_bits() as u64,
            _pad: [0; 11],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_RESIDUAL_ADD (opcode 7)
// words[1]=output(w), [2]=src, [3]=residual, [4]=n(i32)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct ResidualAddInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub src: *const f32,
    pub residual: *const f32,
    pub n: u64,
    pub _pad: [u64; 14],
}
assert_inst_size!(ResidualAddInst);
impl_inst!(ResidualAddInst);

impl ResidualAddInst {
    pub(crate) fn new(grid_x: u32, output: *mut f32, src: *const f32, residual: *const f32, n: i32) -> Self {
        ResidualAddInst {
            opcode_gridx: make_opcode_gridx(OP_RESIDUAL_ADD, grid_x),
            output, src, residual,
            n: n as u64,
            _pad: [0; 14],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_QK_NORM (opcode 8)
// words[1]=q(w), [2]=k(w), [3]=q_norm, [4]=k_norm, [5]=nqh, [6]=nkh, [7]=hd, [8]=eps, [9]=n(batch)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct QkNormInst {
    pub opcode_gridx: u64,
    pub q: *mut f32,
    pub k: *mut f32,
    pub q_norm: *const u16,
    pub k_norm: *const u16,
    pub nqh: u64,
    pub nkh: u64,
    pub hd: u64,
    pub eps_bits: u64,
    pub batch: u64, // only set for n>1; 0 for single-token
    pub _pad: [u64; 9],
}
assert_inst_size!(QkNormInst);
impl_inst!(QkNormInst);

impl QkNormInst {
    pub(crate) fn new(grid_x: u32, q: *mut f32, k: *mut f32, q_norm: *const u16, k_norm: *const u16, nqh: i32, nkh: i32, hd: i32, eps: f32, batch: i32) -> Self {
        QkNormInst {
            opcode_gridx: make_opcode_gridx(OP_QK_NORM, grid_x),
            q, k, q_norm, k_norm,
            nqh: nqh as u64,
            nkh: nkh as u64,
            hd: hd as u64,
            eps_bits: eps.to_bits() as u64,
            batch: batch as u64,
            _pad: [0; 9],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_MROPE (opcode 9)
// words[1]=q(w), [2]=k(w), [3]=inv_freq, [4]=pos_ids, [5]=nqh, [6]=nkh,
//        [7]=hd, [8]=rd, [9]=s0, [10]=s1, [11]=s2, [12]=batch
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct MropeInst {
    pub opcode_gridx: u64,
    pub q: *mut f32,
    pub k: *mut f32,
    pub inv_freq: *const f32,
    pub pos_ids: *const i32,
    pub nqh: u64,
    pub nkh: u64,
    pub hd: u64,
    pub rd: u64,
    pub s0: u64,
    pub s1: u64,
    pub s2: u64,
    pub batch: u64,
    pub dump: *mut u32, // 5ax diag: optional u32 dump buffer (null disables)
    pub _pad: [u64; 5],
}
assert_inst_size!(MropeInst);
impl_inst!(MropeInst);

impl MropeInst {
    pub(crate) fn new(grid_x: u32, q: *mut f32, k: *mut f32, inv_freq: *const f32, pos_ids: *const i32, nqh: i32, nkh: i32, hd: i32, rd: i32, s0: i32, s1: i32, s2: i32, batch: i32) -> Self {
        MropeInst {
            opcode_gridx: make_opcode_gridx(OP_MROPE, grid_x),
            q, k, inv_freq, pos_ids,
            nqh: nqh as u64,
            nkh: nkh as u64,
            hd: hd as u64,
            rd: rd as u64,
            s0: s0 as u64,
            s1: s1 as u64,
            s2: s2 as u64,
            batch: batch as u64,
            dump: std::ptr::null_mut(),
            _pad: [0; 5],
        }
    }

}

// ─────────────────────────────────────────────────────────────────────────────
// OP_GQA_ATTN (opcode 10)
// words[1]=out(w), [2]=q, [3]=k_cache, [4]=v_cache, [5]=nqh, [6]=nkh,
//        [7]=hd, [8]=seq_len, [9]=max_seq_len, [10]=q_head_start
// (q_head_start is only for multi-GPU head-parallel; defaults 0)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct GqaAttnInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub q: *const f32,
    pub k_cache: *const f32,
    pub v_cache: *const f32,
    pub nqh: u64,
    pub nkh: u64,
    pub hd: u64,
    pub seq_len: u64,
    pub max_seq_len: u64,
    pub q_head_start: u64,
    pub _pad: [u64; 8],
}
assert_inst_size!(GqaAttnInst);
impl_inst!(GqaAttnInst);

impl GqaAttnInst {
    pub(crate) fn new(grid_x: u32, output: *mut f32, q: *const f32, k_cache: *const f32, v_cache: *const f32, nqh: i32, nkh: i32, hd: i32, seq_len: i32, max_seq_len: i32) -> Self {
        GqaAttnInst {
            opcode_gridx: make_opcode_gridx(OP_GQA_ATTN, grid_x),
            output, q, k_cache, v_cache,
            nqh: nqh as u64,
            nkh: nkh as u64,
            hd: hd as u64,
            seq_len: seq_len as u64,
            max_seq_len: max_seq_len as u64,
            q_head_start: 0,
            _pad: [0; 8],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_OUTPUT_GATE (opcode 11)
// words[1]=output(w), [2]=attn_out, [3]=gate, [4]=size(i32)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct OutputGateInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub attn_out: *const f32,
    pub gate: *const f32,
    pub size: u64,
    pub _pad: [u64; 14],
}
assert_inst_size!(OutputGateInst);
impl_inst!(OutputGateInst);

impl OutputGateInst {
    pub(crate) fn new(grid_x: u32, output: *mut f32, attn_out: *const f32, gate: *const f32, size: i32) -> Self {
        OutputGateInst {
            opcode_gridx: make_opcode_gridx(OP_OUTPUT_GATE, grid_x),
            output, attn_out, gate,
            size: size as u64,
            _pad: [0; 14],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_FFN_GATE_UP / OP_FFN_GATE_UP_RNF4 (opcodes 12, 30)
// words[1]=out(w), [2]=hidden, [3]=norm_weight, [4]=w_gate, [5]=w_up,
//        [6]=hs(i32), [7]=is(i32), [8]=eps, [9]=batch(i32)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct FfnGateUpInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub hidden: *const f32,
    pub norm_weight: *const u16,
    pub w_gate: *const u8, // bf16 or packed
    pub w_up: *const u8,
    pub hs: u64,
    pub intermediate: u64,
    pub eps_bits: u64,
    pub batch: u64,
    pub _pad: [u64; 9],
}
assert_inst_size!(FfnGateUpInst);
impl_inst!(FfnGateUpInst);

impl FfnGateUpInst {
    pub(crate) fn new(opcode: u32, grid_x: u32, output: *mut f32, hidden: *const f32, norm_weight: *const u16, w_gate: *const u8, w_up: *const u8, hs: i32, intermediate: i32, eps: f32, batch: i32) -> Self {
        FfnGateUpInst {
            opcode_gridx: make_opcode_gridx(opcode, grid_x),
            output, hidden, norm_weight, w_gate, w_up,
            hs: hs as u64,
            intermediate: intermediate as u64,
            eps_bits: eps.to_bits() as u64,
            batch: batch as u64,
            _pad: [0; 9],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_FFN_DOWN_RES / OP_FFN_DOWN_RES_RNF4 (opcodes 13, 31)
// words[1]=out(w), [2]=residual, [3]=w_down, [4]=ffn_act, [5]=hs(i32), [6]=is(i32), [7]=batch(i32)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct FfnDownResInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub residual: *const f32,
    pub w_down: *const u8,
    pub ffn_act: *const f32,
    pub hs: u64,
    pub intermediate: u64,
    pub batch: u64,
    pub _pad: [u64; 11],
}
assert_inst_size!(FfnDownResInst);
impl_inst!(FfnDownResInst);

impl FfnDownResInst {
    pub(crate) fn new(opcode: u32, grid_x: u32, output: *mut f32, residual: *const f32, w_down: *const u8, ffn_act: *const f32, hs: i32, intermediate: i32, batch: i32) -> Self {
        FfnDownResInst {
            opcode_gridx: make_opcode_gridx(opcode, grid_x),
            output, residual, w_down, ffn_act,
            hs: hs as u64,
            intermediate: intermediate as u64,
            batch: batch as u64,
            _pad: [0; 11],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_EMBEDDING (opcode 14)
// words[1]=output(w), [2]=embed_weight, [3]=token_id(i32), [4]=hs(i32)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct EmbeddingInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub embed_weight: *const u16, // bf16
    pub token_id: u64,
    pub hs: u64,
    pub _pad: [u64; 14],
}
assert_inst_size!(EmbeddingInst);
impl_inst!(EmbeddingInst);

impl EmbeddingInst {
    pub(crate) fn new(grid_x: u32, output: *mut f32, embed_weight: *const u16, token_id: i32, hs: i32) -> Self {
        EmbeddingInst {
            opcode_gridx: make_opcode_gridx(OP_EMBEDDING, grid_x),
            output,
            embed_weight,
            token_id: token_id as u64,
            hs: hs as u64,
            _pad: [0; 14],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_LM_HEAD (opcode 15) — same layout as OP_LINEAR_PROJ (reuses LinearProjInst)
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// OP_HALT (opcode 16)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct HaltInst {
    pub opcode_gridx: u64,
    pub _pad: [u64; 18],
}
assert_inst_size!(HaltInst);
impl_inst!(HaltInst);

impl HaltInst {
    pub(crate) fn new() -> Self {
        HaltInst {
            opcode_gridx: make_opcode_gridx(OP_HALT, 0),
            _pad: [0; 18],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_D2D_COPY (opcode 17)
// words[1]=dst(w), [2]=src, [3]=n_elems(i32)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct D2dCopyInst {
    pub opcode_gridx: u64,
    pub dst: *mut f32,
    pub src: *const f32,
    pub n_elems: u64,
    // bd braidinfer-sm16 sentinel wait (consumer role): when wait_ptr != 0,
    // spin-wait at entry until *(u32*)wait_ptr == wait_seq before reading src.
    pub wait_ptr: u64,
    pub wait_seq: u64,
    // bd braidinfer-sm16 sentinel signal (producer role): when signal_ptr != 0,
    // write signal_seq to *(u32*)signal_ptr after the copy completes (release).
    pub signal_ptr: u64,
    pub signal_seq: u64,
    pub _pad: [u64; 11],
}
assert_inst_size!(D2dCopyInst);
impl_inst!(D2dCopyInst);

impl D2dCopyInst {
    pub(crate) fn new(grid_x: u32, dst: *mut f32, src: *const f32, n_elems: i32) -> Self {
        D2dCopyInst {
            opcode_gridx: make_opcode_gridx(OP_D2D_COPY, grid_x),
            dst, src,
            n_elems: n_elems as u64,
            wait_ptr: 0,
            wait_seq: 0,
            signal_ptr: 0,
            signal_seq: 0,
            _pad: [0; 11],
        }
    }

    pub(crate) fn with_wait(mut self, sentinel_ptr: *const u32, seq: u32) -> Self {
        self.wait_ptr = sentinel_ptr as u64;
        self.wait_seq = seq as u64;
        self
    }

    pub(crate) fn with_signal(mut self, sentinel_ptr: *mut u32, seq: u32) -> Self {
        self.signal_ptr = sentinel_ptr as u64;
        self.signal_seq = seq as u64;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_ATTN_PAGED (opcode 18)
// words[1]=out(w), [2]=q, [3]=page_table(ptr, patched), [4]=pos_table(ptr, patched),
//        [5]=inv_freq, [6]=nqh, [7]=nkh, [8]=hd, [9]=seq_len, [10]=chunk_tokens,
//        [11]=rd, [12]=layer_k_offset(raw u64), [13]=layer_v_offset(raw u64),
//        [14]=partial_state(patched), [16]=k_norm
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct AttnPagedInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub q: *const f32,
    pub page_table: u64, // patched per step
    pub pos_table: u64,  // patched per step
    pub inv_freq: *const f32,
    pub nqh: u64,
    pub nkh: u64,
    pub hd: u64,
    pub seq_len: u64,
    pub chunk_tokens: u64,
    pub rd: u64,
    pub layer_k_offset: u64, // raw byte offset for layer within paged buffer
    pub layer_v_offset: u64,
    pub partial_state: u64, // patched when quantized KV enabled
    pub mrope_sections: u64, // packed: low 32 = section0_pairs, high 32 = section1_pairs
    pub k_norm: *const u16, // null if no QK-norm
    pub eps_bits: u64,       // f32 rms_norm_eps as u64 (low 32 bits)
    // bd srg6.2: head-slice for head-parallel paged read.
    // Encoding: bits 0-15 = local_nqh, bits 16-31 = local_q_head_start, bits 32-63 = reserved (0).
    // Single-GPU: (0 << 16) | nqh. Multi-GPU (Phase 4): per-worker slice.
    pub head_slice: u64,
}
assert_inst_size!(AttnPagedInst);
impl_inst!(AttnPagedInst);

impl AttnPagedInst {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        grid_x: u32,
        output: *mut f32,
        q: *const f32,
        inv_freq: *const f32,
        nqh: i32,
        nkh: i32,
        hd: i32,
        seq_len: i32,
        chunk_tokens: i32,
        rd: i32,
        layer_k_offset: u64,
        layer_v_offset: u64,
        k_norm: *const u16,
        eps: f32,
        mrope_section0_pairs: i32,
        mrope_section1_pairs: i32,
        local_q_head_start: u16,
        local_nqh: u16,
    ) -> Self {
        let mrope_sections = (mrope_section0_pairs as u64) | ((mrope_section1_pairs as u64) << 32);
        let head_slice = (local_nqh as u64) | ((local_q_head_start as u64) << 16);
        AttnPagedInst {
            opcode_gridx: make_opcode_gridx(OP_ATTN_PAGED, grid_x),
            output,
            q,
            page_table: 0,
            pos_table: 0,
            inv_freq,
            nqh: nqh as u64,
            nkh: nkh as u64,
            hd: hd as u64,
            seq_len: seq_len as u64,
            chunk_tokens: chunk_tokens as u64,
            rd: rd as u64,
            layer_k_offset,
            layer_v_offset,
            partial_state: 0,
            mrope_sections,
            k_norm,
            eps_bits: eps.to_bits() as u64,
            head_slice,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_ATTN_PREFILL (opcode 19)
// words[1]=out(w), [2]=q, [3]=k_cache, [4]=v_cache, [5]=nqh, [6]=nkh, [7]=hd,
//        [8]=start_pos, [9]=n, [10]=max_seq_len
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct AttnPrefillInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub q: *const f32,
    pub k_cache: *const f32,
    pub v_cache: *const f32,
    pub nqh: u64,
    pub nkh: u64,
    pub hd: u64,
    pub start_pos: u64,
    pub n: u64,
    pub max_seq_len: u64,
    pub _pad: [u64; 8],
}
assert_inst_size!(AttnPrefillInst);
impl_inst!(AttnPrefillInst);

impl AttnPrefillInst {
    pub(crate) fn new(grid_x: u32, output: *mut f32, q: *const f32, k_cache: *const f32, v_cache: *const f32, nqh: i32, nkh: i32, hd: i32, start_pos: i32, n: i32, max_seq_len: i32) -> Self {
        AttnPrefillInst {
            opcode_gridx: make_opcode_gridx(OP_ATTN_PREFILL, grid_x),
            output, q, k_cache, v_cache,
            nqh: nqh as u64,
            nkh: nkh as u64,
            hd: hd as u64,
            start_pos: start_pos as u64,
            n: n as u64,
            max_seq_len: max_seq_len as u64,
            _pad: [0; 8],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_DEINTERLEAVE (opcode 20)
// words[1]=dst_q(w), [2]=dst_gate(w), [3]=src, [4]=num_heads, [5]=head_dim, [6]=batch
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct DeinterleaveInst {
    pub opcode_gridx: u64,
    pub dst_q: *mut f32,
    pub dst_gate: *mut f32,
    pub src: *const f32,
    pub num_heads: u64,
    pub head_dim: u64,
    pub batch: u64,
    pub _pad: [u64; 12],
}
assert_inst_size!(DeinterleaveInst);
impl_inst!(DeinterleaveInst);

impl DeinterleaveInst {
    pub(crate) fn new(grid_x: u32, dst_q: *mut f32, dst_gate: *mut f32, src: *const f32, num_heads: i32, head_dim: i32, batch: i32) -> Self {
        DeinterleaveInst {
            opcode_gridx: make_opcode_gridx(OP_DEINTERLEAVE, grid_x),
            dst_q, dst_gate, src,
            num_heads: num_heads as u64,
            head_dim: head_dim as u64,
            batch: batch as u64,
            _pad: [0; 12],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_KV_QUANTIZE (opcode 21)
// words[1]=src(f32), [2]=q1_data(u8*), [3]=q1_scale(f32*), [4]=r_data(u8*),
//        [5]=r_scale(f32*), [6]=num_kv_heads, [7]=head_dim, [8]=chunk_tokens
// grid_x = nkh * head_dim
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct KvQuantizeInst {
    pub opcode_gridx: u64,
    pub src:          *const f32,
    pub q1_data:      *mut u8,
    pub q1_scale:     *mut f32,
    pub r_data:       *mut u8,
    pub r_scale:      *mut f32,
    pub num_kv_heads: i32,
    pub head_dim:     i32,
    pub chunk_tokens: i32,
    pub _pad0:        i32,
    pub _pad:         [u64; 11],
}
assert_inst_size!(KvQuantizeInst);
impl_inst!(KvQuantizeInst);

// ─────────────────────────────────────────────────────────────────────────────
// OP_ATTN_PAGED_Q (opcode 22)
// words[1]=scratch_ptr(patched), [2]=q, [3]=quant_page_table(patched),
//        [4]=pos_table(patched), [5]=inv_freq, [6]=nqh, [7]=nkh, [8]=hd,
//        [9]=quant_seq_len(patched), [10]=chunk_tokens, [11]=rd,
//        [12]=q1d, [13]=q1s, [14]=rd_off, [15]=rs, [16]=k_norm
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct AttnPagedQInst {
    pub opcode_gridx: u64,
    pub scratch: u64, // patched when quantized KV enabled
    pub q: *const f32,
    pub quant_page_table: u64, // patched per step
    pub pos_table: u64,        // patched per step
    pub inv_freq: *const f32,
    pub nqh: u64,
    pub nkh: u64,
    pub hd: u64,
    pub quant_seq_len: u64, // patched per step
    pub chunk_tokens: u64,
    pub rd: u64,
    pub q1d: u64,
    pub q1s: u64,
    pub rd_off: u64,
    pub rs: u64,
    pub k_norm: *const u16, // null if no QK-norm
    // bd srg6.2: head-slice (same encoding as AttnPagedInst.head_slice).
    pub head_slice: u64,
    pub _pad: u64,
}
assert_inst_size!(AttnPagedQInst);
impl_inst!(AttnPagedQInst);

impl AttnPagedQInst {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(q: *const f32, inv_freq: *const f32, nqh: i32, nkh: i32, hd: i32, chunk_tokens: i32, rd: i32, q1d: u64, q1s: u64, rd_off: u64, rs: u64, k_norm: *const u16, local_q_head_start: u16, local_nqh: u16) -> Self {
        let head_slice = (local_nqh as u64) | ((local_q_head_start as u64) << 16);
        AttnPagedQInst {
            opcode_gridx: make_opcode_gridx(OP_ATTN_PAGED_Q, 0),
            scratch: 0,
            q,
            quant_page_table: 0,
            pos_table: 0,
            inv_freq,
            nqh: nqh as u64,
            nkh: nkh as u64,
            hd: hd as u64,
            quant_seq_len: 0,
            chunk_tokens: chunk_tokens as u64,
            rd: rd as u64,
            q1d,
            q1s,
            rd_off,
            rs,
            k_norm,
            head_slice,
            _pad: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_MOE_GATE (opcode 23)
// words[1]=scores, [2]=expert_ids(w), [3]=expert_weights(w), [4]=ne, [5]=k,
//        [6]=gate_mode, [7]=rsf(f32 bits), [8]=bias_ptr
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct MoeGateInst {
    pub opcode_gridx: u64,
    pub scores: *const f32,
    pub expert_ids: *mut i32,     // output
    pub expert_weights: *mut f32, // output
    pub ne: u64,
    pub k: u64,
    pub gate_mode: u64,
    pub rsf_bits: u64, // routed_scaling_factor as f32 bits
    pub bias: *const u8,
    pub _pad: [u64; 10],
}
assert_inst_size!(MoeGateInst);
impl_inst!(MoeGateInst);

impl MoeGateInst {
    pub(crate) fn new(scores: *const f32, expert_ids: *mut i32, expert_weights: *mut f32, ne: i32, k: i32, gate_mode: u32, rsf: f32, bias: *const u8) -> Self {
        MoeGateInst {
            opcode_gridx: make_opcode_gridx(OP_MOE_GATE, 1),
            scores, expert_ids, expert_weights,
            ne: ne as u64,
            k: k as u64,
            gate_mode: gate_mode as u64,
            rsf_bits: rsf.to_bits() as u64,
            bias,
            _pad: [0; 10],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_MOE_FFN (opcode 24)
// words[1]=expert_ids, [2]=expert_weights, [3]=normed, [4]=ffn_down(w),
//        [5]=gate_up_data, [6]=gate_up_expert_stride(raw), [7]=down_data,
//        [8]=down_expert_stride(raw), [9]=k, [10]=hs|eis<<16, [11]=flags,
//        [12]=moe_expert_gate, [13]=moe_expert_up, [14]=moe_expert_act, [15]=moe_expert_out,
//        [16]=gate_up_row_stride(raw)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct MoeFfnInst {
    pub opcode_gridx: u64,
    pub expert_ids: *const i32,
    pub expert_weights: *const f32,
    pub normed: *const f32,
    pub ffn_down: *mut f32,
    pub gate_up_data: *const u8,
    pub gate_up_expert_stride: u64,
    pub down_data: *const u8,
    pub down_expert_stride: u64,
    pub k: u64,
    pub hs_eis: u64, // hs | (eis << 16)
    pub flags: u64,
    pub expert_gate: *const f32,
    pub expert_up: *const f32,
    pub expert_act: *const f32,
    pub expert_out: *const f32,
    pub gate_up_row_stride: u64,
    pub _pad: [u64; 2],
}
assert_inst_size!(MoeFfnInst);
impl_inst!(MoeFfnInst);

// ─────────────────────────────────────────────────────────────────────────────
// OP_SIGMOID_WEIGHTED_ADD (opcode 32)
// words[1]=output(w), [2]=scalar_ptr, [3]=input, [4]=n(i32)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct SigmoidWeightedAddInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub scalar: *const f32,
    pub input: *const f32,
    pub n: u64,
    pub _pad: [u64; 14],
}
assert_inst_size!(SigmoidWeightedAddInst);
impl_inst!(SigmoidWeightedAddInst);

impl SigmoidWeightedAddInst {
    pub(crate) fn new(grid_x: u32, output: *mut f32, scalar: *const f32, input: *const f32, n: i32) -> Self {
        SigmoidWeightedAddInst {
            opcode_gridx: make_opcode_gridx(OP_SIGMOID_WEIGHTED_ADD, grid_x),
            output, scalar, input,
            n: n as u64,
            _pad: [0; 14],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_DOT_SIGMOID_SCALE_ADD (opcode 46) — bd 9gmh: fused replacement for
// OP_LINEAR_PROJ(1×hs)+OP_SIGMOID_WEIGHTED_ADD which had cross-block L0 stale
// reads on intermediate scratch[0].
// Single-block: grid_x=1, vblock=0. Block-local LDS broadcast of sigmoid scale.
// words[1]=output(w), [2]=src, [3]=input(f32), [4]=weight(bf16/u16), [5]=size
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct DotSigmoidScaleAddInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub src: *const f32,
    pub input: *const f32,
    pub weight: *const u16,
    pub size: u64,
    pub _pad: [u64; 13],
}
assert_inst_size!(DotSigmoidScaleAddInst);
impl_inst!(DotSigmoidScaleAddInst);

impl DotSigmoidScaleAddInst {
    pub(crate) fn new(output: *mut f32, src: *const f32, input: *const f32, weight: *const u16, size: i32) -> Self {
        DotSigmoidScaleAddInst {
            opcode_gridx: make_opcode_gridx(OP_DOT_SIGMOID_SCALE_ADD, 1),
            output, src, input, weight,
            size: size as u64,
            _pad: [0; 13],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MoeDispatchInst — POST-ONLY CARRIER (post bd 0hu.3-b, commit b406640).
//
// OP_MOE_DISPATCH (opcode 34) is RETIRED — GPU 0's MoE PRE-compute now uses
// OP_MOE_FFN_REMOTE (MoeFfnRemoteInst) emitted by compile_inner_p2p via
// `build_ffn_remote_inst_gpu0`. The only remaining consumer of this layout
// is OP_MOE_DISPATCH_POST (opcode 45), which reads exactly four slots:
//   output_slots [word 2], final_output [3], num_workers_hs [7], num_gpus [15],
//   gate_up_in_dim [16].
// All other fields below (work_queue, expert_ids/weights, seq_counter,
// layer_k, eis_gate, activation, layer_config_ptrs, scratch_*, gpu0_acc)
// are VESTIGIAL — preserved only for layout compatibility with the
// kernel-side C struct in `kernels/megakernel_common.h`. Emission sites
// set them to zero or arbitrary values; POST never dereferences them.
//
// TODO: bd to rename to MoeDispatchPostInst and shrink to {opcode_gridx,
// _pad1, output_slots, final_output, _pad2[3], num_workers_hs, _pad3[7],
// num_gpus, gate_up_in_dim, _pad4} — requires coordinating C struct in
// megakernel_common.h.
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct MoeDispatchInst {
    pub opcode_gridx: u64,
    pub work_queue: u64,
    pub output_slots: u64,
    pub final_output: u64,
    pub expert_ids: u64,
    pub expert_weights: u64,
    pub seq_counter: u64,
    pub num_workers_hs: u64, // (num_workers << 32) | hidden_size
    pub layer_k: u64,        // (layer_idx << 32) | k
    pub eis_gate: u64,       // (eis << 32) | has_gate
    pub activation: u64,
    pub layer_config_ptrs: u64,
    pub scratch_gate: u64,
    pub scratch_up: u64,
    pub scratch_act: u64,
    pub num_gpus: u64,
    pub gate_up_in_dim: u64,
    /// Cached-local accumulator buffer (size = hs). op_moe_dispatch stages
    /// the per-expert accumulation here and does a final barrierless copy
    /// to UC `output_slots[0..gupd]`. See moe_p2p.rs `MoeP2pContext::gpu0_acc`.
    /// Unused by OP_MOE_DISPATCH_POST (it only reads output_slots).
    pub gpu0_acc: u64,
    pub _pad: u64,
}
assert_inst_size!(MoeDispatchInst);
impl_inst!(MoeDispatchInst);

// ─────────────────────────────────────────────────────────────────────────────
// OP_SCALE_ADD (opcode 36)
// output[i] += scale * src[i]; args: [1]=output, [2]=src, [3]=scale(f32 bits), [4]=size
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct ScaleAddInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub src: *const f32,
    pub scale_bits: u64,
    pub size: u64,
    pub _pad: [u64; 14],
}
assert_inst_size!(ScaleAddInst);
impl_inst!(ScaleAddInst);

impl ScaleAddInst {
    pub(crate) fn new(grid_x: u32, output: *mut f32, src: *const f32, scale: f32, size: i32) -> Self {
        let scale_bits = scale.to_bits() as u64;
        ScaleAddInst {
            opcode_gridx: make_opcode_gridx(OP_SCALE_ADD, grid_x),
            output, src, scale_bits,
            size: size as u64,
            _pad: [0; 14],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_RELU_SQ (opcode 37)
// output[i] = relu(input[i])^2; args: [1]=output, [2]=input, [3]=size
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct ReluSqInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub input: *const f32,
    pub size: u64,
    pub _pad: [u64; 15],
}
assert_inst_size!(ReluSqInst);
impl_inst!(ReluSqInst);

impl ReluSqInst {
    pub(crate) fn new(grid_x: u32, output: *mut f32, input: *const f32, size: i32) -> Self {
        ReluSqInst {
            opcode_gridx: make_opcode_gridx(OP_RELU_SQ, grid_x),
            output, input,
            size: size as u64,
            _pad: [0; 15],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_MAMBA2_CONV1D (opcode 38)
// args: [1]=state(w), [2]=input, [3]=weight(u16), [4]=bias(f32 ptr), [5]=output(w),
//       [6]=conv_dim(i32), [7]=kernel_size(i32)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct Mamba2Conv1dInst {
    pub opcode_gridx: u64,
    pub state: *mut f32,
    pub input: *const f32,
    pub weight: *const u16, // f16/bf16
    pub bias: *const f32,
    pub output: *mut f32,
    pub conv_dim: u64,
    pub kernel_size: u64,
    pub _pad: [u64; 11],
}
assert_inst_size!(Mamba2Conv1dInst);
impl_inst!(Mamba2Conv1dInst);

impl Mamba2Conv1dInst {
    pub(crate) fn new(grid_x: u32, state: *mut f32, input: *const f32, weight: *const u16, bias: *const f32, output: *mut f32, conv_dim: i32, kernel_size: i32) -> Self {
        Mamba2Conv1dInst {
            opcode_gridx: make_opcode_gridx(OP_MAMBA2_CONV1D, grid_x),
            state, input, weight, bias, output,
            conv_dim: conv_dim as u64,
            kernel_size: kernel_size as u64,
            _pad: [0; 11],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_MAMBA2_NORM_GATED (opcode 39)
// rms_norm(x*silu(z))*weight; args: [1]=output(w), [2]=x, [3]=z, [4]=weight,
//   [5]=num_heads, [6]=value_dim, [7]=eps
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct Mamba2NormGatedInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub x: *const f32,
    pub z: *const f32,
    pub weight: *const f32, // f32 (norm_weight loaded as f32)
    pub num_heads: u64,
    pub value_dim: u64,
    pub eps_bits: u64,
    pub _pad: [u64; 11],
}
assert_inst_size!(Mamba2NormGatedInst);
impl_inst!(Mamba2NormGatedInst);

impl Mamba2NormGatedInst {
    pub(crate) fn new(grid_x: u32, output: *mut f32, x: *const f32, z: *const f32, weight: *const f32, num_heads: i32, value_dim: i32, eps: f32) -> Self {
        Mamba2NormGatedInst {
            opcode_gridx: make_opcode_gridx(OP_MAMBA2_NORM_GATED, grid_x),
            output, x, z, weight,
            num_heads: num_heads as u64,
            value_dim: value_dim as u64,
            eps_bits: eps.to_bits() as u64,
            _pad: [0; 11],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_SSM_UPDATE (opcode 29)
// args: [1]=state(w), [2]=x(conv_out), [3]=dt, [4]=dt_bias, [5]=a_log,
//       [6]=B, [7]=C, [8]=d_weight, [9]=output(w), [10]=nh, [11]=hd, [12]=sd, [13]=ng
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct SsmUpdateInst {
    pub opcode_gridx: u64,
    pub state: *mut f32,
    pub x: *const f32,
    pub dt: *const f32,
    pub dt_bias: *const f32,
    pub a_log: *const f32,
    pub b: *const f32,
    pub c: *const f32,
    pub d_weight: *const f32,
    pub output: *mut f32,
    pub nh: u64,
    pub hd: u64,
    pub sd: u64,
    pub ng: u64,
    pub _pad: [u64; 5],
}
assert_inst_size!(SsmUpdateInst);
impl_inst!(SsmUpdateInst);

impl SsmUpdateInst {
    pub(crate) fn new(grid_x: u32, state: *mut f32, x: *const f32, dt: *const f32, dt_bias: *const f32, a_log: *const f32, b: *const f32, c: *const f32, d_weight: *const f32, output: *mut f32, nh: i32, hd: i32, sd: i32, ng: i32) -> Self {
        SsmUpdateInst {
            opcode_gridx: make_opcode_gridx(OP_SSM_UPDATE, grid_x),
            state, x, dt, dt_bias, a_log, b, c, d_weight, output,
            nh: nh as u64,
            hd: hd as u64,
            sd: sd as u64,
            ng: ng as u64,
            _pad: [0; 5],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_SILU_MUL (opcode 28)
// output[i] = silu(gate[i]) * up[i]; args: [1]=output, [2]=gate, [3]=up, [4]=size
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct SiluMulInst {
    pub opcode_gridx: u64,
    pub output: *mut f32,
    pub gate: *const f32,
    pub up: *const f32,
    pub size: u64,
    pub _pad: [u64; 14],
}
assert_inst_size!(SiluMulInst);
impl_inst!(SiluMulInst);

impl SiluMulInst {
    pub(crate) fn new(grid_x: u32, output: *mut f32, gate: *const f32, up: *const f32, size: i32) -> Self {
        SiluMulInst {
            opcode_gridx: make_opcode_gridx(OP_SILU_MUL, grid_x),
            output, gate, up,
            size: size as u64,
            _pad: [0; 14],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_BARRIER (internal IR, opcode 33)
// words[1]=barrier_flag_ptr, [2]=resume_flag_ptr, [3]=layer_idx
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct BarrierInst {
    pub opcode_gridx: u64,
    pub barrier_flag: *const u32,
    pub resume_flag: *const u32,
    pub layer_idx: u64,
    pub _pad: [u64; 15],
}
assert_inst_size!(BarrierInst);
impl_inst!(BarrierInst);

impl BarrierInst {
    pub(crate) fn new(layer_idx: i32) -> Self {
        BarrierInst {
            opcode_gridx: make_opcode_gridx(OP_BARRIER, 1),
            barrier_flag: std::ptr::null(),
            resume_flag: std::ptr::null(),
            layer_idx: layer_idx as u64,
            _pad: [0; 15],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_CONV1D_3X (opcode 40) — fused Q+K+V causal conv1d
// grid_x = 2*blocks_qk + blocks_v
// vb routing: < blocks_qk → Q, < 2*blocks_qk → K, else → V
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct Conv1d3xInst {
    pub opcode_gridx: u64,
    pub q_state:  *mut f32,
    pub q_input:  *const f32,
    pub q_weight: *const u16,
    pub q_output: *mut f32,
    pub k_state:  *mut f32,
    pub k_input:  *const f32,
    pub k_weight: *const u16,
    pub k_output: *mut f32,
    pub v_state:  *mut f32,
    pub v_input:  *const f32,
    pub v_weight: *const u16,
    pub v_output: *mut f32,
    pub qk_dim:     i64,
    pub v_dim:      i64,
    pub kernel_size: i64,
    pub blocks_qk_v: u64, // low32=blocks_qk, high32=blocks_v
    pub _pad: [u64; 2],
}
assert_inst_size!(Conv1d3xInst);
impl_inst!(Conv1d3xInst);

impl Conv1d3xInst {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        q_state: *mut f32, q_input: *const f32, q_weight: *const u16, q_output: *mut f32,
        k_state: *mut f32, k_input: *const f32, k_weight: *const u16, k_output: *mut f32,
        v_state: *mut f32, v_input: *const f32, v_weight: *const u16, v_output: *mut f32,
        qk_dim: i32, v_dim: i32, kernel_size: i32,
    ) -> Self {
        let blocks_qk = (qk_dim as u32).div_ceil(256);
        let blocks_v  = (v_dim as u32).div_ceil(256);
        let grid_x = 2 * blocks_qk + blocks_v;
        Conv1d3xInst {
            opcode_gridx: make_opcode_gridx(OP_CONV1D_3X, grid_x),
            q_state, q_input, q_weight, q_output,
            k_state, k_input, k_weight, k_output,
            v_state, v_input, v_weight, v_output,
            qk_dim: qk_dim as i64,
            v_dim: v_dim as i64,
            kernel_size: kernel_size as i64,
            blocks_qk_v: (blocks_qk as u64) | ((blocks_v as u64) << 32),
            _pad: [0; 2],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_LINEAR_PROJ_2X (opcode 43) — bf16 fused two linear projections sharing input.
// grid_x = 2 * out_dim. vb < out_dim → A; vb < 2*out_dim → B.
// Used for GDN w_a + w_b (always bf16; see bqnt_quantize.rs SKIP_PATTERNS).
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct LinearProj2xInst {
    pub opcode_gridx: u64,
    pub output_a: *mut f32,
    pub output_b: *mut f32,
    pub weight_a: *const u16,
    pub weight_b: *const u16,
    pub input: *const f32,
    pub out_dim: i64, // same for A and B
    pub in_dim: i64,
    pub batch: i64, // 0 or 1 → single token
    pub _pad: [u64; 10],
}
assert_inst_size!(LinearProj2xInst);
impl_inst!(LinearProj2xInst);

impl LinearProj2xInst {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        output_a: *mut f32,
        output_b: *mut f32,
        weight_a: *const u16,
        weight_b: *const u16,
        input: *const f32,
        out_dim: i32,
        in_dim: i32,
        batch: i32,
    ) -> Self {
        let grid_x = 2 * (out_dim as u32);
        LinearProj2xInst {
            opcode_gridx: make_opcode_gridx(OP_LINEAR_PROJ_2X, grid_x),
            output_a,
            output_b,
            weight_a,
            weight_b,
            input,
            out_dim: out_dim as i64,
            in_dim: in_dim as i64,
            batch: batch as i64,
            _pad: [0; 10],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OP_MOE_FFN_REMOTE (opcode 44)
// Cross-GPU MoE expert compute on a worker GPU's persistent_worker. Reads
// activation from GPU 0 VRAM (P2P), runs experts that are local to this GPU
// (per-eid lookup in MoeWorkerConfig.entries), accumulates weighted outputs
// in worker VRAM, P2P-writes to this worker's slot in GPU 0's output_slots.
// Layout must match MoeFfnRemoteInst in kernels/megakernel_common.h.
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub(crate) struct MoeFfnRemoteInst {
    pub opcode_gridx: u64,
    pub activation_p2p: *const f32,
    pub output_slot_p2p: *mut f32,
    pub expert_ids: *const i32,
    pub expert_weights: *const f32,
    // bd 0hu3-b: per-context array of MoeWorkerConfig*, indexed by layer_idx
    // INSIDE the kernel. Each consumer (GPU 0's compiled megakernel, each
    // worker's mailbox-dispatch) sees its OWN VA pointing at its OWN array
    // (whose entries are also valid VAs in that consumer's context).
    // Mirrors the OP_MOE_DISPATCH layer_config_ptrs-as-array pattern. Replaces
    // the prior `config: *const c_void` which baked a single producer-side VA
    // into the shared instruction bytes (faulted when consumed cross-context).
    pub config_array: *const *const std::ffi::c_void,
    pub local_activation: *mut f32,
    pub local_output: *mut f32,
    pub scratch_gate: *mut f32,
    pub scratch_up: *mut f32,
    pub scratch_act: *mut f32,
    pub k_eis: u64,    // low 32: k, high 32: eis
    pub hs_gupd: u64,  // low 32: hs, high 32: gupd
    pub flags: u64,    // bit0=has_gate, bit1=relu_sq
    // bd el1f Phase A: acquire-side of Step 1 → Step 2.5 drain barrier.
    // Worker acquire-spins on *wait_ptr == wait_seq before reading
    // activation_p2p. Pair with Step 1's D2dCopyInst::with_signal(sentinel, seq).
    // wait_ptr=null disables the acquire (used for non-prefill dispatches).
    pub wait_ptr: *const u32,
    pub wait_seq: u64,
    // bd 0hu3-b: layer index into config_array (above). Resolved inside the
    // kernel via config_array[layer_idx] so each consumer dereferences a VA
    // that is valid in its own context.
    pub layer_idx: u64,
    pub _pad: [u64; 2],
}
assert_inst_size!(MoeFfnRemoteInst);
impl_inst!(MoeFfnRemoteInst);

impl MoeFfnRemoteInst {
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // wired in Phase 3 (decode + prefill MoE caller migration)
    pub(crate) fn new(
        grid_x: u32,
        activation_p2p: *const f32,
        output_slot_p2p: *mut f32,
        expert_ids: *const i32,
        expert_weights: *const f32,
        config_array: *const *const std::ffi::c_void,
        layer_idx: u64,
        local_activation: *mut f32,
        local_output: *mut f32,
        scratch_gate: *mut f32,
        scratch_up: *mut f32,
        scratch_act: *mut f32,
        k: u32,
        eis: u32,
        hs: u32,
        gupd: u32,
        has_gate_proj: bool,
        relu_sq: bool,
    ) -> Self {
        let flags = (has_gate_proj as u64) | ((relu_sq as u64) << 1);
        MoeFfnRemoteInst {
            opcode_gridx: make_opcode_gridx(OP_MOE_FFN_REMOTE, grid_x),
            activation_p2p,
            output_slot_p2p,
            expert_ids,
            expert_weights,
            config_array,
            local_activation,
            local_output,
            scratch_gate,
            scratch_up,
            scratch_act,
            k_eis: (k as u64) | ((eis as u64) << 32),
            hs_gupd: (hs as u64) | ((gupd as u64) << 32),
            flags,
            wait_ptr: std::ptr::null(),
            wait_seq: 0,
            layer_idx,
            _pad: [0; 2],
        }
    }

    /// bd el1f: attach acquire-load on the Step 1 drain sentinel.
    /// `seq` should be `layer_idx + 1` to match the producer's monotonic
    /// `with_signal((layer_idx as u64) + 1)` on the Step 1 D2D-copy.
    #[allow(dead_code)]
    pub(crate) fn with_wait(mut self, ptr: *const u32, seq: u64) -> Self {
        self.wait_ptr = ptr;
        self.wait_seq = seq;
        self
    }
}

