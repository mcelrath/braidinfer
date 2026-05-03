	.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
	.amdhsa_code_object_version 6
	.text
	.protected	repro_kernel            ; -- Begin function repro_kernel
	.globl	repro_kernel
	.p2align	8
	.type	repro_kernel,@function
repro_kernel:                           ; @repro_kernel
; %bb.0:
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x10
	v_lshlrev_b32_e32 v1, 2, v0
	v_mbcnt_lo_u32_b32 v4, -1, 0
	s_waitcnt lgkmcnt(0)
	s_clause 0x2
	global_load_b32 v2, v1, s[4:5]
	global_load_b32 v3, v1, s[6:7]
	global_load_b32 v1, v1, s[2:3]
	v_mbcnt_hi_u32_b32 v4, -1, v4
	s_mov_b64 s[2:3], exec
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v5, v4, 2, 0x80
	s_waitcnt vmcnt(1)
	v_mul_f32_e32 v6, v2, v3
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v7, v1, v1
	ds_bpermute_b32 v6, v5, v6
	ds_bpermute_b32 v5, v5, v7
	v_and_b32_e32 v7, 63, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cmp_gt_u32_e32 vcc, 48, v7
	v_cndmask_b32_e64 v8, 0, 16, vcc
	v_cmp_gt_u32_e32 vcc, 56, v7
	v_add_lshl_u32 v8, v8, v4, 2
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v6, v2, v3
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v5, v1, v1
	v_cndmask_b32_e64 v3, 0, 8, vcc
	v_cmp_gt_u32_e32 vcc, 60, v7
	ds_bpermute_b32 v1, v8, v6
	ds_bpermute_b32 v2, v8, v5
	v_add_lshl_u32 v3, v3, v4, 2
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v1, v6, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v5, v2
	v_cndmask_b32_e64 v6, 0, 4, vcc
	v_cmp_gt_u32_e32 vcc, 62, v7
	ds_bpermute_b32 v5, v3, v1
	ds_bpermute_b32 v3, v3, v2
	v_add_lshl_u32 v6, v6, v4, 2
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v1, v1, v5
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v3
	ds_bpermute_b32 v3, v6, v1
	ds_bpermute_b32 v5, v6, v2
	v_cndmask_b32_e64 v6, 0, 2, vcc
	v_cmp_ne_u32_e32 vcc, 63, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v6, v6, v4, 2
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc
	v_lshlrev_b32_e32 v4, 2, v4
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v1, v1, v3
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v5
	ds_bpermute_b32 v3, v6, v1
	ds_bpermute_b32 v5, v6, v2
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v1, v1, v3
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v3, v2, v5
	v_and_b32_e32 v5, 63, v0
	ds_bpermute_b32 v2, v4, v1
	ds_bpermute_b32 v4, v4, v3
	v_cmpx_eq_u32_e32 0, v5
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_b32 s2, s[0:1], 0x24
	v_lshrrev_b32_e32 v0, 5, v0
	s_load_b64 s[0:1], s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v3, v3, v4
	v_add_f32_e32 v2, v1, v2
	v_lshl_or_b32 v5, s2, 3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc, s0, v5
	v_add_co_ci_u32_e64 v5, null, s1, v6, vcc
	global_store_b64 v[4:5], v[2:3], off
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel repro_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 9
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size 4
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	repro_kernel, .Lfunc_end0-repro_kernel
                                        ; -- End function
	.set repro_kernel.num_vgpr, 9
	.set repro_kernel.num_agpr, 0
	.set repro_kernel.numbered_sgpr, 8
	.set repro_kernel.num_named_barrier, 0
	.set repro_kernel.private_seg_size, 0
	.set repro_kernel.uses_vcc, 1
	.set repro_kernel.uses_flat_scratch, 0
	.set repro_kernel.has_dyn_sized_stack, 0
	.set repro_kernel.has_recursion, 0
	.set repro_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 492
; TotalNumSgprs: 10
; NumVgprs: 9
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 9
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_92567b89c091ec70,@object ; @__hip_cuid_92567b89c091ec70
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_92567b89c091ec70
__hip_cuid_92567b89c091ec70:
	.byte	0                               ; 0x0
	.size	__hip_cuid_92567b89c091ec70, 1

	.ident	"AMD clang version 22.0.0git (/srcdest/rocm-llvm f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_92567b89c091ec70
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           repro_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         repro_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     9
    .vgpr_spill_count: 0
    .wavefront_size: 64
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
