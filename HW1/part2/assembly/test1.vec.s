	.text
	.file	"test1.cpp"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               # -- Begin function _Z5test1PfS_S_i
.LCPI0_0:
	.quad	4472406533629990549     # double 1.0000000000000001E-9
	.text
	.globl	_Z5test1PfS_S_i
	.p2align	4, 0x90
	.type	_Z5test1PfS_S_i,@function
_Z5test1PfS_S_i:                        # @_Z5test1PfS_S_i
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	subq	$32, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %rbx
	leaq	8(%rsp), %rsi
	movl	$1, %edi
	callq	clock_gettime
	testl	%eax, %eax
	jne	.LBB0_8
# %bb.1:
	leaq	4096(%r14), %rax
	leaq	4096(%rbx), %rcx
	cmpq	%r14, %rcx
	seta	%cl
	leaq	4096(%r15), %rsi
	cmpq	%rbx, %rax
	seta	%dl
	andb	%cl, %dl
	cmpq	%r14, %rsi
	seta	%cl
	movq	8(%rsp), %r13
	cmpq	%r15, %rax
	seta	%al
	movq	16(%rsp), %r12
	andb	%cl, %al
	orb	%dl, %al
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #     Child Loop BB0_7 Depth 2
	movq	$-1024, %rdx            # imm = 0xFC00
	testb	%al, %al
	je	.LBB0_3
	.p2align	4, 0x90
.LBB0_7:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	4096(%rbx,%rdx,4), %xmm0 # xmm0 = mem[0],zero,zero,zero
	addss	4096(%r15,%rdx,4), %xmm0
	movss	%xmm0, 4096(%r14,%rdx,4)
	movss	4100(%rbx,%rdx,4), %xmm0 # xmm0 = mem[0],zero,zero,zero
	addss	4100(%r15,%rdx,4), %xmm0
	movss	%xmm0, 4100(%r14,%rdx,4)
	movss	4104(%rbx,%rdx,4), %xmm0 # xmm0 = mem[0],zero,zero,zero
	addss	4104(%r15,%rdx,4), %xmm0
	movss	%xmm0, 4104(%r14,%rdx,4)
	movss	4108(%rbx,%rdx,4), %xmm0 # xmm0 = mem[0],zero,zero,zero
	addss	4108(%r15,%rdx,4), %xmm0
	movss	%xmm0, 4108(%r14,%rdx,4)
	addq	$4, %rdx
	jne	.LBB0_7
	jmp	.LBB0_4
	.p2align	4, 0x90
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movups	4096(%rbx,%rdx,4), %xmm0
	movups	4112(%rbx,%rdx,4), %xmm1
	movups	4096(%r15,%rdx,4), %xmm2
	addps	%xmm0, %xmm2
	movups	4112(%r15,%rdx,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, 4096(%r14,%rdx,4)
	movups	%xmm0, 4112(%r14,%rdx,4)
	movups	4128(%rbx,%rdx,4), %xmm0
	movups	4144(%rbx,%rdx,4), %xmm1
	movups	4128(%r15,%rdx,4), %xmm2
	addps	%xmm0, %xmm2
	movups	4144(%r15,%rdx,4), %xmm0
	movups	%xmm2, 4128(%r14,%rdx,4)
	addps	%xmm1, %xmm0
	movups	%xmm0, 4144(%r14,%rdx,4)
	addq	$16, %rdx
	jne	.LBB0_3
.LBB0_4:                                #   in Loop: Header=BB0_2 Depth=1
	addl	$1, %ecx
	cmpl	$20000000, %ecx         # imm = 0x1312D00
	jne	.LBB0_2
# %bb.5:
	leaq	8(%rsp), %rsi
	movl	$1, %edi
	callq	clock_gettime
	testl	%eax, %eax
	jne	.LBB0_8
# %bb.6:
	movq	8(%rsp), %rax
	subq	%r13, %rax
	movq	16(%rsp), %rcx
	subq	%r12, %rcx
	xorps	%xmm0, %xmm0
	cvtsi2sdq	%rax, %xmm0
	xorps	%xmm1, %xmm1
	cvtsi2sdq	%rcx, %xmm1
	mulsd	.LCPI0_0(%rip), %xmm1
	addsd	%xmm0, %xmm1
	movsd	%xmm1, 24(%rsp)         # 8-byte Spill
	movl	$_ZSt4cout, %edi
	movl	$.L.str, %esi
	movl	$47, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movsd	24(%rsp), %xmm0         # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rbx
	movl	$.L.str.1, %esi
	movl	$8, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$1024, %esi             # imm = 0x400
	movq	%rbx, %rdi
	callq	_ZNSolsEi
	movq	%rax, %rbx
	movl	$.L.str.2, %esi
	movl	$5, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$20000000, %esi         # imm = 0x1312D00
	movq	%rbx, %rdi
	callq	_ZNSolsEi
	movl	$.L.str.3, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	addq	$32, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	retq
.LBB0_8:
	movl	$.L.str.4, %edi
	movl	$.L.str.5, %esi
	movl	$75, %edx
	movl	$.L__PRETTY_FUNCTION__._ZL7gettimev, %ecx
	callq	__assert_fail
.Lfunc_end0:
	.size	_Z5test1PfS_S_i, .Lfunc_end0-_Z5test1PfS_S_i
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90         # -- Begin function _GLOBAL__sub_I_test1.cpp
	.type	_GLOBAL__sub_I_test1.cpp,@function
_GLOBAL__sub_I_test1.cpp:               # @_GLOBAL__sub_I_test1.cpp
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	$_ZStL8__ioinit, %edi
	callq	_ZNSt8ios_base4InitC1Ev
	movl	$_ZNSt8ios_base4InitD1Ev, %edi
	movl	$_ZStL8__ioinit, %esi
	movl	$__dso_handle, %edx
	popq	%rax
	jmp	__cxa_atexit            # TAILCALL
.Lfunc_end1:
	.size	_GLOBAL__sub_I_test1.cpp, .Lfunc_end1-_GLOBAL__sub_I_test1.cpp
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object  # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	.L.str,@object          # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Elapsed execution time of the loop in test1():\n"
	.size	.L.str, 48

	.type	.L.str.1,@object        # @.str.1
.L.str.1:
	.asciz	"sec (N: "
	.size	.L.str.1, 9

	.type	.L.str.2,@object        # @.str.2
.L.str.2:
	.asciz	", I: "
	.size	.L.str.2, 6

	.type	.L.str.3,@object        # @.str.3
.L.str.3:
	.asciz	")\n"
	.size	.L.str.3, 3

	.type	.L.str.4,@object        # @.str.4
.L.str.4:
	.asciz	"r == 0"
	.size	.L.str.4, 7

	.type	.L.str.5,@object        # @.str.5
.L.str.5:
	.asciz	"./fasttime.h"
	.size	.L.str.5, 13

	.type	.L__PRETTY_FUNCTION__._ZL7gettimev,@object # @__PRETTY_FUNCTION__._ZL7gettimev
.L__PRETTY_FUNCTION__._ZL7gettimev:
	.asciz	"fasttime_t gettime()"
	.size	.L__PRETTY_FUNCTION__._ZL7gettimev, 21

	.section	.init_array,"aw",@init_array
	.p2align	3
	.quad	_GLOBAL__sub_I_test1.cpp

	.ident	"clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)"
	.section	".note.GNU-stack","",@progbits
