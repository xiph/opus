/* Copyright (c) 2026 Lynne */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 * Shared AArch64 GAS macros for libopus NEON assembly, #included into the
 * .S files (the C preprocessor runs on .S, so __APPLE__/__ELF__ are visible).
 * A minimal subset of FFmpeg's libavutil/aarch64/asm.S covering both ELF
 * (Linux/Android) and Mach-O (Apple): function/endfunc/const/endconst/movrel,
 * with leading-underscore symbol prefixing on Apple. Windows/BTI/PAC omitted.
 */

#ifndef CELT_ARM_ASM_H
#define CELT_ARM_ASM_H

#ifndef __APPLE__
        .arch   armv8-a
#endif

/* EXTERN_ASM is the C-symbol prefix: a leading underscore on Apple, nothing
   on ELF. X(sym) applies it, for referencing exported symbols (bl X(name)). */
#if defined(__APPLE__)
#  define EXTERN_ASM _
#else
#  define EXTERN_ASM
#endif
#define OPUS_GLUE(a, b) a ## b
#define OPUS_JOIN(a, b) OPUS_GLUE(a, b)
#define X(s) OPUS_JOIN(EXTERN_ASM, s)

        .macro  function name, export=1, align=4
        .text
        .align  \align
        .if \export
        .global EXTERN_ASM\name
#ifdef __ELF__
        .type   EXTERN_ASM\name, %function
#endif
EXTERN_ASM\name\():
        .else
\name\():
        .endif
        .endm

        .macro  endfunc name
#ifdef __ELF__
        .size   EXTERN_ASM\name, . - EXTERN_ASM\name
#endif
        .endm

        .macro  const name, align=4
#ifdef __MACH__
        .const_data
#else
        .section .rodata
#endif
        .align  \align
\name\():
        .endm

        .macro  endconst name
#ifdef __ELF__
        .size   \name, . - \name
#endif
        .text
        .endm

/* movrel rd, sym[, off] : address of a local symbol (ELF :lo12: / Apple @PAGE). */
        .macro  movrel rd, val, offset=0
#if defined(__APPLE__)
        adrp    \rd, \val+(\offset)@PAGE
        add     \rd, \rd, \val+(\offset)@PAGEOFF
#else
        adrp    \rd, \val+(\offset)
        add     \rd, \rd, :lo12:\val+(\offset)
#endif
        .endm

/* Pointer-authentication hints for functions that spill/reload lr, as in
   FFmpeg's asm.S; no-ops unless the build enables branch protection. */
        .macro  AARCH64_SIGN_LINK_REGISTER
#if defined(__ARM_FEATURE_PAC_DEFAULT)
#if (__ARM_FEATURE_PAC_DEFAULT & 1)
        paciasp
#elif (__ARM_FEATURE_PAC_DEFAULT & 2)
        pacibsp
#endif
#endif
        .endm

        .macro  AARCH64_VALIDATE_LINK_REGISTER
#if defined(__ARM_FEATURE_PAC_DEFAULT)
#if (__ARM_FEATURE_PAC_DEFAULT & 1)
        autiasp
#elif (__ARM_FEATURE_PAC_DEFAULT & 2)
        autibsp
#endif
#endif
        .endm

#endif /* CELT_ARM_ASM_H */
