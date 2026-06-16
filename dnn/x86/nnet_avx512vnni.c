/* Copyright (c) 2018-2019 Mozilla
                 2023 Amazon */
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
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "x86/x86_arch_macros.h"

/* This file is compiled with AVX512-VNNI (+ AVX512VL) enabled so that
   vec_avx.h emits the 256-bit EVEX-encoded vpdpbusd instruction for the int8
   GEMV (compute_linear with quantized weights) instead of the AVX2
   multiply-add emulation. We deliberately keep everything 256-bit wide
   (-mprefer-vector-width=256): the win comes from the single-instruction int8
   dot product, not from wider vectors. MSVC does not expose a feature macro
   for this, so define it here when the build has detected support. */
#if defined(_MSC_VER) && defined(OPUS_X86_MAY_HAVE_AVX512VNNI) && !defined(__AVX512VNNI__)
#define __AVX512VNNI__
#endif

#ifndef __AVX512VNNI__
#error nnet_avx512vnni.c is being compiled without AVX512-VNNI enabled
#endif

#define RTCD_ARCH avx512vnni

#include "nnet_arch.h"
