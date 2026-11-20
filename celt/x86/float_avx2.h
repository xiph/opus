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

/* Common definitions for the float AVX2 kernels in this directory.

   These kernels are bit-exact with the C code they replace: every output
   value goes through the same IEEE single-precision operations, in the same
   order, as in the scalar code. The only rewrites used are exact ones:
   commuting the operands of a single add or mul, and x+(-y) for x-y.

   The kernels are built with -mfma, so the compiler must be prevented from
   contracting a mul followed by an add into an FMA (which skips the
   intermediate rounding). Include this header before any code. */

#ifndef FLOAT_AVX2_H
#define FLOAT_AVX2_H

#if defined(__clang__)
# pragma clang fp contract(off)
#elif defined(__GNUC__)
# pragma GCC optimize ("fp-contract=off")
#elif defined(_MSC_VER)
# pragma fp_contract (off)
#endif

#include <immintrin.h>
#include "kiss_fft.h"

/* Complex values are stored interleaved as r,i,r,i,... so a __m256 holds four
   of them. */

/* (r,i) -> (i,r) for each complex. */
#define CPX_SWAP_AVX2(x) _mm256_permute_ps((x), 0xB1)

/* (a.r*b - a.i*c, a.i*b + a.r*c), where b and c hold one value per complex
   (duplicated over its r and i lanes). With b = w.r and c = w.i, this is
   C_MUL(a, w) from _kiss_fft_guts.h, with the operands of the final add
   commuted. */
static OPUS_INLINE __m256 cpx_rot_avx2(__m256 a, __m256 b, __m256 c)
{
   return _mm256_addsub_ps(_mm256_mul_ps(a, b), _mm256_mul_ps(CPX_SWAP_AVX2(a), c));
}

/* Loads the complex values p[0], p[s], p[2*s], p[3*s] (p points to
   interleaved r,i floats, s counts complex values). */
static OPUS_INLINE __m256 cpx_gather4_avx2(const float *p, int s)
{
   __m128 lo, hi;
   if (s == 1)
      return _mm256_loadu_ps(p);
   lo = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(const void*)p);
   lo = _mm_loadh_pi(lo, (const __m64*)(const void*)(p + 2*s));
   hi = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(const void*)(p + 4*s));
   hi = _mm_loadh_pi(hi, (const __m64*)(const void*)(p + 6*s));
   return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}

/* Stores the four complex values of x at y[idx[0]], ..., y[idx[3]]. */
static OPUS_INLINE void cpx_scatter4_avx2(kiss_fft_cpx *y, const opus_int16 *idx, __m256 x)
{
   __m128 lo = _mm256_castps256_ps128(x);
   __m128 hi = _mm256_extractf128_ps(x, 1);
   _mm_storel_pi((__m64*)(void*)&y[idx[0]], lo);
   _mm_storeh_pi((__m64*)(void*)&y[idx[1]], lo);
   _mm_storel_pi((__m64*)(void*)&y[idx[2]], hi);
   _mm_storeh_pi((__m64*)(void*)&y[idx[3]], hi);
}

#endif
