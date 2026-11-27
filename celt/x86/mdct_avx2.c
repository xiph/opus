/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2008 Xiph.Org Foundation
   Copyright (c) 2026,     Advanced Micro Devices, Inc.
   Written by Jean-Marc Valin */
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

/* AVX2/FMA forward MDCT (float path).
 *
 * The window/fold stage is left scalar; the pre- and post-rotation are
 * vectorised four complex pairs at a time with _mm256_fmaddsub_ps, performing
 * the same arithmetic as the scalar reference in celt/mdct.c.
 *
 * Twiddle layout: interleaved complex pairs, trig[2*i] = -sin,
 * trig[2*i+1] = cos.  The FMA pre-/post-rotation loads are deinterleaved
 * with _mm_shuffle_ps; the fmaddsub sign pattern then produces the correct
 * rotated values directly.
 *
 * The N/4 complex FFT in the middle is the shared opus_fft_impl(), which
 * itself dispatches to the AVX2 butterflies.  The result is not bit-identical
 * to the scalar C path, matching the other float SIMD paths.
 *
 * Built at the -mavx2/-mfma baseline through CELT_SOURCES_AVX2 and entered
 * only when opus_select_arch() reports AVX2 (arch >= 4, via the
 * CLT_MDCT_FORWARD_IMPL dispatch table); other CPUs take the scalar
 * clt_mdct_forward path. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "arch.h"

#if defined(OPUS_X86_MAY_HAVE_AVX2) && !defined(FIXED_POINT)

#include <immintrin.h>

#include "mdct.h"
#include "kiss_fft.h"
#include "_kiss_fft_guts.h"
#include "os_support.h"
#include "mathops.h"
#include "stack_alloc.h"

void clt_mdct_forward_avx2(const mdct_lookup *l, kiss_fft_scalar *in,
      kiss_fft_scalar * OPUS_RESTRICT out,
      const celt_coef *window, int overlap, int shift, int stride, int arch)
{
   int i;
   int N, N2, N4;
   VARDECL(kiss_fft_scalar, f);
   VARDECL(kiss_fft_cpx, f2);
   const kiss_fft_state *st = l->kfft[shift];
   const kiss_twiddle_scalar *trig;
   opus_val16 scale;
   SAVE_STACK;
   scale = st->scale;

   N = l->n;
   trig = l->trig;
   for (i=0;i<shift;i++)
   {
      N >>= 1;
      trig += N;
   }
   N2 = N>>1;
   N4 = N>>2;

   ALLOC(f, N2, kiss_fft_scalar);
   ALLOC(f2, N4, kiss_fft_cpx);

   /* Window, shuffle, fold (scalar, as in the reference). */
   {
      const kiss_fft_scalar * OPUS_RESTRICT xp1 = in+(overlap>>1);
      const kiss_fft_scalar * OPUS_RESTRICT xp2 = in+N2-1+(overlap>>1);
      kiss_fft_scalar * OPUS_RESTRICT yp = f;
      const opus_val16 * OPUS_RESTRICT wp1 = window+(overlap>>1);
      const opus_val16 * OPUS_RESTRICT wp2 = window+(overlap>>1)-1;
      for(i=0;i<((overlap+3)>>2);i++)
      {
         *yp++ = MULT16_32_Q15(*wp2, xp1[N2]) + MULT16_32_Q15(*wp1,*xp2);
         *yp++ = MULT16_32_Q15(*wp1, *xp1)    - MULT16_32_Q15(*wp2, xp2[-N2]);
         xp1+=2;
         xp2-=2;
         wp1+=2;
         wp2-=2;
      }
      wp1 = window;
      wp2 = window+overlap-1;
      for(;i<N4-((overlap+3)>>2);i++)
      {
         *yp++ = *xp2;
         *yp++ = *xp1;
         xp1+=2;
         xp2-=2;
      }
      for(;i<N4;i++)
      {
         *yp++ =  -MULT16_32_Q15(*wp1, xp1[-N2]) + MULT16_32_Q15(*wp2, *xp2);
         *yp++ = MULT16_32_Q15(*wp2, *xp1)     + MULT16_32_Q15(*wp1, xp2[N2]);
         xp1+=2;
         xp2-=2;
         wp1+=2;
         wp2-=2;
      }
   }

   /* Pre-rotation, four complex pairs per iteration with FMA.
      tc = cos (from trig[2*i+1]), ts = -sin (from trig[2*i]).
      Even (real): data*tc - data_swap*ts = re*cos + im*sin.
      Odd  (imag): data*tc + data_swap*ts = im*cos - re*sin. */
   {
      kiss_fft_scalar * OPUS_RESTRICT yp = f;
      const kiss_twiddle_scalar *t = &trig[0];
#ifndef ENABLE_QEXT
      const __m256 scale_v = _mm256_set1_ps(scale);
#endif
      int N4_vec = N4 & ~3;

      for(i=0;i<N4_vec;i+=4)
      {
         __m256 data = _mm256_loadu_ps(&yp[2*i]);

         __m128 lo = _mm_loadu_ps(&t[2*i]);
         __m128 hi = _mm_loadu_ps(&t[2*i+4]);
         __m128 tc4 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3,1,3,1));
         __m128 ts4 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2,0,2,0));

         __m256 tc = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_unpacklo_ps(tc4, tc4)),
            _mm_unpackhi_ps(tc4, tc4), 1);
         __m256 ts = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_unpacklo_ps(ts4, ts4)),
            _mm_unpackhi_ps(ts4, ts4), 1);

         __m256 data_swap = _mm256_shuffle_ps(data, data, _MM_SHUFFLE(2,3,0,1));

         __m256 mul_ds = _mm256_mul_ps(data_swap, ts);
         __m256 result = _mm256_fmaddsub_ps(data, tc, mul_ds);

         float tmp[8];
         /* For QEXT the scaling is folded into the post-rotation twiddles;
            otherwise it is applied here.  For float the two are equivalent. */
#ifndef ENABLE_QEXT
         result = _mm256_mul_ps(result, scale_v);
#endif
         _mm256_storeu_ps(tmp, result);
         f2[st->bitrev[i  ]].r = tmp[0]; f2[st->bitrev[i  ]].i = tmp[1];
         f2[st->bitrev[i+1]].r = tmp[2]; f2[st->bitrev[i+1]].i = tmp[3];
         f2[st->bitrev[i+2]].r = tmp[4]; f2[st->bitrev[i+2]].i = tmp[5];
         f2[st->bitrev[i+3]].r = tmp[6]; f2[st->bitrev[i+3]].i = tmp[7];
      }

      for(;i<N4;i++)
      {
         kiss_fft_cpx yc;
         kiss_fft_scalar re, im;
         re = yp[2*i];
         im = yp[2*i+1];
#ifdef ENABLE_QEXT
         yc.r = re*t[2*i+1] - im*t[2*i];
         yc.i = im*t[2*i+1] + re*t[2*i];
#else
         yc.r = (re*t[2*i+1] - im*t[2*i]) * scale;
         yc.i = (im*t[2*i+1] + re*t[2*i]) * scale;
#endif
         f2[st->bitrev[i]] = yc;
      }
   }

   opus_fft_impl(st, f2, arch);

   /* Post-rotation, four complex pairs per iteration with FMA.
      tc = cos (from trig[2*i+1]), ts = -sin (from trig[2*i]).
      yr = fp.i*ts - fp.r*tc ; yi = fp.r*ts + fp.i*tc. */
   {
      const kiss_fft_cpx * OPUS_RESTRICT fp = f2;
      kiss_fft_scalar * OPUS_RESTRICT yp1 = out;
      kiss_fft_scalar * OPUS_RESTRICT yp2 = out+stride*(N2-1);
      const kiss_twiddle_scalar *t = &trig[0];
      const int stride2 = 2*stride;
      int N4_vec = N4 & ~3;

      for(i=0;i<N4_vec;i+=4)
      {
         __m256 data = _mm256_loadu_ps((const float*)&fp[i]);

         __m128 lo = _mm_loadu_ps(&t[2*i]);
         __m128 hi = _mm_loadu_ps(&t[2*i+4]);
         __m128 tc4 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3,1,3,1));
         __m128 ts4 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2,0,2,0));
#ifdef ENABLE_QEXT
         tc4 = _mm_mul_ps(tc4, _mm_set1_ps(scale));
         ts4 = _mm_mul_ps(ts4, _mm_set1_ps(scale));
#endif

         __m256 tc = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_unpacklo_ps(tc4, tc4)),
            _mm_unpackhi_ps(tc4, tc4), 1);
         __m256 ts = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_unpacklo_ps(ts4, ts4)),
            _mm_unpackhi_ps(ts4, ts4), 1);

         __m256 data_swap = _mm256_shuffle_ps(data, data, _MM_SHUFFLE(2,3,0,1));

         __m256 mul_dt = _mm256_mul_ps(data, tc);
         __m256 result = _mm256_fmaddsub_ps(data_swap, ts, mul_dt);

         float tmp[8];
         _mm256_storeu_ps(tmp, result);
         yp1[0]           = tmp[0]; yp2[0]           = tmp[1];
         yp1[stride2]     = tmp[2]; yp2[-stride2]    = tmp[3];
         yp1[2*stride2]   = tmp[4]; yp2[-2*stride2]  = tmp[5];
         yp1[3*stride2]   = tmp[6]; yp2[-3*stride2]  = tmp[7];

         yp1 += 4*stride2;
         yp2 -= 4*stride2;
      }

      for(;i<N4;i++)
      {
         kiss_fft_scalar yr, yi;
#ifdef ENABLE_QEXT
         yr = fp[i].i*(t[2*i]*scale) - fp[i].r*(t[2*i+1]*scale);
         yi = fp[i].r*(t[2*i]*scale) + fp[i].i*(t[2*i+1]*scale);
#else
         yr = fp[i].i*t[2*i] - fp[i].r*t[2*i+1];
         yi = fp[i].r*t[2*i] + fp[i].i*t[2*i+1];
#endif
         *yp1 = yr;
         *yp2 = yi;
         yp1 += stride2;
         yp2 -= stride2;
      }
   }
   RESTORE_STACK;
}

#endif /* OPUS_X86_MAY_HAVE_AVX2 && !FIXED_POINT */
