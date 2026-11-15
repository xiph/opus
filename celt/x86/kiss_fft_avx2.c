/*Copyright (c) 2003-2004, Mark Borgerding
  Lots of modifications by Jean-Marc Valin
  Copyright (c) 2005-2007, Xiph.Org Foundation
  Copyright (c) 2008,      Xiph.Org Foundation, CSIRO
  Copyright (c) 2026,      Advanced Micro Devices, Inc.

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.*/

/* AVX2/FMA implementations of the CELT FFT butterflies kf_bfly{2,3,4}.
 *
 * This is an op-by-op translation of the scalar reference in celt/kiss_fft.c:
 * each scalar S_MUL / C_MUL / C_ADD / C_SUB / C_ADDTO / C_MULBYSCALAR has a
 * corresponding lane-wise vector intrinsic performing the same arithmetic in
 * the same order.  Lane k of every __m256 holds the k-th complex pair the
 * scalar inner loop would process.  The result is not bit-identical to the
 * scalar C path, matching the other float SIMD paths.
 *
 * The file is compiled at the AVX2 baseline (-mavx2 -mfma, via
 * CELT_SOURCES_AVX2) and is entered through opus_fft_impl() only when
 * opus_select_arch() reports AVX2 (arch >= 4); other CPUs take the scalar
 * kf_bfly path. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "arch.h"

#if !defined(FIXED_POINT) && defined(OPUS_X86_MAY_HAVE_AVX2)

#include <immintrin.h>

#include "_kiss_fft_guts.h"
#include "kiss_fft.h"

/* Complex multiply: dst[k] = a[k] * b[k] for the four complex pairs in a,b.
   Mirrors scalar C_MUL(dst,a,b): dst.r = a.r*b.r - a.i*b.i ;
                                  dst.i = a.r*b.i + a.i*b.r.
   _mm256_fmaddsub_ps gives even(real) = a.r*b.r - a.i*b.i and
   odd(imag) = a.r*b.i + a.i*b.r in a single rounded FMA per lane. */
static OPUS_INLINE __m256 cmul_avx(__m256 a, __m256 b)
{
   __m256 a_rr = _mm256_moveldup_ps(a);
   __m256 a_ii = _mm256_movehdup_ps(a);
   __m256 b_sw = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
   __m256 mul2 = _mm256_mul_ps(a_ii, b_sw);
   return _mm256_fmaddsub_ps(a_rr, b, mul2);
}

/* Gather four stride-sk complex twiddles into one __m256, matching the
   scalar inner loop's tw walk of sk entries per complex pair. */
static OPUS_INLINE __m256 load_twiddle4(const kiss_twiddle_cpx *p, size_t sk)
{
   return _mm256_set_ps(
      p[3 * sk].i, p[3 * sk].r,
      p[2 * sk].i, p[2 * sk].r,
      p[1 * sk].i, p[1 * sk].r,
      p[0].i,      p[0].r);
}

/* Maximum twiddle batches cached for one kf_bfly{3,4} call.  m <= N/4 with
   N <= 480 (largest opus FFT), so m/4 <= 30; 32 is a safe stack-friendly
   bound.  Beyond it we fall back to per-iteration gathers. */
#define KFFT_AVX2_TW_CACHE_MAX 32

/* radix-2, m==4 special case (the radix-2 stage always follows a radix-4,
   so m==4).  The four per-k transforms are too irregular to vectorise
   lane-wise, so the four `t` values are built in scalar (exact rounding)
   and the C_SUB / C_ADDTO are done as packed ops. */
void kf_bfly2_avx2(kiss_fft_cpx *Fout, int m, int N)
{
   const float tw_s = 0.7071067812f;     /* sqrt(2)/2 ; same as scalar tw */
   int i;
   (void)m;
   for (i = 0; i < N; i++)
   {
      kiss_fft_cpx *Fout2 = Fout + 4;

      const float t0r = Fout2[0].r;
      const float t0i = Fout2[0].i;
      const float t1r = (Fout2[1].r + Fout2[1].i) * tw_s;
      const float t1i = (Fout2[1].i - Fout2[1].r) * tw_s;
      const float t2r = Fout2[2].i;
      const float t2i = -Fout2[2].r;
      const float t3r = (Fout2[3].i - Fout2[3].r) * tw_s;
      const float t3i = -(Fout2[3].i + Fout2[3].r) * tw_s;

      __m256 t  = _mm256_setr_ps(t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i);
      __m256 vf = _mm256_loadu_ps((const float *)Fout);

      __m256 vf2_new = _mm256_sub_ps(vf, t);
      __m256 vf_new  = _mm256_add_ps(vf, t);

      _mm256_storeu_ps((float *)Fout,  vf_new);
      _mm256_storeu_ps((float *)Fout2, vf2_new);

      Fout += 8;
   }
}

/* radix-4, m>=4 main path.  Scalar reference: celt/kiss_fft.c kf_bfly4
   m>1 branch, inner k-loop.  Radix-4 is also used under RADIX_TWO_ONLY, so
   this kernel is compiled unconditionally. */
void kf_bfly4_avx2(
      kiss_fft_cpx *Fout,
      const size_t fstride,
      const kiss_fft_state *st,
      int m,
      int N,
      int mm)
{
   const int m2 = 2 * m;
   const int m3 = 3 * m;
   kiss_fft_cpx *const Fout_beg = Fout;
   /* Negates odd (imag) lanes so a swapped vector becomes the cross term. */
   const __m256 sign_im_neg = _mm256_setr_ps(
      0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);

   /* Twiddle hoist: load the strided twiddles once per call and reuse over
      the N outer iterations; fall back to per-batch gathers when m is too
      large to cache (never happens for opus sizes). */
   const int n_batches = m / 4;
   const int can_cache = (n_batches <= KFFT_AVX2_TW_CACHE_MAX);
   __m256 vtw1_cache[KFFT_AVX2_TW_CACHE_MAX];
   __m256 vtw2_cache[KFFT_AVX2_TW_CACHE_MAX];
   __m256 vtw3_cache[KFFT_AVX2_TW_CACHE_MAX];
   int i;
   if (can_cache) {
      const kiss_twiddle_cpx *p1 = st->twiddles;
      const kiss_twiddle_cpx *p2 = st->twiddles;
      const kiss_twiddle_cpx *p3 = st->twiddles;
      int b;
      for (b = 0; b < n_batches; b++) {
         vtw1_cache[b] = load_twiddle4(p1, fstride);
         vtw2_cache[b] = load_twiddle4(p2, fstride * 2);
         vtw3_cache[b] = load_twiddle4(p3, fstride * 3);
         p1 += 4 * fstride;
         p2 += 4 * fstride * 2;
         p3 += 4 * fstride * 3;
      }
   }

   for (i = 0; i < N; i++)
   {
      const kiss_twiddle_cpx *tw1 = st->twiddles;
      const kiss_twiddle_cpx *tw2 = st->twiddles;
      const kiss_twiddle_cpx *tw3 = st->twiddles;
      int j;
      int b = 0;
      Fout = Fout_beg + i * mm;

      for (j = 0; j + 4 <= m; j += 4, b++)
      {
         kiss_fft_cpx *const f = Fout + j;

         __m256 vf   = _mm256_loadu_ps((const float *)&f[0]);
         __m256 vfm  = _mm256_loadu_ps((const float *)&f[m]);
         __m256 vfm2 = _mm256_loadu_ps((const float *)&f[m2]);
         __m256 vfm3 = _mm256_loadu_ps((const float *)&f[m3]);

         __m256 vtw1, vtw2, vtw3;
         if (can_cache) {
            vtw1 = vtw1_cache[b];
            vtw2 = vtw2_cache[b];
            vtw3 = vtw3_cache[b];
         } else {
            vtw1 = load_twiddle4(tw1, fstride);
            vtw2 = load_twiddle4(tw2, fstride * 2);
            vtw3 = load_twiddle4(tw3, fstride * 3);
         }

         /* C_MUL(scratch[0..2], Fout[m,m2,m3], *tw{1,2,3}) */
         __m256 vs0 = cmul_avx(vfm,  vtw1);
         __m256 vs1 = cmul_avx(vfm2, vtw2);
         __m256 vs2 = cmul_avx(vfm3, vtw3);

         __m256 vs5      = _mm256_sub_ps(vf, vs1);         /* scratch[5] */
         __m256 vf_add1  = _mm256_add_ps(vf, vs1);         /* C_ADDTO(*Fout,s1) */
         __m256 vs3      = _mm256_add_ps(vs0, vs2);        /* scratch[3] */
         __m256 vs4      = _mm256_sub_ps(vs0, vs2);        /* scratch[4] */
         __m256 vfm2_new = _mm256_sub_ps(vf_add1, vs3);    /* Fout[m2] */
         __m256 vf_new   = _mm256_add_ps(vf_add1, vs3);    /* *Fout */

         /* Advance twiddle pointers every batch (not only on the no-cache
            path) so a scalar tail below reads the correct twiddles. */
         tw1 += fstride * 4;
         tw2 += fstride * 2 * 4;
         tw3 += fstride * 3 * 4;

         /* Fout[m]  = (s5.r + s4.i, s5.i - s4.r); Fout[m3] = (s5.r - s4.i, s5.i + s4.r). */
         __m256 vs4_swap  = _mm256_shuffle_ps(vs4, vs4, _MM_SHUFFLE(2, 3, 0, 1));
         __m256 vs4_cross = _mm256_xor_ps(vs4_swap, sign_im_neg);
         __m256 vfm_new   = _mm256_add_ps(vs5, vs4_cross);
         __m256 vfm3_new  = _mm256_sub_ps(vs5, vs4_cross);

         _mm256_storeu_ps((float *)&f[0],  vf_new);
         _mm256_storeu_ps((float *)&f[m],  vfm_new);
         _mm256_storeu_ps((float *)&f[m2], vfm2_new);
         _mm256_storeu_ps((float *)&f[m3], vfm3_new);
      }

      /* Scalar tail when m % 4 != 0 (does not trigger for opus sizes, kept
         for correctness).  tw{1,2,3} already point at offset j by the
         always-advance above. */
      for (; j < m; j++)
      {
         kiss_fft_cpx *f = Fout + j;
         kiss_fft_cpx scratch[6];

         C_MUL(scratch[0], f[m],  *tw1);
         C_MUL(scratch[1], f[m2], *tw2);
         C_MUL(scratch[2], f[m3], *tw3);

         C_SUB(scratch[5], *f, scratch[1]);
         C_ADDTO(*f, scratch[1]);
         C_ADD(scratch[3], scratch[0], scratch[2]);
         C_SUB(scratch[4], scratch[0], scratch[2]);
         C_SUB(f[m2], *f, scratch[3]);
         tw1 += fstride;
         tw2 += fstride * 2;
         tw3 += fstride * 3;
         C_ADDTO(*f, scratch[3]);

         f[m].r  = scratch[5].r + scratch[4].i;
         f[m].i  = scratch[5].i - scratch[4].r;
         f[m3].r = scratch[5].r - scratch[4].i;
         f[m3].i = scratch[5].i + scratch[4].r;
      }
   }
}

#ifndef RADIX_TWO_ONLY
/* radix-3, m>=4 main path.  Scalar reference: celt/kiss_fft.c kf_bfly3. */
void kf_bfly3_avx2(
      kiss_fft_cpx *Fout,
      const size_t fstride,
      const kiss_fft_state *st,
      int m,
      int N,
      int mm)
{
   const int m2 = 2 * m;
   const kiss_twiddle_cpx epi3 = st->twiddles[fstride * m];
   const __m256 epi3_i_v = _mm256_set1_ps(epi3.i);
   const __m256 half_v   = _mm256_set1_ps(0.5f);   /* HALF_OF(x) == x*0.5f */
   const __m256 sign_im_neg = _mm256_setr_ps(
      0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
   kiss_fft_cpx *const Fout_beg = Fout;

   const int n_batches = m / 4;
   const int can_cache = (n_batches <= KFFT_AVX2_TW_CACHE_MAX);
   __m256 vtw1_cache[KFFT_AVX2_TW_CACHE_MAX];
   __m256 vtw2_cache[KFFT_AVX2_TW_CACHE_MAX];
   int i;
   if (can_cache) {
      const kiss_twiddle_cpx *p1 = st->twiddles;
      const kiss_twiddle_cpx *p2 = st->twiddles;
      int b;
      for (b = 0; b < n_batches; b++) {
         vtw1_cache[b] = load_twiddle4(p1, fstride);
         vtw2_cache[b] = load_twiddle4(p2, fstride * 2);
         p1 += 4 * fstride;
         p2 += 4 * fstride * 2;
      }
   }

   for (i = 0; i < N; i++)
   {
      const kiss_twiddle_cpx *tw1 = st->twiddles;
      const kiss_twiddle_cpx *tw2 = st->twiddles;
      int j;
      int b = 0;
      Fout = Fout_beg + i * mm;

      for (j = 0; j + 4 <= m; j += 4, b++)
      {
         kiss_fft_cpx *const f = Fout + j;

         __m256 vf   = _mm256_loadu_ps((const float *)&f[0]);
         __m256 vfm  = _mm256_loadu_ps((const float *)&f[m]);
         __m256 vfm2 = _mm256_loadu_ps((const float *)&f[m2]);

         __m256 vtw1, vtw2;
         if (can_cache) {
            vtw1 = vtw1_cache[b];
            vtw2 = vtw2_cache[b];
         } else {
            vtw1 = load_twiddle4(tw1, fstride);
            vtw2 = load_twiddle4(tw2, fstride * 2);
         }

         __m256 vs1 = cmul_avx(vfm,  vtw1);    /* scratch[1] */
         __m256 vs2 = cmul_avx(vfm2, vtw2);    /* scratch[2] */
         __m256 vs3 = _mm256_add_ps(vs1, vs2); /* scratch[3] */
         __m256 vs0 = _mm256_sub_ps(vs1, vs2); /* scratch[0] */

         tw1 += fstride * 4;
         tw2 += fstride * 2 * 4;

         /* Fout[m] = Fout - HALF_OF(scratch[3]) ; single-rounded fnmadd. */
         __m256 vfm_new = _mm256_fnmadd_ps(vs3, half_v, vf);
         /* C_MULBYSCALAR(scratch[0], epi3.i) */
         vs0 = _mm256_mul_ps(vs0, epi3_i_v);
         /* C_ADDTO(*Fout, scratch[3]) */
         __m256 vf_new = _mm256_add_ps(vf, vs3);

         /* Fout[m2] = Fout[m] + (s0.i,-s0.r); Fout[m] = Fout[m] - (s0.i,-s0.r). */
         __m256 vs0_swap  = _mm256_shuffle_ps(vs0, vs0, _MM_SHUFFLE(2, 3, 0, 1));
         __m256 vs0_cross = _mm256_xor_ps(vs0_swap, sign_im_neg);
         __m256 vfm2_new  = _mm256_add_ps(vfm_new, vs0_cross);
         __m256 vfm_final = _mm256_sub_ps(vfm_new, vs0_cross);

         _mm256_storeu_ps((float *)&f[0],  vf_new);
         _mm256_storeu_ps((float *)&f[m],  vfm_final);
         _mm256_storeu_ps((float *)&f[m2], vfm2_new);
      }

      /* Scalar tail (m not a multiple of 4; does not trigger for opus). */
      for (; j < m; j++)
      {
         kiss_fft_cpx *f = Fout + j;
         kiss_fft_cpx scratch[5];

         C_MUL(scratch[1], f[m],  *tw1);
         C_MUL(scratch[2], f[m2], *tw2);
         C_ADD(scratch[3], scratch[1], scratch[2]);
         C_SUB(scratch[0], scratch[1], scratch[2]);
         tw1 += fstride;
         tw2 += fstride * 2;

         f[m].r = f->r - HALF_OF(scratch[3].r);
         f[m].i = f->i - HALF_OF(scratch[3].i);

         C_MULBYSCALAR(scratch[0], epi3.i);

         C_ADDTO(*f, scratch[3]);

         f[m2].r = f[m].r + scratch[0].i;
         f[m2].i = f[m].i - scratch[0].r;
         f[m].r  = f[m].r - scratch[0].i;
         f[m].i  = f[m].i + scratch[0].r;
      }
   }
}

#endif /* RADIX_TWO_ONLY */

#endif /* !FIXED_POINT && OPUS_X86_MAY_HAVE_AVX2 */
