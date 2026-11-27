/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Copyright (c) 2007-2016 Jean-Marc Valin
   Copyright (c) 2026 Advanced Micro Devices, Inc. */
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include "celt_lpc.h"
#include "stack_alloc.h"
#include "mathops.h"
#include "vq.h"
#include "x86cpu.h"


#ifndef FIXED_POINT

/* ---- Portable inline / noinline attributes ----
 * pvq_search_small is kept always-inline (its nv argument is a compile-time
 * constant at every call site, so its phase loops fully unroll); the generic
 * large-N fallback is kept out-of-line.  The AVX2 mid-size kernel now lives in
 * its own translation unit (celt/x86/vq_avx2.c), so no per-function AVX2 target
 * attribute is needed here. */
#if defined(__GNUC__) || defined(__clang__)
# define PVQ_ALWAYS_INLINE static inline __attribute__((always_inline))
# define PVQ_NOINLINE      static __attribute__((noinline))
#elif defined(_MSC_VER)
# define PVQ_ALWAYS_INLINE static __forceinline
# define PVQ_NOINLINE      static __declspec(noinline)
#else
# define PVQ_ALWAYS_INLINE static inline
# define PVQ_NOINLINE      static
#endif

/* ---- ISA-adaptive blend / compare intrinsics ----
 * blendv is SSE4.1.  The signed 64-bit compare _mm_cmpgt_epi64 (PCMPGTQ) is
 * SSE4.2, so the hardware form is used only when AVX2 (which implies SSE4.2)
 * is presumed or the compiler already targets SSE4.2; otherwise an SSE2
 * emulation is used.  The emulation is valid because both 32-bit halves of
 * every PVQ argmax key are non-negative (float-bits of max >= 0 in the high
 * half, index < 48 in the low half). */
#if defined(OPUS_X86_PRESUME_AVX2) || defined(__SSE4_2__)
# define pvq_cmpgt_epi64(a,b)      _mm_cmpgt_epi64((a),(b))
#else
static inline __m128i pvq_cmpgt_epi64_sse2(__m128i a, __m128i b) {
   __m128i gt32  = _mm_cmpgt_epi32(a, b);
   __m128i eq32  = _mm_cmpeq_epi32(a, b);
   __m128i lo_gt = _mm_shuffle_epi32(gt32, _MM_SHUFFLE(2, 2, 0, 0));
   __m128i hi_gt = _mm_shuffle_epi32(gt32, _MM_SHUFFLE(3, 3, 1, 1));
   __m128i hi_eq = _mm_shuffle_epi32(eq32, _MM_SHUFFLE(3, 3, 1, 1));
   return _mm_or_si128(hi_gt, _mm_and_si128(hi_eq, lo_gt));
}
# define pvq_cmpgt_epi64(a,b)      pvq_cmpgt_epi64_sse2((a),(b))
#endif

#if defined(OPUS_X86_PRESUME_SSE4_1) || defined(OPUS_X86_PRESUME_AVX2)
# define pvq_blendv_epi8(a,b,m)    _mm_blendv_epi8((a),(b),(m))
# define pvq_blendv_ps(a,b,m)      _mm_blendv_ps((a),(b),(m))
#else
/* SSE2-only variable blend: selects b where mask is all-ones, a where zero. */
static inline __m128i pvq_blendv_epi8_sse2(__m128i a, __m128i b, __m128i mask) {
   return _mm_or_si128(_mm_and_si128(mask, b), _mm_andnot_si128(mask, a));
}
static inline __m128 pvq_blendv_ps_sse2(__m128 a, __m128 b, __m128 mask) {
   return _mm_or_ps(_mm_and_ps(mask, b), _mm_andnot_ps(mask, a));
}
# define pvq_blendv_epi8(a,b,m)    pvq_blendv_epi8_sse2((a),(b),(m))
# define pvq_blendv_ps(a,b,m)      pvq_blendv_ps_sse2((a),(b),(m))
#endif

/* ======================================================================
 * Register-resident PVQ search for small vectors (N <= 16).
 *
 * The whole working set (X, the doubled y, the signs, and the running pulse
 * counts) is held in SSE registers across every phase, so the only memory
 * traffic is the final iy[] store.  Every arithmetic step -- the abs via
 * andnot, the sum reduction order, _mm_rcp_ps in the pyramid projection,
 * _mm_cvttps_epi32 truncation, the per-lane _mm_rsqrt_ps, the running argmax,
 * the horizontal-max tie-break, and the xor/add sign restore -- matches the
 * SSE2 reference op_pvq_search exactly, so iy and the returned yy are
 * bit-identical.
 *
 * nv = ceil(N/4) (<=4) is a compile-time constant at every call site
 * (dispatched by the switch in op_pvq_search_sse2), so the `if (nv>k)` guards
 * and the phase loops fully unroll.
 * ====================================================================== */
PVQ_ALWAYS_INLINE
opus_val16 pvq_search_small(const celt_norm * OPUS_RESTRICT _X,
                            int * OPUS_RESTRICT iy, int K, int N,
                            const int nv, const int pad)
{
   const __m128 signmask = _mm_set_ps1(-0.f);
   const __m128 two = _mm_set1_ps(2.f);
   const __m128 zero = _mm_setzero_ps();
   const __m128i idx0 = _mm_set_epi32(3, 2, 1, 0);
   const __m128i idx1 = _mm_set_epi32(7, 6, 5, 4);
   const __m128i idx2 = _mm_set_epi32(11, 10, 9, 8);
   const __m128i idx3 = _mm_set_epi32(15, 14, 13, 12);
   __m128 Xr0, Xr1, Xr2, Xr3;
   __m128 yr0, yr1, yr2, yr3;
   __m128 sum4;
   __m128 xyv, yyv;              /* xy/yy broadcast across all 4 lanes */
   int sgv[16];                  /* signs live only until phase F -> keep off registers */
   celt_norm xloc[16];           /* zero-padded copy of X for a partial last vector */
   const celt_norm * OPUS_RESTRICT Xp;
   float yy;
   int pulsesLeft, i, extra0 = 0;
   const __m128i nm = _mm_set1_epi32(N);

   Xr1 = Xr2 = Xr3 = zero;
   yr0 = yr1 = yr2 = yr3 = zero;

   /* For a partial last vector (pad != 0, N not a multiple of 4) copy X into a
    * zero-padded local buffer so the 4-wide loads below never read past the
    * caller's X[0..N-1]; the tail is then masked out of the reductions. */
   Xp = _X;
   if (pad)
   {
      int kk;
      for (kk = 0; kk < 16; kk++) xloc[kk] = 0;
      for (kk = 0; kk < N; kk++) xloc[kk] = _X[kk];
      Xp = xloc;
   }

   /* -------- Phase A: abs / sign / sum (register-resident) --------
    * The tail of a partial last vector is zero-masked before the sign/abs pass
    * so the padding never enters the sum (which drives the pyramid allocation). */
   Xr0 = _mm_loadu_ps(&Xp[0]);
   if (pad && nv <= 1) Xr0 = _mm_and_ps(Xr0, _mm_castsi128_ps(_mm_cmplt_epi32(idx0, nm)));
   _mm_storeu_si128((__m128i*)(void*)&sgv[0], _mm_castps_si128(_mm_cmplt_ps(Xr0, zero)));
   Xr0 = _mm_andnot_ps(signmask, Xr0);
   sum4 = Xr0;
   if (nv > 1) {
      Xr1 = _mm_loadu_ps(&Xp[4]);
      if (pad && nv <= 2)
         Xr1 = _mm_and_ps(Xr1, _mm_castsi128_ps(_mm_cmplt_epi32(idx1, nm)));
      _mm_storeu_si128((__m128i*)(void*)&sgv[4],
                       _mm_castps_si128(_mm_cmplt_ps(Xr1, zero)));
      Xr1 = _mm_andnot_ps(signmask, Xr1);
      sum4 = _mm_add_ps(sum4, Xr1);
   }
   if (nv > 2) {
      Xr2 = _mm_loadu_ps(&Xp[8]);
      if (pad && nv <= 3)
         Xr2 = _mm_and_ps(Xr2, _mm_castsi128_ps(_mm_cmplt_epi32(idx2, nm)));
      _mm_storeu_si128((__m128i*)(void*)&sgv[8],
                       _mm_castps_si128(_mm_cmplt_ps(Xr2, zero)));
      Xr2 = _mm_andnot_ps(signmask, Xr2);
      sum4 = _mm_add_ps(sum4, Xr2);
   }
   if (nv > 3) {
      Xr3 = _mm_loadu_ps(&Xp[12]);
      if (pad && nv <= 4)
         Xr3 = _mm_and_ps(Xr3, _mm_castsi128_ps(_mm_cmplt_epi32(idx3, nm)));
      _mm_storeu_si128((__m128i*)(void*)&sgv[12],
                       _mm_castps_si128(_mm_cmplt_ps(Xr3, zero)));
      Xr3 = _mm_andnot_ps(signmask, Xr3);
      sum4 = _mm_add_ps(sum4, Xr3);
   }
   sum4 = _mm_add_ps(sum4, _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(1, 0, 3, 2)));
   sum4 = _mm_add_ps(sum4, _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(2, 3, 0, 1)));

   xyv = yyv = zero;
   pulsesLeft = K;

   /* -------- Phase B: pyramid pre-search (register-resident) -------- */
   if (K > (N >> 1))
   {
      __m128i psum;
      __m128 xy4, yy4, rcp4, sms;
      float sum = _mm_cvtss_f32(sum4);
      sms = sum4;
      if (!(sum > EPSILON && sum < 64))
      {
         Xr0 = _mm_set_ps(0.f, 0.f, 0.f, QCONST16(1.f, 14));
         Xr1 = Xr2 = Xr3 = zero;
         sms = _mm_set_ps1(1.f);
      }
      rcp4 = _mm_mul_ps(_mm_set_ps1((float)(K + .8)), _mm_rcp_ps(sms));
      xy4 = yy4 = zero;
      psum = _mm_setzero_si128();

      {
         __m128i p4 = _mm_cvttps_epi32(_mm_mul_ps(Xr0, rcp4));
         __m128 y4 = _mm_cvtepi32_ps(p4);
         psum = _mm_add_epi32(psum, p4);
         xy4 = _mm_add_ps(xy4, _mm_mul_ps(Xr0, y4));
         yy4 = _mm_add_ps(yy4, _mm_mul_ps(y4, y4));
         yr0 = _mm_add_ps(y4, y4);
      }
      if (nv > 1) {
         __m128i p4 = _mm_cvttps_epi32(_mm_mul_ps(Xr1, rcp4));
         psum = _mm_add_epi32(psum, p4);
         __m128 y4 = _mm_cvtepi32_ps(p4);
         xy4 = _mm_add_ps(xy4, _mm_mul_ps(Xr1, y4));
         yy4 = _mm_add_ps(yy4, _mm_mul_ps(y4, y4));
         yr1 = _mm_add_ps(y4, y4);
      }
      if (nv > 2) {
         __m128i p4 = _mm_cvttps_epi32(_mm_mul_ps(Xr2, rcp4));
         psum = _mm_add_epi32(psum, p4);
         __m128 y4 = _mm_cvtepi32_ps(p4);
         xy4 = _mm_add_ps(xy4, _mm_mul_ps(Xr2, y4));
         yy4 = _mm_add_ps(yy4, _mm_mul_ps(y4, y4));
         yr2 = _mm_add_ps(y4, y4);
      }
      if (nv > 3) {
         __m128i p4 = _mm_cvttps_epi32(_mm_mul_ps(Xr3, rcp4));
         psum = _mm_add_epi32(psum, p4);
         __m128 y4 = _mm_cvtepi32_ps(p4);
         xy4 = _mm_add_ps(xy4, _mm_mul_ps(Xr3, y4));
         yy4 = _mm_add_ps(yy4, _mm_mul_ps(y4, y4));
         yr3 = _mm_add_ps(y4, y4);
      }

      psum = _mm_add_epi32(psum, _mm_shuffle_epi32(psum, _MM_SHUFFLE(1, 0, 3, 2)));
      psum = _mm_add_epi32(psum, _mm_shuffle_epi32(psum, _MM_SHUFFLE(2, 3, 0, 1)));
      pulsesLeft -= _mm_cvtsi128_si32(psum);
      xy4 = _mm_add_ps(xy4, _mm_shuffle_ps(xy4, xy4, _MM_SHUFFLE(1, 0, 3, 2)));
      xy4 = _mm_add_ps(xy4, _mm_shuffle_ps(xy4, xy4, _MM_SHUFFLE(2, 3, 0, 1)));
      yy4 = _mm_add_ps(yy4, _mm_shuffle_ps(yy4, yy4, _MM_SHUFFLE(1, 0, 3, 2)));
      yy4 = _mm_add_ps(yy4, _mm_shuffle_ps(yy4, yy4, _MM_SHUFFLE(2, 3, 0, 1)));
      /* xy4/yy4 already hold the sum broadcast to all 4 lanes. */
      xyv = xy4;
      yyv = yy4;
   }

   /* -------- Phase C: pad lanes >= N with X=-100, y=100 --------
    * Only needed when N is not a multiple of 4 (partial last vector); for the
    * exact widths (N=4/8/12/16) `pad` is 0 and the whole block folds away. */
   if (pad)
   {
      const __m128i nm1 = _mm_set1_epi32(N - 1);
      const __m128 neghund = _mm_set1_ps(-100.f);
      const __m128 hund = _mm_set1_ps(100.f);
      __m128 pmf;
      pmf = _mm_castsi128_ps(_mm_cmpgt_epi32(idx0, nm1));
      Xr0 = _mm_or_ps(_mm_andnot_ps(pmf, Xr0), _mm_and_ps(pmf, neghund));
      yr0 = _mm_or_ps(_mm_andnot_ps(pmf, yr0), _mm_and_ps(pmf, hund));
      if (nv > 1) {
         pmf = _mm_castsi128_ps(_mm_cmpgt_epi32(idx1, nm1));
         Xr1 = _mm_or_ps(_mm_andnot_ps(pmf, Xr1), _mm_and_ps(pmf, neghund));
         yr1 = _mm_or_ps(_mm_andnot_ps(pmf, yr1), _mm_and_ps(pmf, hund));
      }
      if (nv > 2) {
         pmf = _mm_castsi128_ps(_mm_cmpgt_epi32(idx2, nm1));
         Xr2 = _mm_or_ps(_mm_andnot_ps(pmf, Xr2), _mm_and_ps(pmf, neghund));
         yr2 = _mm_or_ps(_mm_andnot_ps(pmf, yr2), _mm_and_ps(pmf, hund));
      }
      if (nv > 3) {
         pmf = _mm_castsi128_ps(_mm_cmpgt_epi32(idx3, nm1));
         Xr3 = _mm_or_ps(_mm_andnot_ps(pmf, Xr3), _mm_and_ps(pmf, neghund));
         yr3 = _mm_or_ps(_mm_andnot_ps(pmf, yr3), _mm_and_ps(pmf, hund));
      }
   }

   /* -------- Phase D: silence fallback (rare) -------- */
   if (pulsesLeft > N + 3)
   {
      opus_val16 tmp = (opus_val16)pulsesLeft;
      yy = _mm_cvtss_f32(yyv);
      yy = MAC16_16(yy, tmp, tmp);
      yy = MAC16_16(yy, tmp, _mm_cvtss_f32(yr0));
      yyv = _mm_set1_ps(yy);
      extra0 = pulsesLeft;
      pulsesLeft = 0;
   }

   /* -------- Phase E: greedy pulse placement (payload-carrying argmax) --------
    * y[best]+=2 is done as an in-register lane blend.  Two 64-bit keys are
    * reduced together under a SHARED compare mask:
    *   keyA = (float_bits(max) << 32) | index   -> argmax + best_id
    *   keyB = (float_bits(X)   << 32) | float_bits(y)  (of the column winner)
    * Because max>=0 its bits are monotone, a signed-i64 max of keyA gives the
    * largest value with ties to the largest index -- exactly the reference
    * tie-break.  keyB rides the same blend masks, so after the reduction its
    * low 32 bits hold y[best] and its high 32 bits hold X[best]. */
   {
   const __m128 one = _mm_set1_ps(1.f);
   for (i = 0; i < pulsesLeft; i++)
   {
      __m128 max, r0, r1, r2, r3, nbf, xsel, ysel;
      __m128i pos, bb, m, kAlo, kAhi, kA, kBlo, kBhi, kB, ksw;

      yyv = _mm_add_ps(yyv, one);

      r0 = _mm_mul_ps(_mm_add_ps(Xr0, xyv), _mm_rsqrt_ps(_mm_add_ps(yr0, yyv)));
      max = _mm_max_ps(zero, r0);
      pos = _mm_and_si128(idx0, _mm_castps_si128(_mm_cmpgt_ps(r0, zero)));
      xsel = Xr0; ysel = yr0;
      if (nv > 1) {
         r1 = _mm_mul_ps(_mm_add_ps(Xr1, xyv), _mm_rsqrt_ps(_mm_add_ps(yr1, yyv)));
         nbf = _mm_cmpgt_ps(r1, max);
         pos = pvq_blendv_epi8(pos, idx1, _mm_castps_si128(nbf));
         xsel = pvq_blendv_ps(xsel, Xr1, nbf);
         ysel = pvq_blendv_ps(ysel, yr1, nbf);
         max = _mm_max_ps(max, r1);
      }
      if (nv > 2) {
         r2 = _mm_mul_ps(_mm_add_ps(Xr2, xyv), _mm_rsqrt_ps(_mm_add_ps(yr2, yyv)));
         nbf = _mm_cmpgt_ps(r2, max);
         pos = pvq_blendv_epi8(pos, idx2, _mm_castps_si128(nbf));
         xsel = pvq_blendv_ps(xsel, Xr2, nbf);
         ysel = pvq_blendv_ps(ysel, yr2, nbf);
         max = _mm_max_ps(max, r2);
      }
      if (nv > 3) {
         r3 = _mm_mul_ps(_mm_add_ps(Xr3, xyv), _mm_rsqrt_ps(_mm_add_ps(yr3, yyv)));
         nbf = _mm_cmpgt_ps(r3, max);
         pos = pvq_blendv_epi8(pos, idx3, _mm_castps_si128(nbf));
         xsel = pvq_blendv_ps(xsel, Xr3, nbf);
         ysel = pvq_blendv_ps(ysel, yr3, nbf);
         max = _mm_max_ps(max, r3);
      }

      kAlo = _mm_unpacklo_epi32(pos, _mm_castps_si128(max));
      kAhi = _mm_unpackhi_epi32(pos, _mm_castps_si128(max));
      kBlo = _mm_unpacklo_epi32(_mm_castps_si128(ysel), _mm_castps_si128(xsel));
      kBhi = _mm_unpackhi_epi32(_mm_castps_si128(ysel), _mm_castps_si128(xsel));
      m  = pvq_cmpgt_epi64(kAlo, kAhi);
      kA = pvq_blendv_epi8(kAhi, kAlo, m);
      kB = pvq_blendv_epi8(kBhi, kBlo, m);
      ksw = _mm_shuffle_epi32(kA, _MM_SHUFFLE(1, 0, 3, 2));
      m  = pvq_cmpgt_epi64(kA, ksw);
      kA = pvq_blendv_epi8(ksw, kA, m);
      kB = pvq_blendv_epi8(_mm_shuffle_epi32(kB, _MM_SHUFFLE(1, 0, 3, 2)), kB, m);
      bb = _mm_shuffle_epi32(kA, 0);

      yyv = _mm_add_ps(yyv, _mm_castsi128_ps(_mm_shuffle_epi32(kB, _MM_SHUFFLE(0, 0, 0, 0))));
      xyv = _mm_add_ps(xyv, _mm_castsi128_ps(_mm_shuffle_epi32(kB, _MM_SHUFFLE(1, 1, 1, 1))));

      m = _mm_cmpeq_epi32(idx0, bb);
      yr0 = _mm_add_ps(yr0, _mm_and_ps(_mm_castsi128_ps(m), two));
      if (nv > 1) {
         m = _mm_cmpeq_epi32(idx1, bb);
         yr1 = _mm_add_ps(yr1, _mm_and_ps(_mm_castsi128_ps(m), two));
      }
      if (nv > 2) {
         m = _mm_cmpeq_epi32(idx2, bb);
         yr2 = _mm_add_ps(yr2, _mm_and_ps(_mm_castsi128_ps(m), two));
      }
      if (nv > 3) {
         m = _mm_cmpeq_epi32(idx3, bb);
         yr3 = _mm_add_ps(yr3, _mm_and_ps(_mm_castsi128_ps(m), two));
      }
   }
   yy = _mm_cvtss_f32(yyv);
   }

   /* -------- Phase F: iy = y/2 (+ silence extra), sign restore, store --------
    * y == 2*iy is maintained throughout, so iy is recovered as trunc(y*0.5)
    * rather than carrying separate pulse-count vectors (as in the YMM path). */
   {
      const __m128 half = _mm_set1_ps(0.5f);
      __m128i e0 = _mm_set_epi32(0, 0, 0, extra0);
      __m128i ic, s0 = _mm_loadu_si128((const __m128i*)(const void*)&sgv[0]);
      ic = _mm_add_epi32(_mm_cvttps_epi32(_mm_mul_ps(yr0, half)), e0);
      _mm_storeu_si128((__m128i*)(void*)&iy[0], _mm_xor_si128(_mm_add_epi32(ic, s0), s0));
      if (nv > 1) {
         __m128i s1 = _mm_loadu_si128((const __m128i*)(const void*)&sgv[4]);
         ic = _mm_cvttps_epi32(_mm_mul_ps(yr1, half));
         _mm_storeu_si128((__m128i*)(void*)&iy[4],
                          _mm_xor_si128(_mm_add_epi32(ic, s1), s1));
      }
      if (nv > 2) {
         __m128i s2 = _mm_loadu_si128((const __m128i*)(const void*)&sgv[8]);
         ic = _mm_cvttps_epi32(_mm_mul_ps(yr2, half));
         _mm_storeu_si128((__m128i*)(void*)&iy[8],
                          _mm_xor_si128(_mm_add_epi32(ic, s2), s2));
      }
      if (nv > 3) {
         __m128i s3 = _mm_loadu_si128((const __m128i*)(const void*)&sgv[12]);
         ic = _mm_cvttps_epi32(_mm_mul_ps(yr3, half));
         _mm_storeu_si128((__m128i*)(void*)&iy[12],
                          _mm_xor_si128(_mm_add_epi32(ic, s3), s3));
      }
   }

   return yy;
}

/* The AVX2 mid-size (13 <= N <= 48) PVQ kernel pvq_search_ymm() lives in its
   own AVX2-baseline translation unit, celt/x86/vq_avx2.c (declared in
   vq_sse.h), so it is built with -mavx2/-mfma (or /arch:AVX2 on MSVC) instead
   of relying on a per-function target attribute. */

/* Generic memory-resident fallback for the large-N tail (N > 48), kept
 * out-of-line from the small/mid dispatch below. */
PVQ_NOINLINE
opus_val16 pvq_search_generic(celt_norm *_X, int *iy, int K, int N, int arch)
{
   int i, j;
   int pulsesLeft;
   float xy, yy;
   VARDECL(celt_norm, y);
   VARDECL(celt_norm, X);
   VARDECL(float, signy);
   __m128 signmask;
   __m128 sums;
   __m128i fours;
   const __m128 two = _mm_set1_ps(2.f);
   SAVE_STACK;

   (void)arch;

   /* All bits set to zero, except for the sign bit. */
   signmask = _mm_set_ps1(-0.f);
   fours = _mm_set_epi32(4, 4, 4, 4);
   ALLOC(y, N+3, celt_norm);
   ALLOC(X, N+3, celt_norm);
   ALLOC(signy, N+3, float);

   OPUS_COPY(X, _X, N);
   X[N] = X[N+1] = X[N+2] = 0;

   sums = _mm_setzero_ps();
   for (j=0;j<N;j+=4)
   {
      __m128 x4, s4;
      x4 = _mm_loadu_ps(&X[j]);
      s4 = _mm_cmplt_ps(x4, _mm_setzero_ps());
      /* Get rid of the sign */
      x4 = _mm_andnot_ps(signmask, x4);
      sums = _mm_add_ps(sums, x4);
      /* Clear y and iy in case we don't do the projection. */
      _mm_storeu_ps(&y[j], _mm_setzero_ps());
      _mm_storeu_si128((__m128i*)(void*)&iy[j], _mm_setzero_si128());
      _mm_storeu_ps(&X[j], x4);
      _mm_storeu_ps(&signy[j], s4);
   }
   sums = _mm_add_ps(sums, _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2)));
   sums = _mm_add_ps(sums, _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 3, 0, 1)));

   xy = yy = 0;

   pulsesLeft = K;

   /* Do a pre-search by projecting on the pyramid */
   if (K > (N>>1))
   {
      __m128i pulses_sum;
      __m128 yy4, xy4;
      __m128 rcp4;
      opus_val32 sum = _mm_cvtss_f32(sums);
      /* If X is too small, just replace it with a pulse at 0 */
      /* Prevents infinities and NaNs from causing too many pulses
         to be allocated. 64 is an approximation of infinity here. */
      if (!(sum > EPSILON && sum < 64))
      {
         X[0] = QCONST16(1.f,14);
         j=1; do
            X[j]=0;
         while (++j<N);
         sums = _mm_set_ps1(1.f);
      }
      /* Using K+e with e < 1 guarantees we cannot get more than K pulses. */
      rcp4 = _mm_mul_ps(_mm_set_ps1((float)(K+.8)), _mm_rcp_ps(sums));
      xy4 = yy4 = _mm_setzero_ps();
      pulses_sum = _mm_setzero_si128();
      for (j=0;j<N;j+=4)
      {
         __m128 rx4, x4, y4;
         __m128i iy4;
         x4 = _mm_loadu_ps(&X[j]);
         rx4 = _mm_mul_ps(x4, rcp4);
         iy4 = _mm_cvttps_epi32(rx4);
         pulses_sum = _mm_add_epi32(pulses_sum, iy4);
         _mm_storeu_si128((__m128i*)(void*)&iy[j], iy4);
         y4 = _mm_cvtepi32_ps(iy4);
         xy4 = _mm_add_ps(xy4, _mm_mul_ps(x4, y4));
         yy4 = _mm_add_ps(yy4, _mm_mul_ps(y4, y4));
         /* double the y[] vector so we don't have to do it in the search loop. */
         _mm_storeu_ps(&y[j], _mm_add_ps(y4, y4));
      }
      pulses_sum = _mm_add_epi32(pulses_sum, _mm_shuffle_epi32(pulses_sum, _MM_SHUFFLE(1, 0, 3, 2)));
      pulses_sum = _mm_add_epi32(pulses_sum, _mm_shuffle_epi32(pulses_sum, _MM_SHUFFLE(2, 3, 0, 1)));
      pulsesLeft -= _mm_cvtsi128_si32(pulses_sum);
      xy4 = _mm_add_ps(xy4, _mm_shuffle_ps(xy4, xy4, _MM_SHUFFLE(1, 0, 3, 2)));
      xy4 = _mm_add_ps(xy4, _mm_shuffle_ps(xy4, xy4, _MM_SHUFFLE(2, 3, 0, 1)));
      xy = _mm_cvtss_f32(xy4);
      yy4 = _mm_add_ps(yy4, _mm_shuffle_ps(yy4, yy4, _MM_SHUFFLE(1, 0, 3, 2)));
      yy4 = _mm_add_ps(yy4, _mm_shuffle_ps(yy4, yy4, _MM_SHUFFLE(2, 3, 0, 1)));
      yy = _mm_cvtss_f32(yy4);
   }
   X[N] = X[N+1] = X[N+2] = -100;
   y[N] = y[N+1] = y[N+2] = 100;
   celt_sig_assert(pulsesLeft>=0);

   /* This should never happen, but just in case it does (e.g. on silence)
      we fill the first bin with pulses. */
   if (pulsesLeft > N+3)
   {
      opus_val16 tmp = (opus_val16)pulsesLeft;
      yy = MAC16_16(yy, tmp, tmp);
      yy = MAC16_16(yy, tmp, y[0]);
      iy[0] += pulsesLeft;
      pulsesLeft=0;
   }

   for (i=0;i<pulsesLeft;i++)
   {
      int best_id;
      __m128 xy4, yy4;
      __m128 max;
      __m128i count;
      __m128i pos;
      /* The squared magnitude term gets added anyway, so we might as well
         add it outside the loop */
      yy = ADD16(yy, 1);
      xy4 = _mm_set1_ps(xy);
      yy4 = _mm_set1_ps(yy);
      max = _mm_setzero_ps();
      pos = _mm_setzero_si128();
      count = _mm_set_epi32(3, 2, 1, 0);
      for (j=0;j<N;j+=4)
      {
         __m128 x4, y4, r4;
         x4 = _mm_loadu_ps(&X[j]);
         y4 = _mm_loadu_ps(&y[j]);
         x4 = _mm_add_ps(x4, xy4);
         y4 = _mm_add_ps(y4, yy4);
         y4 = _mm_rsqrt_ps(y4);
         r4 = _mm_mul_ps(x4, y4);
         /* Update the index of the max. */
         pos = _mm_max_epi16(pos, _mm_and_si128(count, _mm_castps_si128(_mm_cmpgt_ps(r4, max))));
         /* Update the max. */
         max = _mm_max_ps(max, r4);
         /* Update the indices (+4) */
         count = _mm_add_epi32(count, fours);
      }
      /* Horizontal argmax via lossless 64-bit keys (value in high 32, index in
         low 32).  max>=0 so float bits are monotone => signed-i64 max picks the
         largest value, ties to largest index.  Uses SSE2-only helpers so the
         generic fallback compiles under the base -msse2 flag. */
      {
         __m128i mi = _mm_castps_si128(max);
         __m128i klo = _mm_unpacklo_epi32(pos, mi);
         __m128i khi = _mm_unpackhi_epi32(pos, mi);
         __m128i km  = pvq_cmpgt_epi64(klo, khi);
         __m128i kk  = pvq_blendv_epi8(khi, klo, km);
         __m128i ksw = _mm_shuffle_epi32(kk, _MM_SHUFFLE(1, 0, 3, 2));
         km = pvq_cmpgt_epi64(kk, ksw);
         kk = pvq_blendv_epi8(ksw, kk, km);
         best_id = _mm_cvtsi128_si32(kk);
      }

      /* Updating the sums of the new pulse(s) */
      xy = ADD32(xy, EXTEND32(X[best_id]));
      /* We're multiplying y[j] by two so we don't have to do it here */
      yy = ADD16(yy, y[best_id]);
      y[best_id] += 2;
      iy[best_id]++;
   }
   (void)two;

   /* Put the original sign back */
   for (j=0;j<N;j+=4)
   {
      __m128i y4;
      __m128i s4;
      y4 = _mm_loadu_si128((__m128i*)(void*)&iy[j]);
      s4 = _mm_castps_si128(_mm_loadu_ps(&signy[j]));
      y4 = _mm_xor_si128(_mm_add_epi32(y4, s4), s4);
      _mm_storeu_si128((__m128i*)(void*)&iy[j], y4);
   }
   RESTORE_STACK;
   return yy;
}

opus_val16 op_pvq_search_sse2(celt_norm *_X, int *iy, int K, int N, int arch)
{
   /* pvq_search_small is pure SSE2 — always safe on this dispatch path.
      pvq_search_ymm requires AVX2: when PRESUME_AVX2 the compiler already
      has -mavx2, so no runtime arch check is needed; otherwise gate on
      arch >= 4 (set by opus_select_arch via CPUID). */
   if (N <= 12)
   {
      switch (N)
      {
         case 4:  return pvq_search_small(_X, iy, K, N, 1, 0);
         case 8:  return pvq_search_small(_X, iy, K, N, 2, 0);
         case 12: return pvq_search_small(_X, iy, K, N, 3, 0);
         default: return pvq_search_small(_X, iy, K, N, (N + 3) >> 2, 1);
      }
   }
#if defined(OPUS_X86_MAY_HAVE_AVX2)
#ifdef OPUS_X86_PRESUME_AVX2
   if (N <= 48)
#else
   if (N <= 48 && arch >= 4)
#endif
   {
      const int pad = (N & 7) != 0;
      if (N <= 16) return pvq_search_ymm(_X, iy, K, N, 2, pad);
      if (N <= 24) return pvq_search_ymm(_X, iy, K, N, 3, pad);
      if (N <= 32) return pvq_search_ymm(_X, iy, K, N, 4, pad);
      if (N <= 40) return pvq_search_ymm(_X, iy, K, N, 5, pad);
      return pvq_search_ymm(_X, iy, K, N, 6, pad);
   }
#endif
   return pvq_search_generic(_X, iy, K, N, arch);
}

#endif
