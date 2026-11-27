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

/* AVX2 mid-size PVQ codebook search (pvq_search_ymm), selected at run time by
   op_pvq_search_sse2() when opus_select_arch() reports AVX2. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "arch.h"

#if !defined(FIXED_POINT) && defined(OPUS_X86_MAY_HAVE_AVX2)

#include <immintrin.h>
#include "vq.h"

/* AVX2 is a strict superset of SSE4.1/SSE4.2, so the variable blend (SSE4.1)
   and the signed 64-bit compare PCMPGTQ (SSE4.2) used by the payload-carrying
   argmax reduction are the native hardware intrinsics in this translation
   unit. */
#define pvq_cmpgt_epi64(a, b)    _mm_cmpgt_epi64((a), (b))
#define pvq_blendv_epi8(a, b, m) _mm_blendv_epi8((a), (b), (m))
#define pvq_blendv_ps(a, b, m)   _mm_blendv_ps((a), (b), (m))

/* ======================================================================
 * Register-resident PVQ search for the mid-size calls (13 <= N <= 48).
 *
 * The whole working set is held in <=6 __m256 pairs (X and the doubled y),
 * resident across every phase; the only memory traffic is the final iy[]
 * store.
 *
 * BIT-EXACTNESS: the SSE2 reference reduces sum/xy/yy and the per-pulse argmax
 * in 4-lane blocks, in the order [0..3],[4..7],[8..11],...  An 8-lane grouping
 * would change the float reduction order (hence the sum, hence the pyramid
 * allocation) and the argmax tie-break (which index wins an exact tie).  So we
 * compute 8 lanes at a time (one __m256 rsqrt/mul) but always REDUCE by
 * splitting each __m256 into its low/high 128-bit halves and folding them in
 * that exact block order.  This reproduces the reference arithmetic and
 * tie-break byte-for-byte.  iy is recovered as y/2 (the code keeps y == 2*iy
 * throughout) so no separate pulse-count vectors are needed.
 * ====================================================================== */
opus_val16 pvq_search_ymm(const celt_norm * OPUS_RESTRICT _X,
                          int * OPUS_RESTRICT iy, int K, int N,
                          const int nvy, const int pad)
{
   const __m256 signmask8 = _mm256_set1_ps(-0.f);
   const __m256 zero8 = _mm256_setzero_ps();
   const __m256 two8  = _mm256_set1_ps(2.f);
   const __m256 half8 = _mm256_set1_ps(0.5f);
   const __m128 zero  = _mm_setzero_ps();
   /* global lane indices; the fold derives its per-half counts from the
    * lo/hi halves of these. */
   const __m256i ix0=_mm256_setr_epi32(0,1,2,3,4,5,6,7);
   const __m256i ix1=_mm256_setr_epi32(8,9,10,11,12,13,14,15);
   const __m256i ix2=_mm256_setr_epi32(16,17,18,19,20,21,22,23);
   const __m256i ix3=_mm256_setr_epi32(24,25,26,27,28,29,30,31);
   const __m256i ix4=_mm256_setr_epi32(32,33,34,35,36,37,38,39);
   const __m256i ix5=_mm256_setr_epi32(40,41,42,43,44,45,46,47);
   __m256 Xy0,Xy1,Xy2,Xy3,Xy4,Xy5, Yy0,Yy1,Yy2,Yy3,Yy4,Yy5;
   __m256 xy8, yy8;              /* xy/yy broadcast across all 8 lanes */
   __m128 sums4;
   int sgv[48];
   float yy, sum;
   int pulsesLeft, i, extra0 = 0;
   const __m256i nm8 = _mm256_set1_epi32(N);

   Xy1=Xy2=Xy3=Xy4=Xy5=zero8;
   Yy0=Yy1=Yy2=Yy3=Yy4=Yy5=zero8;

   /* -------- Phase A: abs / sign / sum --------
    * Zero-mask the tail of the final vector before sign/abs so that
    * garbage from adjacent memory never enters the sum. */
#define YMM_LOAD_ABS(idx_, Xk_, sgvp_) do {                            \
      if (pad && nvy <= (idx_)+1)                                       \
         Xk_ = _mm256_maskload_ps(&_X[(idx_)*8],                        \
                  _mm256_cmpgt_epi32(nm8, ix##idx_));                    \
      else                                                              \
         Xk_ = _mm256_loadu_ps(&_X[(idx_)*8]);                          \
      _mm256_storeu_si256((__m256i*)(void*)(sgvp_),                     \
         _mm256_castps_si256(_mm256_cmp_ps(Xk_, zero8, _CMP_LT_OQ)));  \
      Xk_ = _mm256_andnot_ps(signmask8, Xk_);                          \
   } while(0)
   YMM_LOAD_ABS(0, Xy0, &sgv[0]);
   if (nvy>1) YMM_LOAD_ABS(1, Xy1, &sgv[8]);
   if (nvy>2) YMM_LOAD_ABS(2, Xy2, &sgv[16]);
   if (nvy>3) YMM_LOAD_ABS(3, Xy3, &sgv[24]);
   if (nvy>4) YMM_LOAD_ABS(4, Xy4, &sgv[32]);
   if (nvy>5) YMM_LOAD_ABS(5, Xy5, &sgv[40]);
#undef YMM_LOAD_ABS
   /* sum in reference 4-lane block order: lo,hi of each vector */
   sums4 = _mm_add_ps(_mm256_castps256_ps128(Xy0), _mm256_extractf128_ps(Xy0, 1));
   if (nvy > 1) {
      sums4 = _mm_add_ps(sums4, _mm256_castps256_ps128(Xy1));
      sums4 = _mm_add_ps(sums4, _mm256_extractf128_ps(Xy1, 1));
   }
   if (nvy > 2) {
      sums4 = _mm_add_ps(sums4, _mm256_castps256_ps128(Xy2));
      sums4 = _mm_add_ps(sums4, _mm256_extractf128_ps(Xy2, 1));
   }
   if (nvy > 3) {
      sums4 = _mm_add_ps(sums4, _mm256_castps256_ps128(Xy3));
      sums4 = _mm_add_ps(sums4, _mm256_extractf128_ps(Xy3, 1));
   }
   if (nvy > 4) {
      sums4 = _mm_add_ps(sums4, _mm256_castps256_ps128(Xy4));
      sums4 = _mm_add_ps(sums4, _mm256_extractf128_ps(Xy4, 1));
   }
   if (nvy > 5) {
      sums4 = _mm_add_ps(sums4, _mm256_castps256_ps128(Xy5));
      sums4 = _mm_add_ps(sums4, _mm256_extractf128_ps(Xy5, 1));
   }
   sums4 = _mm_add_ps(sums4, _mm_shuffle_ps(sums4,sums4,_MM_SHUFFLE(1,0,3,2)));
   sums4 = _mm_add_ps(sums4, _mm_shuffle_ps(sums4,sums4,_MM_SHUFFLE(2,3,0,1)));
   sum = _mm_cvtss_f32(sums4);

   xy8 = yy8 = zero8;
   pulsesLeft = K;

   /* -------- Phase B: pyramid pre-search -------- */
   if (K > (N >> 1))
   {
      __m256 rcp8;
      __m128 xy4, yy4;
      __m128i ps4;
      if (!(sum > EPSILON && sum < 64))
      {
         Xy0 = _mm256_setr_ps(QCONST16(1.f,14),0,0,0,0,0,0,0);
         Xy1=Xy2=Xy3=Xy4=Xy5=zero8;
         sums4 = _mm_set_ps1(1.f);
      }
      {
         __m128 rcp4 = _mm_mul_ps(_mm_set_ps1((float)(K + .8)), _mm_rcp_ps(sums4));
         rcp8 = _mm256_broadcastss_ps(rcp4);
      }
      ps4 = _mm_setzero_si128();
      xy4 = yy4 = zero;
#define YPYR(Xk, Yk, add_first) do {                                   \
         __m256i ii = _mm256_cvttps_epi32(_mm256_mul_ps(Xk, rcp8));     \
         __m256  yv = _mm256_cvtepi32_ps(ii);                          \
         __m256  xyk= _mm256_mul_ps(Xk, yv);                           \
         __m256  yyk= _mm256_mul_ps(yv, yv);                           \
         __m128i il = _mm256_castsi256_si128(ii), ih=_mm256_extractf128_si256(ii,1); \
         __m128  xl = _mm256_castps256_ps128(xyk), xh=_mm256_extractf128_ps(xyk,1);  \
         __m128  yl = _mm256_castps256_ps128(yyk), yh=_mm256_extractf128_ps(yyk,1);  \
         if (add_first) { ps4=il; } else { ps4=_mm_add_epi32(ps4, il); }            \
         ps4=_mm_add_epi32(ps4, ih);                                   \
         xy4=_mm_add_ps(xy4, xl); xy4=_mm_add_ps(xy4, xh);             \
         yy4=_mm_add_ps(yy4, yl); yy4=_mm_add_ps(yy4, yh);             \
         Yk = _mm256_add_ps(yv, yv);                                   \
      } while(0)
      YPYR(Xy0, Yy0, 1);
      if (nvy>1) YPYR(Xy1, Yy1, 0);
      if (nvy>2) YPYR(Xy2, Yy2, 0);
      if (nvy>3) YPYR(Xy3, Yy3, 0);
      if (nvy>4) YPYR(Xy4, Yy4, 0);
      if (nvy>5) YPYR(Xy5, Yy5, 0);
#undef YPYR
      ps4 = _mm_add_epi32(ps4, _mm_shuffle_epi32(ps4,_MM_SHUFFLE(1,0,3,2)));
      ps4 = _mm_add_epi32(ps4, _mm_shuffle_epi32(ps4,_MM_SHUFFLE(2,3,0,1)));
      pulsesLeft -= _mm_cvtsi128_si32(ps4);
      xy4 = _mm_add_ps(xy4, _mm_shuffle_ps(xy4,xy4,_MM_SHUFFLE(1,0,3,2)));
      xy4 = _mm_add_ps(xy4, _mm_shuffle_ps(xy4,xy4,_MM_SHUFFLE(2,3,0,1)));
      yy4 = _mm_add_ps(yy4, _mm_shuffle_ps(yy4,yy4,_MM_SHUFFLE(1,0,3,2)));
      yy4 = _mm_add_ps(yy4, _mm_shuffle_ps(yy4,yy4,_MM_SHUFFLE(2,3,0,1)));
      /* xy4/yy4 hold the sum in all 4 lanes; broadcast straight to 8 lanes. */
      xy8 = _mm256_broadcastss_ps(xy4);
      yy8 = _mm256_broadcastss_ps(yy4);
   }

   /* -------- Phase C: pad lanes >= N with X=-100, y=100 -------- */
   if (pad)
   {
      const __m256i nm1 = _mm256_set1_epi32(N - 1);
      const __m256 nh = _mm256_set1_ps(-100.f), hh = _mm256_set1_ps(100.f);
      __m256 pm;
      pm = _mm256_castsi256_ps(_mm256_cmpgt_epi32(ix0, nm1));
      Xy0 = _mm256_blendv_ps(Xy0, nh, pm);
      Yy0 = _mm256_blendv_ps(Yy0, hh, pm);
      if (nvy > 1) {
         pm = _mm256_castsi256_ps(_mm256_cmpgt_epi32(ix1, nm1));
         Xy1 = _mm256_blendv_ps(Xy1, nh, pm);
         Yy1 = _mm256_blendv_ps(Yy1, hh, pm);
      }
      if (nvy > 2) {
         pm = _mm256_castsi256_ps(_mm256_cmpgt_epi32(ix2, nm1));
         Xy2 = _mm256_blendv_ps(Xy2, nh, pm);
         Yy2 = _mm256_blendv_ps(Yy2, hh, pm);
      }
      if (nvy > 3) {
         pm = _mm256_castsi256_ps(_mm256_cmpgt_epi32(ix3, nm1));
         Xy3 = _mm256_blendv_ps(Xy3, nh, pm);
         Yy3 = _mm256_blendv_ps(Yy3, hh, pm);
      }
      if (nvy > 4) {
         pm = _mm256_castsi256_ps(_mm256_cmpgt_epi32(ix4, nm1));
         Xy4 = _mm256_blendv_ps(Xy4, nh, pm);
         Yy4 = _mm256_blendv_ps(Yy4, hh, pm);
      }
      if (nvy > 5) {
         pm = _mm256_castsi256_ps(_mm256_cmpgt_epi32(ix5, nm1));
         Xy5 = _mm256_blendv_ps(Xy5, nh, pm);
         Yy5 = _mm256_blendv_ps(Yy5, hh, pm);
      }
   }

   /* -------- Phase D: silence fallback (rare) -------- */
   if (pulsesLeft > N + 3)
   {
      opus_val16 tmp = (opus_val16)pulsesLeft;
      /* yy += tmp*tmp; yy += tmp*Yy0[0].  This TU is built at the -mavx2
         baseline (which also enables -mfma), so a plain c + a*b would be
         contracted into a single-rounding FMA and diverge from the SSE2
         reference.  Compute it as scalar _mm_mul_ss/_mm_add_ss (opaque, never
         fused) so the two roundings match the reference bit-for-bit. */
      __m128 yys = _mm256_castps256_ps128(yy8);
      __m128 ts  = _mm_set_ss((float)tmp);
      yys = _mm_add_ss(yys, _mm_mul_ss(ts, ts));
      yys = _mm_add_ss(yys, _mm_mul_ss(ts, _mm256_castps256_ps128(Yy0)));
      yy  = _mm_cvtss_f32(yys);
      yy8 = _mm256_set1_ps(yy);
      extra0 = pulsesLeft;
      pulsesLeft = 0;
   }

   /* -------- Phase E: greedy pulse placement (payload-carrying argmax) --------
    * Scores are computed 8-wide (one YMM rsqrt+mul) then folded into a 4-lane
    * running argmax with payload (xsel/ysel) in lo/hi block order for the
    * bit-exact tie-break.  The xy/yy update broadcasts from the kB payload
    * register. */
   {
   const __m256 one8 = _mm256_set1_ps(1.f);
   for (i = 0; i < pulsesLeft; i++)
   {
      __m128 max4, xsel, ysel, rl, rh, nb;
      __m128i pos4, kA, kAlo, kAhi, kBlo, kBhi, kB, m, ksw, bb;
      int best_id;

      yy8 = _mm256_add_ps(yy8, one8);
      max4 = zero; pos4 = _mm_setzero_si128(); xsel = zero; ysel = zero;

#define ACC(rh_, Xh_, Yh_, cnt_) do {                     \
         nb   = _mm_cmpgt_ps(rh_, max4);                   \
         pos4 = pvq_blendv_epi8(pos4, cnt_, _mm_castps_si128(nb)); \
         xsel = pvq_blendv_ps(xsel, Xh_, nb);              \
         ysel = pvq_blendv_ps(ysel, Yh_, nb);             \
         max4 = _mm_max_ps(max4, rh_);                     \
      } while(0)
#define YSCAN(Xk, Yk, ixk) do {                                             \
         __m256 r8 = _mm256_mul_ps(_mm256_add_ps(Xk, xy8),                   \
                                   _mm256_rsqrt_ps(_mm256_add_ps(Yk, yy8))); \
         rl = _mm256_castps256_ps128(r8); rh = _mm256_extractf128_ps(r8,1);  \
         ACC(rl, _mm256_castps256_ps128(Xk), _mm256_castps256_ps128(Yk), _mm256_castsi256_si128(ixk)); \
         ACC(rh, _mm256_extractf128_ps(Xk,1), _mm256_extractf128_ps(Yk,1), _mm256_extractf128_si256(ixk,1)); \
      } while(0)
      YSCAN(Xy0, Yy0, ix0);
      if (nvy>1) YSCAN(Xy1, Yy1, ix1);
      if (nvy>2) YSCAN(Xy2, Yy2, ix2);
      if (nvy>3) YSCAN(Xy3, Yy3, ix3);
      if (nvy>4) YSCAN(Xy4, Yy4, ix4);
      if (nvy>5) YSCAN(Xy5, Yy5, ix5);
#undef YSCAN
#undef ACC

      /* 4-lane horizontal argmax with payload */
      kAlo = _mm_unpacklo_epi32(pos4, _mm_castps_si128(max4));
      kAhi = _mm_unpackhi_epi32(pos4, _mm_castps_si128(max4));
      kBlo = _mm_unpacklo_epi32(_mm_castps_si128(ysel), _mm_castps_si128(xsel));
      kBhi = _mm_unpackhi_epi32(_mm_castps_si128(ysel), _mm_castps_si128(xsel));
      m  = pvq_cmpgt_epi64(kAlo, kAhi);
      kA = pvq_blendv_epi8(kAhi, kAlo, m);
      kB = pvq_blendv_epi8(kBhi, kBlo, m);
      ksw = _mm_shuffle_epi32(kA, _MM_SHUFFLE(1,0,3,2));
      m  = pvq_cmpgt_epi64(kA, ksw);
      kA = pvq_blendv_epi8(ksw, kA, m);
      kB = pvq_blendv_epi8(_mm_shuffle_epi32(kB,_MM_SHUFFLE(1,0,3,2)), kB, m);
      best_id = _mm_cvtsi128_si32(kA);

      /* update xy/yy from payload (all register, no memory) */
      yy8 = _mm256_add_ps(yy8, _mm256_broadcastss_ps(_mm_castsi128_ps(kB)));
      xy8 = _mm256_add_ps(xy8, _mm256_broadcastss_ps(_mm_castsi128_ps(_mm_shuffle_epi32(kB,_MM_SHUFFLE(1,1,1,1)))));

      /* y[best] += 2 in-register */
      bb = _mm_set1_epi32(best_id);
      {
         __m256i bb8 = _mm256_broadcastd_epi32(bb);
         __m256 mm;
         mm = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ix0, bb8));
         Yy0 = _mm256_add_ps(Yy0, _mm256_and_ps(mm, two8));
         if (nvy > 1) {
            mm = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ix1, bb8));
            Yy1 = _mm256_add_ps(Yy1, _mm256_and_ps(mm, two8));
         }
         if (nvy > 2) {
            mm = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ix2, bb8));
            Yy2 = _mm256_add_ps(Yy2, _mm256_and_ps(mm, two8));
         }
         if (nvy > 3) {
            mm = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ix3, bb8));
            Yy3 = _mm256_add_ps(Yy3, _mm256_and_ps(mm, two8));
         }
         if (nvy > 4) {
            mm = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ix4, bb8));
            Yy4 = _mm256_add_ps(Yy4, _mm256_and_ps(mm, two8));
         }
         if (nvy > 5) {
            mm = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ix5, bb8));
            Yy5 = _mm256_add_ps(Yy5, _mm256_and_ps(mm, two8));
         }
      }
   }
   yy = _mm_cvtss_f32(_mm256_castps256_ps128(yy8));
   }

   /* -------- Phase F: iy = y/2 (+ silence extra), sign restore, store --------
    * The final vector uses a masked store when pad != 0 so we never write
    * beyond the caller's N+3-element iy allocation. */
   {
      __m256i e0 = _mm256_setr_epi32(extra0,0,0,0,0,0,0,0);
      __m256i ic, s8, res;
      /* Unrolled per last-vector case; a masked store keeps writes within iy. */
      ic = _mm256_add_epi32(_mm256_cvttps_epi32(_mm256_mul_ps(Yy0, half8)), e0);
      s8 = _mm256_loadu_si256((const __m256i*)(const void*)&sgv[0]);
      res = _mm256_xor_si256(_mm256_add_epi32(ic, s8), s8);
      if (pad && nvy <= 1)
         _mm256_maskstore_epi32((int*)&iy[0], _mm256_cmpgt_epi32(nm8, ix0), res);
      else
         _mm256_storeu_si256((__m256i*)(void*)&iy[0], res);
      if (nvy > 1) {
         ic = _mm256_cvttps_epi32(_mm256_mul_ps(Yy1, half8));
         s8 = _mm256_loadu_si256((const __m256i*)(const void*)&sgv[8]);
         res = _mm256_xor_si256(_mm256_add_epi32(ic, s8), s8);
         if (pad && nvy <= 2)
            _mm256_maskstore_epi32((int*)&iy[8],
                                   _mm256_cmpgt_epi32(nm8, ix1), res);
         else
            _mm256_storeu_si256((__m256i*)(void*)&iy[8], res);
      }
      if (nvy > 2) {
         ic = _mm256_cvttps_epi32(_mm256_mul_ps(Yy2, half8));
         s8 = _mm256_loadu_si256((const __m256i*)(const void*)&sgv[16]);
         res = _mm256_xor_si256(_mm256_add_epi32(ic, s8), s8);
         if (pad && nvy <= 3)
            _mm256_maskstore_epi32((int*)&iy[16],
                                   _mm256_cmpgt_epi32(nm8, ix2), res);
         else
            _mm256_storeu_si256((__m256i*)(void*)&iy[16], res);
      }
      if (nvy > 3) {
         ic = _mm256_cvttps_epi32(_mm256_mul_ps(Yy3, half8));
         s8 = _mm256_loadu_si256((const __m256i*)(const void*)&sgv[24]);
         res = _mm256_xor_si256(_mm256_add_epi32(ic, s8), s8);
         if (pad && nvy <= 4)
            _mm256_maskstore_epi32((int*)&iy[24],
                                   _mm256_cmpgt_epi32(nm8, ix3), res);
         else
            _mm256_storeu_si256((__m256i*)(void*)&iy[24], res);
      }
      if (nvy > 4) {
         ic = _mm256_cvttps_epi32(_mm256_mul_ps(Yy4, half8));
         s8 = _mm256_loadu_si256((const __m256i*)(const void*)&sgv[32]);
         res = _mm256_xor_si256(_mm256_add_epi32(ic, s8), s8);
         if (pad && nvy <= 5)
            _mm256_maskstore_epi32((int*)&iy[32],
                                   _mm256_cmpgt_epi32(nm8, ix4), res);
         else
            _mm256_storeu_si256((__m256i*)(void*)&iy[32], res);
      }
      if (nvy > 5) {
         ic = _mm256_cvttps_epi32(_mm256_mul_ps(Yy5, half8));
         s8 = _mm256_loadu_si256((const __m256i*)(const void*)&sgv[40]);
         res = _mm256_xor_si256(_mm256_add_epi32(ic, s8), s8);
         if (pad && nvy <= 6)
            _mm256_maskstore_epi32((int*)&iy[40],
                                   _mm256_cmpgt_epi32(nm8, ix5), res);
         else
            _mm256_storeu_si256((__m256i*)(void*)&iy[40], res);
      }
   }

   return yy;
}

#endif /* !FIXED_POINT && OPUS_X86_MAY_HAVE_AVX2 */
