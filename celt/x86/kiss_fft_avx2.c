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

#include "x86/kiss_fft_avx2.h"

#ifdef KISS_FFT_HAVE_AVX2

#include "x86/float_avx2.h"
#include "_kiss_fft_guts.h"

/* The butterflies below are vectorised over the inner loop of their C
   counterpart in kiss_fft.c (four values of j/u at a time), except the
   radix-4 with m==1 which is vectorised over i. */

/* Mask flipping the sign of the imaginary parts. */
static OPUS_INLINE __m256 neg_imag(void)
{
   return _mm256_setr_ps(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f);
}

/* x - i*y = (x.r + y.i, x.i - y.r), where x.i - y.r is computed as
   x.i + (-y.r), which is exact. */
static OPUS_INLINE __m256 sub_mul_i(__m256 x, __m256 y)
{
   return _mm256_add_ps(x, _mm256_xor_ps(CPX_SWAP_AVX2(y), neg_imag()));
}

/* x + i*y = (x.r - y.i, x.i + y.r). */
static OPUS_INLINE __m256 add_mul_i(__m256 x, __m256 y)
{
   return _mm256_addsub_ps(x, CPX_SWAP_AVX2(y));
}

/* C_MUL(a, w[0]), C_MUL(a, w[s]), C_MUL(a, w[2*s]), C_MUL(a, w[3*s]). */
static OPUS_INLINE __m256 twiddle4(__m256 a, const kiss_twiddle_cpx *w, int s)
{
   __m256 tw = cpx_gather4_avx2((const float*)w, s);
   return cpx_rot_avx2(a, _mm256_moveldup_ps(tw), _mm256_movehdup_ps(tw));
}

#define LOAD(p) _mm256_loadu_ps((const float*)(p))
#define STORE(p, v) _mm256_storeu_ps((float*)(p), (v))

/* Radix-2 with m==4, the only case the C code supports for non-custom
   modes. Each iteration is one butterfly: Fout[0..3] with Fout[4..7]. The
   four twiddles are 1, (1-i)/sqrt(2), -i and -(1+i)/sqrt(2). */
static void kf_bfly2_m4(kiss_fft_cpx *Fout, int N)
{
   int i;
   const __m256 tw = _mm256_set1_ps(.7071067812f);
   for (i=0;i<N;i++)
   {
      __m256 a = LOAD(Fout);
      __m256 b = LOAD(Fout + 4);
      __m256 bs = CPX_SWAP_AVX2(b);
      /* Build t, the twiddled b, from these candidates:
         sum = (r+i, i+r), dif = (r-i, i-r), fid = (i-r, r-i). */
      __m256 sum = _mm256_add_ps(b, bs);
      __m256 dif = _mm256_sub_ps(b, bs);
      __m256 fid = _mm256_sub_ps(bs, b);
      /* t1 = (r+i, i-r)*tw and t3 = (i-r, -(i+r))*tw, in the upper half. */
      __m256 t13 = _mm256_blend_ps(_mm256_blend_ps(sum, dif, 0xAA),
            _mm256_blend_ps(fid, _mm256_xor_ps(sum, neg_imag()), 0xAA), 0xF0);
      /* t0 = (r, i) and t2 = (i, -r). */
      __m256 t02 = _mm256_blend_ps(b, _mm256_xor_ps(bs, neg_imag()), 0xF0);
      __m256 t = _mm256_blend_ps(t02, _mm256_mul_ps(t13, tw), 0xCC);
      STORE(Fout + 4, _mm256_sub_ps(a, t));
      STORE(Fout, _mm256_add_ps(a, t));
      Fout += 8;
   }
}

/* Core of the radix-4 butterfly, on four butterflies at a time. f1, f2 and
   f3 are already twiddled. */
static OPUS_INLINE void bfly4_core(__m256 *f0, __m256 *f1, __m256 *f2, __m256 *f3)
{
   __m256 s5 = _mm256_sub_ps(*f0, *f2);
   __m256 a = _mm256_add_ps(*f0, *f2);
   __m256 s3 = _mm256_add_ps(*f1, *f3);
   __m256 s4 = _mm256_sub_ps(*f1, *f3);
   *f2 = _mm256_sub_ps(a, s3);
   *f0 = _mm256_add_ps(a, s3);
   *f1 = sub_mul_i(s5, s4);
   *f3 = add_mul_i(s5, s4);
}

/* Transposes a 4x4 matrix of complex values. */
static OPUS_INLINE void transpose4(__m256 *a, __m256 *b, __m256 *c, __m256 *d)
{
   __m256d ab_lo = _mm256_unpacklo_pd(_mm256_castps_pd(*a), _mm256_castps_pd(*b));
   __m256d ab_hi = _mm256_unpackhi_pd(_mm256_castps_pd(*a), _mm256_castps_pd(*b));
   __m256d cd_lo = _mm256_unpacklo_pd(_mm256_castps_pd(*c), _mm256_castps_pd(*d));
   __m256d cd_hi = _mm256_unpackhi_pd(_mm256_castps_pd(*c), _mm256_castps_pd(*d));
   *a = _mm256_castpd_ps(_mm256_permute2f128_pd(ab_lo, cd_lo, 0x20));
   *b = _mm256_castpd_ps(_mm256_permute2f128_pd(ab_hi, cd_hi, 0x20));
   *c = _mm256_castpd_ps(_mm256_permute2f128_pd(ab_lo, cd_lo, 0x31));
   *d = _mm256_castpd_ps(_mm256_permute2f128_pd(ab_hi, cd_hi, 0x31));
}

/* Radix-4 with m==1 (no twiddles): the N butterflies are on consecutive
   groups of four values, so do four butterflies at a time by transposing. */
static void kf_bfly4_m1(kiss_fft_cpx *Fout, int N)
{
   int i;
   for (i=0;i+4<=N;i+=4)
   {
      __m256 f0 = LOAD(Fout);
      __m256 f1 = LOAD(Fout + 4);
      __m256 f2 = LOAD(Fout + 8);
      __m256 f3 = LOAD(Fout + 12);
      transpose4(&f0, &f1, &f2, &f3);
      bfly4_core(&f0, &f1, &f2, &f3);
      transpose4(&f0, &f1, &f2, &f3);
      STORE(Fout, f0);
      STORE(Fout + 4, f1);
      STORE(Fout + 8, f2);
      STORE(Fout + 12, f3);
      Fout += 16;
   }
   for (;i<N;i++)
   {
      kiss_fft_cpx scratch0, scratch1;
      C_SUB(scratch0, *Fout, Fout[2]);
      C_ADDTO(*Fout, Fout[2]);
      C_ADD(scratch1, Fout[1], Fout[3]);
      C_SUB(Fout[2], *Fout, scratch1);
      C_ADDTO(*Fout, scratch1);
      C_SUB(scratch1, Fout[1], Fout[3]);
      Fout[1].r = scratch0.r + scratch1.i;
      Fout[1].i = scratch0.i - scratch1.r;
      Fout[3].r = scratch0.r - scratch1.i;
      Fout[3].i = scratch0.i + scratch1.r;
      Fout += 4;
   }
}

static void kf_bfly4_avx2(kiss_fft_cpx *Fout, int fstride,
      const kiss_fft_state *st, int m, int N, int mm)
{
   int i, j;
   const kiss_twiddle_cpx *tw = st->twiddles;
   for (i=0;i<N;i++)
   {
      kiss_fft_cpx *F = Fout + i*mm;
      for (j=0;j<m;j+=4)
      {
         __m256 f0 = LOAD(F + j);
         __m256 f1 = twiddle4(LOAD(F + m + j), tw + j*fstride, fstride);
         __m256 f2 = twiddle4(LOAD(F + 2*m + j), tw + 2*j*fstride, 2*fstride);
         __m256 f3 = twiddle4(LOAD(F + 3*m + j), tw + 3*j*fstride, 3*fstride);
         bfly4_core(&f0, &f1, &f2, &f3);
         STORE(F + j, f0);
         STORE(F + m + j, f1);
         STORE(F + 2*m + j, f2);
         STORE(F + 3*m + j, f3);
      }
   }
}

static void kf_bfly3_avx2(kiss_fft_cpx *Fout, int fstride,
      const kiss_fft_state *st, int m, int N, int mm)
{
   int i, j;
   const kiss_twiddle_cpx *tw = st->twiddles;
   const __m256 epi3i = _mm256_set1_ps(st->twiddles[fstride*m].i);
   const __m256 half = _mm256_set1_ps(.5f);
   for (i=0;i<N;i++)
   {
      kiss_fft_cpx *F = Fout + i*mm;
      for (j=0;j<m;j+=4)
      {
         __m256 f0 = LOAD(F + j);
         __m256 s1 = twiddle4(LOAD(F + m + j), tw + j*fstride, fstride);
         __m256 s2 = twiddle4(LOAD(F + 2*m + j), tw + 2*j*fstride, 2*fstride);
         __m256 s3 = _mm256_add_ps(s1, s2);
         __m256 s0 = _mm256_mul_ps(_mm256_sub_ps(s1, s2), epi3i);
         __m256 fm = _mm256_sub_ps(f0, _mm256_mul_ps(s3, half));
         STORE(F + j, _mm256_add_ps(f0, s3));
         STORE(F + m + j, add_mul_i(fm, s0));
         STORE(F + 2*m + j, sub_mul_i(fm, s0));
      }
   }
}

static void kf_bfly5_avx2(kiss_fft_cpx *Fout, int fstride,
      const kiss_fft_state *st, int m, int N, int mm)
{
   int i, u;
   const kiss_twiddle_cpx *tw = st->twiddles;
   const kiss_twiddle_cpx ya = st->twiddles[fstride*m];
   const kiss_twiddle_cpx yb = st->twiddles[fstride*2*m];
   const __m256 yar = _mm256_set1_ps(ya.r), yai = _mm256_set1_ps(ya.i);
   const __m256 ybr = _mm256_set1_ps(yb.r), ybi = _mm256_set1_ps(yb.i);
   for (i=0;i<N;i++)
   {
      kiss_fft_cpx *F = Fout + i*mm;
      for (u=0;u<m;u+=4)
      {
         __m256 s0 = LOAD(F + u);
         __m256 s1 = twiddle4(LOAD(F + m + u), tw + u*fstride, fstride);
         __m256 s2 = twiddle4(LOAD(F + 2*m + u), tw + 2*u*fstride, 2*fstride);
         __m256 s3 = twiddle4(LOAD(F + 3*m + u), tw + 3*u*fstride, 3*fstride);
         __m256 s4 = twiddle4(LOAD(F + 4*m + u), tw + 4*u*fstride, 4*fstride);
         __m256 s7 = _mm256_add_ps(s1, s4);
         __m256 s10 = _mm256_sub_ps(s1, s4);
         __m256 s8 = _mm256_add_ps(s2, s3);
         __m256 s9 = _mm256_sub_ps(s2, s3);
         __m256 s5, s6, s11, s12, a, b;

         STORE(F + u, _mm256_add_ps(s0, _mm256_add_ps(s7, s8)));

         s5 = _mm256_add_ps(s0, _mm256_add_ps(_mm256_mul_ps(s7, yar), _mm256_mul_ps(s8, ybr)));
         /* s6 = (a.i, -a.r) with a = s10*ya.i + s9*yb.i. */
         a = _mm256_add_ps(_mm256_mul_ps(s10, yai), _mm256_mul_ps(s9, ybi));
         s6 = _mm256_xor_ps(CPX_SWAP_AVX2(a), neg_imag());
         STORE(F + m + u, _mm256_sub_ps(s5, s6));
         STORE(F + 4*m + u, _mm256_add_ps(s5, s6));

         s11 = _mm256_add_ps(s0, _mm256_add_ps(_mm256_mul_ps(s7, ybr), _mm256_mul_ps(s8, yar)));
         /* s12 = (a.i, b.r) with a = s9*ya.i - s10*yb.i and b = -a computed
            as in the C code (not by negating a, to keep the sign of 0). */
         a = _mm256_sub_ps(_mm256_mul_ps(s9, yai), _mm256_mul_ps(s10, ybi));
         b = _mm256_sub_ps(_mm256_mul_ps(s10, ybi), _mm256_mul_ps(s9, yai));
         s12 = CPX_SWAP_AVX2(_mm256_blend_ps(a, b, 0x55));
         STORE(F + 2*m + u, _mm256_add_ps(s11, s12));
         STORE(F + 3*m + u, _mm256_sub_ps(s11, s12));
      }
   }
}

int kf_bfly_avx2(kiss_fft_cpx *Fout, int p, size_t fstride,
      const kiss_fft_state *st, int m, int N, int mm)
{
   if (p == 4 && m == 1)
      kf_bfly4_m1(Fout, N);
   else if (p == 2 && m == 4)
      kf_bfly2_m4(Fout, N);
   /* The other kernels need m to be a multiple of 4, which is always the
      case except for custom modes. */
   else if ((m&3) != 0)
      return 0;
   else if (p == 3)
      kf_bfly3_avx2(Fout, (int)fstride, st, m, N, mm);
   else if (p == 4)
      kf_bfly4_avx2(Fout, (int)fstride, st, m, N, mm);
   else if (p == 5)
      kf_bfly5_avx2(Fout, (int)fstride, st, m, N, mm);
   else
      return 0;
   return 1;
}

#endif
