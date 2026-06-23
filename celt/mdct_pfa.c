/* Copyright (c) 2026 Xiph.Org Foundation */
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

#if defined(ENABLE_PFA)

#include "mdct.h"
#include "arm/mdct_arm.h"
#if defined(OPUS_ARM_TX_MDCT) && defined(OPUS_ARM_PRESUME_NEON_INTR)
#define USE_ARM_TX_MDCT
#endif
#include "kiss_fft.h"
#include "_kiss_fft_guts.h"
#include "stack_alloc.h"
#include "mathops.h"
#include <math.h>


typedef kiss_fft_cpx cpx;


#ifdef FIXED_POINT
#ifdef ENABLE_QEXT
#define SIN_2PI_3  1859775393L /* 0.866025388 * 2147483648 */
#define COS_2PI_5  663608942L  /* 0.309017003 * 2147483648 */
#define SIN_2PI_5  2042378317L /* 0.951056540 * 2147483648 */
#define COS_4PI_5  1737350766L /* 0.809016994 * 2147483648 */
#define SIN_4PI_5  1262259218L /* 0.587785252 * 2147483648 */
#else
#define SIN_2PI_3  28378  /* 0.866025388 * 32768 */
#define COS_2PI_5  10126  /* 0.309017003 * 32768 */
#define SIN_2PI_5  31164  /* 0.951056540 * 32768 */
#define COS_4PI_5  26509  /* 0.809016994 * 32768 */
#define SIN_4PI_5  19261  /* 0.587785252 * 32768 */
#endif

static void pfa_downshift(cpx *x, int N, int *total, int step) {
   int i;
   int shift = IMIN(step, *total);
   *total -= shift;
   if (shift == 1) {
      for (i = 0; i < N; i++) {
         x[i].r = SHR32(x[i].r, 1);
         x[i].i = SHR32(x[i].i, 1);
      }
   } else if (shift > 0) {
      for (i = 0; i < N; i++) {
         x[i].r = PSHR32(x[i].r, shift);
         x[i].i = PSHR32(x[i].i, shift);
      }
   }
}
#define PFA_DOWNSHIFT(x, N, total, step) pfa_downshift(x, N, total, step)
#else
#define SIN_2PI_3  0.866025388f
#define COS_2PI_5  0.309017003f
#define SIN_2PI_5  0.951056540f
#define COS_4PI_5  0.809016994f
#define SIN_4PI_5  0.587785252f
#define PFA_DOWNSHIFT(x, N, total, step) (void)(x); (void)(N); (void)(total); (void)(step)
#endif

typedef struct OpusTXContext OpusTXContext;
typedef void (*opus_tx_fn)(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride ARG_FIXED(int downshift));

struct OpusTXContext {
   opus_int32 len;
   opus_int32 inv;
   const opus_int16 *map;
   const void *exp;
   void *tmp;
   const struct OpusTXContext *sub;
   opus_tx_fn fn;
};

#include <stddef.h>
#include "celt_tx_tables.h"
static const opus_int16 p4[4]   = { 0, 2, 1, 3 };
static const opus_int16 p8[8]   = { 0, 4, 2, 6, 1, 5, 7, 3 };
static const opus_int16 p16[16] = { 0, 8, 4, 12, 2, 10, 14, 6, 1, 9, 5, 13, 15, 7, 3, 11 };
static const opus_int16 p32[32] = { 0, 16, 8, 24, 4, 20, 28, 12, 2, 18, 10, 26, 30, 14, 6, 22, 1, 17, 9, 25, 5, 21, 29, 13, 31, 15, 7, 23, 3, 19, 27, 11 };
static const opus_int16 p64[64] = { 0, 32, 16, 48, 8, 40, 56, 24, 4, 36, 20, 52, 60, 28, 12, 44, 2, 34, 18, 50, 10, 42, 58, 26, 62, 30, 14, 46, 6, 38, 54, 22, 1, 33, 17, 49, 9, 41, 57, 25, 5, 37, 21, 53, 61, 29, 13, 45, 63, 31, 15, 47, 7, 39, 55, 23, 3, 35, 19, 51, 59, 27, 11, 43 };

static OPUS_INLINE const opus_int16 *get_sr_perm_table(int M) {
   if (M == 4) return p4;
   if (M == 8) return p8;
   if (M == 16) return p16;
   if (M == 32) return p32;
   if (M == 64) return p64;
   return NULL;
}

#ifndef USE_ARM_TX_MDCT
static void celt_tx_fft_pfa_15xM_ns_c(const struct OpusTXContext *s, void *out, void *in, ptrdiff_t stride ARG_FIXED(int downshift));
#endif

#ifndef USE_ARM_TX_MDCT
#define BF(x, y, a, b) \
    do { \
        x = SUB32_ovflw(a, b); \
        y = ADD32_ovflw(a, b); \
    } while (0)

#define CMUL(dre, dim, are, aim, bre, bim) \
    do { \
        (dre) = SUB32_ovflw(S_MUL(are, bre), S_MUL(aim, bim)); \
        (dim) = ADD32_ovflw(S_MUL(are, bim), S_MUL(aim, bre)); \
    } while (0)

#define CMUL_CONJ(dre, dim, are, aim, bre, bim) \
    do { \
        (dre) = ADD32_ovflw(S_MUL(are, bre), S_MUL(aim, bim)); \
        (dim) = SUB32_ovflw(S_MUL(aim, bre), S_MUL(are, bim)); \
    } while (0)

#define BUTTERFLIES(a0, a1, a2, a3) \
    do { \
        r0 = a0.r; \
        i0 = a0.i; \
        r1 = a1.r; \
        i1 = a1.i; \
        BF(t3, t5, t5, t1); \
        BF(a2.r, a0.r, r0, t5); \
        BF(a3.i, a1.i, i1, t3); \
        BF(t4, t6, t2, t6); \
        BF(a3.r, a1.r, r1, t4); \
        BF(a2.i, a0.i, i0, t6); \
    } while (0)

#define TRANSFORM(a0, a1, a2, a3, wre, wim) \
    do { \
        CMUL_CONJ(t1, t2, a2.r, a2.i, wre, wim); \
        CMUL(t5, t6, a3.r, a3.i, wre, wim); \
        BUTTERFLIES(a0, a1, a2, a3); \
    } while (0)

static OPUS_INLINE void celt_tx_fft_sr_combine(cpx *z, const kiss_twiddle_scalar *cos, int len)
{
    int o1 = 2*len;
    int o2 = 4*len;
    int o3 = 6*len;
    const kiss_twiddle_scalar *wim = cos + o1 - 7;
    kiss_fft_scalar t1, t2, t3, t4, t5, t6, r0, i0, r1, i1;
    int i;

    for (i = 0; i < len; i += 4) {
        TRANSFORM(z[0], z[o1 + 0], z[o2 + 0], z[o3 + 0], cos[0], wim[7]);
        TRANSFORM(z[2], z[o1 + 2], z[o2 + 2], z[o3 + 2], cos[2], wim[5]);
        TRANSFORM(z[4], z[o1 + 4], z[o2 + 4], z[o3 + 4], cos[4], wim[3]);
        TRANSFORM(z[6], z[o1 + 6], z[o2 + 6], z[o3 + 6], cos[6], wim[1]);

        TRANSFORM(z[1], z[o1 + 1], z[o2 + 1], z[o3 + 1], cos[1], wim[6]);
        TRANSFORM(z[3], z[o1 + 3], z[o2 + 3], z[o3 + 3], cos[3], wim[4]);
        TRANSFORM(z[5], z[o1 + 5], z[o2 + 5], z[o3 + 5], cos[5], wim[2]);
        TRANSFORM(z[7], z[o1 + 7], z[o2 + 7], z[o3 + 7], cos[7], wim[0]);

        z   += 2*4;
        cos += 2*4;
        wim -= 2*4;
    }
}

static OPUS_INLINE void celt_tx_fft2(cpx *dst, const cpx *src)
{
    cpx tmp;
    BF(tmp.r, dst[0].r, src[0].r, src[1].r);
    BF(tmp.i, dst[0].i, src[0].i, src[1].i);
    dst[1] = tmp;
}

static OPUS_INLINE void celt_tx_fft4(cpx *dst, const cpx *src)
{
    kiss_fft_scalar t1, t2, t3, t4, t5, t6, t7, t8;

    BF(t3, t1, src[0].r, src[1].r);
    BF(t8, t6, src[3].r, src[2].r);
    BF(dst[2].r, dst[0].r, t1, t6);
    BF(t4, t2, src[0].i, src[1].i);
    BF(t7, t5, src[2].i, src[3].i);
    BF(dst[3].i, dst[1].i, t4, t8);
    BF(dst[3].r, dst[1].r, t3, t7);
    BF(dst[2].i, dst[0].i, t2, t5);
}

static OPUS_INLINE void celt_tx_fft8(cpx *dst, const cpx *src)
{
    kiss_fft_scalar t1, t2, t3, t4, t5, t6, r0, i0, r1, i1;
    kiss_twiddle_scalar cos = celt_tx_tab_32[4];

    celt_tx_fft4(dst, src);

    t1 = ADD32_ovflw(src[4].r, src[5].r);
    dst[5].r = SUB32_ovflw(src[4].r, src[5].r);
    t2 = ADD32_ovflw(src[4].i, src[5].i);
    dst[5].i = SUB32_ovflw(src[4].i, src[5].i);
    t5 = ADD32_ovflw(src[6].r, src[7].r);
    dst[7].r = SUB32_ovflw(src[6].r, src[7].r);
    t6 = ADD32_ovflw(src[6].i, src[7].i);
    dst[7].i = SUB32_ovflw(src[6].i, src[7].i);

    BUTTERFLIES(dst[0], dst[2], dst[4], dst[6]);
    TRANSFORM(dst[1], dst[3], dst[5], dst[7], cos, cos);
}

static OPUS_INLINE void celt_tx_fft16(cpx *dst, const cpx *src)
{
    kiss_fft_scalar t1, t2, t3, t4, t5, t6, r0, i0, r1, i1;
    kiss_twiddle_scalar cos_16_1 = celt_tx_tab_32[2];
    kiss_twiddle_scalar cos_16_2 = celt_tx_tab_32[4];
    kiss_twiddle_scalar cos_16_3 = celt_tx_tab_32[6];

    celt_tx_fft8(dst +  0, src +  0);
    celt_tx_fft4(dst +  8, src +  8);
    celt_tx_fft4(dst + 12, src + 12);

    t1 = dst[ 8].r;
    t2 = dst[ 8].i;
    t5 = dst[12].r;
    t6 = dst[12].i;
    BUTTERFLIES(dst[0], dst[4], dst[8], dst[12]);

    TRANSFORM(dst[ 2], dst[ 6], dst[10], dst[14], cos_16_2, cos_16_2);
    TRANSFORM(dst[ 1], dst[ 5], dst[ 9], dst[13], cos_16_1, cos_16_3);
    TRANSFORM(dst[ 3], dst[ 7], dst[11], dst[15], cos_16_3, cos_16_1);
}

static OPUS_INLINE void celt_tx_fft32(cpx *dst, const cpx *src)
{
    const kiss_twiddle_scalar *cos = celt_tx_tab_32;
    celt_tx_fft16(dst, src);
    celt_tx_fft8(dst + 16, src + 16);
    celt_tx_fft8(dst + 24, src + 24);
    celt_tx_fft_sr_combine(dst, cos, 4);
}

static OPUS_INLINE void celt_tx_fft64(cpx *dst, const cpx *src)
{
    const kiss_twiddle_scalar *cos = celt_tx_tab_64;
    celt_tx_fft32(dst, src);
    celt_tx_fft16(dst + 32, src + 32);
    celt_tx_fft16(dst + 48, src + 48);
    celt_tx_fft_sr_combine(dst, cos, 8);
}

static void celt_tx_fft_sr_c(cpx *dst, const cpx *src, int N ARG_FIXED(int *downshift_ptr))
{
#ifdef FIXED_POINT
   int stages = celt_ilog2(N);
   PFA_DOWNSHIFT((cpx*)src, N, downshift_ptr, stages);
#endif
   switch (N) {
      case   2: celt_tx_fft2(dst, src); break;
      case   4: celt_tx_fft4(dst, src); break;
      case   8: celt_tx_fft8(dst, src); break;
      case  16: celt_tx_fft16(dst, src); break;
      case  32: celt_tx_fft32(dst, src); break;
      case  64: celt_tx_fft64(dst, src); break;
      default: celt_assert2(0, "Unsupported Split-Radix FFT size");
   }
#ifdef FIXED_POINT
   PFA_DOWNSHIFT(dst, N, downshift_ptr, *downshift_ptr);
#endif
}

#undef BF
#undef CMUL
#undef CMUL_CONJ
#undef BUTTERFLIES
#undef TRANSFORM

/*
 * 15-point Good-Thomas Prime Factor Algorithm (PFA) DFT core.
 * Mathematically identical to FFT15_CORE from celt_tx_neon.S.
 */
static void winograd_fft3(const cpx *in0, const cpx *in1, const cpx *in2, cpx *out0, cpx *out1, cpx *out2) {
   kiss_fft_scalar r_sum12, r_diff12, i_sum12, i_diff12;
   kiss_fft_scalar t1_r, t1_i, t2_r, t2_i;

   r_sum12  = ADD32_ovflw(in1->r, in2->r);
   r_diff12 = SUB32_ovflw(in1->r, in2->r);
   i_sum12  = ADD32_ovflw(in1->i, in2->i);
   i_diff12 = SUB32_ovflw(in1->i, in2->i);

   out0->r = ADD32_ovflw(in0->r, r_sum12);
   out0->i = ADD32_ovflw(in0->i, i_sum12);

   t1_r = S_MUL(i_diff12, SIN_2PI_3);
   t1_i = S_MUL(r_diff12, SIN_2PI_3);
   t2_r = HALF32(r_sum12);
   t2_i = HALF32(i_sum12);

   out1->r = ADD32_ovflw(SUB32_ovflw(in0->r, t2_r), t1_r);
   out1->i = SUB32_ovflw(SUB32_ovflw(in0->i, t2_i), t1_i);

   out2->r = SUB32_ovflw(SUB32_ovflw(in0->r, t2_r), t1_r);
   out2->i = ADD32_ovflw(SUB32_ovflw(in0->i, t2_i), t1_i);
}

static OPUS_INLINE void decl_fft5(const cpx *in, int idx0, int idx1, int idx2, int idx3, int idx4, cpx *out, int stride) {
   cpx dc;
   kiss_fft_scalar r_sum14, r_diff14, i_sum14, i_diff14;
   kiss_fft_scalar r_sum23, r_diff23, i_sum23, i_diff23;
   kiss_fft_scalar r_t4, r_t0, i_t4, i_t0;
   kiss_fft_scalar r_t5, r_t1, i_t5, i_t1;
   int s0 = idx0 * stride;
   int s1 = idx1 * stride;
   int s2 = idx2 * stride;
   int s3 = idx3 * stride;
   int s4 = idx4 * stride;

   dc = in[0];

   r_diff14 = SUB32_ovflw(in[1].r, in[4].r);
   r_sum14  = ADD32_ovflw(in[1].r, in[4].r);
   i_diff14 = SUB32_ovflw(in[1].i, in[4].i);
   i_sum14  = ADD32_ovflw(in[1].i, in[4].i);

   r_diff23 = SUB32_ovflw(in[2].r, in[3].r);
   r_sum23  = ADD32_ovflw(in[2].r, in[3].r);
   i_diff23 = SUB32_ovflw(in[2].i, in[3].i);
   i_sum23  = ADD32_ovflw(in[2].i, in[3].i);

   out[s0].r = ADD32_ovflw(dc.r, ADD32_ovflw(r_sum14, r_sum23));
   out[s0].i = ADD32_ovflw(dc.i, ADD32_ovflw(i_sum14, i_sum23));

   r_t4 = SUB32_ovflw(S_MUL(r_sum14, COS_2PI_5), S_MUL(r_sum23, COS_4PI_5));
   r_t0 = SUB32_ovflw(S_MUL(r_sum23, COS_2PI_5), S_MUL(r_sum14, COS_4PI_5));
   i_t4 = SUB32_ovflw(S_MUL(i_sum14, COS_2PI_5), S_MUL(i_sum23, COS_4PI_5));
   i_t0 = SUB32_ovflw(S_MUL(i_sum23, COS_2PI_5), S_MUL(i_sum14, COS_4PI_5));

   r_t5 = ADD32_ovflw(S_MUL(i_diff14, SIN_2PI_5), S_MUL(i_diff23, SIN_4PI_5));
   r_t1 = SUB32_ovflw(S_MUL(i_diff14, SIN_4PI_5), S_MUL(i_diff23, SIN_2PI_5));
   i_t5 = NEG32_ovflw(ADD32_ovflw(S_MUL(r_diff14, SIN_2PI_5), S_MUL(r_diff23, SIN_4PI_5)));
   i_t1 = SUB32_ovflw(S_MUL(r_diff23, SIN_2PI_5), S_MUL(r_diff14, SIN_4PI_5));

   out[s1].r = ADD32_ovflw(dc.r, ADD32_ovflw(r_t4, r_t5));
   out[s1].i = ADD32_ovflw(dc.i, ADD32_ovflw(i_t4, i_t5));

   out[s2].r = ADD32_ovflw(dc.r, ADD32_ovflw(r_t0, r_t1));
   out[s2].i = ADD32_ovflw(dc.i, ADD32_ovflw(i_t0, i_t1));

   out[s3].r = ADD32_ovflw(dc.r, SUB32_ovflw(r_t0, r_t1));
   out[s3].i = ADD32_ovflw(dc.i, SUB32_ovflw(i_t0, i_t1));

   out[s4].r = ADD32_ovflw(dc.r, SUB32_ovflw(r_t4, r_t5));
   out[s4].i = ADD32_ovflw(dc.i, SUB32_ovflw(i_t4, i_t5));
}

static void celt_tx_fft15_c(const cpx *in, cpx *out, int stride) {
   cpx tmp[15];

   /* c5 = 0 */
   winograd_fft3(&in[2], &in[0], &in[1], &tmp[0], &tmp[5], &tmp[10]);

   /* c5 = 1 */
   winograd_fft3(&in[13], &in[5], &in[9], &tmp[1], &tmp[6], &tmp[11]);

   /* c5 = 2 */
   winograd_fft3(&in[11], &in[3], &in[7], &tmp[2], &tmp[7], &tmp[12]);

   /* c5 = 3 */
   winograd_fft3(&in[14], &in[6], &in[10], &tmp[3], &tmp[8], &tmp[13]);

   /* c5 = 4 */
   winograd_fft3(&in[12], &in[4], &in[8], &tmp[4], &tmp[9], &tmp[14]);

   decl_fft5(tmp, 0, 3, 6, 9, 12, out, stride);
   decl_fft5(tmp + 5, 5, 8, 11, 14, 2, out, stride);
   decl_fft5(tmp + 10, 10, 13, 1, 4, 7, out, stride);
}

static void celt_tx_fft_pfa_15xM_ns_c(const struct OpusTXContext *s, void *out, void *in, ptrdiff_t stride ARG_FIXED(int downshift)) {
   int i, j;
   int len = s->len;
   int M = s->sub->len;
   const opus_int16 *perm;
   cpx *tmp = (cpx *)s->tmp;
   const cpx *in_cpx = (const cpx *)in;
   cpx *out_cpx = (cpx *)out;
#ifndef FIXED_POINT
   int downshift = 0;
#endif

   (void)stride;
#ifndef FIXED_POINT
   (void)downshift;
#endif

   perm = get_sr_perm_table(M);
   celt_assert(perm != NULL);
   PFA_DOWNSHIFT((cpx*)in, len, &downshift, 3);
   for (i = 0; i < M; i++) {
      celt_tx_fft15_c(in_cpx + 15 * perm[i], tmp + i, M);
   }
   PFA_DOWNSHIFT(tmp, len, &downshift, 2);

   {
#ifdef FIXED_POINT
      int sub_shift = downshift;
#endif
      for (j = 0; j < 15; j++) {
         cpx *row = tmp + j * M;
#ifdef FIXED_POINT
         sub_shift = downshift;
#endif
         celt_tx_fft_sr_c(row, row, M ARG_FIXED(&sub_shift));
      }
#ifdef FIXED_POINT
      downshift = sub_shift;
#endif
   }

   for (i = 0; i < len; i++) {
      out_cpx[i] = tmp[s->map[i]];
   }
}

static const struct OpusTXContext celt_tx_p2_4_c   = {  4, 1, celt_tx_p2_map_4,  NULL, NULL, NULL, NULL };
static const struct OpusTXContext celt_tx_p2_8_c   = {  8, 1, celt_tx_p2_map_8,  NULL, NULL, NULL, NULL };
static const struct OpusTXContext celt_tx_p2_16_c  = { 16, 1, celt_tx_p2_map_16, NULL, NULL, NULL, NULL };
static const struct OpusTXContext celt_tx_p2_32_c  = { 32, 1, celt_tx_p2_map_32, NULL, NULL, NULL, NULL };
#if defined(ENABLE_QEXT)
static const struct OpusTXContext celt_tx_p2_64_c  = { 64, 1, celt_tx_p2_map_64, NULL, NULL, NULL, NULL };
#endif

static const struct OpusTXContext celt_tx_pfa_60_c  = {  60, 1, celt_tx_pfa_map_60,  NULL, NULL, &celt_tx_p2_4_c,  NULL };
static const struct OpusTXContext celt_tx_pfa_120_c = { 120, 1, celt_tx_pfa_map_120, NULL, NULL, &celt_tx_p2_8_c,  NULL };
static const struct OpusTXContext celt_tx_pfa_240_c = { 240, 1, celt_tx_pfa_map_240, NULL, NULL, &celt_tx_p2_16_c, NULL };
static const struct OpusTXContext celt_tx_pfa_480_c = { 480, 1, celt_tx_pfa_map_480, NULL, NULL, &celt_tx_p2_32_c, NULL };
#if defined(ENABLE_QEXT)
static const struct OpusTXContext celt_tx_pfa_960_c = { 960, 1, celt_tx_pfa_map_960, NULL, NULL, &celt_tx_p2_64_c, NULL };
#endif

static const struct OpusTXContext celt_tx_mdct_120_c  = {  120, 1, celt_tx_mdct_map_120,  NULL,  NULL, &celt_tx_pfa_60_c,  celt_tx_fft_pfa_15xM_ns_c };
static const struct OpusTXContext celt_tx_mdct_240_c  = {  240, 1, celt_tx_mdct_map_240,  NULL,  NULL, &celt_tx_pfa_120_c, celt_tx_fft_pfa_15xM_ns_c };
static const struct OpusTXContext celt_tx_mdct_480_c  = {  480, 1, celt_tx_mdct_map_480,  NULL,  NULL, &celt_tx_pfa_240_c, celt_tx_fft_pfa_15xM_ns_c };
static const struct OpusTXContext celt_tx_mdct_960_c  = {  960, 1, celt_tx_mdct_map_960,  NULL,  NULL, &celt_tx_pfa_480_c, celt_tx_fft_pfa_15xM_ns_c };
#if defined(ENABLE_QEXT)
static const struct OpusTXContext celt_tx_mdct_1920_c = { 1920, 1, celt_tx_mdct_map_1920, NULL, NULL, &celt_tx_pfa_960_c, celt_tx_fft_pfa_15xM_ns_c };
#endif

static const struct OpusTXContext *celt_tx_mdct_kernel_c(int len)
{
   switch (len) {
      case  120: return &celt_tx_mdct_120_c;
      case  240: return &celt_tx_mdct_240_c;
      case  480: return &celt_tx_mdct_480_c;
      case  960: return &celt_tx_mdct_960_c;
#if defined(ENABLE_QEXT)
      case 1920: return &celt_tx_mdct_1920_c;
#endif
      default:   return NULL;
   }
}
#endif

#if defined(ENABLE_PFA)

static OPUS_INLINE void pfa_copy_bitrev_input(const kiss_fft_state *st, const kiss_fft_cpx *fin, kiss_fft_cpx *fout)
{
   int i;
   int nfft = st->nfft;
   if (fin == fout) {
      VARDECL(kiss_fft_cpx, tmp_perm);
      SAVE_STACK;
      ALLOC(tmp_perm, nfft, kiss_fft_cpx);
      for (i = 0; i < nfft; i++)
         tmp_perm[i] = fin[i];
      for (i = 0; i < nfft; i++)
         fout[st->bitrev[i]] = tmp_perm[i];
      RESTORE_STACK;
   } else {
      for (i = 0; i < nfft; i++)
         fout[st->bitrev[i]] = fin[i];
   }
}

static OPUS_INLINE const struct OpusTXContext *get_pfa_context(int nfft, const struct OpusTXContext **mdct_tpl)
{
#if defined(USE_ARM_TX_MDCT)
   *mdct_tpl = celt_tx_mdct_kernel(2 * nfft);
   return *mdct_tpl ? (*mdct_tpl)->sub : NULL;
#else
   *mdct_tpl = celt_tx_mdct_kernel_c(2 * nfft);
   return (*mdct_tpl && (*mdct_tpl)->fn == celt_tx_fft_pfa_15xM_ns_c) ? (*mdct_tpl)->sub : NULL;
#endif
}

static void opus_pfa_impl(const kiss_fft_state *st, const kiss_fft_cpx *fin, kiss_fft_cpx *fout, int is_inverse ARG_FIXED(int shift))
{
   int i;
   int nfft = st->nfft;
   const opus_int16 *pfa_map;
   const struct OpusTXContext *mdct_tpl;
   const struct OpusTXContext *tpl = get_pfa_context(nfft, &mdct_tpl);
   struct OpusTXContext pfa;
   VARDECL(cpx, tmp);
   VARDECL(cpx, in_perm);
   SAVE_STACK;

#if !defined(ENABLE_PFA) || defined(CUSTOM_MODES) || defined(ENABLE_OPUS_CUSTOM_API) || defined(ENABLE_DEEP_PLC)
   if (tpl == NULL) {
      pfa_copy_bitrev_input(st, fin, fout);
      if (is_inverse) {
         for (i = 0; i < nfft; i++)
            fout[i].i = -fout[i].i;
      }
      opus_fft_impl(st, fout ARG_FIXED(shift));
      if (is_inverse) {
         for (i = 0; i < nfft; i++)
            fout[i].i = -fout[i].i;
      }
      RESTORE_STACK;
      return;
   }
#else
   celt_assert2(tpl != NULL, "PFA FFT/IFFT called with unsupported size in non-custom mode");
#endif

   ALLOC(tmp, nfft, cpx);
   ALLOC(in_perm, nfft, cpx);

   pfa_map = mdct_tpl->map + nfft;

   for (i = 0; i < nfft; i++)
      in_perm[pfa_map[i]] = fin[i];

   pfa = *tpl;
   pfa.tmp = tmp;

#if defined(USE_ARM_TX_MDCT)
   mdct_tpl->fn(&pfa, fout, in_perm, sizeof(kiss_fft_cpx) ARG_FIXED(shift));
   /* Time-reverse the output from index 1 to N-1 to obtain forward DFT
      because the Neon assembly computes Inverse DFT by default */
   if (!is_inverse) {
      for (i = 1; i < (nfft + 1) / 2; i++) {
         cpx t = fout[i];
         fout[i] = fout[nfft - i];
         fout[nfft - i] = t;
      }
   }
#else
   celt_tx_fft_pfa_15xM_ns_c(&pfa, fout, in_perm, 1 ARG_FIXED(shift));

   /* Time-reverse the output from index 1 to N-1 to obtain inverse DFT */
   if (is_inverse) {
      for (i = 1; i < (nfft + 1) / 2; i++) {
         cpx t = fout[i];
         fout[i] = fout[nfft - i];
         fout[nfft - i] = t;
      }
   }
#endif

   RESTORE_STACK;
}

void opus_fft_pfa_c(const kiss_fft_state *st, const kiss_fft_cpx *fin, kiss_fft_cpx *fout ARG_FIXED(int downshift))
{
   opus_pfa_impl(st, fin, fout, 0 ARG_FIXED(downshift));
}

void opus_ifft_pfa_c(const kiss_fft_state *st, const kiss_fft_cpx *fin, kiss_fft_cpx *fout ARG_FIXED(int fft_shift))
{
   opus_pfa_impl(st, fin, fout, 1 ARG_FIXED(fft_shift));
}
#endif

#endif /* ENABLE_PFA */
