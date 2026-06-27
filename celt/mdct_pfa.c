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

#include "mdct.h"
#include "_kiss_fft_guts.h"
#include "stack_alloc.h"
#include "mathops.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef kiss_fft_cpx cpx;

static const int celt_tx_pfa_P[15] = {2, 5, 7, 14, 4, 1, 13, 3, 10, 12, 0, 9, 11, 6, 8};

#ifdef FIXED_POINT
#ifdef ENABLE_QEXT
#define SIN_2PI_3  1859775393L /* 0.866025388 * 2147483648 */
#define COS_2PI_5  663608942L  /* 0.309017003 * 2147483648 */
#define SIN_2PI_5  2042378317L /* 0.951056540 * 2147483648 */
#define COS_4PI_5  1737350766L /* 0.809016994 * 2147483648 */
#define SIN_4PI_5  1262259218L /* 0.587785252 * 2147483648 */
#define HALF_VAL   1073741824L /* 0.5 * 2147483648 */
#else
#define SIN_2PI_3  28378  /* 0.866025388 * 32768 */
#define COS_2PI_5  10126  /* 0.309017003 * 32768 */
#define SIN_2PI_5  31164  /* 0.951056540 * 32768 */
#define COS_4PI_5  26509  /* 0.809016994 * 32768 */
#define SIN_4PI_5  19261  /* 0.587785252 * 32768 */
#define HALF_VAL   16384  /* 0.5 * 32768 */
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
#define HALF_VAL   0.5f
#define PFA_DOWNSHIFT(x, N, total, step) (void)(x); (void)(N); (void)(total); (void)(step)
#endif

static OPUS_INLINE cpx crot(cpx a) {
   return (cpx){-a.i, a.r};
}

#include "celt_tx.h"


#include <stddef.h>
#include "celt_tx_tables.h"
#include "celt_tx_twiddles.h"


static void celt_tx_fft_pfa_15xM_ns_c(const struct OpusTXContext *s, void *out, void *in, ptrdiff_t stride ARG_FIXED(int downshift));
static void get_pfa_crt_params(int M, int *K3, int *K4);


void celt_tx_fft_p2_c(cpx *out, const cpx *in, int N ARG_FIXED(int *downshift_ptr)) {
   int i, j, len, half_len;
#ifndef FIXED_POINT
   int downshift_val = 0;
   int *downshift_ptr = &downshift_val;
   (void)downshift_ptr;
#endif
   SAVE_STACK;
   VARDECL(cpx, buf);
   ALLOC(buf, N, cpx);

   /* Bit-reversal permutation */
   int log2N = 0;
   for (i = N >> 1; i > 0; i >>= 1) log2N++;

   for (i = 0; i < N; i++) {
      int rev = 0;
      for (j = 0; j < log2N; j++) {
         if ((i >> j) & 1)
            rev |= (1 << (log2N - 1 - j));
      }
      buf[rev] = in[i];
   }

   /* Cooley-Tukey butterfly stages using pre-baked twiddle table */
   /* We use C_MULC to multiply by conjugate twiddles, computing DFT directly */
   for (len = 2; len <= N; len <<= 1) {
      half_len = len >> 1;
      int tw_offset = half_len - 1;
      PFA_DOWNSHIFT(buf, N, downshift_ptr, 1);
      for (i = 0; i < N; i += len) {
         for (j = 0; j < half_len; j++) {
            cpx w = p2_twiddle_table[tw_offset + j];
            cpx u = buf[i + j];
            cpx t;
            C_MULC(t, buf[i + j + half_len], w);
            C_ADD(buf[i + j], u, t);
            C_SUB(buf[i + j + half_len], u, t);
         }
      }
   }
   PFA_DOWNSHIFT(buf, N, downshift_ptr, *downshift_ptr);

   for (i = 0; i < N; i++) out[i] = buf[i];
   RESTORE_STACK;
}

/*
 * 15-point Good-Thomas Prime Factor Algorithm (PFA) DFT core.
 * Mathematically identical to FFT15_CORE from celt_tx_neon.S.
 */
static void winograd_fft3(const cpx *in, cpx *out) {
   kiss_fft_scalar r_sum12, r_diff12, i_sum12, i_diff12;

   r_sum12  = ADD32_ovflw(in[1].r, in[2].r);
   r_diff12 = SUB32_ovflw(in[1].r, in[2].r);
   i_sum12  = ADD32_ovflw(in[1].i, in[2].i);
   i_diff12 = SUB32_ovflw(in[1].i, in[2].i);

   out[0].r = ADD32_ovflw(in[0].r, r_sum12);
   out[0].i = ADD32_ovflw(in[0].i, i_sum12);

   kiss_fft_scalar t1_r, t1_i, t2_r, t2_i;
   t1_r = S_MUL(i_diff12, SIN_2PI_3);
   t1_i = S_MUL(r_diff12, SIN_2PI_3);
   t2_r = S_MUL(r_sum12, HALF_VAL);
   t2_i = S_MUL(i_sum12, HALF_VAL);

   out[1].r = ADD32_ovflw(SUB32_ovflw(in[0].r, t2_r), t1_r);
   out[1].i = SUB32_ovflw(SUB32_ovflw(in[0].i, t2_i), t1_i);

   out[2].r = SUB32_ovflw(SUB32_ovflw(in[0].r, t2_r), t1_r);
   out[2].i = ADD32_ovflw(SUB32_ovflw(in[0].i, t2_i), t1_i);
}

static void decl_fft5(const cpx *in, int r3, cpx *out, int stride) {
   cpx dc = in[0];

   kiss_fft_scalar r_sum14, r_diff14, i_sum14, i_diff14;
   kiss_fft_scalar r_sum23, r_diff23, i_sum23, i_diff23;

   r_diff14 = SUB32_ovflw(in[1].r, in[4].r);
   r_sum14  = ADD32_ovflw(in[1].r, in[4].r);
   i_diff14 = SUB32_ovflw(in[1].i, in[4].i);
   i_sum14  = ADD32_ovflw(in[1].i, in[4].i);

   r_diff23 = SUB32_ovflw(in[2].r, in[3].r);
   r_sum23  = ADD32_ovflw(in[2].r, in[3].r);
   i_diff23 = SUB32_ovflw(in[2].i, in[3].i);
   i_sum23  = ADD32_ovflw(in[2].i, in[3].i);

   int idx0 = (5 * r3) % 15;
   if (idx0 < 0) idx0 += 15;
   out[idx0 * stride].r = ADD32_ovflw(dc.r, ADD32_ovflw(r_sum14, r_sum23));
   out[idx0 * stride].i = ADD32_ovflw(dc.i, ADD32_ovflw(i_sum14, i_sum23));

   kiss_fft_scalar r_t4, r_t0, i_t4, i_t0;
   r_t4 = SUB32_ovflw(S_MUL(r_sum14, COS_2PI_5), S_MUL(r_sum23, COS_4PI_5));
   r_t0 = SUB32_ovflw(S_MUL(r_sum23, COS_2PI_5), S_MUL(r_sum14, COS_4PI_5));
   i_t4 = SUB32_ovflw(S_MUL(i_sum14, COS_2PI_5), S_MUL(i_sum23, COS_4PI_5));
   i_t0 = SUB32_ovflw(S_MUL(i_sum23, COS_2PI_5), S_MUL(i_sum14, COS_4PI_5));

   kiss_fft_scalar r_t5, r_t1, i_t5, i_t1;
   r_t5 = ADD32_ovflw(S_MUL(i_diff14, SIN_2PI_5), S_MUL(i_diff23, SIN_4PI_5));
   r_t1 = SUB32_ovflw(S_MUL(i_diff14, SIN_4PI_5), S_MUL(i_diff23, SIN_2PI_5));
   i_t5 = NEG32_ovflw(ADD32_ovflw(S_MUL(r_diff14, SIN_2PI_5), S_MUL(r_diff23, SIN_4PI_5)));
   i_t1 = SUB32_ovflw(S_MUL(r_diff23, SIN_2PI_5), S_MUL(r_diff14, SIN_4PI_5));

   int idx1 = (5 * r3 + 3) % 15; if (idx1 < 0) idx1 += 15;
   int idx2 = (5 * r3 + 6) % 15; if (idx2 < 0) idx2 += 15;
   int idx3 = (5 * r3 + 9) % 15; if (idx3 < 0) idx3 += 15;
   int idx4 = (5 * r3 + 12) % 15; if (idx4 < 0) idx4 += 15;

   out[idx1 * stride].r = ADD32_ovflw(dc.r, ADD32_ovflw(r_t4, r_t5));
   out[idx1 * stride].i = ADD32_ovflw(dc.i, ADD32_ovflw(i_t4, i_t5));

   out[idx2 * stride].r = ADD32_ovflw(dc.r, ADD32_ovflw(r_t0, r_t1));
   out[idx2 * stride].i = ADD32_ovflw(dc.i, ADD32_ovflw(i_t0, i_t1));

   out[idx3 * stride].r = ADD32_ovflw(dc.r, SUB32_ovflw(r_t0, r_t1));
   out[idx3 * stride].i = ADD32_ovflw(dc.i, SUB32_ovflw(i_t0, i_t1));

   out[idx4 * stride].r = ADD32_ovflw(dc.r, SUB32_ovflw(r_t4, r_t5));
   out[idx4 * stride].i = ADD32_ovflw(dc.i, SUB32_ovflw(i_t4, i_t5));
}

static void celt_tx_fft15_c(const cpx *in, cpx *out, int stride) {
   cpx tmp[15];

   for (int c5 = 0; c5 < 5; c5++) {
      cpx in_col[3];
      cpx out_col[3];
      for (int r3 = 0; r3 < 3; r3++) {
         int r_pfa = celt_tx_pfa_P[(10 * r3 + 6 * c5) % 15];
         in_col[r3] = in[r_pfa];
      }
      winograd_fft3(in_col, out_col);
      tmp[c5] = out_col[0];
      tmp[c5 + 5] = out_col[1];
      tmp[c5 + 10] = out_col[2];
   }

   decl_fft5(tmp, 0, out, stride);
   decl_fft5(tmp + 5, 1, out, stride);
   decl_fft5(tmp + 10, 2, out, stride);
}

static void celt_tx_fft_pfa_15xM_ns_c(const struct OpusTXContext *s, void *out, void *in, ptrdiff_t stride ARG_FIXED(int downshift)) {
   int i, j;
   int len = s->len;
   int M = s->sub->len;
#ifndef FIXED_POINT
   int downshift = 0;
   (void)downshift;
#endif
   cpx *tmp = (cpx *)s->tmp;
   const cpx *in_cpx = (const cpx *)in;
   cpx *out_cpx = (cpx *)out;
   (void)stride;

   PFA_DOWNSHIFT((cpx*)in, len, &downshift, 3);
   for (i = 0; i < M; i++) {
      celt_tx_fft15_c(in_cpx + 15 * i, tmp + i, M);
   }
   PFA_DOWNSHIFT(tmp, len, &downshift, 2);

   for (j = 0; j < 15; j++) {
#ifdef FIXED_POINT
      int sub_shift = downshift;
      celt_tx_fft_p2_c(tmp + j * M, tmp + j * M, M ARG_FIXED(&sub_shift));
      if (j == 14) downshift = sub_shift;
#else
      celt_tx_fft_p2_c(tmp + j * M, tmp + j * M, M);
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

static const struct OpusTXContext celt_tx_mdct_120_c  = {  120, 1, celt_tx_mdct_map_120,  (const void *)celt_tx_mdct_exp_120,  NULL, &celt_tx_pfa_60_c,  (void *)celt_tx_fft_pfa_15xM_ns_c };
static const struct OpusTXContext celt_tx_mdct_240_c  = {  240, 1, celt_tx_mdct_map_240,  (const void *)celt_tx_mdct_exp_240,  NULL, &celt_tx_pfa_120_c, (void *)celt_tx_fft_pfa_15xM_ns_c };
static const struct OpusTXContext celt_tx_mdct_480_c  = {  480, 1, celt_tx_mdct_map_480,  (const void *)celt_tx_mdct_exp_480,  NULL, &celt_tx_pfa_240_c, (void *)celt_tx_fft_pfa_15xM_ns_c };
static const struct OpusTXContext celt_tx_mdct_960_c  = {  960, 1, celt_tx_mdct_map_960,  (const void *)celt_tx_mdct_exp_960,  NULL, &celt_tx_pfa_480_c, (void *)celt_tx_fft_pfa_15xM_ns_c };
#if defined(ENABLE_QEXT)
static const struct OpusTXContext celt_tx_mdct_1920_c = { 1920, 1, celt_tx_mdct_map_1920, (const void *)celt_tx_mdct_exp_1920, NULL, &celt_tx_pfa_960_c, (void *)celt_tx_fft_pfa_15xM_ns_c };
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

static void celt_tx_mdct_inv_c(const struct OpusTXContext *s, kiss_fft_scalar *out, const kiss_fft_scalar *in, int stride ARG_FIXED(int pre_shift) ARG_FIXED(int fft_shift) ARG_FIXED(int post_shift)) {
   int i;
   int N2 = s->len;
   int N4 = N2 >> 1;
   cpx *z = (cpx *)out;
   const kiss_twiddle_scalar *trig = (const kiss_twiddle_scalar *)s->exp;
   (void)stride;

   for (i = 0; i < N4; i++) {
      int k = s->map[i];
      int j = k >> 1;
      kiss_fft_scalar im = in[k * stride];
      kiss_fft_scalar re = in[(N2 - 1 - k) * stride];
      kiss_twiddle_scalar trig_r = trig[2 * j + 1]; /* -sin(theta_j) */
      kiss_twiddle_scalar trig_i = trig[2 * j];     /*  cos(theta_j) */
#ifdef FIXED_POINT
      re = SHL32_ovflw(re, pre_shift);
      im = SHL32_ovflw(im, pre_shift);
      z[i].r = SUB32_ovflw(S_MUL(im, trig_i), S_MUL(re, trig_r));
      z[i].i = ADD32_ovflw(S_MUL(re, trig_i), S_MUL(im, trig_r));
#else
      z[i] = (cpx){im * trig_i - re * trig_r, re * trig_i + im * trig_r};
#endif
   }

   if (s->fn == (void *)celt_tx_fft_pfa_15xM_ns_c) {
      celt_tx_fft_pfa_15xM_ns_c(s->sub, z, z, 1 ARG_FIXED(fft_shift));
   } else {
      celt_tx_fft_p2_c(z, z, N4 ARG_FIXED(&fft_shift));
   }

   for (i = 0; i < N4 / 2; i++) {
      int i0 = i;
      int i1 = N4 - 1 - i;
      cpx z0 = z[i0];
      cpx z1 = z[i1];
      kiss_twiddle_scalar e0_r = trig[2 * i0 + 1];
      kiss_twiddle_scalar e0_i = trig[2 * i0];
      kiss_twiddle_scalar e1_r = trig[2 * i1 + 1];
      kiss_twiddle_scalar e1_i = trig[2 * i1];
#ifdef FIXED_POINT
      kiss_fft_scalar r0_r = PSHR32_ovflw(ADD32_ovflw(S_MUL(z0.i, e0_i), S_MUL(z0.r, e0_r)), post_shift);
      kiss_fft_scalar r0_i = PSHR32_ovflw(SUB32_ovflw(S_MUL(z0.i, e0_r), S_MUL(z0.r, e0_i)), post_shift);
      kiss_fft_scalar r1_r = PSHR32_ovflw(ADD32_ovflw(S_MUL(z1.i, e1_i), S_MUL(z1.r, e1_r)), post_shift);
      kiss_fft_scalar r1_i = PSHR32_ovflw(SUB32_ovflw(S_MUL(z1.i, e1_r), S_MUL(z1.r, e1_i)), post_shift);
      z[i0] = (cpx){r0_r, r1_i};
      z[i1] = (cpx){r1_r, r0_i};
#else
      cpx r0 = (cpx){z0.i * e0_i + z0.r * e0_r, z0.i * e0_r - z0.r * e0_i};
      cpx r1 = (cpx){z1.i * e1_i + z1.r * e1_r, z1.i * e1_r - z1.r * e1_i};
      z[i0] = (cpx){r0.r, r1.i};
      z[i1] = (cpx){r1.r, r0.i};
#endif
   }
}

void clt_mdct_backward_pfa_c(const mdct_lookup *l, kiss_fft_scalar *in,
      kiss_fft_scalar * OPUS_RESTRICT out,
      const celt_coef * OPUS_RESTRICT window,
      int overlap, int shift, int stride, int arch)
{
   int i;
   int N = l->n >> shift;
   int N2 = N >> 1;
   const struct OpusTXContext *tpl = celt_tx_mdct_kernel_c(N2);
   (void)arch;

   if (tpl == NULL) {
#if defined(CUSTOM_MODES) || defined(ENABLE_OPUS_CUSTOM_API)
      clt_mdct_backward_c(l, in, out, window, overlap, shift, stride, arch);
      return;
#else
      celt_assert2(0, "PFA IMDCT called with unsupported size in non-custom mode");
#endif
   }

   SAVE_STACK;
   const kiss_twiddle_scalar *trig = l->trig;
   int cur_n = l->n;
   for (i = 0; i < shift; i++) {
      cur_n >>= 1;
      trig += cur_n;
   }


   int pre_shift = 0;
   int post_shift = 0;
   int fft_shift = 0;
#ifdef FIXED_POINT
   {
      opus_val32 sumval = N2;
      opus_val32 maxval = 0;
      for (i = 0; i < N2; i++) {
         maxval = MAX32(maxval, ABS32(in[i * stride]));
         sumval = ADD32_ovflw(sumval, ABS32(SHR32(in[i * stride], 11)));
      }
      pre_shift = IMAX(0, 29 - celt_zlog2(1 + maxval));
      post_shift = IMAX(0, 19 - celt_ilog2(ABS32(sumval)));
      post_shift = IMIN(post_shift, pre_shift);
      fft_shift = pre_shift - post_shift;
   }
#else
   (void)pre_shift;
   (void)post_shift;
   (void)fft_shift;
#endif

   if (tpl->sub != NULL) {
      VARDECL(cpx, tmp);
      ALLOC(tmp, N2 / 2, cpx);
      struct OpusTXContext mdct = *tpl;
      struct OpusTXContext pfa = *tpl->sub;
      pfa.tmp = tmp;
      mdct.sub = &pfa;
      mdct.exp = trig;
      celt_tx_mdct_inv_c(&mdct, out + (overlap >> 1), in, stride ARG_FIXED(pre_shift) ARG_FIXED(fft_shift) ARG_FIXED(post_shift));
   } else {
      struct OpusTXContext mdct = *tpl;
      mdct.exp = trig;
      celt_tx_mdct_inv_c(&mdct, out + (overlap >> 1), in, stride ARG_FIXED(pre_shift) ARG_FIXED(fft_shift) ARG_FIXED(post_shift));
   }

   {
      kiss_fft_scalar * OPUS_RESTRICT xp1 = out + overlap - 1;
      kiss_fft_scalar * OPUS_RESTRICT yp1 = out;
      const celt_coef * OPUS_RESTRICT wp1 = window;
      const celt_coef * OPUS_RESTRICT wp2 = window + overlap - 1;

      for (i = 0; i < overlap / 2; i++) {
         kiss_fft_scalar x1 = *xp1;
         kiss_fft_scalar x2 = *yp1;
         *yp1++ = SUB32_ovflw(S_MUL(x2, *wp2), S_MUL(x1, *wp1));
         *xp1-- = ADD32_ovflw(S_MUL(x2, *wp1), S_MUL(x1, *wp2));
         wp1++;
         wp2--;
      }
   }
   RESTORE_STACK;
}

static void celt_tx_mdct_fwd_c(const struct OpusTXContext *s, kiss_fft_scalar *out, const kiss_fft_scalar *in,
      const celt_coef *window, int overlap, int stride ARG_FIXED(celt_coef scale) ARG_FIXED(int scale_shift)) {
   int i;
   int N2 = s->len;
   int N4 = N2 >> 1;
   SAVE_STACK;
   VARDECL(kiss_fft_scalar, f);
   VARDECL(cpx, z);
   ALLOC(f, N2, kiss_fft_scalar);
   ALLOC(z, N4, cpx);
   const kiss_twiddle_scalar *trig = (const kiss_twiddle_scalar *)s->exp;
#ifdef FIXED_POINT
   int headroom = 0;
#endif

   /* Window, shuffle, fold */
   {
      const kiss_fft_scalar * OPUS_RESTRICT xp1 = in + (overlap >> 1);
      const kiss_fft_scalar * OPUS_RESTRICT xp2 = in + N2 - 1 + (overlap >> 1);
      kiss_fft_scalar * OPUS_RESTRICT yp = f;
      const celt_coef * OPUS_RESTRICT wp1 = window + (overlap >> 1);
      const celt_coef * OPUS_RESTRICT wp2 = window + (overlap >> 1) - 1;
      for (i = 0; i < ((overlap + 3) >> 2); i++) {
         *yp++ = ADD32_ovflw(S_MUL(xp1[N2], *wp2), S_MUL(*xp2, *wp1));
         *yp++ = SUB32_ovflw(S_MUL(*xp1, *wp1), S_MUL(xp2[-N2], *wp2));
         xp1 += 2; xp2 -= 2; wp1 += 2; wp2 -= 2;
      }
      wp1 = window;
      wp2 = window + overlap - 1;
      for (; i < N4 - ((overlap + 3) >> 2); i++) {
         *yp++ = *xp2;
         *yp++ = *xp1;
         xp1 += 2; xp2 -= 2;
      }
      for (; i < N4; i++) {
         *yp++ = ADD32_ovflw(S_MUL(-xp1[-N2], *wp1), S_MUL(*xp2, *wp2));
         *yp++ = ADD32_ovflw(S_MUL(*xp1, *wp2), S_MUL(xp2[N2], *wp1));
         xp1 += 2; xp2 -= 2; wp1 += 2; wp2 -= 2;
      }
   }

   /* Pre-rotation and scaling */
   {
#ifndef FIXED_POINT
      float scale = 1.0f / N4;
#else
      opus_val32 maxval = 1;
#endif
      for (i = 0; i < N4; i++) {
         int k = s->map[i];
         int j = k >> 1;
         kiss_twiddle_scalar t0 = trig[2 * j];
         kiss_twiddle_scalar t1 = trig[2 * j + 1];
         kiss_fft_scalar re = f[k];
         kiss_fft_scalar im = f[k + 1];
#ifdef FIXED_POINT
         kiss_fft_scalar yr = SUB32_ovflw(S_MUL(re, t0), S_MUL(im, t1));
         kiss_fft_scalar yi = ADD32_ovflw(S_MUL(im, t0), S_MUL(re, t1));
#ifdef ENABLE_QEXT
         z[i].r = yr;
         z[i].i = yi;
#else
         z[i].r = S_MUL2(yr, scale);
         z[i].i = S_MUL2(yi, scale);
#endif
         maxval = MAX32(maxval, MAX32(ABS32(z[i].r), ABS32(z[i].i)));
#else
         float yr = re * t0 - im * t1;
         float yi = im * t0 + re * t1;
         z[i] = (cpx){yr * scale, yi * scale};
#endif
      }
#ifdef FIXED_POINT
      headroom = IMAX(0, IMIN(scale_shift, 28 - celt_ilog2(maxval)));
#endif
   }
   /* Transform core */
#ifdef FIXED_POINT
   int fft_shift = scale_shift - headroom;
#endif
   if (s->fn == (void *)celt_tx_fft_pfa_15xM_ns_c) {
      celt_tx_fft_pfa_15xM_ns_c(s->sub, z, z, 1 ARG_FIXED(fft_shift));
   } else {
      celt_tx_fft_p2_c(z, z, N4 ARG_FIXED(&fft_shift));
   }

   /* Post-rotation */
   for (i = 0; i < N4; i++) {
      kiss_twiddle_scalar t0 = trig[2 * i];
      kiss_twiddle_scalar t1 = trig[2 * i + 1];
      cpx fp = z[i];
      kiss_fft_scalar fp_r = fp.r;
      kiss_fft_scalar fp_i = fp.i;
#ifdef FIXED_POINT
#ifdef ENABLE_QEXT
      t0 = S_MUL2(t0, scale);
      t1 = S_MUL2(t1, scale);
#endif
      kiss_fft_scalar yr = PSHR32(SUB32_ovflw(S_MUL(fp_i, t1), S_MUL(fp_r, t0)), headroom);
      kiss_fft_scalar yi = PSHR32(ADD32_ovflw(S_MUL(fp_r, t1), S_MUL(fp_i, t0)), headroom);
#else
      float yr = fp_i * t1 - fp_r * t0;
      float yi = fp_r * t1 + fp_i * t0;
#endif
      out[2 * i * stride] = yr;
      out[(N2 - 1 - 2 * i) * stride] = yi;
   }
   RESTORE_STACK;
}

void clt_mdct_forward_pfa_c(const mdct_lookup *l, kiss_fft_scalar *in,
      kiss_fft_scalar * OPUS_RESTRICT out,
      const celt_coef *window, int overlap, int shift, int stride, int arch) {
   int i;
   int N = l->n >> shift;
   int N2 = N >> 1;
   const struct OpusTXContext *tpl = celt_tx_mdct_kernel_c(N2);
   (void)arch;

   if (tpl == NULL) {
#if defined(CUSTOM_MODES) || defined(ENABLE_OPUS_CUSTOM_API)
      clt_mdct_forward_c(l, in, out, window, overlap, shift, stride, arch);
      return;
#else
      celt_assert2(0, "PFA MDCT called with unsupported size in non-custom mode");
#endif
   }

   SAVE_STACK;
   const kiss_twiddle_scalar *trig = l->trig;
   int cur_n = l->n;
   for (i = 0; i < shift; i++) {
      cur_n >>= 1;
      trig += cur_n;
   }

   const kiss_fft_state *st = l->kfft[shift];
   int scale_shift = 0;
   celt_coef scale = 0;
#ifdef FIXED_POINT
   scale_shift = st->scale_shift - 1;
   scale = st->scale;
#else
   (void)st;
   (void)scale_shift;
   (void)scale;
#endif

   if (tpl->sub != NULL) {
      VARDECL(cpx, tmp);
      ALLOC(tmp, N2 / 2, cpx);
      struct OpusTXContext mdct = *tpl;
      struct OpusTXContext pfa = *tpl->sub;
      pfa.tmp = tmp;
      mdct.sub = &pfa;
      mdct.exp = trig;
      celt_tx_mdct_fwd_c(&mdct, out, in, window, overlap, stride ARG_FIXED(scale) ARG_FIXED(scale_shift));
   } else {
      struct OpusTXContext mdct = *tpl;
      mdct.exp = trig;
      celt_tx_mdct_fwd_c(&mdct, out, in, window, overlap, stride ARG_FIXED(scale) ARG_FIXED(scale_shift));
   }
   RESTORE_STACK;
}

#if defined(OPUS_USE_PFA_MDCT)
static void get_pfa_crt_params(int M, int *K3, int *K4)
{
   switch (M) {
      case 4:  *K3 = 3;  *K4 = 4;  break;
      case 8:  *K3 = 7;  *K4 = 2;  break;
      case 16: *K3 = 15; *K4 = 1;  break;
      case 32: *K3 = 15; *K4 = 8;  break;
      case 64: *K3 = 47; *K4 = 4;  break;
      default: *K3 = 0;  *K4 = 0;  break;
   }
}

void opus_fft_pfa_c(const kiss_fft_state *st, const kiss_fft_cpx *fin, kiss_fft_cpx *fout)
{
   int i;
   int nfft = st->nfft;
   int M = nfft / 15;
   int K3, K4;
   const struct OpusTXContext *mdct_tpl = celt_tx_mdct_kernel_c(2 * nfft);
   const struct OpusTXContext *tpl = mdct_tpl ? mdct_tpl->sub : NULL;

   SAVE_STACK;
   VARDECL(cpx, tmp);
   VARDECL(cpx, in_perm);
   ALLOC(tmp, nfft, cpx);
   ALLOC(in_perm, nfft, cpx);

   get_pfa_crt_params(M, &K3, &K4);

   /* 1. Scale and permute input to grid order: dest = c + r * M */
   celt_coef scale = st->scale;
   for (i = 0; i < nfft; i++) {
      int r = celt_tx_pfa_P[(i * K4) % 15];
      int c = (i * K3) % M;
      int dest = r + c * 15;
#ifdef FIXED_POINT
      in_perm[dest].r = S_MUL2(fin[i].r, scale);
      in_perm[dest].i = S_MUL2(fin[i].i, scale);
#else
      in_perm[dest].r = fin[i].r * scale;
      in_perm[dest].i = fin[i].i * scale;
#endif
   }

   /* 2. Run the PFA core forward FFT */
   struct OpusTXContext pfa;
   if (tpl == NULL) {
      RESTORE_STACK;
      return;
   }
   pfa = *tpl;
   pfa.tmp = tmp;

   int downshift = 0;
#ifdef FIXED_POINT
   downshift = st->scale_shift - 1;
#endif
   celt_tx_fft_pfa_15xM_ns_c(&pfa, fout, in_perm, 1 ARG_FIXED(downshift));

   RESTORE_STACK;
}

void opus_ifft_pfa_c(const kiss_fft_state *st, const kiss_fft_cpx *fin, kiss_fft_cpx *fout)
{
   int i;
   int nfft = st->nfft;
   int M = nfft / 15;
   int K3, K4;
   const struct OpusTXContext *mdct_tpl = celt_tx_mdct_kernel_c(2 * nfft);
   const struct OpusTXContext *tpl = mdct_tpl ? mdct_tpl->sub : NULL;

   SAVE_STACK;
   VARDECL(cpx, tmp);
   VARDECL(cpx, in_perm);
   ALLOC(tmp, nfft, cpx);
   ALLOC(in_perm, nfft, cpx);

   get_pfa_crt_params(M, &K3, &K4);

   /* 1. Permute input to grid order (IDFT input is pre-scaled by caller) */
   for (i = 0; i < nfft; i++) {
      int r = celt_tx_pfa_P[(i * K4) % 15];
      int c = (i * K3) % M;
      int dest = r + c * 15;
      in_perm[dest] = fin[i];
   }

   /* 2. Run the PFA core forward FFT (downshift = 0) */
   struct OpusTXContext pfa;
   if (tpl == NULL) {
      RESTORE_STACK;
      return;
   }
   pfa = *tpl;
   pfa.tmp = tmp;
   celt_tx_fft_pfa_15xM_ns_c(&pfa, fout, in_perm, 1 ARG_FIXED(0));

   /* 3. Time-reverse the output from index 1 to N-1 to obtain inverse DFT */
   for (i = 1; i < (nfft + 1) / 2; i++) {
      cpx t = fout[i];
      fout[i] = fout[nfft - i];
      fout[nfft - i] = t;
   }

   RESTORE_STACK;
}
#endif
