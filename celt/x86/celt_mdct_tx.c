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
#include "x86/mdct_x86.h"

#if defined(OPUS_X86_TX_MDCT)

#include <stddef.h>
#include "celt_tx.h"
#include "celt_tx_tables.h"


/* Declare SSE functions */
void celt_tx_fft4_fwd_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
void celt_tx_fft8_ns_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
void celt_tx_fft16_ns_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
void celt_tx_fft32_ns_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
void celt_tx_fft_sr_ns_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
void celt_tx_fft_pfa_15xM_ns_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
void celt_tx_mdct_inv_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);

/* M-point power-of-two sub-FFTs. */
static const OpusTXContext celt_tx_p2_4   = {  4, 1, celt_tx_p2_map_4,  NULL, NULL, NULL, NULL };
static const OpusTXContext celt_tx_p2_8   = {  8, 1, celt_tx_p2_map_8,  NULL, NULL, NULL, NULL };
static const OpusTXContext celt_tx_p2_16  = { 16, 1, celt_tx_p2_map_16, NULL, NULL, NULL, NULL };
static const OpusTXContext celt_tx_p2_32  = { 32, 1, celt_tx_p2_map_32, NULL, NULL, NULL, NULL };
#if defined(ENABLE_QEXT)
static const OpusTXContext celt_tx_p2_64  = { 64, 1, celt_tx_p2_map_64, NULL, NULL, NULL, NULL };
#endif

/* 15xM PFA sub-FFTs */
static const OpusTXContext celt_tx_pfa_60  = {  60, 1, celt_tx_pfa_map_60,  NULL, NULL, &celt_tx_p2_4,  celt_tx_fft4_fwd_float_sse };
static const OpusTXContext celt_tx_pfa_120 = { 120, 1, celt_tx_pfa_map_120, NULL, NULL, &celt_tx_p2_8,  celt_tx_fft8_ns_float_sse };
static const OpusTXContext celt_tx_pfa_240 = { 240, 1, celt_tx_pfa_map_240, NULL, NULL, &celt_tx_p2_16, celt_tx_fft16_ns_float_sse };
static const OpusTXContext celt_tx_pfa_480 = { 480, 1, celt_tx_pfa_map_480, NULL, NULL, &celt_tx_p2_32, celt_tx_fft32_ns_float_sse };
#if defined(ENABLE_QEXT)
static const OpusTXContext celt_tx_pfa_960 = { 960, 1, celt_tx_pfa_map_960, NULL, NULL, &celt_tx_p2_64, celt_tx_fft_sr_ns_float_sse };
#endif

#if defined(CUSTOM_MODES) && 0 /* Disabled: requires fft_sr_ns_sse */
static const OpusTXContext celt_tx_sr_32  = {  32, 1, NULL, NULL, NULL, NULL, NULL };
static const OpusTXContext celt_tx_sr_64  = {  64, 1, NULL, NULL, NULL, NULL, NULL };
static const OpusTXContext celt_tx_sr_128 = { 128, 1, NULL, NULL, NULL, NULL, NULL };
static const OpusTXContext celt_tx_sr_256 = { 256, 1, NULL, NULL, NULL, NULL, NULL };
static const OpusTXContext celt_tx_sr_512 = { 512, 1, NULL, NULL, NULL, NULL, NULL };
#endif

/* Inverse MDCT roots */
static const OpusTXContext celt_tx_mdct_120  = {  120, 1, celt_tx_mdct_map_120,  (const void *)celt_tx_mdct_exp_120,  NULL, &celt_tx_pfa_60,  celt_tx_fft_pfa_15xM_ns_float_sse };
static const OpusTXContext celt_tx_mdct_240  = {  240, 1, celt_tx_mdct_map_240,  (const void *)celt_tx_mdct_exp_240,  NULL, &celt_tx_pfa_120, celt_tx_fft_pfa_15xM_ns_float_sse };
static const OpusTXContext celt_tx_mdct_480  = {  480, 1, celt_tx_mdct_map_480,  (const void *)celt_tx_mdct_exp_480,  NULL, &celt_tx_pfa_240, celt_tx_fft_pfa_15xM_ns_float_sse };
static const OpusTXContext celt_tx_mdct_960  = {  960, 1, celt_tx_mdct_map_960,  (const void *)celt_tx_mdct_exp_960,  NULL, &celt_tx_pfa_480, celt_tx_fft_pfa_15xM_ns_float_sse };
#if defined(ENABLE_QEXT)
static const OpusTXContext celt_tx_mdct_1920 = { 1920, 1, celt_tx_mdct_map_1920, (const void *)celt_tx_mdct_exp_1920, NULL, &celt_tx_pfa_960, celt_tx_fft_pfa_15xM_ns_float_sse };
#endif

#if defined(CUSTOM_MODES) && 0 /* Disabled: requires fft_sr_ns_sse */
static const OpusTXContext celt_tx_mdct_64   = {   64, 1, celt_tx_mdct_map_64,   (const void *)celt_tx_mdct_exp_64,   NULL, &celt_tx_sr_32,  celt_tx_fft32_ns_float_sse };
static const OpusTXContext celt_tx_mdct_128  = {  128, 1, celt_tx_mdct_map_128,  (const void *)celt_tx_mdct_exp_128,  NULL, &celt_tx_sr_64,  celt_tx_fft_sr_ns_float_sse };
static const OpusTXContext celt_tx_mdct_256  = {  256, 1, celt_tx_mdct_map_256,  (const void *)celt_tx_mdct_exp_256,  NULL, &celt_tx_sr_128, celt_tx_fft_sr_ns_float_sse };
static const OpusTXContext celt_tx_mdct_512  = {  512, 1, celt_tx_mdct_map_512,  (const void *)celt_tx_mdct_exp_512,  NULL, &celt_tx_sr_256, celt_tx_fft_sr_ns_float_sse };
static const OpusTXContext celt_tx_mdct_1024 = { 1024, 1, celt_tx_mdct_map_1024, (const void *)celt_tx_mdct_exp_1024, NULL, &celt_tx_sr_512, celt_tx_fft_sr_ns_float_sse };
#endif

static const OpusTXContext *celt_tx_mdct_kernel(int len)
{
   switch (len) {
      case  120: return &celt_tx_mdct_120;
      case  240: return &celt_tx_mdct_240;
      case  480: return &celt_tx_mdct_480;
      case  960: return &celt_tx_mdct_960;
#if defined(ENABLE_QEXT)
      case 1920: return &celt_tx_mdct_1920;
#endif
#if defined(CUSTOM_MODES) && 0 /* Disabled: requires fft_sr_ns_sse */
      case   64: return &celt_tx_mdct_64;
      case  128: return &celt_tx_mdct_128;
      case  256: return &celt_tx_mdct_256;
      case  512: return &celt_tx_mdct_512;
      case 1024: return &celt_tx_mdct_1024;
#endif
      default:   return NULL;
   }
}

void clt_mdct_backward_sse(const mdct_lookup *l, kiss_fft_scalar *in,
      kiss_fft_scalar * OPUS_RESTRICT out,
      const celt_coef * OPUS_RESTRICT window,
      int overlap, int shift, int stride, int arch)
{
   int i;
   int N = l->n >> shift;
   int N2 = N >> 1;
   const OpusTXContext *tpl = celt_tx_mdct_kernel(N2);
    
   if (tpl == NULL)
   {
#if defined(CUSTOM_MODES) || defined(ENABLE_OPUS_CUSTOM_API)
      clt_mdct_backward_c(l, in, out, window, overlap, shift, stride, arch);
      return;
#else
      celt_assert2(0, "PFA IMDCT called with unsupported size in non-custom mode");
#endif
   }
   (void)arch;

   if (tpl->fn == celt_tx_fft_pfa_15xM_ns_float_sse)
   {
      OpusTXContext mdct, pfa;
      VARDECL(kiss_fft_scalar, tmp);
      SAVE_STACK;
      ALLOC(tmp, N2, kiss_fft_scalar);   /* N/4 complex values */

      mdct = *tpl;
      pfa = *tpl->sub;
      pfa.tmp = tmp;
      mdct.sub = &pfa;
      celt_tx_mdct_inv_float_sse(&mdct, out+(overlap>>1), in,
                                  (ptrdiff_t)stride*sizeof(kiss_fft_scalar));
      RESTORE_STACK;
   } else {
      celt_tx_mdct_inv_float_sse(tpl, out+(overlap>>1), in,
                                  (ptrdiff_t)stride*sizeof(kiss_fft_scalar));
   }

   /* Mirror on both sides for TDAC (same as mdct.c) */
   {
      kiss_fft_scalar * OPUS_RESTRICT xp1 = out+overlap-1;
      kiss_fft_scalar * OPUS_RESTRICT yp1 = out;
      const celt_coef * OPUS_RESTRICT wp1 = window;
      const celt_coef * OPUS_RESTRICT wp2 = window+overlap-1;

      for(i = 0; i < overlap/2; i++)
      {
         kiss_fft_scalar x1, x2;
         x1 = *xp1;
         x2 = *yp1;
         *yp1++ = SUB32_ovflw(S_MUL(x2, *wp2), S_MUL(x1, *wp1));
         *xp1-- = ADD32_ovflw(S_MUL(x2, *wp1), S_MUL(x1, *wp2));
         wp1++;
         wp2--;
      }
   }
}

#endif /* OPUS_X86_TX_MDCT */
