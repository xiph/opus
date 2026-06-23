/* Copyright (c) 2026 Lynne */
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

/*
 * Glue between CELT's clt_mdct_backward() and the AArch64 NEON inverse MDCT
 * ported from FFmpeg's libavutil/tx (celt_tx_neon.S).
 *
 * The FFmpeg kernels are driven by a small context struct whose leading
 * fields mirror FFmpeg's AVTXContext (the assembly hard-codes the offsets;
 * see OpusTXContext below). FFmpeg builds these contexts and their lookup
 * tables at runtime in av_tx_init(); here everything is pre-baked in
 * celt_tx_tables.h for the CELT inverse MDCT sizes
 *
 *      N/2  = 120, 240, 480, 960 [, 1920 with ENABLE_QEXT]
 *      FFT  =  60, 120, 240, 480 [,  960]   = 15 * M,  M = 4 .. 64
 *
 * plus, under CUSTOM_MODES, the power-of-two custom mode sizes
 * N/2 = {64 .. 1024} (sub-FFT = fft32/fft_sr directly, no PFA stage).
 * Neither gated family is reachable otherwise: 96 kHz modes need qext and
 * custom frame sizes cap at 1024 (2048 under qext).
 *
 * Tables are dumped with scale = -1.0f, which makes the FFmpeg transform
 * numerically identical to clt_mdct_backward_c()'s pre-rotation + FFT +
 * post-rotation. The TDAC
 * mirror / windowing stays in C below (identical to mdct.c); the kernel
 * writes the N/2 raw samples to out[overlap/2 ..] exactly like the C code.
 *
 * The only mutable piece of FFmpeg's context tree is the PFA scratch buffer
 * (tmp). Mode data is shared between decoder instances in different
 * threads, so the two context levels that link to per-call state are copied
 * to the stack and pointed at stack scratch instead (a few dozen bytes).
 *
 * Sizes whose tables are not baked (or are compiled out) fall back to the
 * C path. The forward MDCT is not ported yet and always uses the C path.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mdct.h"
#include "_kiss_fft_guts.h"
#include "stack_alloc.h"

#if defined(OPUS_ARM_TX_MDCT)

#include <stddef.h>

typedef struct OpusTXContext OpusTXContext;
typedef void (*opus_tx_fn)(const OpusTXContext *s, void *out, void *in,
                           ptrdiff_t stride);

/* Mirror of the part of FFmpeg's AVTXContext the assembly reads. The field
   offsets are part of the asm ABI: len@0, inv@4, map@8, exp@16, tmp@24,
   sub@32, fn@40 (LP64). */
struct OpusTXContext {
   opus_int32 len;             /* Length of the transform */
   opus_int32 inv;             /* If transform is inverse */
   const opus_int16 *map;      /* Lookup table(s); int16 (max index 1920 < 2^15) */
   const void *exp;            /* Pre-baked multiplication factors */
   void *tmp;                  /* Temporary buffer, if needed */
   const OpusTXContext *sub;   /* Subtransform context */
   opus_tx_fn fn;              /* Function for the subtransform (fn[0]) */
};

/* The assembly hard-codes the offsets above; fail the build if the ABI
   assumption doesn't hold. */
typedef char opus_tx_check_len[(offsetof(OpusTXContext, len) ==  0) ? 1 : -1];
typedef char opus_tx_check_map[(offsetof(OpusTXContext, map) ==  8) ? 1 : -1];
typedef char opus_tx_check_exp[(offsetof(OpusTXContext, exp) == 16) ? 1 : -1];
typedef char opus_tx_check_tmp[(offsetof(OpusTXContext, tmp) == 24) ? 1 : -1];
typedef char opus_tx_check_sub[(offsetof(OpusTXContext, sub) == 32) ? 1 : -1];
typedef char opus_tx_check_fn [(offsetof(OpusTXContext, fn)  == 40) ? 1 : -1];

#include "celt_tx_tables.h"

void celt_tx_fft4_fwd_float_neon(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
void celt_tx_fft8_ns_float_neon(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
void celt_tx_fft16_ns_float_neon(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
void celt_tx_fft32_ns_float_neon(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
void celt_tx_fft_sr_ns_float_neon(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
void celt_tx_fft_pfa_15xM_ns_float_neon(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
void celt_tx_mdct_inv_float_neon(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);

/* M-point power-of-two sub-FFTs. The PFA kernel reads len and map (the
   split-radix parity scatter map); the preshuffled kernels themselves read
   nothing (fft_sr additionally reads len). */
static const OpusTXContext celt_tx_p2_4   = {  4, 1, celt_tx_p2_map_4,  NULL, NULL, NULL, celt_tx_fft4_fwd_float_neon };
static const OpusTXContext celt_tx_p2_8   = {  8, 1, celt_tx_p2_map_8,  NULL, NULL, NULL, celt_tx_fft8_ns_float_neon };
static const OpusTXContext celt_tx_p2_16  = { 16, 1, celt_tx_p2_map_16, NULL, NULL, NULL, celt_tx_fft16_ns_float_neon };
static const OpusTXContext celt_tx_p2_32  = { 32, 1, celt_tx_p2_map_32, NULL, NULL, NULL, celt_tx_fft32_ns_float_neon };
#if defined(ENABLE_QEXT)
static const OpusTXContext celt_tx_p2_64  = { 64, 1, celt_tx_p2_map_64, NULL, NULL, NULL, celt_tx_fft_sr_ns_float_neon };
#endif

/* 15xM PFA sub-FFTs; .tmp is filled in per call with stack scratch. */
static const OpusTXContext celt_tx_pfa_60  = {  60, 1, celt_tx_pfa_map_60,  NULL, NULL, &celt_tx_p2_4,  celt_tx_fft4_fwd_float_neon };
static const OpusTXContext celt_tx_pfa_120 = { 120, 1, celt_tx_pfa_map_120, NULL, NULL, &celt_tx_p2_8,  celt_tx_fft8_ns_float_neon };
static const OpusTXContext celt_tx_pfa_240 = { 240, 1, celt_tx_pfa_map_240, NULL, NULL, &celt_tx_p2_16, celt_tx_fft16_ns_float_neon };
static const OpusTXContext celt_tx_pfa_480 = { 480, 1, celt_tx_pfa_map_480, NULL, NULL, &celt_tx_p2_32, celt_tx_fft32_ns_float_neon };
#if defined(ENABLE_QEXT)
static const OpusTXContext celt_tx_pfa_960 = { 960, 1, celt_tx_pfa_map_960, NULL, NULL, &celt_tx_p2_64, celt_tx_fft_sr_ns_float_neon };
#endif

#if defined(CUSTOM_MODES)
/* Power-of-two sub-FFTs for the Opus custom mode sizes. The preshuffled
   kernels read only len (the input permutation is folded into the MDCT
   gather map by FFmpeg's init), and there is no PFA stage, hence no map
   and no scratch buffer. */
static const OpusTXContext celt_tx_sr_32  = {  32, 1, NULL, NULL, NULL, NULL, NULL };
static const OpusTXContext celt_tx_sr_64  = {  64, 1, NULL, NULL, NULL, NULL, NULL };
static const OpusTXContext celt_tx_sr_128 = { 128, 1, NULL, NULL, NULL, NULL, NULL };
static const OpusTXContext celt_tx_sr_256 = { 256, 1, NULL, NULL, NULL, NULL, NULL };
static const OpusTXContext celt_tx_sr_512 = { 512, 1, NULL, NULL, NULL, NULL, NULL };
#endif

/* Inverse MDCT roots; for the 15*M sizes .sub is relinked per call to the
   stack copy of the PFA context, the power-of-two ones run fully const. */
static const OpusTXContext celt_tx_mdct_120  = {  120, 1, celt_tx_mdct_map_120,  NULL,  NULL, &celt_tx_pfa_60,  celt_tx_fft_pfa_15xM_ns_float_neon };
static const OpusTXContext celt_tx_mdct_240  = {  240, 1, celt_tx_mdct_map_240,  NULL,  NULL, &celt_tx_pfa_120, celt_tx_fft_pfa_15xM_ns_float_neon };
static const OpusTXContext celt_tx_mdct_480  = {  480, 1, celt_tx_mdct_map_480,  NULL,  NULL, &celt_tx_pfa_240, celt_tx_fft_pfa_15xM_ns_float_neon };
static const OpusTXContext celt_tx_mdct_960  = {  960, 1, celt_tx_mdct_map_960,  NULL,  NULL, &celt_tx_pfa_480, celt_tx_fft_pfa_15xM_ns_float_neon };
#if defined(ENABLE_QEXT)
static const OpusTXContext celt_tx_mdct_1920 = { 1920, 1, celt_tx_mdct_map_1920, NULL, NULL, &celt_tx_pfa_960, celt_tx_fft_pfa_15xM_ns_float_neon };
#endif

#if defined(CUSTOM_MODES)
static const OpusTXContext celt_tx_mdct_64   = {   64, 1, celt_tx_mdct_map_64,   NULL,   NULL, &celt_tx_sr_32,  celt_tx_fft32_ns_float_neon };
static const OpusTXContext celt_tx_mdct_128  = {  128, 1, celt_tx_mdct_map_128,  NULL,  NULL, &celt_tx_sr_64,  celt_tx_fft_sr_ns_float_neon };
static const OpusTXContext celt_tx_mdct_256  = {  256, 1, celt_tx_mdct_map_256,  NULL,  NULL, &celt_tx_sr_128, celt_tx_fft_sr_ns_float_neon };
static const OpusTXContext celt_tx_mdct_512  = {  512, 1, celt_tx_mdct_map_512,  NULL,  NULL, &celt_tx_sr_256, celt_tx_fft_sr_ns_float_neon };
static const OpusTXContext celt_tx_mdct_1024 = { 1024, 1, celt_tx_mdct_map_1024, NULL, NULL, &celt_tx_sr_512, celt_tx_fft_sr_ns_float_neon };
#endif

const OpusTXContext *celt_tx_mdct_kernel(int len)
{
   switch (len) {
      case  120: return &celt_tx_mdct_120;
      case  240: return &celt_tx_mdct_240;
      case  480: return &celt_tx_mdct_480;
      case  960: return &celt_tx_mdct_960;
#if defined(ENABLE_QEXT)
      case 1920: return &celt_tx_mdct_1920;
#endif
#if defined(CUSTOM_MODES)
      case   64: return &celt_tx_mdct_64;
      case  128: return &celt_tx_mdct_128;
      case  256: return &celt_tx_mdct_256;
      case  512: return &celt_tx_mdct_512;
      case 1024: return &celt_tx_mdct_1024;
#endif
      default:   return NULL;
   }
}

void clt_mdct_backward_tx(const mdct_lookup *l, kiss_fft_scalar *in,
      kiss_fft_scalar * OPUS_RESTRICT out,
      const celt_coef * OPUS_RESTRICT window,
      int overlap, int shift, int stride, int arch)
{
   int i;
   int N = l->n >> shift;
   int N2 = N >> 1;
   const OpusTXContext *tpl = celt_tx_mdct_kernel(N2);
   const kiss_twiddle_scalar *trig;
   int cur_n;
   VARDECL(kiss_fft_scalar, tmp);
   SAVE_STACK;

   if (tpl == NULL)
   {
      /* Opus custom mode sizes we have no baked tables for. */
      clt_mdct_backward_c(l, in, out, window, overlap, shift, stride, arch);
      return;
   }
   (void)arch;

   trig = l->trig;
   cur_n = l->n;
   for (i = 0; i < shift; i++) {
      cur_n >>= 1;
      trig += cur_n;
   }

   /* Pre-rotate, N/4 complex FFT and post-rotate, writing the raw N/2
      samples to out[overlap/2 ..] like the C code does. For the 15*M sizes,
      only the PFA scratch buffer is written through the context tree, so
      the two levels that reference it are copied to the stack; the mode
      data stays shared and read-only across threads. The power-of-two
      sizes run their sub-FFT in place on the output and need no fixup. */
   ALLOC(tmp, N2, kiss_fft_scalar);   /* N/4 complex values */

   if (tpl->fn == celt_tx_fft_pfa_15xM_ns_float_neon)
   {
      OpusTXContext mdct, pfa;
      mdct = *tpl;
      mdct.exp = trig;
      mdct.tmp = tmp;
      pfa = *tpl->sub;
      pfa.tmp = tmp;
      mdct.sub = &pfa;
      celt_tx_mdct_inv_float_neon(&mdct, out+(overlap>>1), in,
                                  (ptrdiff_t)stride*sizeof(kiss_fft_scalar));
   } else {
      OpusTXContext mdct = *tpl;
      mdct.exp = trig;
      mdct.tmp = tmp;
      celt_tx_mdct_inv_float_neon(&mdct, out+(overlap>>1), in,
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
   RESTORE_STACK;
}

#endif /* OPUS_ARM_TX_MDCT */
