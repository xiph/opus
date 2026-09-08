/* Copyright (c) 2015 Xiph.Org Foundation
   Written by Viswanath Puttagunta */
/**
   @file mdct_arm.h
   @brief ARM-specific MDCT backend hooks for CELT
 */

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

#if !defined(MDCT_ARM_H)
#define MDCT_ARM_H

#include "mdct.h"

/* The AArch64 NEON inverse MDCT ported from FFmpeg's libavutil/tx
   (celt_tx_neon.S / celt_mdct_tx.c), float only. */
#if !defined(FIXED_POINT) && defined(__aarch64__) && \
    (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) || defined(OPUS_ARM_PRESUME_NEON_INTR))
#define OPUS_ARM_TX_MDCT (1)
#endif

#if defined(OPUS_ARM_TX_MDCT)

/* NEON inverse MDCT for the 48 kHz and 96 kHz mode sizes, falling back to
   clt_mdct_backward_c() for other (custom mode) sizes. The forward MDCT is
   not ported yet. */
struct OpusTXContext;
const struct OpusTXContext *celt_tx_mdct_kernel(int len);

void clt_mdct_backward_tx(const mdct_lookup *l, kiss_fft_scalar *in,
      kiss_fft_scalar * OPUS_RESTRICT out,
      const celt_coef * OPUS_RESTRICT window,
      int overlap, int shift, int stride, int arch);

# if defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_MAY_HAVE_NEON_INTR) \
     && !defined(OPUS_ARM_PRESUME_NEON_INTR)
extern void (*const CLT_MDCT_BACKWARD_IMPL[OPUS_ARCHMASK+1])(
      const mdct_lookup *l, kiss_fft_scalar *in,
      kiss_fft_scalar * OPUS_RESTRICT out,
      const celt_coef * OPUS_RESTRICT window,
      int overlap, int shift, int stride, int arch);

#  define OVERRIDE_CLT_MDCT_BACKWARD (1)
#  define clt_mdct_backward(_l, _in, _out, _window, _overlap, _shift, _stride, _arch) \
      ((*CLT_MDCT_BACKWARD_IMPL[(_arch)&OPUS_ARCHMASK])(_l, _in, _out, _window, _overlap, _shift, _stride, _arch))
# elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#  define OVERRIDE_CLT_MDCT_BACKWARD (1)
#  define clt_mdct_backward(_l, _in, _out, _window, _overlap, _shift, _stride, _arch) \
      clt_mdct_backward_tx(_l, _in, _out, _window, _overlap, _shift, _stride, _arch)
# endif

#endif /* OPUS_ARM_TX_MDCT */

#endif
