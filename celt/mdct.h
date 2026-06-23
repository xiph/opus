/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2008 Xiph.Org Foundation
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

/* This is a simple MDCT implementation that uses a N/4 complex FFT
   to do most of the work. It should be relatively straightforward to
   plug in pretty much and FFT here.

   This replaces the Vorbis FFT (and uses the exact same API), which
   was a bit too messy and that was ending up duplicating code
   (might as well use the same FFT everywhere).

   The algorithm is similar to (and inspired from) Fabrice Bellard's
   MDCT implementation in FFMPEG, but has differences in signs, ordering
   and scaling in many places.
*/

#ifndef MDCT_H
#define MDCT_H

#include "opus_defines.h"
#include "kiss_fft.h"
#include "arch.h"

typedef struct {
   int n;
   int maxshift;
   const kiss_fft_state *kfft[4];
   const kiss_twiddle_scalar * OPUS_RESTRICT trig;
} mdct_lookup;

int clt_mdct_init(mdct_lookup *l,int N, int maxshift, int arch);
void clt_mdct_clear(mdct_lookup *l, int arch);

/** Compute a forward MDCT and scale by 4/N, trashes the input array */
void clt_mdct_forward_c(const mdct_lookup *l, kiss_fft_scalar *in,
                        kiss_fft_scalar * OPUS_RESTRICT out,
                        const celt_coef *window, int overlap,
                        int shift, int stride, int arch);

/** Compute a backward MDCT (no scaling) and performs weighted overlap-add
    (scales implicitly by 1/2) */
void clt_mdct_backward_c(const mdct_lookup *l, kiss_fft_scalar *in,
      kiss_fft_scalar * OPUS_RESTRICT out,
      const celt_coef * OPUS_RESTRICT window,
      int overlap, int shift, int stride, int arch);

void opus_fft_pfa_c(const kiss_fft_state *st, const kiss_fft_cpx *fin, kiss_fft_cpx *fout ARG_FIXED(int downshift));
void opus_ifft_pfa_c(const kiss_fft_state *st, const kiss_fft_cpx *fin, kiss_fft_cpx *fout ARG_FIXED(int fft_shift));

#if defined(OPUS_ARM_MAY_HAVE_NEON_INTR) || defined(OPUS_ARM_PRESUME_NEON_INTR)
#include "arm/mdct_arm.h"
#endif



#if !defined(OVERRIDE_CLT_MDCT_FORWARD)
# define clt_mdct_forward(_l, _in, _out, _window, _overlap, _shift, _stride, _arch) \
   clt_mdct_forward_c(_l, _in, _out, _window, _overlap, _shift, _stride, _arch)
#endif

#if !defined(OVERRIDE_CLT_MDCT_BACKWARD)
# define clt_mdct_backward(_l, _in, _out, _window, _overlap, _shift, _stride, _arch) \
   clt_mdct_backward_c(_l, _in, _out, _window, _overlap, _shift, _stride, _arch)
#endif

#endif
