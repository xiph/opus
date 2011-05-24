/* Copyright (c) 2011 Xiph.Org Foundation
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
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
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

#include "kiss_fft.h"
#include "celt.h"
#include "modes.h"

#define NBANDS 17
const int bands[NBANDS+1] =
{1,  4,  8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96,112,136,160};

void feature_analysis(CELTEncoder *celt_enc, celt_word16 *x)
{
    int i;
    const CELTMode *mode;
    const kiss_fft_state *kfft;
    kiss_fft_cpx in[480], out[480];
    const celt_word16 *window;
    celt_word32 E[NBANDS+1];
    int overlap = 120;
    int N = 480;

    celt_encoder_ctl(celt_enc, CELT_GET_MODE(&mode));

    kfft = mode->mdct.kfft[0];
    window = mode->window;

    for (i=0;i<overlap;i++)
    {
        in[i].r = MULT16_16(window[i], x[i]);
        in[i].i = MULT16_16(window[i], x[N-overlap+i]);
        in[N-i].r = MULT16_16(window[i], x[N-i]);
        in[N-i].i = MULT16_16(window[i], x[2*N-overlap-i]);
    }
    for (;i<N;i++)
    {
        in[i].r = SHL32(EXTEND32(x[i]),15);
        in[i].i = SHL32(EXTEND32(x[N-overlap+i]),15);
    }
    kiss_fft(kfft, in, out);

    for (i=0;i<NBANDS;i++)
    {
        int j;
        E[i] = 0;
        for (j=bands[i];j<bands[i+1];j++)
            E[i] = E[i] + MULT_32_32_Q31(out[  j].r, out[  j].r)
                        + MULT_32_32_Q31(out[  j].i, out[  j].i)
                        + MULT_32_32_Q31(out[N-j].r, out[N-j].r)
                        + MULT_32_32_Q31(out[N-j].i, out[N-j].i);
    }
}
