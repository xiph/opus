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
#include "arch.h"
#include "features.h"
#include "quant_bands.h"

#define NBANDS 17
const int bands[NBANDS+1] =
{1,  4,  8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96,112,136,160};

float dct_table[128] = {
        0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000,
        0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000,
        0.351851, 0.338330, 0.311806, 0.273300, 0.224292, 0.166664, 0.102631, 0.034654,
        -0.034654, -0.102631, -0.166664, -0.224292, -0.273300, -0.311806, -0.338330, -0.351851,
        0.346760, 0.293969, 0.196424, 0.068975, -0.068975, -0.196424, -0.293969, -0.346760,
        -0.346760, -0.293969, -0.196424, -0.068975, 0.068975, 0.196424, 0.293969, 0.346760,
        0.338330, 0.224292, 0.034654, -0.166664, -0.311806, -0.351851, -0.273300, -0.102631,
        0.102631, 0.273300, 0.351851, 0.311806, 0.166664, -0.034654, -0.224292, -0.338330,
        0.326641, 0.135299, -0.135299, -0.326641, -0.326641, -0.135299, 0.135299, 0.326641,
        0.326641, 0.135299, -0.135299, -0.326641, -0.326641, -0.135299, 0.135299, 0.326641,
        0.311806, 0.034654, -0.273300, -0.338330, -0.102631, 0.224292, 0.351851, 0.166664,
        -0.166664, -0.351851, -0.224292, 0.102631, 0.338330, 0.273300, -0.034654, -0.311806,
        0.293969, -0.068975, -0.346760, -0.196424, 0.196424, 0.346760, 0.068975, -0.293969,
        -0.293969, 0.068975, 0.346760, 0.196424, -0.196424, -0.346760, -0.068975, 0.293969,
        0.273300, -0.166664, -0.338330, 0.034654, 0.351851, 0.102631, -0.311806, -0.224292,
        0.224292, 0.311806, -0.102631, -0.351851, -0.034654, 0.338330, 0.166664, -0.273300,
};

static void feature_analysis(CELTEncoder *celt_enc, const celt_word16 *x,
        celt_word16 *features, celt_word16 *mem)
{
    int i;
    const CELTMode *mode;
    const kiss_fft_state *kfft;
    kiss_fft_cpx in[480], out[480];
    const celt_word16 *window;
    celt_word32 E[NBANDS];
    celt_word16 logE[NBANDS];
    celt_word16 BFCC[16];
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
        celt_word32 sum = 0;
        for (j=bands[i];j<bands[i+1];j++)
            sum = sum + MULT32_32_Q31(out[  j].r, out[  j].r)
                      + MULT32_32_Q31(out[  j].i, out[  j].i)
                      + MULT32_32_Q31(out[N-j].r, out[N-j].r)
                      + MULT32_32_Q31(out[N-j].i, out[N-j].i);
        E[i] = MAX32(EPSILON, sum);
        //printf ("%f ", E[i]);
    }
    amp2Log2(mode, NBANDS, NBANDS, E, logE, 1);
    for (i=0;i<NBANDS;i++)
        logE[i] = MAX32(logE[i], -14.);
    //for (i=0;i<16;i++)
    //    printf ("%f ", logE[i]);

    for (i=0;i<8;i++)
    {
        int j;
        float sum = 0;
        for (j=0;j<16;j++)
            sum += dct_table[i*16+j]*logE[j];
        BFCC[i] = sum;
        //printf ("%f ", BFCC[i]);
    }
    for (i=1;i<8;i++)
        features[i-1] = -0.12299*(BFCC[i]+mem[i+24]) + 0.49195*(mem[i]+mem[i+16]) + 0.69693*mem[i+8];

    for (i=0;i<8;i++)
        features[7+i] = 0.63246*(BFCC[i]-mem[i+24]) + 0.31623*(mem[i]-mem[i+16]);
    for (i=0;i<8;i++)
        features[15+i] = 0.53452*(BFCC[i]+mem[i+24]) - 0.26726*(mem[i]+mem[i+16]) -0.53452*mem[i+8];
    for (i=0;i<8;i++)
    {
        mem[i+24] = mem[i+16];
        mem[i+16] = mem[i+8];
        mem[i+8] = mem[i];
        mem[i] = BFCC[i];
    }
    for (i=0;i<23;i++)
        printf ("%f ", features[i]);

    printf("\n");
}

void feature_analysis_fixed(CELTEncoder *celt_enc, const celt_int16 *x)
{
    /* FIXME: Get rid of this static var ASAP! */
    static float mem[32];
    float features[23];
#ifdef FIXED_POINT
    feature_analysis(celt_enc, x);
#else
    int i;
    int N = 960-120;
    celt_word16 x2[960-120];

    for (i=0;i<N;i++)
        x2[i] = x[i];
    feature_analysis(celt_enc, x2, features, mem);
#endif
}
