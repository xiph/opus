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
#include "quant_bands.h"
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.141592653
#endif

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

#define NB_FRAMES 8

#define NB_TBANDS 18
static const int tbands[NB_TBANDS+1] = {
      2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 68, 80, 96, 120
};

#define NB_TONAL_SKIP_BANDS 8

typedef struct {
   float angle[240];
   float d_angle[240];
   float d2_angle[240];
   float prev_band_tonality[NB_TBANDS];
   float prev_tonality;
   float E[NB_FRAMES][NB_TBANDS];
   float lowE[NB_TBANDS], highE[NB_TBANDS];
   float mem[32];
   int E_count;
   int count;
} TonalityAnalysisState;

void tonality_analysis(TonalityAnalysisState *tonal, AnalysisInfo *info, CELTEncoder *celt_enc, const opus_val16 *x, int C)
{
    int i, b;
    const CELTMode *mode;
    const kiss_fft_state *kfft;
    kiss_fft_cpx in[480], out[480];
    int N = 480, N2=240;
    float * restrict A = tonal->angle;
    float * restrict dA = tonal->d_angle;
    float * restrict d2A = tonal->d2_angle;
    float tonality[240];
    float noisiness[240];
    float band_tonality[NB_TBANDS];
    float logE[NB_TBANDS];
    float BFCC[8];
    float features[27];
    float frame_tonality;
    float frame_noisiness;
    const float pi4 = M_PI*M_PI*M_PI*M_PI;
    float slope=0;
    float frame_stationarity;
    float relativeE;
    celt_encoder_ctl(celt_enc, CELT_GET_MODE(&mode));

    kfft = mode->mdct.kfft[0];
    if (C==1)
    {
       for (i=0;i<N2;i++)
       {
          float w = .5-.5*cos(M_PI*(i+1)/N2);
          in[i].r = MULT16_16(w, x[i]);
          in[i].i = MULT16_16(w, x[N-N2+i]);
          in[N-i-1].r = MULT16_16(w, x[N-i-1]);
          in[N-i-1].i = MULT16_16(w, x[2*N-N2-i-1]);
       }
    } else {
       for (i=0;i<N2;i++)
       {
          float w = .5-.5*cos(M_PI*(i+1)/N2);
          in[i].r = MULT16_16(w, x[2*i]+x[2*i+1]);
          in[i].i = MULT16_16(w, x[2*(N-N2+i)]+x[2*(N-N2+i)+1]);
          in[N-i-1].r = MULT16_16(w, x[2*(N-i-1)]+x[2*(N-i-1)+1]);
          in[N-i-1].i = MULT16_16(w, x[2*(2*N-N2-i-1)]+x[2*(2*N-N2-i-1)+1]);
       }
    }
    opus_fft(kfft, in, out);

    for (i=1;i<N2;i++)
    {
       float X1r, X2r, X1i, X2i;
       float angle, d_angle, d2_angle;
       float angle2, d_angle2, d2_angle2;
       float mod1, mod2, avg_mod;
       X1r = out[i].r+out[N-i].r;
       X1i = out[i].i-out[N-i].i;
       X2r = out[i].i+out[N-i].i;
       X2i = out[N-i].r-out[i].r;

       angle = (.5/M_PI)*atan2(X1i, X1r);
       d_angle = angle - A[i];
       d2_angle = d_angle - dA[i];

       angle2 = (.5/M_PI)*atan2(X2i, X2r);
       d_angle2 = angle2 - angle;
       d2_angle2 = d_angle2 - d_angle;

       mod1 = d2_angle - floor(.5+d2_angle);
       noisiness[i] = fabs(mod1);
       mod1 *= mod1;
       mod1 *= mod1;

       mod2 = d2_angle2 - floor(.5+d2_angle2);
       noisiness[i] += fabs(mod2);
       mod2 *= mod2;
       mod2 *= mod2;

       avg_mod = .25*(d2A[i]+2*mod1+mod2);
       tonality[i] = 1./(1+40*16*pi4*avg_mod)-.015;

       A[i] = angle2;
       dA[i] = d_angle2;
       d2A[i] = mod2;
    }

    frame_tonality = 0;
    info->activity = 0;
    frame_noisiness = 0;
    frame_stationarity = 0;
    if (!tonal->count)
    {
       for (b=0;b<NB_TBANDS;b++)
       {
          tonal->lowE[b] = 1e10;
          tonal->highE[b] = -1e10;
       }
    }
    relativeE = 0;
    info->boost_amount[0]=info->boost_amount[1]=0;
    info->boost_band[0]=info->boost_band[1]=0;
    for (b=0;b<NB_TBANDS;b++)
    {
       float E=0, tE=0, nE=0;
       float L1, L2;
       float stationarity;
       for (i=tbands[b];i<tbands[b+1];i++)
       {
          float binE = out[i].r*out[i].r + out[N-i].r*out[N-i].r
                     + out[i].i*out[i].i + out[N-i].i*out[N-i].i;
          E += binE;
          tE += binE*tonality[i];
          nE += binE*2*(.5-noisiness[i]);
       }
       tonal->E[tonal->E_count][b] = E;
       frame_noisiness += nE/(1e-15+E);

       logE[b] = log(E+EPSILON);
       tonal->lowE[b] = MIN32(logE[b], tonal->lowE[b]+.01);
       tonal->highE[b] = MAX32(logE[b], tonal->highE[b]-.1);
       if (tonal->highE[b] < tonal->lowE[b]+1)
       {
          tonal->highE[b]+=.5;
          tonal->lowE[b]-=.5;
       }
       relativeE += (logE[b]-tonal->lowE[b])/(EPSILON+tonal->highE[b]-tonal->lowE[b]);

       L1=L2=0;
       for (i=0;i<NB_FRAMES;i++)
       {
          L1 += sqrt(tonal->E[i][b]);
          L2 += tonal->E[i][b];
       }

       stationarity = MIN16(0.99,L1/sqrt(EPSILON+NB_FRAMES*L2));
       stationarity *= stationarity;
       stationarity *= stationarity;
       frame_stationarity += stationarity;
       /*band_tonality[b] = tE/(1e-15+E)*/;
       band_tonality[b] = MAX16(tE/(EPSILON+E), stationarity*tonal->prev_band_tonality[b]);
       if (b>=NB_TONAL_SKIP_BANDS)
          frame_tonality += band_tonality[b];
       slope += band_tonality[b]*(b-8);
       if (band_tonality[b] > info->boost_amount[1] && b>=7 && b < NB_TBANDS-1)
       {
          if (band_tonality[b] > info->boost_amount[0])
          {
             info->boost_amount[1] = info->boost_amount[0];
             info->boost_band[1] = info->boost_band[0];
             info->boost_amount[0] = band_tonality[b];
             info->boost_band[0] = b;
          } else {
             info->boost_amount[1] = band_tonality[b];
             info->boost_band[1] = b;
          }
       }
       tonal->prev_band_tonality[b] = band_tonality[b];
    }

    for (i=0;i<8;i++)
    {
       float sum=0;
       for (b=0;b<16;b++)
          sum += dct_table[i*16+b]*logE[b];
       BFCC[i] = sum;
    }

    frame_stationarity /= NB_TBANDS;
    relativeE /= NB_TBANDS;
    if (tonal->count<10)
       relativeE = .5;
    frame_noisiness /= NB_TBANDS;
#if 1
    info->activity = frame_noisiness + (1-frame_noisiness)*relativeE;
#else
    info->activity = .5*(1+frame_noisiness-frame_stationarity);
#endif
    frame_tonality /= NB_TBANDS-NB_TONAL_SKIP_BANDS;
    frame_tonality = MAX16(frame_tonality, tonal->prev_tonality*.8);
    tonal->prev_tonality = frame_tonality;
    info->boost_amount[0] -= frame_tonality+.2;
    info->boost_amount[1] -= frame_tonality+.2;
    if (band_tonality[info->boost_band[0]] < band_tonality[info->boost_band[0]+1]+.15
        || band_tonality[info->boost_band[0]] < band_tonality[info->boost_band[0]-1]+.15)
       info->boost_amount[0]=0;
    if (band_tonality[info->boost_band[1]] < band_tonality[info->boost_band[1]+1]+.15
        || band_tonality[info->boost_band[1]] < band_tonality[info->boost_band[1]-1]+.15)
       info->boost_amount[1]=0;

    slope /= 8*8;
    info->tonality_slope = slope;

    tonal->E_count = (tonal->E_count+1)%NB_FRAMES;
    tonal->count++;
    info->tonality = frame_tonality;

    for (i=1;i<8;i++)
        features[i-1] = -0.12299*(BFCC[i]+tonal->mem[i+24]) + 0.49195*(tonal->mem[i]+tonal->mem[i+16]) + 0.69693*tonal->mem[i+8];

    for (i=0;i<8;i++)
        features[7+i] = 0.63246*(BFCC[i]-tonal->mem[i+24]) + 0.31623*(tonal->mem[i]-tonal->mem[i+16]);
    for (i=0;i<8;i++)
        features[15+i] = 0.53452*(BFCC[i]+tonal->mem[i+24]) - 0.26726*(tonal->mem[i]+tonal->mem[i+16]) -0.53452*tonal->mem[i+8];
    for (i=0;i<8;i++)
    {
       tonal->mem[i+24] = tonal->mem[i+16];
       tonal->mem[i+16] = tonal->mem[i+8];
       tonal->mem[i+8] = tonal->mem[i];
       tonal->mem[i] = BFCC[i];
    }
    features[23] = info->tonality;
    features[24] = info->tonality_slope;
    features[25] = info->activity;
    features[26] = frame_stationarity;

    /*for (i=0;i<27;i++)
       printf("%f ", features[i]);
    printf("\n");*/

    info->valid = 1;
}
