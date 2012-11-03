/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2010 Xiph.Org Foundation
   Copyright (c) 2008 Gregory Maxwell
   Written by Jean-Marc Valin and Gregory Maxwell */
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

#define CELT_C

#include "os_support.h"
#include "mdct.h"
#include <math.h>
#include "celt.h"
#include "pitch.h"
#include "bands.h"
#include "modes.h"
#include "entcode.h"
#include "quant_bands.h"
#include "rate.h"
#include "stack_alloc.h"
#include "mathops.h"
#include "float_cast.h"
#include <stdarg.h>
#include "celt_lpc.h"
#include "vq.h"

#ifndef OPUS_VERSION
#define OPUS_VERSION "unknown"
#endif

#ifdef CUSTOM_MODES
#define OPUS_CUSTOM_NOSTATIC
#else
#define OPUS_CUSTOM_NOSTATIC static inline
#endif

static const unsigned char trim_icdf[11] = {126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0};
/* Probs: NONE: 21.875%, LIGHT: 6.25%, NORMAL: 65.625%, AGGRESSIVE: 6.25% */
static const unsigned char spread_icdf[4] = {25, 23, 2, 0};

static const unsigned char tapset_icdf[3]={2,1,0};

#ifdef CUSTOM_MODES
static const unsigned char toOpusTable[20] = {
      0xE0, 0xE8, 0xF0, 0xF8,
      0xC0, 0xC8, 0xD0, 0xD8,
      0xA0, 0xA8, 0xB0, 0xB8,
      0x00, 0x00, 0x00, 0x00,
      0x80, 0x88, 0x90, 0x98,
};

static const unsigned char fromOpusTable[16] = {
      0x80, 0x88, 0x90, 0x98,
      0x40, 0x48, 0x50, 0x58,
      0x20, 0x28, 0x30, 0x38,
      0x00, 0x08, 0x10, 0x18
};

static inline int toOpus(unsigned char c)
{
   int ret=0;
   if (c<0xA0)
      ret = toOpusTable[c>>3];
   if (ret == 0)
      return -1;
   else
      return ret|(c&0x7);
}

static inline int fromOpus(unsigned char c)
{
   if (c<0x80)
      return -1;
   else
      return fromOpusTable[(c>>3)-16] | (c&0x7);
}
#endif /* CUSTOM_MODES */

#define COMBFILTER_MAXPERIOD 1024
#define COMBFILTER_MINPERIOD 15

static int resampling_factor(opus_int32 rate)
{
   int ret;
   switch (rate)
   {
   case 48000:
      ret = 1;
      break;
   case 24000:
      ret = 2;
      break;
   case 16000:
      ret = 3;
      break;
   case 12000:
      ret = 4;
      break;
   case 8000:
      ret = 6;
      break;
   default:
#ifndef CUSTOM_MODES
      celt_assert(0);
#endif
      ret = 0;
      break;
   }
   return ret;
}

/** Encoder state
 @brief Encoder state
 */
struct OpusCustomEncoder {
   const OpusCustomMode *mode;     /**< Mode used by the encoder */
   int overlap;
   int channels;
   int stream_channels;

   int force_intra;
   int clip;
   int disable_pf;
   int complexity;
   int upsample;
   int start, end;

   opus_int32 bitrate;
   int vbr;
   int signalling;
   int constrained_vbr;      /* If zero, VBR can do whatever it likes with the rate */
   int loss_rate;
   int lsb_depth;

   /* Everything beyond this point gets cleared on a reset */
#define ENCODER_RESET_START rng

   opus_uint32 rng;
   int spread_decision;
   opus_val32 delayedIntra;
   int tonal_average;
   int lastCodedBands;
   int hf_average;
   int tapset_decision;

   int prefilter_period;
   opus_val16 prefilter_gain;
   int prefilter_tapset;
#ifdef RESYNTH
   int prefilter_period_old;
   opus_val16 prefilter_gain_old;
   int prefilter_tapset_old;
#endif
   int consec_transient;
   AnalysisInfo analysis;

   opus_val32 preemph_memE[2];
   opus_val32 preemph_memD[2];

   /* VBR-related parameters */
   opus_int32 vbr_reservoir;
   opus_int32 vbr_drift;
   opus_int32 vbr_offset;
   opus_int32 vbr_count;
   opus_val16 overlap_max;
   opus_val16 stereo_saving;
   int intensity;

#ifdef RESYNTH
   celt_sig syn_mem[2][2*MAX_PERIOD];
#endif

   celt_sig in_mem[1]; /* Size = channels*mode->overlap */
   /* celt_sig prefilter_mem[],  Size = channels*COMBFILTER_MAXPERIOD */
   /* opus_val16 oldBandE[],     Size = channels*mode->nbEBands */
   /* opus_val16 oldLogE[],      Size = channels*mode->nbEBands */
   /* opus_val16 oldLogE2[],     Size = channels*mode->nbEBands */
#ifdef RESYNTH
   /* opus_val16 overlap_mem[],  Size = channels*overlap */
#endif
};

int celt_encoder_get_size(int channels)
{
   CELTMode *mode = opus_custom_mode_create(48000, 960, NULL);
   return opus_custom_encoder_get_size(mode, channels);
}

OPUS_CUSTOM_NOSTATIC int opus_custom_encoder_get_size(const CELTMode *mode, int channels)
{
   int size = sizeof(struct CELTEncoder)
         + (channels*mode->overlap-1)*sizeof(celt_sig)    /* celt_sig in_mem[channels*mode->overlap]; */
         + channels*COMBFILTER_MAXPERIOD*sizeof(celt_sig) /* celt_sig prefilter_mem[channels*COMBFILTER_MAXPERIOD]; */
         + 3*channels*mode->nbEBands*sizeof(opus_val16);  /* opus_val16 oldBandE[channels*mode->nbEBands]; */
                                                          /* opus_val16 oldLogE[channels*mode->nbEBands]; */
                                                          /* opus_val16 oldLogE2[channels*mode->nbEBands]; */
#ifdef RESYNTH
   size += channels*mode->overlap*sizeof(celt_sig);       /* celt_sig overlap_mem[channels*mode->nbEBands]; */
#endif
   return size;
}

#ifdef CUSTOM_MODES
CELTEncoder *opus_custom_encoder_create(const CELTMode *mode, int channels, int *error)
{
   int ret;
   CELTEncoder *st = (CELTEncoder *)opus_alloc(opus_custom_encoder_get_size(mode, channels));
   /* init will handle the NULL case */
   ret = opus_custom_encoder_init(st, mode, channels);
   if (ret != OPUS_OK)
   {
      opus_custom_encoder_destroy(st);
      st = NULL;
   }
   if (error)
      *error = ret;
   return st;
}
#endif /* CUSTOM_MODES */

int celt_encoder_init(CELTEncoder *st, opus_int32 sampling_rate, int channels)
{
   int ret;
   ret = opus_custom_encoder_init(st, opus_custom_mode_create(48000, 960, NULL), channels);
   if (ret != OPUS_OK)
      return ret;
   st->upsample = resampling_factor(sampling_rate);
   return OPUS_OK;
}

OPUS_CUSTOM_NOSTATIC int opus_custom_encoder_init(CELTEncoder *st, const CELTMode *mode, int channels)
{
   if (channels < 0 || channels > 2)
      return OPUS_BAD_ARG;

   if (st==NULL || mode==NULL)
      return OPUS_ALLOC_FAIL;

   OPUS_CLEAR((char*)st, opus_custom_encoder_get_size(mode, channels));

   st->mode = mode;
   st->overlap = mode->overlap;
   st->stream_channels = st->channels = channels;

   st->upsample = 1;
   st->start = 0;
   st->end = st->mode->effEBands;
   st->signalling = 1;

   st->constrained_vbr = 1;
   st->clip = 1;

   st->bitrate = OPUS_BITRATE_MAX;
   st->vbr = 0;
   st->force_intra  = 0;
   st->complexity = 5;
   st->lsb_depth=24;

   opus_custom_encoder_ctl(st, OPUS_RESET_STATE);

   return OPUS_OK;
}

#ifdef CUSTOM_MODES
void opus_custom_encoder_destroy(CELTEncoder *st)
{
   opus_free(st);
}
#endif /* CUSTOM_MODES */

static inline opus_val16 SIG2WORD16(celt_sig x)
{
#ifdef FIXED_POINT
   x = PSHR32(x, SIG_SHIFT);
   x = MAX32(x, -32768);
   x = MIN32(x, 32767);
   return EXTRACT16(x);
#else
   return (opus_val16)x;
#endif
}

static int transient_analysis(const opus_val32 * OPUS_RESTRICT in, int len, int C,
                              opus_val16 *tf_estimate, int *tf_chan)
{
   int i;
   VARDECL(opus_val16, tmp);
   opus_val32 mem0,mem1;
   int is_transient = 0;
   opus_int32 mask_metric = 0;
   int c;
   int tf_max;
   /* Table of 6*64/x, trained on real data to minimize the average error */
   static const unsigned char inv_table[128] = {
         255,255,156,110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25,
          23, 22, 21, 20, 19, 18, 17, 16, 16, 15, 15, 14, 13, 13, 12, 12,
          12, 12, 11, 11, 11, 10, 10, 10,  9,  9,  9,  9,  9,  9,  8,  8,
           8,  8,  8,  7,  7,  7,  7,  7,  7,  6,  6,  6,  6,  6,  6,  6,
           6,  6,  6,  6,  6,  6,  6,  6,  6,  5,  5,  5,  5,  5,  5,  5,
           5,  5,  5,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
           4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  3,  3,
           3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,
   };
   SAVE_STACK;
   ALLOC(tmp, len, opus_val16);

   tf_max = 0;
   for (c=0;c<C;c++)
   {
      opus_val32 mean;
      opus_int32 unmask=0;
      opus_val32 norm;
      mem0=0;
      mem1=0;
      /* High-pass filter: (1 - 2*z^-1 + z^-2) / (1 - z^-1 + .5*z^-2) */
      for (i=0;i<len;i++)
      {
         opus_val32 x,y;
         x = SHR32(in[i+c*len],SIG_SHIFT);
         y = ADD32(mem0, x);
#ifdef FIXED_POINT
         mem0 = mem1 + y - SHL32(x,1);
         mem1 = x - SHR32(y,1);
#else
         mem0 = mem1 + y - 2*x;
         mem1 = x - .5f*y;
#endif
         tmp[i] = EXTRACT16(SHR32(y,2));
         /*printf("%f ", tmp[i]);*/
      }
      /*printf("\n");*/
      /* First few samples are bad because we don't propagate the memory */
      for (i=0;i<12;i++)
         tmp[i] = 0;

#ifdef FIXED_POINT
      /* Normalize tmp to max range */
      {
         int shift=0;
         shift = 14-celt_ilog2(1+celt_maxabs16(tmp, len));
         if (shift!=0)
         {
            for (i=0;i<len;i++)
               tmp[i] = SHL16(tmp[i], shift);
         }
      }
#endif

      mean=0;
      mem0=0;
      /*  Grouping by two to reduce complexity */
      len/=2;
      /* Forward pass to compute the post-echo threshold*/
      for (i=0;i<len;i++)
      {
         opus_val16 x2 = PSHR32(MULT16_16(tmp[2*i],tmp[2*i]) + MULT16_16(tmp[2*i+1],tmp[2*i+1]),16);
         mean += x2;
#ifdef FIXED_POINT
         /* FIXME: Use PSHR16() instead */
         tmp[i] = mem0 + PSHR32(x2-mem0,4);
#else
         tmp[i] = mem0 + MULT16_16_P15(QCONST16(.0625f,15),x2-mem0);
#endif
         mem0 = tmp[i];
      }

      mem0=0;
      /* Backward pass to compute the pre-echo threshold */
      for (i=len-1;i>=0;i--)
      {
#ifdef FIXED_POINT
         /* FIXME: Use PSHR16() instead */
         tmp[i] = mem0 + PSHR32(tmp[i]-mem0,3);
#else
         tmp[i] = mem0 + MULT16_16_P15(QCONST16(0.125f,15),tmp[i]-mem0);
#endif
         mem0 = tmp[i];
      }
      /*for (i=0;i<len;i++)printf("%f ", tmp[i]/mean);printf("\n");*/

      /* Compute the ratio of the mean energy over the harmonic mean of the energy.
         This essentially corresponds to a bitrate-normalized temporal noise-to-mask
         ratio */

      /* Inverse of the mean energy in Q15+6 */
      norm = SHL32(EXTEND32(len),6+14)/ADD32(EPSILON,SHR32(mean,1));
      /* Compute harmonic mean discarding the unreliable boundaries
         The data is smooth, so we only take 1/4th of the samples */
      unmask=0;
      for (i=12;i<len-5;i+=4)
      {
         int id;
#ifdef FIXED_POINT
         id = IMAX(0,IMIN(127,MULT16_32_Q15(tmp[i],norm))); /* Do not round to nearest */
#else
         id = IMAX(0,IMIN(127,floor(64*norm*tmp[i]))); /* Do not round to nearest */
#endif
         unmask += inv_table[id];
      }
      /*printf("%d\n", unmask);*/
      /* Normalize, compensate for the 1/4th of the sample and the factor of 6 in the inverse table */
      unmask = 64*unmask*4/(6*(len-17));
      if (unmask>mask_metric)
      {
         *tf_chan = c;
         mask_metric = unmask;
      }
   }
   is_transient = mask_metric>141;

   /* Arbitrary metric for VBR boost */
   tf_max = MAX16(0,celt_sqrt(64*mask_metric)-64);
   /* *tf_estimate = 1 + MIN16(1, sqrt(MAX16(0, tf_max-30))/20); */
   *tf_estimate = QCONST16(1.f, 14) + celt_sqrt(MAX16(0, SHL32(MULT16_16(QCONST16(0.0069,14),IMIN(163,tf_max)),14)-QCONST32(0.139,28)));
   /*printf("%d %f\n", tf_max, mask_metric);*/
   RESTORE_STACK;
#ifdef FUZZING
   is_transient = rand()&0x1;
#endif
   /*printf("%d %f %d\n", is_transient, (float)*tf_estimate, tf_max);*/
   return is_transient;
}

/** Apply window and compute the MDCT for all sub-frames and
    all channels in a frame */
static void compute_mdcts(const CELTMode *mode, int shortBlocks, celt_sig * OPUS_RESTRICT in, celt_sig * OPUS_RESTRICT out, int C, int LM)
{
   if (C==1 && !shortBlocks)
   {
      const int overlap = OVERLAP(mode);
      clt_mdct_forward(&mode->mdct, in, out, mode->window, overlap, mode->maxLM-LM, 1);
   } else {
      const int overlap = OVERLAP(mode);
      int N = mode->shortMdctSize<<LM;
      int B = 1;
      int b, c;
      if (shortBlocks)
      {
         N = mode->shortMdctSize;
         B = shortBlocks;
      }
      c=0; do {
         for (b=0;b<B;b++)
         {
            /* Interleaving the sub-frames while doing the MDCTs */
            clt_mdct_forward(&mode->mdct, in+c*(B*N+overlap)+b*N, &out[b+c*N*B], mode->window, overlap, shortBlocks ? mode->maxLM : mode->maxLM-LM, B);
         }
      } while (++c<C);
   }
}

/** Compute the IMDCT and apply window for all sub-frames and
    all channels in a frame */
static void compute_inv_mdcts(const CELTMode *mode, int shortBlocks, celt_sig *X,
      celt_sig * OPUS_RESTRICT out_mem[],
      celt_sig * OPUS_RESTRICT overlap_mem[], int C, int LM)
{
   int c;
   const int N = mode->shortMdctSize<<LM;
   const int overlap = OVERLAP(mode);
   VARDECL(opus_val32, x);
   SAVE_STACK;

   ALLOC(x, N+overlap, opus_val32);
   c=0; do {
      int j;
      int b;
      int N2 = N;
      int B = 1;

      if (shortBlocks)
      {
         N2 = mode->shortMdctSize;
         B = shortBlocks;
      }
      /* Prevents problems from the imdct doing the overlap-add */
      OPUS_CLEAR(x, overlap);

      for (b=0;b<B;b++)
      {
         /* IMDCT on the interleaved the sub-frames */
         clt_mdct_backward(&mode->mdct, &X[b+c*N2*B], x+N2*b, mode->window, overlap, shortBlocks ? mode->maxLM : mode->maxLM-LM, B);
      }

      for (j=0;j<overlap;j++)
         out_mem[c][j] = x[j] + overlap_mem[c][j];
      for (;j<N;j++)
         out_mem[c][j] = x[j];
      for (j=0;j<overlap;j++)
         overlap_mem[c][j] = x[N+j];
   } while (++c<C);
   RESTORE_STACK;
}

static void preemphasis(const opus_val16 * OPUS_RESTRICT pcmp, celt_sig * OPUS_RESTRICT inp,
                        int N, int CC, int upsample, const opus_val16 *coef, celt_sig *mem, int clip)
{
   int i;
   opus_val16 coef0, coef1;
   celt_sig m;
   int Nu;

   coef0 = coef[0];
   coef1 = coef[1];


   Nu = N/upsample;
   if (upsample!=1)
   {
      for (i=0;i<N;i++)
         inp[i] = 0;
   }
   for (i=0;i<Nu;i++)
   {
      celt_sig x;

      x = SCALEIN(pcmp[CC*i]);
#ifndef FIXED_POINT
      /* Replace NaNs with zeros */
      if (!(x==x))
         x = 0;
#endif
      inp[i*upsample] = x;
   }

#ifndef FIXED_POINT
   if (clip)
   {
      /* Clip input to avoid encoding non-portable files */
      for (i=0;i<Nu;i++)
         inp[i*upsample] = MAX32(-65536.f, MIN32(65536.f,inp[i*upsample]));
   }
#endif
   m = *mem;
   if (coef1 == 0)
   {
      for (i=0;i<N;i++)
      {
         celt_sig x;
         x = SHL32(inp[i], SIG_SHIFT);
         /* Apply pre-emphasis */
         inp[i] = x + m;
         m = - MULT16_32_Q15(coef0, x);
      }
   } else {
      opus_val16 coef2 = coef[2];
      for (i=0;i<N;i++)
      {
         opus_val16 x, tmp;
         x = inp[i];
         /* Apply pre-emphasis */
         tmp = MULT16_16(coef2, x);
         inp[i] = tmp + m;
         m = MULT16_32_Q15(coef1, inp[i]) - MULT16_32_Q15(coef0, tmp);
      }
   }
   *mem = m;
}

static void deemphasis(celt_sig *in[], opus_val16 *pcm, int N, int C, int downsample, const opus_val16 *coef, celt_sig *mem, celt_sig * OPUS_RESTRICT scratch)
{
   int c;
   int Nd;
   opus_val16 coef0, coef1;

   coef0 = coef[0];
   coef1 = coef[1];
   Nd = N/downsample;
   c=0; do {
      int j;
      celt_sig * OPUS_RESTRICT x;
      opus_val16  * OPUS_RESTRICT y;
      celt_sig m = mem[c];
      x =in[c];
      y = pcm+c;
      /* Shortcut for the standard (non-custom modes) case */
      if (coef1 == 0)
      {
         for (j=0;j<N;j++)
         {
            celt_sig tmp = x[j] + m;
            m = MULT16_32_Q15(coef0, tmp);
            scratch[j] = tmp;
         }
      } else {
         opus_val16 coef3 = coef[3];
         for (j=0;j<N;j++)
         {
            celt_sig tmp = x[j] + m;
            m = MULT16_32_Q15(coef0, tmp)
              - MULT16_32_Q15(coef1, x[j]);
            tmp = SHL32(MULT16_32_Q15(coef3, tmp), 2);
            scratch[j] = tmp;
         }
      }
      mem[c] = m;

      /* Perform down-sampling */
      for (j=0;j<Nd;j++)
         y[j*C] = SCALEOUT(SIG2WORD16(scratch[j*downsample]));
   } while (++c<C);
}

static void comb_filter(opus_val32 *y, opus_val32 *x, int T0, int T1, int N,
      opus_val16 g0, opus_val16 g1, int tapset0, int tapset1,
      const opus_val16 *window, int overlap)
{
   int i;
   /* printf ("%d %d %f %f\n", T0, T1, g0, g1); */
   opus_val16 g00, g01, g02, g10, g11, g12;
   opus_val32 x0, x1, x2, x3, x4;
   static const opus_val16 gains[3][3] = {
         {QCONST16(0.3066406250f, 15), QCONST16(0.2170410156f, 15), QCONST16(0.1296386719f, 15)},
         {QCONST16(0.4638671875f, 15), QCONST16(0.2680664062f, 15), QCONST16(0.f, 15)},
         {QCONST16(0.7998046875f, 15), QCONST16(0.1000976562f, 15), QCONST16(0.f, 15)}};

   if (g0==0 && g1==0)
   {
      /* OPT: Happens to work without the OPUS_MOVE(), but only because the current encoder already copies x to y */
      if (x!=y)
         OPUS_MOVE(y, x, N);
      return;
   }
   g00 = MULT16_16_Q15(g0, gains[tapset0][0]);
   g01 = MULT16_16_Q15(g0, gains[tapset0][1]);
   g02 = MULT16_16_Q15(g0, gains[tapset0][2]);
   g10 = MULT16_16_Q15(g1, gains[tapset1][0]);
   g11 = MULT16_16_Q15(g1, gains[tapset1][1]);
   g12 = MULT16_16_Q15(g1, gains[tapset1][2]);
   x1 = x[-T1+1];
   x2 = x[-T1  ];
   x3 = x[-T1-1];
   x4 = x[-T1-2];
   for (i=0;i<overlap;i++)
   {
      opus_val16 f;
      x0=x[i-T1+2];
      f = MULT16_16_Q15(window[i],window[i]);
      y[i] = x[i]
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g00),x[i-T0])
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g01),ADD32(x[i-T0+1],x[i-T0-1]))
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g02),ADD32(x[i-T0+2],x[i-T0-2]))
               + MULT16_32_Q15(MULT16_16_Q15(f,g10),x2)
               + MULT16_32_Q15(MULT16_16_Q15(f,g11),ADD32(x1,x3))
               + MULT16_32_Q15(MULT16_16_Q15(f,g12),ADD32(x0,x4));
      x4=x3;
      x3=x2;
      x2=x1;
      x1=x0;

   }
   if (g1==0)
   {
      /* OPT: Happens to work without the OPUS_MOVE(), but only because the current encoder already copies x to y */
      if (x!=y)
         OPUS_MOVE(y+overlap, x+overlap, N-overlap);
      return;
   }
   /* OPT: For machines where the movs are costly, unroll by 5 */
   for (;i<N;i++)
   {
      x0=x[i-T1+2];
      y[i] = x[i]
               + MULT16_32_Q15(g10,x2)
               + MULT16_32_Q15(g11,ADD32(x1,x3))
               + MULT16_32_Q15(g12,ADD32(x0,x4));
      x4=x3;
      x3=x2;
      x2=x1;
      x1=x0;
   }
}

static const signed char tf_select_table[4][8] = {
      {0, -1, 0, -1,    0,-1, 0,-1},
      {0, -1, 0, -2,    1, 0, 1,-1},
      {0, -2, 0, -3,    2, 0, 1,-1},
      {0, -2, 0, -3,    3, 0, 1,-1},
};

static opus_val32 l1_metric(const celt_norm *tmp, int N, int LM, opus_val16 bias)
{
   int i;
   opus_val32 L1;
   L1 = 0;
   for (i=0;i<N;i++)
      L1 += EXTEND32(ABS16(tmp[i]));
   /* When in doubt, prefer good freq resolution */
   L1 = MAC16_32_Q15(L1, LM*bias, L1);
   return L1;

}

static int tf_analysis(const CELTMode *m, int len, int C, int isTransient,
      int *tf_res, int nbCompressedBytes, celt_norm *X, int N0, int LM,
      int *tf_sum, opus_val16 tf_estimate, int tf_chan)
{
   int i;
   VARDECL(int, metric);
   int cost0;
   int cost1;
   VARDECL(int, path0);
   VARDECL(int, path1);
   VARDECL(celt_norm, tmp);
   VARDECL(celt_norm, tmp_1);
   int lambda;
   int sel;
   int selcost[2];
   int tf_select=0;
   opus_val16 bias;

   SAVE_STACK;
   bias = MULT16_16_Q14(QCONST16(.04f,15), MAX16(-QCONST16(.25f,14), QCONST16(1.5f,14)-tf_estimate));
   /*printf("%f ", bias);*/

   if (nbCompressedBytes<15*C)
   {
      *tf_sum = 0;
      for (i=0;i<len;i++)
         tf_res[i] = isTransient;
      return 0;
   }
   if (nbCompressedBytes<40)
      lambda = 12;
   else if (nbCompressedBytes<60)
      lambda = 6;
   else if (nbCompressedBytes<100)
      lambda = 4;
   else
      lambda = 3;
   lambda*=2;
   ALLOC(metric, len, int);
   ALLOC(tmp, (m->eBands[len]-m->eBands[len-1])<<LM, celt_norm);
   ALLOC(tmp_1, (m->eBands[len]-m->eBands[len-1])<<LM, celt_norm);
   ALLOC(path0, len, int);
   ALLOC(path1, len, int);

   *tf_sum = 0;
   for (i=0;i<len;i++)
   {
      int j, k, N;
      int narrow;
      opus_val32 L1, best_L1;
      int best_level=0;
      N = (m->eBands[i+1]-m->eBands[i])<<LM;
      /* band is too narrow to be split down to LM=-1 */
      narrow = (m->eBands[i+1]-m->eBands[i])==1;
      for (j=0;j<N;j++)
         tmp[j] = X[tf_chan*N0 + j+(m->eBands[i]<<LM)];
      /* Just add the right channel if we're in stereo */
      /*if (C==2)
         for (j=0;j<N;j++)
            tmp[j] = ADD16(SHR16(tmp[j], 1),SHR16(X[N0+j+(m->eBands[i]<<LM)], 1));*/
      L1 = l1_metric(tmp, N, isTransient ? LM : 0, bias);
      best_L1 = L1;
      /* Check the -1 case for transients */
      if (isTransient && !narrow)
      {
         for (j=0;j<N;j++)
            tmp_1[j] = tmp[j];
         haar1(tmp_1, N>>LM, 1<<LM);
         L1 = l1_metric(tmp_1, N, LM+1, bias);
         if (L1<best_L1)
         {
            best_L1 = L1;
            best_level = -1;
         }
      }
      /*printf ("%f ", L1);*/
      for (k=0;k<LM+!(isTransient||narrow);k++)
      {
         int B;

         if (isTransient)
            B = (LM-k-1);
         else
            B = k+1;

         haar1(tmp, N>>k, 1<<k);

         L1 = l1_metric(tmp, N, B, bias);

         if (L1 < best_L1)
         {
            best_L1 = L1;
            best_level = k+1;
         }
      }
      /*printf ("%d ", isTransient ? LM-best_level : best_level);*/
      /* metric is in Q1 to be able to select the mid-point (-0.5) for narrower bands */
      if (isTransient)
         metric[i] = 2*best_level;
      else
         metric[i] = -2*best_level;
      *tf_sum += (isTransient ? LM : 0) - metric[i]/2;
      /* For bands that can't be split to -1, set the metric to the half-way point to avoid
         biasing the decision */
      if (narrow && (metric[i]==0 || metric[i]==-2*LM))
         metric[i]-=1;
      /*printf("%d ", metric[i]);*/
   }
   /*printf("\n");*/
   /* Search for the optimal tf resolution, including tf_select */
   tf_select = 0;
   for (sel=0;sel<2;sel++)
   {
      cost0 = 0;
      cost1 = isTransient ? 0 : lambda;
      for (i=1;i<len;i++)
      {
         int curr0, curr1;
         curr0 = IMIN(cost0, cost1 + lambda);
         curr1 = IMIN(cost0 + lambda, cost1);
         cost0 = curr0 + abs(metric[i]-2*tf_select_table[LM][4*isTransient+2*sel+0]);
         cost1 = curr1 + abs(metric[i]-2*tf_select_table[LM][4*isTransient+2*sel+1]);
      }
      cost0 = IMIN(cost0, cost1);
      selcost[sel]=cost0;
   }
   /* For now, we're conservative and only allow tf_select=1 for transients.
    * If tests confirm it's useful for non-transients, we could allow it. */
   if (selcost[1]<selcost[0] && isTransient)
      tf_select=1;
   cost0 = 0;
   cost1 = isTransient ? 0 : lambda;
   /* Viterbi forward pass */
   for (i=1;i<len;i++)
   {
      int curr0, curr1;
      int from0, from1;

      from0 = cost0;
      from1 = cost1 + lambda;
      if (from0 < from1)
      {
         curr0 = from0;
         path0[i]= 0;
      } else {
         curr0 = from1;
         path0[i]= 1;
      }

      from0 = cost0 + lambda;
      from1 = cost1;
      if (from0 < from1)
      {
         curr1 = from0;
         path1[i]= 0;
      } else {
         curr1 = from1;
         path1[i]= 1;
      }
      cost0 = curr0 + abs(metric[i]-2*tf_select_table[LM][4*isTransient+2*tf_select+0]);
      cost1 = curr1 + abs(metric[i]-2*tf_select_table[LM][4*isTransient+2*tf_select+1]);
   }
   tf_res[len-1] = cost0 < cost1 ? 0 : 1;
   /* Viterbi backward pass to check the decisions */
   for (i=len-2;i>=0;i--)
   {
      if (tf_res[i+1] == 1)
         tf_res[i] = path1[i+1];
      else
         tf_res[i] = path0[i+1];
   }
   /*printf("%d %f\n", *tf_sum, tf_estimate);*/
   RESTORE_STACK;
#ifdef FUZZING
   tf_select = rand()&0x1;
   tf_res[0] = rand()&0x1;
   for (i=1;i<len;i++)
      tf_res[i] = tf_res[i-1] ^ ((rand()&0xF) == 0);
#endif
   return tf_select;
}

static void tf_encode(int start, int end, int isTransient, int *tf_res, int LM, int tf_select, ec_enc *enc)
{
   int curr, i;
   int tf_select_rsv;
   int tf_changed;
   int logp;
   opus_uint32 budget;
   opus_uint32 tell;
   budget = enc->storage*8;
   tell = ec_tell(enc);
   logp = isTransient ? 2 : 4;
   /* Reserve space to code the tf_select decision. */
   tf_select_rsv = LM>0 && tell+logp+1 <= budget;
   budget -= tf_select_rsv;
   curr = tf_changed = 0;
   for (i=start;i<end;i++)
   {
      if (tell+logp<=budget)
      {
         ec_enc_bit_logp(enc, tf_res[i] ^ curr, logp);
         tell = ec_tell(enc);
         curr = tf_res[i];
         tf_changed |= curr;
      }
      else
         tf_res[i] = curr;
      logp = isTransient ? 4 : 5;
   }
   /* Only code tf_select if it would actually make a difference. */
   if (tf_select_rsv &&
         tf_select_table[LM][4*isTransient+0+tf_changed]!=
         tf_select_table[LM][4*isTransient+2+tf_changed])
      ec_enc_bit_logp(enc, tf_select, 1);
   else
      tf_select = 0;
   for (i=start;i<end;i++)
      tf_res[i] = tf_select_table[LM][4*isTransient+2*tf_select+tf_res[i]];
   /*for(i=0;i<end;i++)printf("%d ", isTransient ? tf_res[i] : LM+tf_res[i]);printf("\n");*/
}

static void tf_decode(int start, int end, int isTransient, int *tf_res, int LM, ec_dec *dec)
{
   int i, curr, tf_select;
   int tf_select_rsv;
   int tf_changed;
   int logp;
   opus_uint32 budget;
   opus_uint32 tell;

   budget = dec->storage*8;
   tell = ec_tell(dec);
   logp = isTransient ? 2 : 4;
   tf_select_rsv = LM>0 && tell+logp+1<=budget;
   budget -= tf_select_rsv;
   tf_changed = curr = 0;
   for (i=start;i<end;i++)
   {
      if (tell+logp<=budget)
      {
         curr ^= ec_dec_bit_logp(dec, logp);
         tell = ec_tell(dec);
         tf_changed |= curr;
      }
      tf_res[i] = curr;
      logp = isTransient ? 4 : 5;
   }
   tf_select = 0;
   if (tf_select_rsv &&
     tf_select_table[LM][4*isTransient+0+tf_changed] !=
     tf_select_table[LM][4*isTransient+2+tf_changed])
   {
      tf_select = ec_dec_bit_logp(dec, 1);
   }
   for (i=start;i<end;i++)
   {
      tf_res[i] = tf_select_table[LM][4*isTransient+2*tf_select+tf_res[i]];
   }
}

static void init_caps(const CELTMode *m,int *cap,int LM,int C)
{
   int i;
   for (i=0;i<m->nbEBands;i++)
   {
      int N;
      N=(m->eBands[i+1]-m->eBands[i])<<LM;
      cap[i] = (m->cache.caps[m->nbEBands*(2*LM+C-1)+i]+64)*C*N>>2;
   }
}

static int alloc_trim_analysis(const CELTMode *m, const celt_norm *X,
      const opus_val16 *bandLogE, int end, int LM, int C, int N0,
      AnalysisInfo *analysis, opus_val16 *stereo_saving, opus_val16 tf_estimate,
      int intensity)
{
   int i;
   opus_val32 diff=0;
   int c;
   int trim_index = 5;
   opus_val16 trim = QCONST16(5.f, 8);
   opus_val16 logXC, logXC2;
   if (C==2)
   {
      opus_val16 sum = 0; /* Q10 */
      opus_val16 minXC; /* Q10 */
      /* Compute inter-channel correlation for low frequencies */
      for (i=0;i<8;i++)
      {
         int j;
         opus_val32 partial = 0;
         for (j=m->eBands[i]<<LM;j<m->eBands[i+1]<<LM;j++)
            partial = MAC16_16(partial, X[j], X[N0+j]);
         sum = ADD16(sum, EXTRACT16(SHR32(partial, 18)));
      }
      sum = MULT16_16_Q15(QCONST16(1.f/8, 15), sum);
      sum = MIN16(QCONST16(1.f, 10), ABS16(sum));
      minXC = sum;
      for (i=8;i<intensity;i++)
      {
         int j;
         opus_val32 partial = 0;
         for (j=m->eBands[i]<<LM;j<m->eBands[i+1]<<LM;j++)
            partial = MAC16_16(partial, X[j], X[N0+j]);
         minXC = MIN16(minXC, ABS16(EXTRACT16(SHR32(partial, 18))));
      }
      minXC = MIN16(QCONST16(1.f, 10), ABS16(minXC));
      /*printf ("%f\n", sum);*/
      if (sum > QCONST16(.995f,10))
         trim_index-=4;
      else if (sum > QCONST16(.92f,10))
         trim_index-=3;
      else if (sum > QCONST16(.85f,10))
         trim_index-=2;
      else if (sum > QCONST16(.8f,10))
         trim_index-=1;
      /* mid-side savings estimations based on the LF average*/
      logXC = celt_log2(QCONST32(1.001f, 20)-MULT16_16(sum, sum));
      /* mid-side savings estimations based on min correlation */
      logXC2 = MAX16(HALF16(logXC), celt_log2(QCONST32(1.001f, 20)-MULT16_16(minXC, minXC)));
#ifdef FIXED_POINT
      /* Compensate for Q20 vs Q14 input and convert output to Q8 */
      logXC = PSHR32(logXC-QCONST16(6.f, DB_SHIFT),DB_SHIFT-8);
      logXC2 = PSHR32(logXC2-QCONST16(6.f, DB_SHIFT),DB_SHIFT-8);
#endif

      trim += MAX16(-QCONST16(4.f, 8), MULT16_16_Q15(QCONST16(.75f,15),logXC));
      *stereo_saving = MIN16(*stereo_saving + QCONST16(0.25f, 8), -HALF16(logXC2));
   }

   /* Estimate spectral tilt */
   c=0; do {
      for (i=0;i<end-1;i++)
      {
         diff += bandLogE[i+c*m->nbEBands]*(opus_int32)(2+2*i-end);
      }
   } while (++c<C);
   diff /= C*(end-1);
   /*printf("%f\n", diff);*/
   if (diff > QCONST16(2.f, DB_SHIFT))
      trim_index--;
   if (diff > QCONST16(8.f, DB_SHIFT))
      trim_index--;
   if (diff < -QCONST16(4.f, DB_SHIFT))
      trim_index++;
   if (diff < -QCONST16(10.f, DB_SHIFT))
      trim_index++;
   trim -= MAX16(-QCONST16(2.f, 8), MIN16(QCONST16(2.f, 8), SHR16(diff+QCONST16(1.f, DB_SHIFT),DB_SHIFT-8)/6 ));
   trim -= 2*SHR16(tf_estimate-QCONST16(1.f,14), 14-8);
#ifndef FIXED_POINT
   if (analysis->valid)
   {
      trim -= MAX16(-QCONST16(2.f, 8), MIN16(QCONST16(2.f, 8), 2*(analysis->tonality_slope+.05)));
   }
#endif

#ifdef FIXED_POINT
   trim_index = PSHR32(trim, 8);
#else
   trim_index = floor(.5+trim);
#endif
   if (trim_index<0)
      trim_index = 0;
   if (trim_index>10)
      trim_index = 10;
   /*printf("%d\n", trim_index);*/
#ifdef FUZZING
   trim_index = rand()%11;
#endif
   return trim_index;
}

static int stereo_analysis(const CELTMode *m, const celt_norm *X,
      int LM, int N0)
{
   int i;
   int thetas;
   opus_val32 sumLR = EPSILON, sumMS = EPSILON;

   /* Use the L1 norm to model the entropy of the L/R signal vs the M/S signal */
   for (i=0;i<13;i++)
   {
      int j;
      for (j=m->eBands[i]<<LM;j<m->eBands[i+1]<<LM;j++)
      {
         opus_val32 L, R, M, S;
         /* We cast to 32-bit first because of the -32768 case */
         L = EXTEND32(X[j]);
         R = EXTEND32(X[N0+j]);
         M = ADD32(L, R);
         S = SUB32(L, R);
         sumLR = ADD32(sumLR, ADD32(ABS32(L), ABS32(R)));
         sumMS = ADD32(sumMS, ADD32(ABS32(M), ABS32(S)));
      }
   }
   sumMS = MULT16_32_Q15(QCONST16(0.707107f, 15), sumMS);
   thetas = 13;
   /* We don't need thetas for lower bands with LM<=1 */
   if (LM<=1)
      thetas -= 8;
   return MULT16_32_Q15((m->eBands[13]<<(LM+1))+thetas, sumMS)
         > MULT16_32_Q15(m->eBands[13]<<(LM+1), sumLR);
}

static int run_prefilter(CELTEncoder *st, celt_sig *in, celt_sig *prefilter_mem, int CC, int N,
      int prefilter_tapset, int *pitch, opus_val16 *gain, int *qgain, int enabled, int nbAvailableBytes)
{
   int c;
   VARDECL(celt_sig, _pre);
   celt_sig *pre[2];
   const CELTMode *mode;
   int pitch_index;
   opus_val16 gain1;
   opus_val16 pf_threshold;
   int pf_on;
   int qg;
   SAVE_STACK;

   mode = st->mode;
   ALLOC(_pre, CC*(N+COMBFILTER_MAXPERIOD), celt_sig);

   pre[0] = _pre;
   pre[1] = _pre + (N+COMBFILTER_MAXPERIOD);


   c=0; do {
      OPUS_COPY(pre[c], prefilter_mem+c*COMBFILTER_MAXPERIOD, COMBFILTER_MAXPERIOD);
      OPUS_COPY(pre[c]+COMBFILTER_MAXPERIOD, in+c*(N+st->overlap)+st->overlap, N);
   } while (++c<CC);

   if (enabled)
   {
      VARDECL(opus_val16, pitch_buf);
      ALLOC(pitch_buf, (COMBFILTER_MAXPERIOD+N)>>1, opus_val16);

      pitch_downsample(pre, pitch_buf, COMBFILTER_MAXPERIOD+N, CC);
      /* Don't search for the fir last 1.5 octave of the range because
         there's too many false-positives due to short-term correlation */
      pitch_search(pitch_buf+(COMBFILTER_MAXPERIOD>>1), pitch_buf, N,
            COMBFILTER_MAXPERIOD-3*COMBFILTER_MINPERIOD, &pitch_index);
      pitch_index = COMBFILTER_MAXPERIOD-pitch_index;

      gain1 = remove_doubling(pitch_buf, COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD,
            N, &pitch_index, st->prefilter_period, st->prefilter_gain);
      if (pitch_index > COMBFILTER_MAXPERIOD-2)
         pitch_index = COMBFILTER_MAXPERIOD-2;
      gain1 = MULT16_16_Q15(QCONST16(.7f,15),gain1);
      /*printf("%d %d %f %f\n", pitch_change, pitch_index, gain1, st->analysis.tonality);*/
      if (st->loss_rate>2)
         gain1 = HALF32(gain1);
      if (st->loss_rate>4)
         gain1 = HALF32(gain1);
      if (st->loss_rate>8)
         gain1 = 0;
   } else {
      gain1 = 0;
      pitch_index = COMBFILTER_MINPERIOD;
   }

   /* Gain threshold for enabling the prefilter/postfilter */
   pf_threshold = QCONST16(.2f,15);

   /* Adjusting the threshold based on rate and continuity */
   if (abs(pitch_index-st->prefilter_period)*10>pitch_index)
      pf_threshold += QCONST16(.2f,15);
   if (nbAvailableBytes<25)
      pf_threshold += QCONST16(.1f,15);
   if (nbAvailableBytes<35)
      pf_threshold += QCONST16(.1f,15);
   if (st->prefilter_gain > QCONST16(.4f,15))
      pf_threshold -= QCONST16(.1f,15);
   if (st->prefilter_gain > QCONST16(.55f,15))
      pf_threshold -= QCONST16(.1f,15);

   /* Hard threshold at 0.2 */
   pf_threshold = MAX16(pf_threshold, QCONST16(.2f,15));
   if (gain1<pf_threshold)
   {
      gain1 = 0;
      pf_on = 0;
      qg = 0;
   } else {
      /*This block is not gated by a total bits check only because
        of the nbAvailableBytes check above.*/
      if (ABS16(gain1-st->prefilter_gain)<QCONST16(.1f,15))
         gain1=st->prefilter_gain;

#ifdef FIXED_POINT
      qg = ((gain1+1536)>>10)/3-1;
#else
      qg = (int)floor(.5f+gain1*32/3)-1;
#endif
      qg = IMAX(0, IMIN(7, qg));
      gain1 = QCONST16(0.09375f,15)*(qg+1);
      pf_on = 1;
   }
   /*printf("%d %f\n", pitch_index, gain1);*/

   c=0; do {
      int offset = mode->shortMdctSize-st->overlap;
      st->prefilter_period=IMAX(st->prefilter_period, COMBFILTER_MINPERIOD);
      OPUS_COPY(in+c*(N+st->overlap), st->in_mem+c*(st->overlap), st->overlap);
      if (offset)
         comb_filter(in+c*(N+st->overlap)+st->overlap, pre[c]+COMBFILTER_MAXPERIOD,
               st->prefilter_period, st->prefilter_period, offset, -st->prefilter_gain, -st->prefilter_gain,
               st->prefilter_tapset, st->prefilter_tapset, NULL, 0);

      comb_filter(in+c*(N+st->overlap)+st->overlap+offset, pre[c]+COMBFILTER_MAXPERIOD+offset,
            st->prefilter_period, pitch_index, N-offset, -st->prefilter_gain, -gain1,
            st->prefilter_tapset, prefilter_tapset, mode->window, st->overlap);
      OPUS_COPY(st->in_mem+c*(st->overlap), in+c*(N+st->overlap)+N, st->overlap);

      if (N>COMBFILTER_MAXPERIOD)
      {
         OPUS_MOVE(prefilter_mem+c*COMBFILTER_MAXPERIOD, pre[c]+N, COMBFILTER_MAXPERIOD);
      } else {
         OPUS_MOVE(prefilter_mem+c*COMBFILTER_MAXPERIOD, prefilter_mem+c*COMBFILTER_MAXPERIOD+N, COMBFILTER_MAXPERIOD-N);
         OPUS_MOVE(prefilter_mem+c*COMBFILTER_MAXPERIOD+COMBFILTER_MAXPERIOD-N, pre[c]+COMBFILTER_MAXPERIOD, N);
      }
   } while (++c<CC);

   RESTORE_STACK;
   *gain = gain1;
   *pitch = pitch_index;
   *qgain = qg;
   return pf_on;
}

int celt_encode_with_ec(CELTEncoder * OPUS_RESTRICT st, const opus_val16 * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
   int i, c, N;
   opus_int32 bits;
   ec_enc _enc;
   VARDECL(celt_sig, in);
   VARDECL(celt_sig, freq);
   VARDECL(celt_norm, X);
   VARDECL(celt_ener, bandE);
   VARDECL(opus_val16, bandLogE);
   VARDECL(opus_val16, bandLogE2);
   VARDECL(int, fine_quant);
   VARDECL(opus_val16, error);
   VARDECL(int, pulses);
   VARDECL(int, cap);
   VARDECL(int, offsets);
   VARDECL(int, fine_priority);
   VARDECL(int, tf_res);
   VARDECL(unsigned char, collapse_masks);
   celt_sig *prefilter_mem;
   opus_val16 *oldBandE, *oldLogE, *oldLogE2;
   int shortBlocks=0;
   int isTransient=0;
   const int CC = st->channels;
   const int C = st->stream_channels;
   int LM, M;
   int tf_select;
   int nbFilledBytes, nbAvailableBytes;
   int effEnd;
   int codedBands;
   int tf_sum;
   int alloc_trim;
   int pitch_index=COMBFILTER_MINPERIOD;
   opus_val16 gain1 = 0;
   int dual_stereo=0;
   int effectiveBytes;
   int dynalloc_logp;
   opus_int32 vbr_rate;
   opus_int32 total_bits;
   opus_int32 total_boost;
   opus_int32 balance;
   opus_int32 tell;
   int prefilter_tapset=0;
   int pf_on;
   int anti_collapse_rsv;
   int anti_collapse_on=0;
   int silence=0;
   int tf_chan = 0;
   opus_val16 tf_estimate;
   int pitch_change=0;
   opus_int32 tot_boost=0;
   opus_val16 sample_max;
   opus_val16 maxDepth;
   const OpusCustomMode *mode;
   int nbEBands;
   int overlap;
   const opus_int16 *eBands;
   ALLOC_STACK;

   mode = st->mode;
   nbEBands = mode->nbEBands;
   overlap = mode->overlap;
   eBands = mode->eBands;
   tf_estimate = QCONST16(1.0f,14);
   if (nbCompressedBytes<2 || pcm==NULL)
     return OPUS_BAD_ARG;

   frame_size *= st->upsample;
   for (LM=0;LM<=mode->maxLM;LM++)
      if (mode->shortMdctSize<<LM==frame_size)
         break;
   if (LM>mode->maxLM)
      return OPUS_BAD_ARG;
   M=1<<LM;
   N = M*mode->shortMdctSize;

   prefilter_mem = st->in_mem+CC*(st->overlap);
   oldBandE = (opus_val16*)(st->in_mem+CC*(st->overlap+COMBFILTER_MAXPERIOD));
   oldLogE = oldBandE + CC*nbEBands;
   oldLogE2 = oldLogE + CC*nbEBands;

   if (enc==NULL)
   {
      tell=1;
      nbFilledBytes=0;
   } else {
      tell=ec_tell(enc);
      nbFilledBytes=(tell+4)>>3;
   }

#ifdef CUSTOM_MODES
   if (st->signalling && enc==NULL)
   {
      int tmp = (mode->effEBands-st->end)>>1;
      st->end = IMAX(1, mode->effEBands-tmp);
      compressed[0] = tmp<<5;
      compressed[0] |= LM<<3;
      compressed[0] |= (C==2)<<2;
      /* Convert "standard mode" to Opus header */
      if (mode->Fs==48000 && mode->shortMdctSize==120)
      {
         int c0 = toOpus(compressed[0]);
         if (c0<0)
            return OPUS_BAD_ARG;
         compressed[0] = c0;
      }
      compressed++;
      nbCompressedBytes--;
   }
#else
   celt_assert(st->signalling==0);
#endif

   /* Can't produce more than 1275 output bytes */
   nbCompressedBytes = IMIN(nbCompressedBytes,1275);
   nbAvailableBytes = nbCompressedBytes - nbFilledBytes;

   if (st->vbr && st->bitrate!=OPUS_BITRATE_MAX)
   {
      opus_int32 den=mode->Fs>>BITRES;
      vbr_rate=(st->bitrate*frame_size+(den>>1))/den;
#ifdef CUSTOM_MODES
      if (st->signalling)
         vbr_rate -= 8<<BITRES;
#endif
      effectiveBytes = vbr_rate>>(3+BITRES);
   } else {
      opus_int32 tmp;
      vbr_rate = 0;
      tmp = st->bitrate*frame_size;
      if (tell>1)
         tmp += tell;
      if (st->bitrate!=OPUS_BITRATE_MAX)
         nbCompressedBytes = IMAX(2, IMIN(nbCompressedBytes,
               (tmp+4*mode->Fs)/(8*mode->Fs)-!!st->signalling));
      effectiveBytes = nbCompressedBytes;
   }

   if (enc==NULL)
   {
      ec_enc_init(&_enc, compressed, nbCompressedBytes);
      enc = &_enc;
   }

   if (vbr_rate>0)
   {
      /* Computes the max bit-rate allowed in VBR mode to avoid violating the
          target rate and buffering.
         We must do this up front so that bust-prevention logic triggers
          correctly if we don't have enough bits. */
      if (st->constrained_vbr)
      {
         opus_int32 vbr_bound;
         opus_int32 max_allowed;
         /* We could use any multiple of vbr_rate as bound (depending on the
             delay).
            This is clamped to ensure we use at least two bytes if the encoder
             was entirely empty, but to allow 0 in hybrid mode. */
         vbr_bound = vbr_rate;
         max_allowed = IMIN(IMAX(tell==1?2:0,
               (vbr_rate+vbr_bound-st->vbr_reservoir)>>(BITRES+3)),
               nbAvailableBytes);
         if(max_allowed < nbAvailableBytes)
         {
            nbCompressedBytes = nbFilledBytes+max_allowed;
            nbAvailableBytes = max_allowed;
            ec_enc_shrink(enc, nbCompressedBytes);
         }
      }
   }
   total_bits = nbCompressedBytes*8;

   effEnd = st->end;
   if (effEnd > mode->effEBands)
      effEnd = mode->effEBands;

   ALLOC(in, CC*(N+st->overlap), celt_sig);

   sample_max=MAX16(st->overlap_max, celt_maxabs16(pcm, C*(N-overlap)/st->upsample));
   st->overlap_max=celt_maxabs16(pcm+C*(N-overlap)/st->upsample, C*overlap/st->upsample);
   sample_max=MAX16(sample_max, st->overlap_max);
#ifdef FIXED_POINT
   silence = (sample_max==0);
#else
   silence = (sample_max <= (opus_val16)1/(1<<st->lsb_depth));
#endif
#ifdef FUZZING
   if ((rand()&0x3F)==0)
      silence = 1;
#endif
   if (tell==1)
      ec_enc_bit_logp(enc, silence, 15);
   else
      silence=0;
   if (silence)
   {
      /*In VBR mode there is no need to send more than the minimum. */
      if (vbr_rate>0)
      {
         effectiveBytes=nbCompressedBytes=IMIN(nbCompressedBytes, nbFilledBytes+2);
         total_bits=nbCompressedBytes*8;
         nbAvailableBytes=2;
         ec_enc_shrink(enc, nbCompressedBytes);
      }
      /* Pretend we've filled all the remaining bits with zeros
            (that's what the initialiser did anyway) */
      tell = nbCompressedBytes*8;
      enc->nbits_total+=tell-ec_tell(enc);
   }
   c=0; do {
      preemphasis(pcm+c, in+c*(N+st->overlap)+st->overlap, N, CC, st->upsample,
                  mode->preemph, st->preemph_memE+c, st->clip);
   } while (++c<CC);



   /* Find pitch period and gain */
   {
      int enabled;
      int qg;
      enabled = nbAvailableBytes>12*C && st->start==0 && !silence && !st->disable_pf && st->complexity >= 5;

      prefilter_tapset = st->tapset_decision;
      pf_on = run_prefilter(st, in, prefilter_mem, CC, N, prefilter_tapset, &pitch_index, &gain1, &qg, enabled, nbAvailableBytes);
      if ((gain1 > QCONST16(.4f,15) || st->prefilter_gain > QCONST16(.4f,15)) && st->analysis.tonality > .3
            && (pitch_index > 1.26*st->prefilter_period || pitch_index < .79*st->prefilter_period))
         pitch_change = 1;
      if (pf_on==0)
      {
         if(st->start==0 && tell+16<=total_bits)
            ec_enc_bit_logp(enc, 0, 1);
      } else {
         /*This block is not gated by a total bits check only because
           of the nbAvailableBytes check above.*/
         int octave;
         ec_enc_bit_logp(enc, 1, 1);
         pitch_index += 1;
         octave = EC_ILOG(pitch_index)-5;
         ec_enc_uint(enc, octave, 6);
         ec_enc_bits(enc, pitch_index-(16<<octave), 4+octave);
         pitch_index -= 1;
         ec_enc_bits(enc, qg, 3);
         ec_enc_icdf(enc, prefilter_tapset, tapset_icdf, 2);
      }
   }

   isTransient = 0;
   shortBlocks = 0;
   if (LM>0 && ec_tell(enc)+3<=total_bits)
   {
      if (st->complexity > 1)
      {
         isTransient = transient_analysis(in, N+st->overlap, CC,
                  &tf_estimate, &tf_chan);
         if (isTransient)
            shortBlocks = M;
      }
      ec_enc_bit_logp(enc, isTransient, 3);
   }

   ALLOC(freq, CC*N, celt_sig); /**< Interleaved signal MDCTs */
   ALLOC(bandE,nbEBands*CC, celt_ener);
   ALLOC(bandLogE,nbEBands*CC, opus_val16);
   /* Compute MDCTs */
   compute_mdcts(mode, shortBlocks, in, freq, CC, LM);

   if (CC==2&&C==1)
   {
      for (i=0;i<N;i++)
         freq[i] = ADD32(HALF32(freq[i]), HALF32(freq[N+i]));
      tf_chan = 0;
   }
   if (st->upsample != 1)
   {
      c=0; do
      {
         int bound = N/st->upsample;
         for (i=0;i<bound;i++)
            freq[c*N+i] *= st->upsample;
         for (;i<N;i++)
            freq[c*N+i] = 0;
      } while (++c<C);
   }
   compute_band_energies(mode, freq, bandE, effEnd, C, M);

   amp2Log2(mode, effEnd, st->end, bandE, bandLogE, C);
   /*for (i=0;i<21;i++)
      printf("%f ", bandLogE[i]);
   printf("\n");*/

   ALLOC(bandLogE2, C*nbEBands, opus_val16);
   if (shortBlocks && st->complexity>=8)
   {
      VARDECL(celt_sig, freq2);
      VARDECL(opus_val32, bandE2);
      ALLOC(freq2, CC*N, celt_sig);
      compute_mdcts(mode, 0, in, freq2, CC, LM);
      if (CC==2&&C==1)
      {
         for (i=0;i<N;i++)
            freq2[i] = ADD32(HALF32(freq2[i]), HALF32(freq2[N+i]));
      }
      if (st->upsample != 1)
      {
         c=0; do
         {
            int bound = N/st->upsample;
            for (i=0;i<bound;i++)
               freq2[c*N+i] *= st->upsample;
            for (;i<N;i++)
               freq2[c*N+i] = 0;
         } while (++c<C);
      }
      ALLOC(bandE2, C*nbEBands, opus_val32);
      compute_band_energies(mode, freq2, bandE2, effEnd, C, M);
      amp2Log2(mode, effEnd, st->end, bandE2, bandLogE2, C);
      for (i=0;i<C*nbEBands;i++)
         bandLogE2[i] += HALF16(SHL16(LM, DB_SHIFT));
   } else {
      for (i=0;i<C*nbEBands;i++)
         bandLogE2[i] = bandLogE[i];
   }

   ALLOC(X, C*N, celt_norm);         /**< Interleaved normalised MDCTs */

   /* Band normalisation */
   normalise_bands(mode, freq, X, bandE, effEnd, C, M);

   ALLOC(tf_res, nbEBands, int);
   tf_select = tf_analysis(mode, effEnd, C, isTransient, tf_res, effectiveBytes, X, N, LM, &tf_sum, tf_estimate, tf_chan);
   for (i=effEnd;i<st->end;i++)
      tf_res[i] = tf_res[effEnd-1];

   ALLOC(error, C*nbEBands, opus_val16);
   quant_coarse_energy(mode, st->start, st->end, effEnd, bandLogE,
         oldBandE, total_bits, error, enc,
         C, LM, nbAvailableBytes, st->force_intra,
         &st->delayedIntra, st->complexity >= 4, st->loss_rate);

   tf_encode(st->start, st->end, isTransient, tf_res, LM, tf_select, enc);

   if (ec_tell(enc)+4<=total_bits)
   {
      if (shortBlocks || st->complexity < 3 || nbAvailableBytes < 10*C)
      {
         if (st->complexity == 0)
            st->spread_decision = SPREAD_NONE;
      } else {
         if (st->analysis.valid)
         {
            static const opus_val16 spread_thresholds[3] = {-QCONST16(.6f, 15), -QCONST16(.2f, 15), -QCONST16(.07f, 15)};
            static const opus_val16 spread_histeresis[3] = {QCONST16(.15f, 15), QCONST16(.07f, 15), QCONST16(.02f, 15)};
            static const opus_val16 tapset_thresholds[2] = {QCONST16(.0f, 15), QCONST16(.15f, 15)};
            static const opus_val16 tapset_histeresis[2] = {QCONST16(.1f, 15), QCONST16(.05f, 15)};
            st->spread_decision = hysteresis_decision(-st->analysis.tonality, spread_thresholds, spread_histeresis, 3, st->spread_decision);
            st->tapset_decision = hysteresis_decision(st->analysis.tonality_slope, tapset_thresholds, tapset_histeresis, 2, st->tapset_decision);
         } else {
            st->spread_decision = spreading_decision(mode, X,
                  &st->tonal_average, st->spread_decision, &st->hf_average,
                  &st->tapset_decision, pf_on&&!shortBlocks, effEnd, C, M);
         }
         /*printf("%d %d\n", st->tapset_decision, st->spread_decision);*/
         /*printf("%f %d %f %d\n\n", st->analysis.tonality, st->spread_decision, st->analysis.tonality_slope, st->tapset_decision);*/
      }
      ec_enc_icdf(enc, st->spread_decision, spread_icdf, 5);
   }

   ALLOC(cap, nbEBands, int);
   ALLOC(offsets, nbEBands, int);

   init_caps(mode,cap,LM,C);
   for (i=0;i<nbEBands;i++)
      offsets[i] = 0;
   /* Dynamic allocation code */
   maxDepth=-QCONST16(32.f, DB_SHIFT);
   /* Make sure that dynamic allocation can't make us bust the budget */
   if (effectiveBytes > 50 && LM>=1)
   {
      int last=0;
      VARDECL(opus_val16, follower);
      ALLOC(follower, C*nbEBands, opus_val16);
      c=0;do
      {
         follower[c*nbEBands] = bandLogE2[c*nbEBands];
         for (i=1;i<st->end;i++)
         {
            /* The last band to be at least 3 dB higher than the previous one
               is the last we'll consider. Otherwise, we run into problems on
               bandlimited signals. */
            if (bandLogE2[c*nbEBands+i] > bandLogE2[c*nbEBands+i-1]+QCONST16(.5f,DB_SHIFT))
               last=i;
            follower[c*nbEBands+i] = MIN16(follower[c*nbEBands+i-1]+QCONST16(1.5f,DB_SHIFT), bandLogE2[c*nbEBands+i]);
         }
         for (i=last-1;i>=0;i--)
            follower[c*nbEBands+i] = MIN16(follower[c*nbEBands+i], MIN16(follower[c*nbEBands+i+1]+QCONST16(2.f,DB_SHIFT), bandLogE2[c*nbEBands+i]));
         for (i=0;i<st->end;i++)
         {
            opus_val16 noise_floor;
            /* Noise floor must take into account eMeans, the depth, the width of the bands
               and the preemphasis filter (approx. square of bark band ID) */
            noise_floor = MULT16_16(QCONST16(0.0625f, DB_SHIFT),mode->logN[i])
                  +QCONST16(.5f,DB_SHIFT)+SHL16(9-st->lsb_depth,DB_SHIFT)-SHL16(eMeans[i],6)
                  +MULT16_16(QCONST16(.0062,DB_SHIFT),(i+5)*(i+5));
            follower[c*nbEBands+i] = MAX16(follower[c*nbEBands+i], noise_floor);
            maxDepth = MAX16(maxDepth, bandLogE[c*nbEBands+i]-noise_floor);
         }
      } while (++c<C);
      if (C==2)
      {
         for (i=st->start;i<st->end;i++)
         {
            /* Consider 24 dB "cross-talk" */
            follower[nbEBands+i] = MAX16(follower[nbEBands+i], follower[                   i]-QCONST16(4.f,DB_SHIFT));
            follower[                   i] = MAX16(follower[                   i], follower[nbEBands+i]-QCONST16(4.f,DB_SHIFT));
            follower[i] = HALF16(MAX16(0, bandLogE[i]-follower[i]) + MAX16(0, bandLogE[nbEBands+i]-follower[nbEBands+i]));
         }
      } else {
         for (i=st->start;i<st->end;i++)
         {
            follower[i] = MAX16(0, bandLogE[i]-follower[i]);
         }
      }
      /* For non-transient CBR/CVBR frames, halve the dynalloc contribution */
      if ((!st->vbr || st->constrained_vbr)&&!isTransient)
      {
         for (i=st->start;i<st->end;i++)
            follower[i] = HALF16(follower[i]);
      }
      for (i=st->start;i<st->end;i++)
      {
         int width;
         int boost;
         int boost_bits;

         if (i<8)
            follower[i] *= 2;
         if (i>=12)
            follower[i] = HALF16(follower[i]);
         follower[i] = MIN16(follower[i], QCONST16(4, DB_SHIFT));

         width = C*(eBands[i+1]-eBands[i])<<LM;
         if (width<6)
         {
            boost = SHR32(EXTEND32(follower[i]),DB_SHIFT);
            boost_bits = boost*width<<BITRES;
         } else if (width > 48) {
            boost = SHR32(EXTEND32(follower[i])*8,DB_SHIFT);
            boost_bits = (boost*width<<BITRES)/8;
         } else {
            boost = SHR32(EXTEND32(follower[i])*width/6,DB_SHIFT);
            boost_bits = boost*6<<BITRES;
         }
         /* For CBR and non-transient CVBR frames, limit dynalloc to 1/4 of the bits */
         if ((!st->vbr || (st->constrained_vbr&&!isTransient))
               && (tot_boost+boost_bits)>>BITRES>>3 > effectiveBytes/4)
         {
            offsets[i] = 0;
            break;
         } else {
            offsets[i] = boost;
            tot_boost += boost_bits;
         }
      }
   }
   dynalloc_logp = 6;
   total_bits<<=BITRES;
   total_boost = 0;
   tell = ec_tell_frac(enc);
   for (i=st->start;i<st->end;i++)
   {
      int width, quanta;
      int dynalloc_loop_logp;
      int boost;
      int j;
      width = C*(eBands[i+1]-eBands[i])<<LM;
      /* quanta is 6 bits, but no more than 1 bit/sample
         and no less than 1/8 bit/sample */
      quanta = IMIN(width<<BITRES, IMAX(6<<BITRES, width));
      dynalloc_loop_logp = dynalloc_logp;
      boost = 0;
      for (j = 0; tell+(dynalloc_loop_logp<<BITRES) < total_bits-total_boost
            && boost < cap[i]; j++)
      {
         int flag;
         flag = j<offsets[i];
         ec_enc_bit_logp(enc, flag, dynalloc_loop_logp);
         tell = ec_tell_frac(enc);
         if (!flag)
            break;
         boost += quanta;
         total_boost += quanta;
         dynalloc_loop_logp = 1;
      }
      /* Making dynalloc more likely */
      if (j)
         dynalloc_logp = IMAX(2, dynalloc_logp-1);
      offsets[i] = boost;
   }

   if (C==2)
   {
      int effectiveRate;

      static const opus_val16 intensity_thresholds[21]=
      /* 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19  20  off*/
        { 16,21,23,25,27,29,31,33,35,38,42,46,50,54,58,63,68,75,84,102,130};
      static const opus_val16 intensity_histeresis[21]=
        {  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6,  8, 12};

      /* Always use MS for 2.5 ms frames until we can do a better analysis */
      if (LM!=0)
         dual_stereo = stereo_analysis(mode, X, LM, N);

      /* Account for coarse energy */
      effectiveRate = (8*effectiveBytes - 80)>>LM;

      /* effectiveRate in kb/s */
      effectiveRate = 2*effectiveRate/5;

      st->intensity = hysteresis_decision(effectiveRate, intensity_thresholds, intensity_histeresis, 21, st->intensity);
      st->intensity = IMIN(st->end,IMAX(st->start, st->intensity));
   }

   alloc_trim = 5;
   if (tell+(6<<BITRES) <= total_bits - total_boost)
   {
      alloc_trim = alloc_trim_analysis(mode, X, bandLogE,
            st->end, LM, C, N, &st->analysis, &st->stereo_saving, tf_estimate, st->intensity);
      ec_enc_icdf(enc, alloc_trim, trim_icdf, 7);
      tell = ec_tell_frac(enc);
   }

   /* Variable bitrate */
   if (vbr_rate>0)
   {
     opus_val16 alpha;
     opus_int32 delta;
     /* The target rate in 8th bits per frame */
     opus_int32 target, base_target;
     opus_int32 min_allowed;
     int coded_bins;
     int coded_bands;
     int lm_diff = mode->maxLM - LM;
     coded_bands = st->lastCodedBands ? st->lastCodedBands : nbEBands;
     coded_bins = eBands[coded_bands]<<LM;
     if (C==2)
        coded_bins += eBands[IMIN(st->intensity, coded_bands)]<<LM;

     /* Don't attempt to use more than 510 kb/s, even for frames smaller than 20 ms.
        The CELT allocator will just not be able to use more than that anyway. */
     nbCompressedBytes = IMIN(nbCompressedBytes,1275>>(3-LM));
     target = vbr_rate - ((40*C+20)<<BITRES);
     base_target = target;

     if (st->constrained_vbr)
        target += (st->vbr_offset>>lm_diff);

     /*printf("%f %f %f %f %d %d ", st->analysis.activity, st->analysis.tonality, tf_estimate, st->stereo_saving, tot_boost, coded_bands);*/
#ifndef FIXED_POINT
     if (st->analysis.valid && st->analysis.activity<.4)
        target -= (coded_bins<<BITRES)*1*(.4-st->analysis.activity);
#endif
     /* Stereo savings */
     if (C==2)
     {
        int coded_stereo_bands;
        int coded_stereo_dof;
        coded_stereo_bands = IMIN(st->intensity, coded_bands);
        coded_stereo_dof = (eBands[coded_stereo_bands]<<LM)-coded_stereo_bands;
        /*printf("%d %d %d ", coded_stereo_dof, coded_bins, tot_boost);*/
        target -= MIN32(target/3, SHR16(MULT16_16(st->stereo_saving,(coded_stereo_dof<<BITRES)),8));
        target += MULT16_16_Q15(QCONST16(0.035,15),coded_stereo_dof<<BITRES);
     }
     /* Limits starving of other bands when using dynalloc */
     target += tot_boost;
     /* Compensates for the average transient boost */
     target = MULT16_32_Q15(QCONST16(0.96f,15),target);
     /* Apply transient boost */
     target = SHL32(MULT16_32_Q15(tf_estimate, target),1);

#ifndef FIXED_POINT
     /* Apply tonality boost */
     if (st->analysis.valid) {
        int tonal_target;
        float tonal;

        /* Compensates for the average tonality boost */
        target -= MULT16_16_Q15(QCONST16(0.13f,15),coded_bins<<BITRES);

        tonal = MAX16(0,st->analysis.tonality-.2);
        tonal_target = target + (coded_bins<<BITRES)*2.0f*tonal;
        if (pitch_change)
           tonal_target +=  (coded_bins<<BITRES)*.8;
        /*printf("%f %f ", st->analysis.tonality, tonal);*/
        target = IMAX(tonal_target,target);
     }
#endif

     {
        opus_int32 floor_depth;
        int bins;
        bins = eBands[nbEBands-2]<<LM;
        /*floor_depth = SHR32(MULT16_16((C*bins<<BITRES),celt_log2(SHL32(MAX16(1,sample_max),13))), DB_SHIFT);*/
        floor_depth = SHR32(MULT16_16((C*bins<<BITRES),maxDepth), DB_SHIFT);
        floor_depth = IMAX(floor_depth, target>>2);
        target = IMIN(target, floor_depth);
        /*printf("%f %d\n", maxDepth, floor_depth);*/
     }

     if (st->constrained_vbr || st->bitrate<64000)
     {
        opus_val16 rate_factor;
#ifdef FIXED_POINT
        rate_factor = MAX16(0,(st->bitrate-32000));
#else
        rate_factor = MAX16(0,(1.f/32768)*(st->bitrate-32000));
#endif
        if (st->constrained_vbr)
           rate_factor = MIN16(rate_factor, QCONST16(0.67f, 15));
        target = base_target + MULT16_32_Q15(rate_factor, target-base_target);

     }
     /* Don't allow more than doubling the rate */
     target = IMIN(2*base_target, target);

     /* The current offset is removed from the target and the space used
        so far is added*/
     target=target+tell;
     /* In VBR mode the frame size must not be reduced so much that it would
         result in the encoder running out of bits.
        The margin of 2 bytes ensures that none of the bust-prevention logic
         in the decoder will have triggered so far. */
     min_allowed = ((tell+total_boost+(1<<(BITRES+3))-1)>>(BITRES+3)) + 2 - nbFilledBytes;

     nbAvailableBytes = (target+(1<<(BITRES+2)))>>(BITRES+3);
     nbAvailableBytes = IMAX(min_allowed,nbAvailableBytes);
     nbAvailableBytes = IMIN(nbCompressedBytes,nbAvailableBytes+nbFilledBytes) - nbFilledBytes;

     /* By how much did we "miss" the target on that frame */
     delta = target - vbr_rate;

     target=nbAvailableBytes<<(BITRES+3);

     /*If the frame is silent we don't adjust our drift, otherwise
       the encoder will shoot to very high rates after hitting a
       span of silence, but we do allow the bitres to refill.
       This means that we'll undershoot our target in CVBR/VBR modes
       on files with lots of silence. */
     if(silence)
     {
       nbAvailableBytes = 2;
       target = 2*8<<BITRES;
       delta = 0;
     }

     if (st->vbr_count < 970)
     {
        st->vbr_count++;
        alpha = celt_rcp(SHL32(EXTEND32(st->vbr_count+20),16));
     } else
        alpha = QCONST16(.001f,15);
     /* How many bits have we used in excess of what we're allowed */
     if (st->constrained_vbr)
        st->vbr_reservoir += target - vbr_rate;
     /*printf ("%d\n", st->vbr_reservoir);*/

     /* Compute the offset we need to apply in order to reach the target */
     if (st->constrained_vbr)
     {
        st->vbr_drift += (opus_int32)MULT16_32_Q15(alpha,(delta*(1<<lm_diff))-st->vbr_offset-st->vbr_drift);
        st->vbr_offset = -st->vbr_drift;
     }
     /*printf ("%d\n", st->vbr_drift);*/

     if (st->constrained_vbr && st->vbr_reservoir < 0)
     {
        /* We're under the min value -- increase rate */
        int adjust = (-st->vbr_reservoir)/(8<<BITRES);
        /* Unless we're just coding silence */
        nbAvailableBytes += silence?0:adjust;
        st->vbr_reservoir = 0;
        /*printf ("+%d\n", adjust);*/
     }
     nbCompressedBytes = IMIN(nbCompressedBytes,nbAvailableBytes+nbFilledBytes);
     /*printf("%d\n", nbCompressedBytes*50*8);*/
     /* This moves the raw bits to take into account the new compressed size */
     ec_enc_shrink(enc, nbCompressedBytes);
   }

   /* Bit allocation */
   ALLOC(fine_quant, nbEBands, int);
   ALLOC(pulses, nbEBands, int);
   ALLOC(fine_priority, nbEBands, int);

   /* bits =           packet size                    - where we are - safety*/
   bits = (((opus_int32)nbCompressedBytes*8)<<BITRES) - ec_tell_frac(enc) - 1;
   anti_collapse_rsv = isTransient&&LM>=2&&bits>=((LM+2)<<BITRES) ? (1<<BITRES) : 0;
   bits -= anti_collapse_rsv;
   codedBands = compute_allocation(mode, st->start, st->end, offsets, cap,
         alloc_trim, &st->intensity, &dual_stereo, bits, &balance, pulses,
         fine_quant, fine_priority, C, LM, enc, 1, st->lastCodedBands);
   st->lastCodedBands = codedBands;

   quant_fine_energy(mode, st->start, st->end, oldBandE, error, fine_quant, enc, C);

#ifdef MEASURE_NORM_MSE
   float X0[3000];
   float bandE0[60];
   c=0; do
      for (i=0;i<N;i++)
         X0[i+c*N] = X[i+c*N];
   while (++c<C);
   for (i=0;i<C*nbEBands;i++)
      bandE0[i] = bandE[i];
#endif

   /* Residual quantisation */
   ALLOC(collapse_masks, C*nbEBands, unsigned char);
   quant_all_bands(1, mode, st->start, st->end, X, C==2 ? X+N : NULL, collapse_masks,
         bandE, pulses, shortBlocks, st->spread_decision, dual_stereo, st->intensity, tf_res,
         nbCompressedBytes*(8<<BITRES)-anti_collapse_rsv, balance, enc, LM, codedBands, &st->rng);

   if (anti_collapse_rsv > 0)
   {
      anti_collapse_on = st->consec_transient<2;
#ifdef FUZZING
      anti_collapse_on = rand()&0x1;
#endif
      ec_enc_bits(enc, anti_collapse_on, 1);
   }
   quant_energy_finalise(mode, st->start, st->end, oldBandE, error, fine_quant, fine_priority, nbCompressedBytes*8-ec_tell(enc), enc, C);

   if (silence)
   {
      for (i=0;i<C*nbEBands;i++)
         oldBandE[i] = -QCONST16(28.f,DB_SHIFT);
   }

#ifdef RESYNTH
   /* Re-synthesis of the coded audio if required */
   {
      celt_sig *out_mem[2];
      celt_sig *overlap_mem[2];

      log2Amp(mode, st->start, st->end, bandE, oldBandE, C);
      if (silence)
      {
         for (i=0;i<C*nbEBands;i++)
            bandE[i] = 0;
      }

#ifdef MEASURE_NORM_MSE
      measure_norm_mse(mode, X, X0, bandE, bandE0, M, N, C);
#endif
      if (anti_collapse_on)
      {
         anti_collapse(mode, X, collapse_masks, LM, C, N,
               st->start, st->end, oldBandE, oldLogE, oldLogE2, pulses, st->rng);
      }

      /* Synthesis */
      denormalise_bands(mode, X, freq, bandE, effEnd, C, M);

      OPUS_MOVE(st->syn_mem[0], st->syn_mem[0]+N, MAX_PERIOD);
      if (CC==2)
         OPUS_MOVE(st->syn_mem[1], st->syn_mem[1]+N, MAX_PERIOD);

      c=0; do
         for (i=0;i<M*eBands[st->start];i++)
            freq[c*N+i] = 0;
      while (++c<C);
      c=0; do
         for (i=M*eBands[st->end];i<N;i++)
            freq[c*N+i] = 0;
      while (++c<C);

      if (CC==2&&C==1)
      {
         for (i=0;i<N;i++)
            freq[N+i] = freq[i];
      }

      out_mem[0] = st->syn_mem[0]+MAX_PERIOD;
      if (CC==2)
         out_mem[1] = st->syn_mem[1]+MAX_PERIOD;

      overlap_mem[0] = (celt_sig*)(oldLogE2 + CC*nbEBands);
      if (CC==2)
         overlap_mem[1] = overlap_mem[0] + st->overlap;

      compute_inv_mdcts(mode, shortBlocks, freq, out_mem, overlap_mem, CC, LM);

      c=0; do {
         st->prefilter_period=IMAX(st->prefilter_period, COMBFILTER_MINPERIOD);
         st->prefilter_period_old=IMAX(st->prefilter_period_old, COMBFILTER_MINPERIOD);
         comb_filter(out_mem[c], out_mem[c], st->prefilter_period_old, st->prefilter_period, mode->shortMdctSize,
               st->prefilter_gain_old, st->prefilter_gain, st->prefilter_tapset_old, st->prefilter_tapset,
               mode->window, st->overlap);
         if (LM!=0)
            comb_filter(out_mem[c]+mode->shortMdctSize, out_mem[c]+mode->shortMdctSize, st->prefilter_period, pitch_index, N-mode->shortMdctSize,
                  st->prefilter_gain, gain1, st->prefilter_tapset, prefilter_tapset,
                  mode->window, overlap);
      } while (++c<CC);

      /* We reuse freq[] as scratch space for the de-emphasis */
      deemphasis(out_mem, (opus_val16*)pcm, N, CC, st->upsample, mode->preemph, st->preemph_memD, freq);
      st->prefilter_period_old = st->prefilter_period;
      st->prefilter_gain_old = st->prefilter_gain;
      st->prefilter_tapset_old = st->prefilter_tapset;
   }
#endif

   st->prefilter_period = pitch_index;
   st->prefilter_gain = gain1;
   st->prefilter_tapset = prefilter_tapset;
#ifdef RESYNTH
   if (LM!=0)
   {
      st->prefilter_period_old = st->prefilter_period;
      st->prefilter_gain_old = st->prefilter_gain;
      st->prefilter_tapset_old = st->prefilter_tapset;
   }
#endif

   if (CC==2&&C==1) {
      for (i=0;i<nbEBands;i++)
         oldBandE[nbEBands+i]=oldBandE[i];
   }

   if (!isTransient)
   {
      for (i=0;i<CC*nbEBands;i++)
         oldLogE2[i] = oldLogE[i];
      for (i=0;i<CC*nbEBands;i++)
         oldLogE[i] = oldBandE[i];
   } else {
      for (i=0;i<CC*nbEBands;i++)
         oldLogE[i] = MIN16(oldLogE[i], oldBandE[i]);
   }
   /* In case start or end were to change */
   c=0; do
   {
      for (i=0;i<st->start;i++)
      {
         oldBandE[c*nbEBands+i]=0;
         oldLogE[c*nbEBands+i]=oldLogE2[c*nbEBands+i]=-QCONST16(28.f,DB_SHIFT);
      }
      for (i=st->end;i<nbEBands;i++)
      {
         oldBandE[c*nbEBands+i]=0;
         oldLogE[c*nbEBands+i]=oldLogE2[c*nbEBands+i]=-QCONST16(28.f,DB_SHIFT);
      }
   } while (++c<CC);

   if (isTransient)
      st->consec_transient++;
   else
      st->consec_transient=0;
   st->rng = enc->rng;

   /* If there's any room left (can only happen for very high rates),
      it's already filled with zeros */
   ec_enc_done(enc);

#ifdef CUSTOM_MODES
   if (st->signalling)
      nbCompressedBytes++;
#endif

   RESTORE_STACK;
   if (ec_get_error(enc))
      return OPUS_INTERNAL_ERROR;
   else
      return nbCompressedBytes;
}


#ifdef CUSTOM_MODES

#ifdef FIXED_POINT
int opus_custom_encode(CELTEncoder * OPUS_RESTRICT st, const opus_int16 * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes)
{
   return celt_encode_with_ec(st, pcm, frame_size, compressed, nbCompressedBytes, NULL);
}

#ifndef DISABLE_FLOAT_API
int opus_custom_encode_float(CELTEncoder * OPUS_RESTRICT st, const float * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes)
{
   int j, ret, C, N;
   VARDECL(opus_int16, in);
   ALLOC_STACK;

   if (pcm==NULL)
      return OPUS_BAD_ARG;

   C = st->channels;
   N = frame_size;
   ALLOC(in, C*N, opus_int16);

   for (j=0;j<C*N;j++)
     in[j] = FLOAT2INT16(pcm[j]);

   ret=celt_encode_with_ec(st,in,frame_size,compressed,nbCompressedBytes, NULL);
#ifdef RESYNTH
   for (j=0;j<C*N;j++)
      ((float*)pcm)[j]=in[j]*(1.f/32768.f);
#endif
   RESTORE_STACK;
   return ret;
}
#endif /* DISABLE_FLOAT_API */
#else

int opus_custom_encode(CELTEncoder * OPUS_RESTRICT st, const opus_int16 * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes)
{
   int j, ret, C, N;
   VARDECL(celt_sig, in);
   ALLOC_STACK;

   if (pcm==NULL)
      return OPUS_BAD_ARG;

   C=st->channels;
   N=frame_size;
   ALLOC(in, C*N, celt_sig);
   for (j=0;j<C*N;j++) {
     in[j] = SCALEOUT(pcm[j]);
   }

   ret = celt_encode_with_ec(st,in,frame_size,compressed,nbCompressedBytes, NULL);
#ifdef RESYNTH
   for (j=0;j<C*N;j++)
      ((opus_int16*)pcm)[j] = FLOAT2INT16(in[j]);
#endif
   RESTORE_STACK;
   return ret;
}

int opus_custom_encode_float(CELTEncoder * OPUS_RESTRICT st, const float * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes)
{
   return celt_encode_with_ec(st, pcm, frame_size, compressed, nbCompressedBytes, NULL);
}

#endif

#endif /* CUSTOM_MODES */

int opus_custom_encoder_ctl(CELTEncoder * OPUS_RESTRICT st, int request, ...)
{
   va_list ap;

   va_start(ap, request);
   switch (request)
   {
      case OPUS_SET_COMPLEXITY_REQUEST:
      {
         int value = va_arg(ap, opus_int32);
         if (value<0 || value>10)
            goto bad_arg;
         st->complexity = value;
      }
      break;
      case CELT_SET_START_BAND_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         if (value<0 || value>=st->mode->nbEBands)
            goto bad_arg;
         st->start = value;
      }
      break;
      case CELT_SET_END_BAND_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         if (value<1 || value>st->mode->nbEBands)
            goto bad_arg;
         st->end = value;
      }
      break;
      case CELT_SET_PREDICTION_REQUEST:
      {
         int value = va_arg(ap, opus_int32);
         if (value<0 || value>2)
            goto bad_arg;
         st->disable_pf = value<=1;
         st->force_intra = value==0;
      }
      break;
      case OPUS_SET_PACKET_LOSS_PERC_REQUEST:
      {
         int value = va_arg(ap, opus_int32);
         if (value<0 || value>100)
            goto bad_arg;
         st->loss_rate = value;
      }
      break;
      case OPUS_SET_VBR_CONSTRAINT_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         st->constrained_vbr = value;
      }
      break;
      case OPUS_SET_VBR_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         st->vbr = value;
      }
      break;
      case OPUS_SET_BITRATE_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         if (value<=500 && value!=OPUS_BITRATE_MAX)
            goto bad_arg;
         value = IMIN(value, 260000*st->channels);
         st->bitrate = value;
      }
      break;
      case CELT_SET_CHANNELS_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         if (value<1 || value>2)
            goto bad_arg;
         st->stream_channels = value;
      }
      break;
      case OPUS_SET_LSB_DEPTH_REQUEST:
      {
          opus_int32 value = va_arg(ap, opus_int32);
          if (value<8 || value>24)
             goto bad_arg;
          st->lsb_depth=value;
      }
      break;
      case OPUS_GET_LSB_DEPTH_REQUEST:
      {
          opus_int32 *value = va_arg(ap, opus_int32*);
          *value=st->lsb_depth;
      }
      break;
      case OPUS_RESET_STATE:
      {
         int i;
         opus_val16 *oldBandE, *oldLogE, *oldLogE2;
         oldBandE = (opus_val16*)(st->in_mem+st->channels*(st->overlap+COMBFILTER_MAXPERIOD));
         oldLogE = oldBandE + st->channels*st->mode->nbEBands;
         oldLogE2 = oldLogE + st->channels*st->mode->nbEBands;
         OPUS_CLEAR((char*)&st->ENCODER_RESET_START,
               opus_custom_encoder_get_size(st->mode, st->channels)-
               ((char*)&st->ENCODER_RESET_START - (char*)st));
         for (i=0;i<st->channels*st->mode->nbEBands;i++)
            oldLogE[i]=oldLogE2[i]=-QCONST16(28.f,DB_SHIFT);
         st->vbr_offset = 0;
         st->delayedIntra = 1;
         st->spread_decision = SPREAD_NORMAL;
         st->tonal_average = 256;
         st->hf_average = 0;
         st->tapset_decision = 0;
      }
      break;
#ifdef CUSTOM_MODES
      case CELT_SET_INPUT_CLIPPING_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         st->clip = value;
      }
      break;
#endif
      case CELT_SET_SIGNALLING_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         st->signalling = value;
      }
      break;
      case CELT_SET_ANALYSIS_REQUEST:
      {
         AnalysisInfo *info = va_arg(ap, AnalysisInfo *);
         if (info)
            OPUS_COPY(&st->analysis, info, 1);
      }
      break;
      case CELT_GET_MODE_REQUEST:
      {
         const CELTMode ** value = va_arg(ap, const CELTMode**);
         if (value==0)
            goto bad_arg;
         *value=st->mode;
      }
      break;
      case OPUS_GET_FINAL_RANGE_REQUEST:
      {
         opus_uint32 * value = va_arg(ap, opus_uint32 *);
         if (value==0)
            goto bad_arg;
         *value=st->rng;
      }
      break;
      default:
         goto bad_request;
   }
   va_end(ap);
   return OPUS_OK;
bad_arg:
   va_end(ap);
   return OPUS_BAD_ARG;
bad_request:
   va_end(ap);
   return OPUS_UNIMPLEMENTED;
}

/**********************************************************************/
/*                                                                    */
/*                             DECODER                                */
/*                                                                    */
/**********************************************************************/
#define DECODE_BUFFER_SIZE 2048

/** Decoder state
 @brief Decoder state
 */
struct OpusCustomDecoder {
   const OpusCustomMode *mode;
   int overlap;
   int channels;
   int stream_channels;

   int downsample;
   int start, end;
   int signalling;

   /* Everything beyond this point gets cleared on a reset */
#define DECODER_RESET_START rng

   opus_uint32 rng;
   int error;
   int last_pitch_index;
   int loss_count;
   int postfilter_period;
   int postfilter_period_old;
   opus_val16 postfilter_gain;
   opus_val16 postfilter_gain_old;
   int postfilter_tapset;
   int postfilter_tapset_old;

   celt_sig preemph_memD[2];

   celt_sig _decode_mem[1]; /* Size = channels*(DECODE_BUFFER_SIZE+mode->overlap) */
   /* opus_val16 lpc[],  Size = channels*LPC_ORDER */
   /* opus_val16 oldEBands[], Size = 2*mode->nbEBands */
   /* opus_val16 oldLogE[], Size = 2*mode->nbEBands */
   /* opus_val16 oldLogE2[], Size = 2*mode->nbEBands */
   /* opus_val16 backgroundLogE[], Size = 2*mode->nbEBands */
};

int celt_decoder_get_size(int channels)
{
   const CELTMode *mode = opus_custom_mode_create(48000, 960, NULL);
   return opus_custom_decoder_get_size(mode, channels);
}

OPUS_CUSTOM_NOSTATIC int opus_custom_decoder_get_size(const CELTMode *mode, int channels)
{
   int size = sizeof(struct CELTDecoder)
            + (channels*(DECODE_BUFFER_SIZE+mode->overlap)-1)*sizeof(celt_sig)
            + channels*LPC_ORDER*sizeof(opus_val16)
            + 4*2*mode->nbEBands*sizeof(opus_val16);
   return size;
}

#ifdef CUSTOM_MODES
CELTDecoder *opus_custom_decoder_create(const CELTMode *mode, int channels, int *error)
{
   int ret;
   CELTDecoder *st = (CELTDecoder *)opus_alloc(opus_custom_decoder_get_size(mode, channels));
   ret = opus_custom_decoder_init(st, mode, channels);
   if (ret != OPUS_OK)
   {
      opus_custom_decoder_destroy(st);
      st = NULL;
   }
   if (error)
      *error = ret;
   return st;
}
#endif /* CUSTOM_MODES */

int celt_decoder_init(CELTDecoder *st, opus_int32 sampling_rate, int channels)
{
   int ret;
   ret = opus_custom_decoder_init(st, opus_custom_mode_create(48000, 960, NULL), channels);
   if (ret != OPUS_OK)
      return ret;
   st->downsample = resampling_factor(sampling_rate);
   if (st->downsample==0)
      return OPUS_BAD_ARG;
   else
      return OPUS_OK;
}

OPUS_CUSTOM_NOSTATIC int opus_custom_decoder_init(CELTDecoder *st, const CELTMode *mode, int channels)
{
   if (channels < 0 || channels > 2)
      return OPUS_BAD_ARG;

   if (st==NULL)
      return OPUS_ALLOC_FAIL;

   OPUS_CLEAR((char*)st, opus_custom_decoder_get_size(mode, channels));

   st->mode = mode;
   st->overlap = mode->overlap;
   st->stream_channels = st->channels = channels;

   st->downsample = 1;
   st->start = 0;
   st->end = st->mode->effEBands;
   st->signalling = 1;

   st->loss_count = 0;

   opus_custom_decoder_ctl(st, OPUS_RESET_STATE);

   return OPUS_OK;
}

#ifdef CUSTOM_MODES
void opus_custom_decoder_destroy(CELTDecoder *st)
{
   opus_free(st);
}
#endif /* CUSTOM_MODES */

static void celt_decode_lost(CELTDecoder * OPUS_RESTRICT st, opus_val16 * OPUS_RESTRICT pcm, int N, int LM)
{
   int c;
   int pitch_index;
   opus_val16 fade = Q15ONE;
   int i, len;
   const int C = st->channels;
   int offset;
   celt_sig *out_mem[2];
   celt_sig *decode_mem[2];
   celt_sig *overlap_mem[2];
   opus_val16 *lpc;
   opus_val32 *out_syn[2];
   opus_val16 *oldBandE, *oldLogE, *oldLogE2, *backgroundLogE;
   const OpusCustomMode *mode;
   int nbEBands;
   int overlap;
   const opus_int16 *eBands;
   VARDECL(celt_sig, scratch);
   SAVE_STACK;

   mode = st->mode;
   nbEBands = mode->nbEBands;
   overlap = mode->overlap;
   eBands = mode->eBands;

   c=0; do {
      decode_mem[c] = st->_decode_mem + c*(DECODE_BUFFER_SIZE+st->overlap);
      out_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE-MAX_PERIOD;
      overlap_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE;
   } while (++c<C);
   lpc = (opus_val16*)(st->_decode_mem+(DECODE_BUFFER_SIZE+st->overlap)*C);
   oldBandE = lpc+C*LPC_ORDER;
   oldLogE = oldBandE + 2*nbEBands;
   oldLogE2 = oldLogE + 2*nbEBands;
   backgroundLogE = oldLogE2  + 2*nbEBands;

   out_syn[0] = out_mem[0]+MAX_PERIOD-N;
   if (C==2)
      out_syn[1] = out_mem[1]+MAX_PERIOD-N;

   len = N+overlap;

   if (st->loss_count >= 5 || st->start!=0)
   {
      /* Noise-based PLC/CNG */
      VARDECL(celt_sig, freq);
      VARDECL(celt_norm, X);
      VARDECL(celt_ener, bandE);
      opus_uint32 seed;
      int effEnd;

      effEnd = st->end;
      if (effEnd > mode->effEBands)
         effEnd = mode->effEBands;

      ALLOC(freq, C*N, celt_sig); /**< Interleaved signal MDCTs */
      ALLOC(X, C*N, celt_norm);   /**< Interleaved normalised MDCTs */
      ALLOC(bandE, nbEBands*C, celt_ener);

      if (st->loss_count >= 5)
         log2Amp(mode, st->start, st->end, bandE, backgroundLogE, C);
      else {
         /* Energy decay */
         opus_val16 decay = st->loss_count==0 ? QCONST16(1.5f, DB_SHIFT) : QCONST16(.5f, DB_SHIFT);
         c=0; do
         {
            for (i=st->start;i<st->end;i++)
               oldBandE[c*nbEBands+i] -= decay;
         } while (++c<C);
         log2Amp(mode, st->start, st->end, bandE, oldBandE, C);
      }
      seed = st->rng;
      for (c=0;c<C;c++)
      {
         for (i=0;i<(eBands[st->start]<<LM);i++)
            X[c*N+i] = 0;
         for (i=st->start;i<mode->effEBands;i++)
         {
            int j;
            int boffs;
            int blen;
            boffs = N*c+(eBands[i]<<LM);
            blen = (eBands[i+1]-eBands[i])<<LM;
            for (j=0;j<blen;j++)
            {
               seed = celt_lcg_rand(seed);
               X[boffs+j] = (celt_norm)((opus_int32)seed>>20);
            }
            renormalise_vector(X+boffs, blen, Q15ONE);
         }
         for (i=(eBands[st->end]<<LM);i<N;i++)
            X[c*N+i] = 0;
      }
      st->rng = seed;

      denormalise_bands(mode, X, freq, bandE, mode->effEBands, C, 1<<LM);

      c=0; do
         for (i=0;i<eBands[st->start]<<LM;i++)
            freq[c*N+i] = 0;
      while (++c<C);
      c=0; do {
         int bound = eBands[effEnd]<<LM;
         if (st->downsample!=1)
            bound = IMIN(bound, N/st->downsample);
         for (i=bound;i<N;i++)
            freq[c*N+i] = 0;
      } while (++c<C);
      compute_inv_mdcts(mode, 0, freq, out_syn, overlap_mem, C, LM);
   } else {
      /* Pitch-based PLC */
      VARDECL(opus_val32, e);

      if (st->loss_count == 0)
      {
         opus_val16 pitch_buf[DECODE_BUFFER_SIZE>>1];
         /* Corresponds to a min pitch of 67 Hz. It's possible to save CPU in this
         search by using only part of the decode buffer */
         int poffset = 720;
         pitch_downsample(decode_mem, pitch_buf, DECODE_BUFFER_SIZE, C);
         /* Max pitch is 100 samples (480 Hz) */
         pitch_search(pitch_buf+((poffset)>>1), pitch_buf, DECODE_BUFFER_SIZE-poffset,
               poffset-100, &pitch_index);
         pitch_index = poffset-pitch_index;
         st->last_pitch_index = pitch_index;
      } else {
         pitch_index = st->last_pitch_index;
         fade = QCONST16(.8f,15);
      }

      ALLOC(e, MAX_PERIOD+2*overlap, opus_val32);
      c=0; do {
         opus_val16 exc[MAX_PERIOD];
         opus_val32 ac[LPC_ORDER+1];
         opus_val16 decay = 1;
         opus_val32 S1=0;
         opus_val16 mem[LPC_ORDER]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

         offset = MAX_PERIOD-pitch_index;
         for (i=0;i<MAX_PERIOD;i++)
            exc[i] = ROUND16(out_mem[c][i], SIG_SHIFT);

         if (st->loss_count == 0)
         {
            _celt_autocorr(exc, ac, mode->window, overlap,
                  LPC_ORDER, MAX_PERIOD);

            /* Noise floor -40 dB */
#ifdef FIXED_POINT
            ac[0] += SHR32(ac[0],13);
#else
            ac[0] *= 1.0001f;
#endif
            /* Lag windowing */
            for (i=1;i<=LPC_ORDER;i++)
            {
               /*ac[i] *= exp(-.5*(2*M_PI*.002*i)*(2*M_PI*.002*i));*/
#ifdef FIXED_POINT
               ac[i] -= MULT16_32_Q15(2*i*i, ac[i]);
#else
               ac[i] -= ac[i]*(.008f*i)*(.008f*i);
#endif
            }

            _celt_lpc(lpc+c*LPC_ORDER, ac, LPC_ORDER);
         }
         for (i=0;i<LPC_ORDER;i++)
            mem[i] = ROUND16(out_mem[c][MAX_PERIOD-1-i], SIG_SHIFT);
         celt_fir(exc, lpc+c*LPC_ORDER, exc, MAX_PERIOD, LPC_ORDER, mem);
         /*for (i=0;i<MAX_PERIOD;i++)printf("%d ", exc[i]); printf("\n");*/
         /* Check if the waveform is decaying (and if so how fast) */
         {
            opus_val32 E1=1, E2=1;
            int period;
            if (pitch_index <= MAX_PERIOD/2)
               period = pitch_index;
            else
               period = MAX_PERIOD/2;
            for (i=0;i<period;i++)
            {
               E1 += SHR32(MULT16_16(exc[MAX_PERIOD-period+i],exc[MAX_PERIOD-period+i]),8);
               E2 += SHR32(MULT16_16(exc[MAX_PERIOD-2*period+i],exc[MAX_PERIOD-2*period+i]),8);
            }
            if (E1 > E2)
               E1 = E2;
            decay = celt_sqrt(frac_div32(SHR32(E1,1),E2));
         }

         /* Copy excitation, taking decay into account */
         for (i=0;i<len+overlap;i++)
         {
            opus_val16 tmp;
            if (offset+i >= MAX_PERIOD)
            {
               offset -= pitch_index;
               decay = MULT16_16_Q15(decay, decay);
            }
            e[i] = SHL32(EXTEND32(MULT16_16_Q15(decay, exc[offset+i])), SIG_SHIFT);
            tmp = ROUND16(out_mem[c][offset+i],SIG_SHIFT);
            S1 += SHR32(MULT16_16(tmp,tmp),8);
         }
         for (i=0;i<LPC_ORDER;i++)
            mem[i] = ROUND16(out_mem[c][MAX_PERIOD-1-i], SIG_SHIFT);
         for (i=0;i<len+overlap;i++)
            e[i] = MULT16_32_Q15(fade, e[i]);
         celt_iir(e, lpc+c*LPC_ORDER, e, len+overlap, LPC_ORDER, mem);

         {
            opus_val32 S2=0;
            for (i=0;i<len+overlap;i++)
            {
               opus_val16 tmp = ROUND16(e[i],SIG_SHIFT);
               S2 += SHR32(MULT16_16(tmp,tmp),8);
            }
            /* This checks for an "explosion" in the synthesis */
#ifdef FIXED_POINT
            if (!(S1 > SHR32(S2,2)))
#else
               /* Float test is written this way to catch NaNs at the same time */
               if (!(S1 > 0.2f*S2))
#endif
               {
                  for (i=0;i<len+overlap;i++)
                     e[i] = 0;
               } else if (S1 < S2)
               {
                  opus_val16 ratio = celt_sqrt(frac_div32(SHR32(S1,1)+1,S2+1));
                  for (i=0;i<len+overlap;i++)
                     e[i] = MULT16_32_Q15(ratio, e[i]);
               }
         }

         /* Apply post-filter to the MDCT overlap of the previous frame */
         comb_filter(out_mem[c]+MAX_PERIOD, out_mem[c]+MAX_PERIOD, st->postfilter_period, st->postfilter_period, st->overlap,
               st->postfilter_gain, st->postfilter_gain, st->postfilter_tapset, st->postfilter_tapset,
               NULL, 0);

         for (i=0;i<MAX_PERIOD+overlap-N;i++)
            out_mem[c][i] = out_mem[c][N+i];

         /* Apply TDAC to the concealed audio so that it blends with the
         previous and next frames */
         for (i=0;i<overlap/2;i++)
         {
            opus_val32 tmp;
            tmp = MULT16_32_Q15(mode->window[i],           e[N+overlap-1-i]) +
                  MULT16_32_Q15(mode->window[overlap-i-1], e[N+i          ]);
            out_mem[c][MAX_PERIOD+i] = MULT16_32_Q15(mode->window[overlap-i-1], tmp);
            out_mem[c][MAX_PERIOD+overlap-i-1] = MULT16_32_Q15(mode->window[i], tmp);
         }
         for (i=0;i<N;i++)
            out_mem[c][MAX_PERIOD-N+i] = e[i];

         /* Apply pre-filter to the MDCT overlap for the next frame (post-filter will be applied then) */
         comb_filter(e, out_mem[c]+MAX_PERIOD, st->postfilter_period, st->postfilter_period, st->overlap,
               -st->postfilter_gain, -st->postfilter_gain, st->postfilter_tapset, st->postfilter_tapset,
               NULL, 0);
         for (i=0;i<overlap;i++)
            out_mem[c][MAX_PERIOD+i] = e[i];
      } while (++c<C);
   }

   ALLOC(scratch, N, celt_sig);
   deemphasis(out_syn, pcm, N, C, st->downsample, mode->preemph, st->preemph_memD, scratch);

   st->loss_count++;

   RESTORE_STACK;
}

int celt_decode_with_ec(CELTDecoder * OPUS_RESTRICT st, const unsigned char *data, int len, opus_val16 * OPUS_RESTRICT pcm, int frame_size, ec_dec *dec)
{
   int c, i, N;
   int spread_decision;
   opus_int32 bits;
   ec_dec _dec;
   VARDECL(celt_sig, freq);
   VARDECL(celt_norm, X);
   VARDECL(celt_ener, bandE);
   VARDECL(int, fine_quant);
   VARDECL(int, pulses);
   VARDECL(int, cap);
   VARDECL(int, offsets);
   VARDECL(int, fine_priority);
   VARDECL(int, tf_res);
   VARDECL(unsigned char, collapse_masks);
   celt_sig *out_mem[2];
   celt_sig *decode_mem[2];
   celt_sig *overlap_mem[2];
   celt_sig *out_syn[2];
   opus_val16 *lpc;
   opus_val16 *oldBandE, *oldLogE, *oldLogE2, *backgroundLogE;

   int shortBlocks;
   int isTransient;
   int intra_ener;
   const int CC = st->channels;
   int LM, M;
   int effEnd;
   int codedBands;
   int alloc_trim;
   int postfilter_pitch;
   opus_val16 postfilter_gain;
   int intensity=0;
   int dual_stereo=0;
   opus_int32 total_bits;
   opus_int32 balance;
   opus_int32 tell;
   int dynalloc_logp;
   int postfilter_tapset;
   int anti_collapse_rsv;
   int anti_collapse_on=0;
   int silence;
   int C = st->stream_channels;
   const OpusCustomMode *mode;
   int nbEBands;
   int overlap;
   const opus_int16 *eBands;
   ALLOC_STACK;

   mode = st->mode;
   nbEBands = mode->nbEBands;
   overlap = mode->overlap;
   eBands = mode->eBands;
   frame_size *= st->downsample;

   c=0; do {
      decode_mem[c] = st->_decode_mem + c*(DECODE_BUFFER_SIZE+overlap);
      out_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE-MAX_PERIOD;
      overlap_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE;
   } while (++c<CC);
   lpc = (opus_val16*)(st->_decode_mem+(DECODE_BUFFER_SIZE+overlap)*CC);
   oldBandE = lpc+CC*LPC_ORDER;
   oldLogE = oldBandE + 2*nbEBands;
   oldLogE2 = oldLogE + 2*nbEBands;
   backgroundLogE = oldLogE2  + 2*nbEBands;

#ifdef CUSTOM_MODES
   if (st->signalling && data!=NULL)
   {
      int data0=data[0];
      /* Convert "standard mode" to Opus header */
      if (mode->Fs==48000 && mode->shortMdctSize==120)
      {
         data0 = fromOpus(data0);
         if (data0<0)
            return OPUS_INVALID_PACKET;
      }
      st->end = IMAX(1, mode->effEBands-2*(data0>>5));
      LM = (data0>>3)&0x3;
      C = 1 + ((data0>>2)&0x1);
      data++;
      len--;
      if (LM>mode->maxLM)
         return OPUS_INVALID_PACKET;
      if (frame_size < mode->shortMdctSize<<LM)
         return OPUS_BUFFER_TOO_SMALL;
      else
         frame_size = mode->shortMdctSize<<LM;
   } else {
#else
   {
#endif
      for (LM=0;LM<=mode->maxLM;LM++)
         if (mode->shortMdctSize<<LM==frame_size)
            break;
      if (LM>mode->maxLM)
         return OPUS_BAD_ARG;
   }
   M=1<<LM;

   if (len<0 || len>1275 || pcm==NULL)
      return OPUS_BAD_ARG;

   N = M*mode->shortMdctSize;

   effEnd = st->end;
   if (effEnd > mode->effEBands)
      effEnd = mode->effEBands;

   if (data == NULL || len<=1)
   {
      celt_decode_lost(st, pcm, N, LM);
      RESTORE_STACK;
      return frame_size/st->downsample;
   }

   ALLOC(freq, IMAX(CC,C)*N, celt_sig); /**< Interleaved signal MDCTs */
   ALLOC(X, C*N, celt_norm);   /**< Interleaved normalised MDCTs */
   ALLOC(bandE, nbEBands*C, celt_ener);
   c=0; do
      for (i=0;i<M*eBands[st->start];i++)
         X[c*N+i] = 0;
   while (++c<C);
   c=0; do
      for (i=M*eBands[effEnd];i<N;i++)
         X[c*N+i] = 0;
   while (++c<C);

   if (dec == NULL)
   {
      ec_dec_init(&_dec,(unsigned char*)data,len);
      dec = &_dec;
   }

   if (C==1)
   {
      for (i=0;i<nbEBands;i++)
         oldBandE[i]=MAX16(oldBandE[i],oldBandE[nbEBands+i]);
   }

   total_bits = len*8;
   tell = ec_tell(dec);

   if (tell >= total_bits)
      silence = 1;
   else if (tell==1)
      silence = ec_dec_bit_logp(dec, 15);
   else
      silence = 0;
   if (silence)
   {
      /* Pretend we've read all the remaining bits */
      tell = len*8;
      dec->nbits_total+=tell-ec_tell(dec);
   }

   postfilter_gain = 0;
   postfilter_pitch = 0;
   postfilter_tapset = 0;
   if (st->start==0 && tell+16 <= total_bits)
   {
      if(ec_dec_bit_logp(dec, 1))
      {
         int qg, octave;
         octave = ec_dec_uint(dec, 6);
         postfilter_pitch = (16<<octave)+ec_dec_bits(dec, 4+octave)-1;
         qg = ec_dec_bits(dec, 3);
         if (ec_tell(dec)+2<=total_bits)
            postfilter_tapset = ec_dec_icdf(dec, tapset_icdf, 2);
         postfilter_gain = QCONST16(.09375f,15)*(qg+1);
      }
      tell = ec_tell(dec);
   }

   if (LM > 0 && tell+3 <= total_bits)
   {
      isTransient = ec_dec_bit_logp(dec, 3);
      tell = ec_tell(dec);
   }
   else
      isTransient = 0;

   if (isTransient)
      shortBlocks = M;
   else
      shortBlocks = 0;

   /* Decode the global flags (first symbols in the stream) */
   intra_ener = tell+3<=total_bits ? ec_dec_bit_logp(dec, 3) : 0;
   /* Get band energies */
   unquant_coarse_energy(mode, st->start, st->end, oldBandE,
         intra_ener, dec, C, LM);

   ALLOC(tf_res, nbEBands, int);
   tf_decode(st->start, st->end, isTransient, tf_res, LM, dec);

   tell = ec_tell(dec);
   spread_decision = SPREAD_NORMAL;
   if (tell+4 <= total_bits)
      spread_decision = ec_dec_icdf(dec, spread_icdf, 5);

   ALLOC(pulses, nbEBands, int);
   ALLOC(cap, nbEBands, int);
   ALLOC(offsets, nbEBands, int);
   ALLOC(fine_priority, nbEBands, int);

   init_caps(mode,cap,LM,C);

   dynalloc_logp = 6;
   total_bits<<=BITRES;
   tell = ec_tell_frac(dec);
   for (i=st->start;i<st->end;i++)
   {
      int width, quanta;
      int dynalloc_loop_logp;
      int boost;
      width = C*(eBands[i+1]-eBands[i])<<LM;
      /* quanta is 6 bits, but no more than 1 bit/sample
         and no less than 1/8 bit/sample */
      quanta = IMIN(width<<BITRES, IMAX(6<<BITRES, width));
      dynalloc_loop_logp = dynalloc_logp;
      boost = 0;
      while (tell+(dynalloc_loop_logp<<BITRES) < total_bits && boost < cap[i])
      {
         int flag;
         flag = ec_dec_bit_logp(dec, dynalloc_loop_logp);
         tell = ec_tell_frac(dec);
         if (!flag)
            break;
         boost += quanta;
         total_bits -= quanta;
         dynalloc_loop_logp = 1;
      }
      offsets[i] = boost;
      /* Making dynalloc more likely */
      if (boost>0)
         dynalloc_logp = IMAX(2, dynalloc_logp-1);
   }

   ALLOC(fine_quant, nbEBands, int);
   alloc_trim = tell+(6<<BITRES) <= total_bits ?
         ec_dec_icdf(dec, trim_icdf, 7) : 5;

   bits = (((opus_int32)len*8)<<BITRES) - ec_tell_frac(dec) - 1;
   anti_collapse_rsv = isTransient&&LM>=2&&bits>=((LM+2)<<BITRES) ? (1<<BITRES) : 0;
   bits -= anti_collapse_rsv;
   codedBands = compute_allocation(mode, st->start, st->end, offsets, cap,
         alloc_trim, &intensity, &dual_stereo, bits, &balance, pulses,
         fine_quant, fine_priority, C, LM, dec, 0, 0);

   unquant_fine_energy(mode, st->start, st->end, oldBandE, fine_quant, dec, C);

   /* Decode fixed codebook */
   ALLOC(collapse_masks, C*nbEBands, unsigned char);
   quant_all_bands(0, mode, st->start, st->end, X, C==2 ? X+N : NULL, collapse_masks,
         NULL, pulses, shortBlocks, spread_decision, dual_stereo, intensity, tf_res,
         len*(8<<BITRES)-anti_collapse_rsv, balance, dec, LM, codedBands, &st->rng);

   if (anti_collapse_rsv > 0)
   {
      anti_collapse_on = ec_dec_bits(dec, 1);
   }

   unquant_energy_finalise(mode, st->start, st->end, oldBandE,
         fine_quant, fine_priority, len*8-ec_tell(dec), dec, C);

   if (anti_collapse_on)
      anti_collapse(mode, X, collapse_masks, LM, C, N,
            st->start, st->end, oldBandE, oldLogE, oldLogE2, pulses, st->rng);

   log2Amp(mode, st->start, st->end, bandE, oldBandE, C);

   if (silence)
   {
      for (i=0;i<C*nbEBands;i++)
      {
         bandE[i] = 0;
         oldBandE[i] = -QCONST16(28.f,DB_SHIFT);
      }
   }
   /* Synthesis */
   denormalise_bands(mode, X, freq, bandE, effEnd, C, M);

   OPUS_MOVE(decode_mem[0], decode_mem[0]+N, DECODE_BUFFER_SIZE-N);
   if (CC==2)
      OPUS_MOVE(decode_mem[1], decode_mem[1]+N, DECODE_BUFFER_SIZE-N);

   c=0; do
      for (i=0;i<M*eBands[st->start];i++)
         freq[c*N+i] = 0;
   while (++c<C);
   c=0; do {
      int bound = M*eBands[effEnd];
      if (st->downsample!=1)
         bound = IMIN(bound, N/st->downsample);
      for (i=bound;i<N;i++)
         freq[c*N+i] = 0;
   } while (++c<C);

   out_syn[0] = out_mem[0]+MAX_PERIOD-N;
   if (CC==2)
      out_syn[1] = out_mem[1]+MAX_PERIOD-N;

   if (CC==2&&C==1)
   {
      for (i=0;i<N;i++)
         freq[N+i] = freq[i];
   }
   if (CC==1&&C==2)
   {
      for (i=0;i<N;i++)
         freq[i] = HALF32(ADD32(freq[i],freq[N+i]));
   }

   /* Compute inverse MDCTs */
   compute_inv_mdcts(mode, shortBlocks, freq, out_syn, overlap_mem, CC, LM);

   c=0; do {
      st->postfilter_period=IMAX(st->postfilter_period, COMBFILTER_MINPERIOD);
      st->postfilter_period_old=IMAX(st->postfilter_period_old, COMBFILTER_MINPERIOD);
      comb_filter(out_syn[c], out_syn[c], st->postfilter_period_old, st->postfilter_period, mode->shortMdctSize,
            st->postfilter_gain_old, st->postfilter_gain, st->postfilter_tapset_old, st->postfilter_tapset,
            mode->window, overlap);
      if (LM!=0)
         comb_filter(out_syn[c]+mode->shortMdctSize, out_syn[c]+mode->shortMdctSize, st->postfilter_period, postfilter_pitch, N-mode->shortMdctSize,
               st->postfilter_gain, postfilter_gain, st->postfilter_tapset, postfilter_tapset,
               mode->window, overlap);

   } while (++c<CC);
   st->postfilter_period_old = st->postfilter_period;
   st->postfilter_gain_old = st->postfilter_gain;
   st->postfilter_tapset_old = st->postfilter_tapset;
   st->postfilter_period = postfilter_pitch;
   st->postfilter_gain = postfilter_gain;
   st->postfilter_tapset = postfilter_tapset;
   if (LM!=0)
   {
      st->postfilter_period_old = st->postfilter_period;
      st->postfilter_gain_old = st->postfilter_gain;
      st->postfilter_tapset_old = st->postfilter_tapset;
   }

   if (C==1) {
      for (i=0;i<nbEBands;i++)
         oldBandE[nbEBands+i]=oldBandE[i];
   }

   /* In case start or end were to change */
   if (!isTransient)
   {
      for (i=0;i<2*nbEBands;i++)
         oldLogE2[i] = oldLogE[i];
      for (i=0;i<2*nbEBands;i++)
         oldLogE[i] = oldBandE[i];
      for (i=0;i<2*nbEBands;i++)
         backgroundLogE[i] = MIN16(backgroundLogE[i] + M*QCONST16(0.001f,DB_SHIFT), oldBandE[i]);
   } else {
      for (i=0;i<2*nbEBands;i++)
         oldLogE[i] = MIN16(oldLogE[i], oldBandE[i]);
   }
   c=0; do
   {
      for (i=0;i<st->start;i++)
      {
         oldBandE[c*nbEBands+i]=0;
         oldLogE[c*nbEBands+i]=oldLogE2[c*nbEBands+i]=-QCONST16(28.f,DB_SHIFT);
      }
      for (i=st->end;i<nbEBands;i++)
      {
         oldBandE[c*nbEBands+i]=0;
         oldLogE[c*nbEBands+i]=oldLogE2[c*nbEBands+i]=-QCONST16(28.f,DB_SHIFT);
      }
   } while (++c<2);
   st->rng = dec->rng;

   /* We reuse freq[] as scratch space for the de-emphasis */
   deemphasis(out_syn, pcm, N, CC, st->downsample, mode->preemph, st->preemph_memD, freq);
   st->loss_count = 0;
   RESTORE_STACK;
   if (ec_tell(dec) > 8*len)
      return OPUS_INTERNAL_ERROR;
   if(ec_get_error(dec))
      st->error = 1;
   return frame_size/st->downsample;
}


#ifdef CUSTOM_MODES

#ifdef FIXED_POINT
int opus_custom_decode(CELTDecoder * OPUS_RESTRICT st, const unsigned char *data, int len, opus_int16 * OPUS_RESTRICT pcm, int frame_size)
{
   return celt_decode_with_ec(st, data, len, pcm, frame_size, NULL);
}

#ifndef DISABLE_FLOAT_API
int opus_custom_decode_float(CELTDecoder * OPUS_RESTRICT st, const unsigned char *data, int len, float * OPUS_RESTRICT pcm, int frame_size)
{
   int j, ret, C, N;
   VARDECL(opus_int16, out);
   ALLOC_STACK;

   if (pcm==NULL)
      return OPUS_BAD_ARG;

   C = st->channels;
   N = frame_size;

   ALLOC(out, C*N, opus_int16);
   ret=celt_decode_with_ec(st, data, len, out, frame_size, NULL);
   if (ret>0)
      for (j=0;j<C*ret;j++)
         pcm[j]=out[j]*(1.f/32768.f);

   RESTORE_STACK;
   return ret;
}
#endif /* DISABLE_FLOAT_API */

#else

int opus_custom_decode_float(CELTDecoder * OPUS_RESTRICT st, const unsigned char *data, int len, float * OPUS_RESTRICT pcm, int frame_size)
{
   return celt_decode_with_ec(st, data, len, pcm, frame_size, NULL);
}

int opus_custom_decode(CELTDecoder * OPUS_RESTRICT st, const unsigned char *data, int len, opus_int16 * OPUS_RESTRICT pcm, int frame_size)
{
   int j, ret, C, N;
   VARDECL(celt_sig, out);
   ALLOC_STACK;

   if (pcm==NULL)
      return OPUS_BAD_ARG;

   C = st->channels;
   N = frame_size;
   ALLOC(out, C*N, celt_sig);

   ret=celt_decode_with_ec(st, data, len, out, frame_size, NULL);

   if (ret>0)
      for (j=0;j<C*ret;j++)
         pcm[j] = FLOAT2INT16 (out[j]);

   RESTORE_STACK;
   return ret;
}

#endif
#endif /* CUSTOM_MODES */

int opus_custom_decoder_ctl(CELTDecoder * OPUS_RESTRICT st, int request, ...)
{
   va_list ap;

   va_start(ap, request);
   switch (request)
   {
      case CELT_SET_START_BAND_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         if (value<0 || value>=st->mode->nbEBands)
            goto bad_arg;
         st->start = value;
      }
      break;
      case CELT_SET_END_BAND_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         if (value<1 || value>st->mode->nbEBands)
            goto bad_arg;
         st->end = value;
      }
      break;
      case CELT_SET_CHANNELS_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         if (value<1 || value>2)
            goto bad_arg;
         st->stream_channels = value;
      }
      break;
      case CELT_GET_AND_CLEAR_ERROR_REQUEST:
      {
         opus_int32 *value = va_arg(ap, opus_int32*);
         if (value==NULL)
            goto bad_arg;
         *value=st->error;
         st->error = 0;
      }
      break;
      case OPUS_GET_LOOKAHEAD_REQUEST:
      {
         opus_int32 *value = va_arg(ap, opus_int32*);
         if (value==NULL)
            goto bad_arg;
         *value = st->overlap/st->downsample;
      }
      break;
      case OPUS_RESET_STATE:
      {
         int i;
         opus_val16 *lpc, *oldBandE, *oldLogE, *oldLogE2;
         lpc = (opus_val16*)(st->_decode_mem+(DECODE_BUFFER_SIZE+st->overlap)*st->channels);
         oldBandE = lpc+st->channels*LPC_ORDER;
         oldLogE = oldBandE + 2*st->mode->nbEBands;
         oldLogE2 = oldLogE + 2*st->mode->nbEBands;
         OPUS_CLEAR((char*)&st->DECODER_RESET_START,
               opus_custom_decoder_get_size(st->mode, st->channels)-
               ((char*)&st->DECODER_RESET_START - (char*)st));
         for (i=0;i<2*st->mode->nbEBands;i++)
            oldLogE[i]=oldLogE2[i]=-QCONST16(28.f,DB_SHIFT);
      }
      break;
      case OPUS_GET_PITCH_REQUEST:
      {
         opus_int32 *value = va_arg(ap, opus_int32*);
         if (value==NULL)
            goto bad_arg;
         *value = st->postfilter_period;
      }
      break;
#ifdef OPUS_BUILD
      case CELT_GET_MODE_REQUEST:
      {
         const CELTMode ** value = va_arg(ap, const CELTMode**);
         if (value==0)
            goto bad_arg;
         *value=st->mode;
      }
      break;
      case CELT_SET_SIGNALLING_REQUEST:
      {
         opus_int32 value = va_arg(ap, opus_int32);
         st->signalling = value;
      }
      break;
      case OPUS_GET_FINAL_RANGE_REQUEST:
      {
         opus_uint32 * value = va_arg(ap, opus_uint32 *);
         if (value==0)
            goto bad_arg;
         *value=st->rng;
      }
      break;
#endif
      default:
         goto bad_request;
   }
   va_end(ap);
   return OPUS_OK;
bad_arg:
   va_end(ap);
   return OPUS_BAD_ARG;
bad_request:
      va_end(ap);
  return OPUS_UNIMPLEMENTED;
}



const char *opus_strerror(int error)
{
   static const char * const error_strings[8] = {
      "success",
      "invalid argument",
      "buffer too small",
      "internal error",
      "corrupted stream",
      "request not implemented",
      "invalid state",
      "memory allocation failed"
   };
   if (error > 0 || error < -7)
      return "unknown error";
   else
      return error_strings[-error];
}

const char *opus_get_version_string(void)
{
    return "libopus " OPUS_VERSION
#ifdef FIXED_POINT
          "-fixed"
#endif
#ifdef FUZZING
          "-fuzzing"
#endif
          ;
}
