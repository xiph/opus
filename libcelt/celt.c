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

#define CELT_C

/* Always enable postfilter for Opus */
#if defined(OPUS_BUILD) && !defined(ENABLE_POSTFILTER)
#define ENABLE_POSTFILTER
#endif

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
#include "plc.h"
#include "vq.h"

static const unsigned char trim_icdf[11] = {126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0};
/* Probs: NONE: 21.875%, LIGHT: 6.25%, NORMAL: 65.625%, AGGRESSIVE: 6.25% */
static const unsigned char spread_icdf[4] = {25, 23, 2, 0};

static const unsigned char tapset_icdf[3]={2,1,0};

#define COMBFILTER_MAXPERIOD 1024
#define COMBFILTER_MINPERIOD 15

static int resampling_factor(celt_int32 rate)
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
      ret = 0;
   }
   return ret;
}

/** Encoder state 
 @brief Encoder state
 */
struct CELTEncoder {
   const CELTMode *mode;     /**< Mode used by the encoder */
   int overlap;
   int channels;
   int stream_channels;
   
   int force_intra;
   int clip;
   int disable_pf;
   int complexity;
   int upsample;
   int start, end;

   celt_int32 bitrate;
   int vbr;
   int signalling;
   int constrained_vbr;      /* If zero, VBR can do whatever it likes with the rate */

   /* Everything beyond this point gets cleared on a reset */
#define ENCODER_RESET_START rng

   celt_uint32 rng;
   int spread_decision;
   int delayedIntra;
   int tonal_average;
   int lastCodedBands;
   int hf_average;
   int tapset_decision;

   int prefilter_period;
   celt_word16 prefilter_gain;
   int prefilter_tapset;
#ifdef RESYNTH
   int prefilter_period_old;
   celt_word16 prefilter_gain_old;
   int prefilter_tapset_old;
#endif
   int consec_transient;

   /* VBR-related parameters */
   celt_int32 vbr_reservoir;
   celt_int32 vbr_drift;
   celt_int32 vbr_offset;
   celt_int32 vbr_count;

   celt_word32 preemph_memE[2];
   celt_word32 preemph_memD[2];

#ifdef RESYNTH
   celt_sig syn_mem[2][2*MAX_PERIOD];
#endif

   celt_sig in_mem[1]; /* Size = channels*mode->overlap */
   /* celt_sig prefilter_mem[],  Size = channels*COMBFILTER_PERIOD */
   /* celt_sig overlap_mem[],  Size = channels*mode->overlap */
   /* celt_word16 oldEBands[], Size = 2*channels*mode->nbEBands */
};

int celt_encoder_get_size(int channels)
{
   CELTMode *mode = celt_mode_create(48000, 960, NULL);
   return celt_encoder_get_size_custom(mode, channels);
}

int celt_encoder_get_size_custom(const CELTMode *mode, int channels)
{
   int size = sizeof(struct CELTEncoder)
         + (2*channels*mode->overlap-1)*sizeof(celt_sig)
         + channels*COMBFILTER_MAXPERIOD*sizeof(celt_sig)
         + 3*channels*mode->nbEBands*sizeof(celt_word16);
   return size;
}

CELTEncoder *celt_encoder_create(int sampling_rate, int channels, int *error)
{
   CELTEncoder *st;
   st = (CELTEncoder *)celt_alloc(celt_encoder_get_size(channels));
   if (st!=NULL && celt_encoder_init(st, sampling_rate, channels, error)==NULL)
   {
      celt_encoder_destroy(st);
      st = NULL;
   }
   return st;
}

CELTEncoder *celt_encoder_create_custom(const CELTMode *mode, int channels, int *error)
{
   CELTEncoder *st = (CELTEncoder *)celt_alloc(celt_encoder_get_size_custom(mode, channels));
   if (st!=NULL && celt_encoder_init_custom(st, mode, channels, error)==NULL)
   {
      celt_encoder_destroy(st);
      st = NULL;
   }
   return st;
}

CELTEncoder *celt_encoder_init(CELTEncoder *st, int sampling_rate, int channels, int *error)
{
   celt_encoder_init_custom(st, celt_mode_create(48000, 960, NULL), channels, error);
   st->upsample = resampling_factor(sampling_rate);
   if (st->upsample==0)
   {
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }
   return st;
}

CELTEncoder *celt_encoder_init_custom(CELTEncoder *st, const CELTMode *mode, int channels, int *error)
{
   if (channels < 0 || channels > 2)
   {
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }

   if (st==NULL || mode==NULL)
   {
      if (error)
         *error = CELT_ALLOC_FAIL;
      return NULL;
   }

   CELT_MEMSET((char*)st, 0, celt_encoder_get_size_custom(mode, channels));
   
   st->mode = mode;
   st->overlap = mode->overlap;
   st->stream_channels = st->channels = channels;

   st->upsample = 1;
   st->start = 0;
   st->end = st->mode->effEBands;
   st->signalling = 1;

   st->constrained_vbr = 1;
   st->clip = 1;

   st->bitrate = 255000*channels;
   st->vbr = 0;
   st->vbr_offset = 0;
   st->force_intra  = 0;
   st->delayedIntra = 1;
   st->tonal_average = 256;
   st->spread_decision = SPREAD_NORMAL;
   st->hf_average = 0;
   st->tapset_decision = 0;
   st->complexity = 5;

   if (error)
      *error = CELT_OK;
   return st;
}

void celt_encoder_destroy(CELTEncoder *st)
{
   celt_free(st);
}

static inline celt_int16 FLOAT2INT16(float x)
{
   x = x*CELT_SIG_SCALE;
   x = MAX32(x, -32768);
   x = MIN32(x, 32767);
   return (celt_int16)float2int(x);
}

static inline celt_word16 SIG2WORD16(celt_sig x)
{
#ifdef FIXED_POINT
   x = PSHR32(x, SIG_SHIFT);
   x = MAX32(x, -32768);
   x = MIN32(x, 32767);
   return EXTRACT16(x);
#else
   return (celt_word16)x;
#endif
}

static int transient_analysis(const celt_word32 * restrict in, int len, int C,
                              int overlap)
{
   int i;
   VARDECL(celt_word16, tmp);
   celt_word32 mem0=0,mem1=0;
   int is_transient = 0;
   int block;
   int N;
   /* FIXME: Make that smaller */
   celt_word16 bins[50];
   SAVE_STACK;
   ALLOC(tmp, len, celt_word16);

   block = overlap/2;
   N=len/block;
   if (C==1)
   {
      for (i=0;i<len;i++)
         tmp[i] = SHR32(in[i],SIG_SHIFT);
   } else {
      for (i=0;i<len;i++)
         tmp[i] = SHR32(ADD32(in[i],in[i+len]), SIG_SHIFT+1);
   }

   /* High-pass filter: (1 - 2*z^-1 + z^-2) / (1 - z^-1 + .5*z^-2) */
   for (i=0;i<len;i++)
   {
      celt_word32 x,y;
      x = tmp[i];
      y = ADD32(mem0, x);
#ifdef FIXED_POINT
      mem0 = mem1 + y - SHL32(x,1);
      mem1 = x - SHR32(y,1);
#else
      mem0 = mem1 + y - 2*x;
      mem1 = x - .5f*y;
#endif
      tmp[i] = EXTRACT16(SHR(y,2));
   }
   /* First few samples are bad because we don't propagate the memory */
   for (i=0;i<12;i++)
      tmp[i] = 0;

   for (i=0;i<N;i++)
   {
      int j;
      celt_word16 max_abs=0;
      for (j=0;j<block;j++)
         max_abs = MAX16(max_abs, ABS16(tmp[i*block+j]));
      bins[i] = max_abs;
   }
   for (i=0;i<N;i++)
   {
      int j;
      int conseq=0;
      celt_word16 t1, t2, t3;

      t1 = MULT16_16_Q15(QCONST16(.15f, 15), bins[i]);
      t2 = MULT16_16_Q15(QCONST16(.4f, 15), bins[i]);
      t3 = MULT16_16_Q15(QCONST16(.15f, 15), bins[i]);
      for (j=0;j<i;j++)
      {
         if (bins[j] < t1)
            conseq++;
         if (bins[j] < t2)
            conseq++;
         else
            conseq = 0;
      }
      if (conseq>=3)
         is_transient=1;
      conseq = 0;
      for (j=i+1;j<N;j++)
      {
         if (bins[j] < t3)
            conseq++;
         else
            conseq = 0;
      }
      if (conseq>=7)
         is_transient=1;
   }
   RESTORE_STACK;
   return is_transient;
}

/** Apply window and compute the MDCT for all sub-frames and 
    all channels in a frame */
static void compute_mdcts(const CELTMode *mode, int shortBlocks, celt_sig * restrict in, celt_sig * restrict out, int _C, int LM)
{
   const int C = CHANNELS(_C);
   if (C==1 && !shortBlocks)
   {
      const int overlap = OVERLAP(mode);
      clt_mdct_forward(&mode->mdct, in, out, mode->window, overlap, mode->maxLM-LM);
   } else {
      const int overlap = OVERLAP(mode);
      int N = mode->shortMdctSize<<LM;
      int B = 1;
      int b, c;
      VARDECL(celt_word32, tmp);
      SAVE_STACK;
      if (shortBlocks)
      {
         /*lookup = &mode->mdct[0];*/
         N = mode->shortMdctSize;
         B = shortBlocks;
      }
      ALLOC(tmp, N, celt_word32);
      c=0; do {
         for (b=0;b<B;b++)
         {
            int j;
            clt_mdct_forward(&mode->mdct, in+c*(B*N+overlap)+b*N, tmp, mode->window, overlap, shortBlocks ? mode->maxLM : mode->maxLM-LM);
            /* Interleaving the sub-frames */
            for (j=0;j<N;j++)
               out[(j*B+b)+c*N*B] = tmp[j];
         }
      } while (++c<C);
      RESTORE_STACK;
   }
}

/** Compute the IMDCT and apply window for all sub-frames and 
    all channels in a frame */
static void compute_inv_mdcts(const CELTMode *mode, int shortBlocks, celt_sig *X,
      celt_sig * restrict out_mem[],
      celt_sig * restrict overlap_mem[], int _C, int LM)
{
   int c;
   const int C = CHANNELS(_C);
   const int N = mode->shortMdctSize<<LM;
   const int overlap = OVERLAP(mode);
   c=0; do {
      int j;
         VARDECL(celt_word32, x);
         VARDECL(celt_word32, tmp);
         int b;
         int N2 = N;
         int B = 1;
         SAVE_STACK;
         
         ALLOC(x, N+overlap, celt_word32);
         ALLOC(tmp, N, celt_word32);

         if (shortBlocks)
         {
            N2 = mode->shortMdctSize;
            B = shortBlocks;
         }
         /* Prevents problems from the imdct doing the overlap-add */
         CELT_MEMSET(x, 0, overlap);

         for (b=0;b<B;b++)
         {
            /* De-interleaving the sub-frames */
            for (j=0;j<N2;j++)
               tmp[j] = X[(j*B+b)+c*N2*B];
            clt_mdct_backward(&mode->mdct, tmp, x+N2*b, mode->window, overlap, shortBlocks ? mode->maxLM : mode->maxLM-LM);
         }

         for (j=0;j<overlap;j++)
            out_mem[c][j] = x[j] + overlap_mem[c][j];
         for (;j<N;j++)
            out_mem[c][j] = x[j];
         for (j=0;j<overlap;j++)
            overlap_mem[c][j] = x[N+j];
         RESTORE_STACK;
   } while (++c<C);
}

static void deemphasis(celt_sig *in[], celt_word16 *pcm, int N, int _C, int downsample, const celt_word16 *coef, celt_sig *mem)
{
   const int C = CHANNELS(_C);
   int c;
   int count=0;
   c=0; do {
      int j;
      celt_sig * restrict x;
      celt_word16  * restrict y;
      celt_sig m = mem[c];
      x =in[c];
      y = pcm+c;
      for (j=0;j<N;j++)
      {
         celt_sig tmp = *x + m;
         m = MULT16_32_Q15(coef[0], tmp)
           - MULT16_32_Q15(coef[1], *x);
         tmp = SHL32(MULT16_32_Q15(coef[3], tmp), 2);
         x++;
         /* Technically the store could be moved outside of the if because
            the stores we don't want will just be overwritten */
         if (++count==downsample)
         {
            *y = SCALEOUT(SIG2WORD16(tmp));
            y+=C;
            count=0;
         }
      }
      mem[c] = m;
   } while (++c<C);
}

#ifdef ENABLE_POSTFILTER
static void comb_filter(celt_word32 *y, celt_word32 *x, int T0, int T1, int N,
      celt_word16 g0, celt_word16 g1, int tapset0, int tapset1,
      const celt_word16 *window, int overlap)
{
   int i;
   /* printf ("%d %d %f %f\n", T0, T1, g0, g1); */
   celt_word16 g00, g01, g02, g10, g11, g12;
   static const celt_word16 gains[3][3] = {
         {QCONST16(0.3066406250f, 15), QCONST16(0.2170410156f, 15), QCONST16(0.1296386719f, 15)},
         {QCONST16(0.4638671875f, 15), QCONST16(0.2680664062f, 15), QCONST16(0.f, 15)},
         {QCONST16(0.7998046875f, 15), QCONST16(0.1000976562f, 15), QCONST16(0.f, 15)}};
   g00 = MULT16_16_Q15(g0, gains[tapset0][0]);
   g01 = MULT16_16_Q15(g0, gains[tapset0][1]);
   g02 = MULT16_16_Q15(g0, gains[tapset0][2]);
   g10 = MULT16_16_Q15(g1, gains[tapset1][0]);
   g11 = MULT16_16_Q15(g1, gains[tapset1][1]);
   g12 = MULT16_16_Q15(g1, gains[tapset1][2]);
   for (i=0;i<overlap;i++)
   {
      celt_word16 f;
      f = MULT16_16_Q15(window[i],window[i]);
      y[i] = x[i]
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g00),x[i-T0])
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g01),x[i-T0-1])
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g01),x[i-T0+1])
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g02),x[i-T0-2])
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g02),x[i-T0+2])
               + MULT16_32_Q15(MULT16_16_Q15(f,g10),x[i-T1])
               + MULT16_32_Q15(MULT16_16_Q15(f,g11),x[i-T1-1])
               + MULT16_32_Q15(MULT16_16_Q15(f,g11),x[i-T1+1])
               + MULT16_32_Q15(MULT16_16_Q15(f,g12),x[i-T1-2])
               + MULT16_32_Q15(MULT16_16_Q15(f,g12),x[i-T1+2]);

   }
   for (i=overlap;i<N;i++)
      y[i] = x[i]
               + MULT16_32_Q15(g10,x[i-T1])
               + MULT16_32_Q15(g11,x[i-T1-1])
               + MULT16_32_Q15(g11,x[i-T1+1])
               + MULT16_32_Q15(g12,x[i-T1-2])
               + MULT16_32_Q15(g12,x[i-T1+2]);
}
#endif /* ENABLE_POSTFILTER */

static const signed char tf_select_table[4][8] = {
      {0, -1, 0, -1,    0,-1, 0,-1},
      {0, -1, 0, -2,    1, 0, 1,-1},
      {0, -2, 0, -3,    2, 0, 1,-1},
      {0, -2, 0, -3,    3, 0, 1,-1},
};

static celt_word32 l1_metric(const celt_norm *tmp, int N, int LM, int width)
{
   int i, j;
   static const celt_word16 sqrtM_1[4] = {Q15ONE, QCONST16(.70710678f,15), QCONST16(0.5f,15), QCONST16(0.35355339f,15)};
   celt_word32 L1;
   celt_word16 bias;
   L1=0;
   for (i=0;i<1<<LM;i++)
   {
      celt_word32 L2 = 0;
      for (j=0;j<N>>LM;j++)
         L2 = MAC16_16(L2, tmp[(j<<LM)+i], tmp[(j<<LM)+i]);
      L1 += celt_sqrt(L2);
   }
   L1 = MULT16_32_Q15(sqrtM_1[LM], L1);
   if (width==1)
      bias = QCONST16(.12f,15)*LM;
   else if (width==2)
      bias = QCONST16(.05f,15)*LM;
   else
      bias = QCONST16(.02f,15)*LM;
   L1 = MAC16_32_Q15(L1, bias, L1);
   return L1;
}

static int tf_analysis(const CELTMode *m, celt_word16 *bandLogE, celt_word16 *oldBandE,
      int len, int C, int isTransient, int *tf_res, int nbCompressedBytes, celt_norm *X,
      int N0, int LM, int *tf_sum)
{
   int i;
   VARDECL(int, metric);
   int cost0;
   int cost1;
   VARDECL(int, path0);
   VARDECL(int, path1);
   VARDECL(celt_norm, tmp);
   int lambda;
   int tf_select=0;
   SAVE_STACK;

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

   ALLOC(metric, len, int);
   ALLOC(tmp, (m->eBands[len]-m->eBands[len-1])<<LM, celt_norm);
   ALLOC(path0, len, int);
   ALLOC(path1, len, int);

   *tf_sum = 0;
   for (i=0;i<len;i++)
   {
      int j, k, N;
      celt_word32 L1, best_L1;
      int best_level=0;
      N = (m->eBands[i+1]-m->eBands[i])<<LM;
      for (j=0;j<N;j++)
         tmp[j] = X[j+(m->eBands[i]<<LM)];
      /* Just add the right channel if we're in stereo */
      if (C==2)
         for (j=0;j<N;j++)
            tmp[j] = ADD16(tmp[j],X[N0+j+(m->eBands[i]<<LM)]);
      L1 = l1_metric(tmp, N, isTransient ? LM : 0, N>>LM);
      best_L1 = L1;
      /*printf ("%f ", L1);*/
      for (k=0;k<LM;k++)
      {
         int B;

         if (isTransient)
            B = (LM-k-1);
         else
            B = k+1;

         if (isTransient)
            haar1(tmp, N>>(LM-k), 1<<(LM-k));
         else
            haar1(tmp, N>>k, 1<<k);

         L1 = l1_metric(tmp, N, B, N>>LM);

         if (L1 < best_L1)
         {
            best_L1 = L1;
            best_level = k+1;
         }
      }
      /*printf ("%d ", isTransient ? LM-best_level : best_level);*/
      if (isTransient)
         metric[i] = best_level;
      else
         metric[i] = -best_level;
      *tf_sum += metric[i];
   }
   /*printf("\n");*/
   /* FIXME: Figure out how to set this */
   tf_select = 0;

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
      cost0 = curr0 + abs(metric[i]-tf_select_table[LM][4*isTransient+2*tf_select+0]);
      cost1 = curr1 + abs(metric[i]-tf_select_table[LM][4*isTransient+2*tf_select+1]);
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
   RESTORE_STACK;
   return tf_select;
}

static void tf_encode(int start, int end, int isTransient, int *tf_res, int LM, int tf_select, ec_enc *enc)
{
   int curr, i;
   int tf_select_rsv;
   int tf_changed;
   int logp;
   celt_uint32 budget;
   celt_uint32 tell;
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
   /*printf("%d %d ", isTransient, tf_select); for(i=0;i<end;i++)printf("%d ", tf_res[i]);printf("\n");*/
}

static void tf_decode(int start, int end, int isTransient, int *tf_res, int LM, ec_dec *dec)
{
   int i, curr, tf_select;
   int tf_select_rsv;
   int tf_changed;
   int logp;
   celt_uint32 budget;
   celt_uint32 tell;

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
      const celt_word16 *bandLogE, int end, int LM, int C, int N0)
{
   int i;
   celt_word32 diff=0;
   int c;
   int trim_index = 5;
   if (C==2)
   {
      celt_word16 sum = 0; /* Q10 */
      /* Compute inter-channel correlation for low frequencies */
      for (i=0;i<8;i++)
      {
         int j;
         celt_word32 partial = 0;
         for (j=m->eBands[i]<<LM;j<m->eBands[i+1]<<LM;j++)
            partial = MAC16_16(partial, X[j], X[N0+j]);
         sum = ADD16(sum, EXTRACT16(SHR32(partial, 18)));
      }
      sum = MULT16_16_Q15(QCONST16(1.f/8, 15), sum);
      /*printf ("%f\n", sum);*/
      if (sum > QCONST16(.995f,10))
         trim_index-=4;
      else if (sum > QCONST16(.92f,10))
         trim_index-=3;
      else if (sum > QCONST16(.85f,10))
         trim_index-=2;
      else if (sum > QCONST16(.8f,10))
         trim_index-=1;
   }

   /* Estimate spectral tilt */
   c=0; do {
      for (i=0;i<end-1;i++)
      {
         diff += bandLogE[i+c*m->nbEBands]*(celt_int32)(2+2*i-m->nbEBands);
      }
   } while (++c<0);
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

   if (trim_index<0)
      trim_index = 0;
   if (trim_index>10)
      trim_index = 10;
   return trim_index;
}

static int stereo_analysis(const CELTMode *m, const celt_norm *X,
      int LM, int N0)
{
   int i;
   int thetas;
   celt_word32 sumLR = EPSILON, sumMS = EPSILON;

   /* Use the L1 norm to model the entropy of the L/R signal vs the M/S signal */
   for (i=0;i<13;i++)
   {
      int j;
      for (j=m->eBands[i]<<LM;j<m->eBands[i+1]<<LM;j++)
      {
         celt_word16 L, R, M, S;
         L = X[j];
         R = X[N0+j];
         M = L+R;
         S = L-R;
         sumLR += EXTEND32(ABS16(L)) + EXTEND32(ABS16(R));
         sumMS += EXTEND32(ABS16(M)) + EXTEND32(ABS16(S));
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

#ifdef FIXED_POINT
CELT_STATIC
int celt_encode_with_ec(CELTEncoder * restrict st, const celt_int16 * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
#else
CELT_STATIC
int celt_encode_with_ec_float(CELTEncoder * restrict st, const celt_sig * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
#endif
   int i, c, N;
   celt_int32 bits;
   ec_enc _enc;
   VARDECL(celt_sig, in);
   VARDECL(celt_sig, freq);
   VARDECL(celt_norm, X);
   VARDECL(celt_ener, bandE);
   VARDECL(celt_word16, bandLogE);
   VARDECL(int, fine_quant);
   VARDECL(celt_word16, error);
   VARDECL(int, pulses);
   VARDECL(int, cap);
   VARDECL(int, offsets);
   VARDECL(int, fine_priority);
   VARDECL(int, tf_res);
   VARDECL(unsigned char, collapse_masks);
   celt_sig *_overlap_mem;
   celt_sig *prefilter_mem;
   celt_word16 *oldBandE, *oldLogE, *oldLogE2;
   int shortBlocks=0;
   int isTransient=0;
   int resynth;
   const int CC = CHANNELS(st->channels);
   const int C = CHANNELS(st->stream_channels);
   int LM, M;
   int tf_select;
   int nbFilledBytes, nbAvailableBytes;
   int effEnd;
   int codedBands;
   int tf_sum;
   int alloc_trim;
   int pitch_index=COMBFILTER_MINPERIOD;
   celt_word16 gain1 = 0;
   int intensity=0;
   int dual_stereo=0;
   int effectiveBytes;
   celt_word16 pf_threshold;
   int dynalloc_logp;
   celt_int32 vbr_rate;
   celt_int32 total_bits;
   celt_int32 total_boost;
   celt_int32 balance;
   celt_int32 tell;
   int prefilter_tapset=0;
   int pf_on;
   int anti_collapse_rsv;
   int anti_collapse_on=0;
   int silence=0;
   ALLOC_STACK;
   SAVE_STACK;

   if (nbCompressedBytes<2 || pcm==NULL)
     return CELT_BAD_ARG;

   frame_size *= st->upsample;
   for (LM=0;LM<=st->mode->maxLM;LM++)
      if (st->mode->shortMdctSize<<LM==frame_size)
         break;
   if (LM>st->mode->maxLM)
      return CELT_BAD_ARG;
   M=1<<LM;
   N = M*st->mode->shortMdctSize;

   prefilter_mem = st->in_mem+CC*(st->overlap);
   _overlap_mem = prefilter_mem+CC*COMBFILTER_MAXPERIOD;
   /*_overlap_mem = st->in_mem+C*(st->overlap);*/
   oldBandE = (celt_word16*)(st->in_mem+CC*(2*st->overlap+COMBFILTER_MAXPERIOD));
   oldLogE = oldBandE + CC*st->mode->nbEBands;
   oldLogE2 = oldLogE + CC*st->mode->nbEBands;

   if (enc==NULL)
   {
      tell=1;
      nbFilledBytes=0;
   } else {
      tell=ec_tell(enc);
      nbFilledBytes=(tell+4)>>3;
   }

   if (st->signalling && enc==NULL)
   {
      int tmp = (st->mode->effEBands-st->end)>>1;
      st->end = IMAX(1, st->mode->effEBands-tmp);
      compressed[0] = tmp<<5;
      compressed[0] |= LM<<3;
      compressed[0] |= (C==2)<<2;
      compressed++;
      nbCompressedBytes--;
   }

   /* Can't produce more than 1275 output bytes */
   nbCompressedBytes = IMIN(nbCompressedBytes,1275);
   nbAvailableBytes = nbCompressedBytes - nbFilledBytes;

   if (st->vbr)
   {
      celt_int32 den=st->mode->Fs>>BITRES;
      vbr_rate=(st->bitrate*frame_size+(den>>1))/den;
      if (st->signalling)
         vbr_rate -= 8<<BITRES;
      effectiveBytes = vbr_rate>>(3+BITRES);
   } else {
      celt_int32 tmp;
      vbr_rate = 0;
      tmp = st->bitrate*frame_size;
      if (tell>1)
         tmp += tell;
      nbCompressedBytes = IMAX(2, IMIN(nbCompressedBytes,
            (tmp+4*st->mode->Fs)/(8*st->mode->Fs)-!!st->signalling));
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
         celt_int32 vbr_bound;
         celt_int32 max_allowed;
         /* We could use any multiple of vbr_rate as bound (depending on the
             delay).
            This is clamped to ensure we use at least two bytes if the encoder
             was entirely empty, but to allow 0 in hybrid mode. */
         vbr_bound = vbr_rate;
         max_allowed = IMIN(IMAX(tell==1?2:0,
               vbr_rate+vbr_bound-st->vbr_reservoir>>(BITRES+3)),
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
   if (effEnd > st->mode->effEBands)
      effEnd = st->mode->effEBands;

   ALLOC(in, CC*(N+st->overlap), celt_sig);

   /* Find pitch period and gain */
   {
      VARDECL(celt_sig, _pre);
      celt_sig *pre[2];
      SAVE_STACK;
      ALLOC(_pre, CC*(N+COMBFILTER_MAXPERIOD), celt_sig);

      pre[0] = _pre;
      pre[1] = _pre + (N+COMBFILTER_MAXPERIOD);

      silence = 1;
      c=0; do {
         int count = 0;
         const celt_word16 * restrict pcmp = pcm+c;
         celt_sig * restrict inp = in+c*(N+st->overlap)+st->overlap;

         for (i=0;i<N;i++)
         {
            celt_sig x, tmp;

            x = SCALEIN(*pcmp);
#ifndef FIXED_POINT
            if (st->clip)
               x = MAX32(-65536.f, MIN32(65536.f,x));
#endif
            if (++count==st->upsample)
            {
               count=0;
               pcmp+=CC;
            } else {
               x = 0;
            }
            /* Apply pre-emphasis */
            tmp = MULT16_16(st->mode->preemph[2], x);
            *inp = tmp + st->preemph_memE[c];
            st->preemph_memE[c] = MULT16_32_Q15(st->mode->preemph[1], *inp)
                                   - MULT16_32_Q15(st->mode->preemph[0], tmp);
            silence = silence && *inp == 0;
            inp++;
         }
         CELT_COPY(pre[c], prefilter_mem+c*COMBFILTER_MAXPERIOD, COMBFILTER_MAXPERIOD);
         CELT_COPY(pre[c]+COMBFILTER_MAXPERIOD, in+c*(N+st->overlap)+st->overlap, N);
      } while (++c<CC);

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
#ifdef ENABLE_POSTFILTER
      if (nbAvailableBytes>12*C && st->start==0 && !silence && !st->disable_pf && st->complexity >= 5)
      {
         VARDECL(celt_word16, pitch_buf);
         ALLOC(pitch_buf, (COMBFILTER_MAXPERIOD+N)>>1, celt_word16);

         pitch_downsample(pre, pitch_buf, COMBFILTER_MAXPERIOD+N, CC);
         pitch_search(pitch_buf+(COMBFILTER_MAXPERIOD>>1), pitch_buf, N,
               COMBFILTER_MAXPERIOD-COMBFILTER_MINPERIOD, &pitch_index);
         pitch_index = COMBFILTER_MAXPERIOD-pitch_index;

         gain1 = remove_doubling(pitch_buf, COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD,
               N, &pitch_index, st->prefilter_period, st->prefilter_gain);
         if (pitch_index > COMBFILTER_MAXPERIOD-2)
            pitch_index = COMBFILTER_MAXPERIOD-2;
         gain1 = MULT16_16_Q15(QCONST16(.7f,15),gain1);
         prefilter_tapset = st->tapset_decision;
      } else {
         gain1 = 0;
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
         if(st->start==0 && tell+16<=total_bits)
            ec_enc_bit_logp(enc, 0, 1);
         gain1 = 0;
         pf_on = 0;
      } else {
         /*This block is not gated by a total bits check only because
           of the nbAvailableBytes check above.*/
         int qg;
         int octave;

         if (ABS16(gain1-st->prefilter_gain)<QCONST16(.1f,15))
            gain1=st->prefilter_gain;

#ifdef FIXED_POINT
         qg = ((gain1+1536)>>10)/3-1;
#else
         qg = floor(.5+gain1*32/3)-1;
#endif
         qg = IMAX(0, IMIN(7, qg));
         ec_enc_bit_logp(enc, 1, 1);
         pitch_index += 1;
         octave = EC_ILOG(pitch_index)-5;
         ec_enc_uint(enc, octave, 6);
         ec_enc_bits(enc, pitch_index-(16<<octave), 4+octave);
         pitch_index -= 1;
         ec_enc_bits(enc, qg, 3);
         if (ec_tell(enc)+2<=total_bits)
            ec_enc_icdf(enc, prefilter_tapset, tapset_icdf, 2);
         else
           prefilter_tapset = 0;
         gain1 = QCONST16(0.09375f,15)*(qg+1);
         pf_on = 1;
      }
      /*printf("%d %f\n", pitch_index, gain1);*/
#else /* ENABLE_POSTFILTER */
      if(st->start==0 && tell+16<=total_bits)
         ec_enc_bit_logp(enc, 0, 1);
      pf_on = 0;
#endif /* ENABLE_POSTFILTER */

      c=0; do {
         int offset = st->mode->shortMdctSize-st->mode->overlap;
         st->prefilter_period=IMAX(st->prefilter_period, COMBFILTER_MINPERIOD);
         CELT_COPY(in+c*(N+st->overlap), st->in_mem+c*(st->overlap), st->overlap);
#ifdef ENABLE_POSTFILTER
         if (offset)
            comb_filter(in+c*(N+st->overlap)+st->overlap, pre[c]+COMBFILTER_MAXPERIOD,
                  st->prefilter_period, st->prefilter_period, offset, -st->prefilter_gain, -st->prefilter_gain,
                  st->prefilter_tapset, st->prefilter_tapset, NULL, 0);

         comb_filter(in+c*(N+st->overlap)+st->overlap+offset, pre[c]+COMBFILTER_MAXPERIOD+offset,
               st->prefilter_period, pitch_index, N-offset, -st->prefilter_gain, -gain1,
               st->prefilter_tapset, prefilter_tapset, st->mode->window, st->mode->overlap);
#endif /* ENABLE_POSTFILTER */
         CELT_COPY(st->in_mem+c*(st->overlap), in+c*(N+st->overlap)+N, st->overlap);

#ifdef ENABLE_POSTFILTER
         if (N>COMBFILTER_MAXPERIOD)
         {
            CELT_MOVE(prefilter_mem+c*COMBFILTER_MAXPERIOD, pre[c]+N, COMBFILTER_MAXPERIOD);
         } else {
            CELT_MOVE(prefilter_mem+c*COMBFILTER_MAXPERIOD, prefilter_mem+c*COMBFILTER_MAXPERIOD+N, COMBFILTER_MAXPERIOD-N);
            CELT_MOVE(prefilter_mem+c*COMBFILTER_MAXPERIOD+COMBFILTER_MAXPERIOD-N, pre[c]+COMBFILTER_MAXPERIOD, N);
         }
#endif /* ENABLE_POSTFILTER */
      } while (++c<CC);

      RESTORE_STACK;
   }

#ifdef RESYNTH
   resynth = 1;
#else
   resynth = 0;
#endif

   isTransient = 0;
   shortBlocks = 0;
   if (LM>0 && ec_tell(enc)+3<=total_bits)
   {
      if (st->complexity > 1)
      {
         isTransient = transient_analysis(in, N+st->overlap, CC,
                  st->overlap);
         if (isTransient)
            shortBlocks = M;
      }
      ec_enc_bit_logp(enc, isTransient, 3);
   }

   ALLOC(freq, CC*N, celt_sig); /**< Interleaved signal MDCTs */
   ALLOC(bandE,st->mode->nbEBands*CC, celt_ener);
   ALLOC(bandLogE,st->mode->nbEBands*CC, celt_word16);
   /* Compute MDCTs */
   compute_mdcts(st->mode, shortBlocks, in, freq, CC, LM);

   if (CC==2&&C==1)
   {
      for (i=0;i<N;i++)
         freq[i] = ADD32(HALF32(freq[i]), HALF32(freq[N+i]));
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
   ALLOC(X, C*N, celt_norm);         /**< Interleaved normalised MDCTs */

   compute_band_energies(st->mode, freq, bandE, effEnd, C, M);

   amp2Log2(st->mode, effEnd, st->end, bandE, bandLogE, C);

   /* Band normalisation */
   normalise_bands(st->mode, freq, X, bandE, effEnd, C, M);

   ALLOC(tf_res, st->mode->nbEBands, int);
   /* Needs to be before coarse energy quantization because otherwise the energy gets modified */
   tf_select = tf_analysis(st->mode, bandLogE, oldBandE, effEnd, C, isTransient, tf_res, effectiveBytes, X, N, LM, &tf_sum);
   for (i=effEnd;i<st->end;i++)
      tf_res[i] = tf_res[effEnd-1];

   ALLOC(error, C*st->mode->nbEBands, celt_word16);
   quant_coarse_energy(st->mode, st->start, st->end, effEnd, bandLogE,
         oldBandE, total_bits, error, enc,
         C, LM, nbAvailableBytes, st->force_intra,
         &st->delayedIntra, st->complexity >= 4);

   tf_encode(st->start, st->end, isTransient, tf_res, LM, tf_select, enc);

   st->spread_decision = SPREAD_NORMAL;
   if (ec_tell(enc)+4<=total_bits)
   {
      if (shortBlocks || st->complexity < 3 || nbAvailableBytes < 10*C)
      {
         if (st->complexity == 0)
            st->spread_decision = SPREAD_NONE;
      } else {
         st->spread_decision = spreading_decision(st->mode, X,
               &st->tonal_average, st->spread_decision, &st->hf_average,
               &st->tapset_decision, pf_on&&!shortBlocks, effEnd, C, M);
      }
      ec_enc_icdf(enc, st->spread_decision, spread_icdf, 5);
   }

   ALLOC(cap, st->mode->nbEBands, int);
   ALLOC(offsets, st->mode->nbEBands, int);

   init_caps(st->mode,cap,LM,C);
   for (i=0;i<st->mode->nbEBands;i++)
      offsets[i] = 0;
   /* Dynamic allocation code */
   /* Make sure that dynamic allocation can't make us bust the budget */
   if (effectiveBytes > 50 && LM>=1)
   {
      int t1, t2;
      if (LM <= 1)
      {
         t1 = 3;
         t2 = 5;
      } else {
         t1 = 2;
         t2 = 4;
      }
      for (i=st->start+1;i<st->end-1;i++)
      {
         celt_word32 d2;
         d2 = 2*bandLogE[i]-bandLogE[i-1]-bandLogE[i+1];
         if (C==2)
            d2 = HALF32(d2 + 2*bandLogE[i+st->mode->nbEBands]-
                  bandLogE[i-1+st->mode->nbEBands]-bandLogE[i+1+st->mode->nbEBands]);
         if (d2 > SHL16(t1,DB_SHIFT))
            offsets[i] += 1;
         if (d2 > SHL16(t2,DB_SHIFT))
            offsets[i] += 1;
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
      width = C*(st->mode->eBands[i+1]-st->mode->eBands[i])<<LM;
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
   alloc_trim = 5;
   if (tell+(6<<BITRES) <= total_bits - total_boost)
   {
      alloc_trim = alloc_trim_analysis(st->mode, X, bandLogE,
            st->end, LM, C, N);
      ec_enc_icdf(enc, alloc_trim, trim_icdf, 7);
      tell = ec_tell_frac(enc);
   }

   /* Variable bitrate */
   if (vbr_rate>0)
   {
     celt_word16 alpha;
     celt_int32 delta;
     /* The target rate in 8th bits per frame */
     celt_int32 target;
     celt_int32 min_allowed;

     target = vbr_rate + st->vbr_offset - ((40*C+20)<<BITRES);

     /* Shortblocks get a large boost in bitrate, but since they
        are uncommon long blocks are not greatly affected */
     if (shortBlocks || tf_sum < -2*(st->end-st->start))
        target = 7*target/4;
     else if (tf_sum < -(st->end-st->start))
        target = 3*target/2;
     else if (M > 1)
        target-=(target+14)/28;

     /* The current offset is removed from the target and the space used
        so far is added*/
     target=target+tell;

     /* In VBR mode the frame size must not be reduced so much that it would
         result in the encoder running out of bits.
        The margin of 2 bytes ensures that none of the bust-prevention logic
         in the decoder will have triggered so far. */
     min_allowed = (tell+total_boost+(1<<BITRES+3)-1>>(BITRES+3)) + 2 - nbFilledBytes;

     nbAvailableBytes = target+(1<<(BITRES+2))>>(BITRES+3);
     nbAvailableBytes = IMAX(min_allowed,nbAvailableBytes);
     nbAvailableBytes = IMIN(nbCompressedBytes,nbAvailableBytes+nbFilledBytes) - nbFilledBytes;

     if(silence)
     {
       nbAvailableBytes = 2;
       target = 2*8<<BITRES;
     }

     /* By how much did we "miss" the target on that frame */
     delta = target - vbr_rate;

     target=nbAvailableBytes<<(BITRES+3);

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
     st->vbr_drift += (celt_int32)MULT16_32_Q15(alpha,delta-st->vbr_offset-st->vbr_drift);
     st->vbr_offset = -st->vbr_drift;
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
     /* This moves the raw bits to take into account the new compressed size */
     ec_enc_shrink(enc, nbCompressedBytes);
   }
   if (C==2)
   {
      int effectiveRate;

      /* Always use MS for 2.5 ms frames until we can do a better analysis */
      if (LM!=0)
         dual_stereo = stereo_analysis(st->mode, X, LM, N);

      /* Account for coarse energy */
      effectiveRate = (8*effectiveBytes - 80)>>LM;

      /* effectiveRate in kb/s */
      effectiveRate = 2*effectiveRate/5;
      if (effectiveRate<35)
         intensity = 8;
      else if (effectiveRate<50)
         intensity = 12;
      else if (effectiveRate<68)
         intensity = 16;
      else if (effectiveRate<84)
         intensity = 18;
      else if (effectiveRate<102)
         intensity = 19;
      else if (effectiveRate<130)
         intensity = 20;
      else
         intensity = 100;
      intensity = IMIN(st->end,IMAX(st->start, intensity));
   }

   /* Bit allocation */
   ALLOC(fine_quant, st->mode->nbEBands, int);
   ALLOC(pulses, st->mode->nbEBands, int);
   ALLOC(fine_priority, st->mode->nbEBands, int);

   /* bits =           packet size                    - where we are - safety*/
   bits = ((celt_int32)nbCompressedBytes*8<<BITRES) - ec_tell_frac(enc) - 1;
   anti_collapse_rsv = isTransient&&LM>=2&&bits>=(LM+2<<BITRES) ? (1<<BITRES) : 0;
   bits -= anti_collapse_rsv;
   codedBands = compute_allocation(st->mode, st->start, st->end, offsets, cap,
         alloc_trim, &intensity, &dual_stereo, bits, &balance, pulses,
         fine_quant, fine_priority, C, LM, enc, 1, st->lastCodedBands);
   st->lastCodedBands = codedBands;

   quant_fine_energy(st->mode, st->start, st->end, oldBandE, error, fine_quant, enc, C);

#ifdef MEASURE_NORM_MSE
   float X0[3000];
   float bandE0[60];
   c=0; do 
      for (i=0;i<N;i++)
         X0[i+c*N] = X[i+c*N];
   while (++c<C);
   for (i=0;i<C*st->mode->nbEBands;i++)
      bandE0[i] = bandE[i];
#endif

   /* Residual quantisation */
   ALLOC(collapse_masks, C*st->mode->nbEBands, unsigned char);
   quant_all_bands(1, st->mode, st->start, st->end, X, C==2 ? X+N : NULL, collapse_masks,
         bandE, pulses, shortBlocks, st->spread_decision, dual_stereo, intensity, tf_res, resynth,
         nbCompressedBytes*(8<<BITRES)-anti_collapse_rsv, balance, enc, LM, codedBands, &st->rng);

   if (anti_collapse_rsv > 0)
   {
      anti_collapse_on = st->consec_transient<2;
      ec_enc_bits(enc, anti_collapse_on, 1);
   }
   quant_energy_finalise(st->mode, st->start, st->end, oldBandE, error, fine_quant, fine_priority, nbCompressedBytes*8-ec_tell(enc), enc, C);

   if (silence)
   {
      for (i=0;i<C*st->mode->nbEBands;i++)
         oldBandE[i] = -QCONST16(28.f,DB_SHIFT);
   }

#ifdef RESYNTH
   /* Re-synthesis of the coded audio if required */
   if (resynth)
   {
      celt_sig *out_mem[2];
      celt_sig *overlap_mem[2];

      log2Amp(st->mode, st->start, st->end, bandE, oldBandE, C);
      if (silence)
      {
         for (i=0;i<C*st->mode->nbEBands;i++)
            bandE[i] = 0;
      }

#ifdef MEASURE_NORM_MSE
      measure_norm_mse(st->mode, X, X0, bandE, bandE0, M, N, C);
#endif
      if (anti_collapse_on)
      {
         anti_collapse(st->mode, X, collapse_masks, LM, C, CC, N,
               st->start, st->end, oldBandE, oldLogE, oldLogE2, pulses, st->rng);
      }

      /* Synthesis */
      denormalise_bands(st->mode, X, freq, bandE, effEnd, C, M);

      CELT_MOVE(st->syn_mem[0], st->syn_mem[0]+N, MAX_PERIOD);
      if (CC==2)
         CELT_MOVE(st->syn_mem[1], st->syn_mem[1]+N, MAX_PERIOD);

      c=0; do
         for (i=0;i<M*st->mode->eBands[st->start];i++)
            freq[c*N+i] = 0;
      while (++c<C);
      c=0; do
         for (i=M*st->mode->eBands[st->end];i<N;i++)
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

      c=0; do
         overlap_mem[c] = _overlap_mem + c*st->overlap;
      while (++c<CC);

      compute_inv_mdcts(st->mode, shortBlocks, freq, out_mem, overlap_mem, CC, LM);

#ifdef ENABLE_POSTFILTER
      c=0; do {
         st->prefilter_period=IMAX(st->prefilter_period, COMBFILTER_MINPERIOD);
         st->prefilter_period_old=IMAX(st->prefilter_period_old, COMBFILTER_MINPERIOD);
         comb_filter(out_mem[c], out_mem[c], st->prefilter_period_old, st->prefilter_period, st->mode->shortMdctSize,
               st->prefilter_gain_old, st->prefilter_gain, st->prefilter_tapset_old, st->prefilter_tapset,
               st->mode->window, st->overlap);
         if (LM!=0)
            comb_filter(out_mem[c]+st->mode->shortMdctSize, out_mem[c]+st->mode->shortMdctSize, st->prefilter_period, pitch_index, N-st->mode->shortMdctSize,
                  st->prefilter_gain, gain1, st->prefilter_tapset, prefilter_tapset,
                  st->mode->window, st->mode->overlap);
      } while (++c<CC);
#endif /* ENABLE_POSTFILTER */

      deemphasis(out_mem, (celt_word16*)pcm, N, CC, st->upsample, st->mode->preemph, st->preemph_memD);
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
      for (i=0;i<st->mode->nbEBands;i++)
         oldBandE[st->mode->nbEBands+i]=oldBandE[i];
   }

   /* In case start or end were to change */
   c=0; do
   {
      for (i=0;i<st->start;i++)
         oldBandE[c*st->mode->nbEBands+i]=0;
      for (i=st->end;i<st->mode->nbEBands;i++)
         oldBandE[c*st->mode->nbEBands+i]=0;
   } while (++c<CC);
   if (!isTransient)
   {
      for (i=0;i<CC*st->mode->nbEBands;i++)
         oldLogE2[i] = oldLogE[i];
      for (i=0;i<CC*st->mode->nbEBands;i++)
         oldLogE[i] = oldBandE[i];
   } else {
      for (i=0;i<CC*st->mode->nbEBands;i++)
         oldLogE[i] = MIN16(oldLogE[i], oldBandE[i]);
   }
   if (isTransient)
      st->consec_transient++;
   else
      st->consec_transient=0;
   st->rng = enc->rng;

   /* If there's any room left (can only happen for very high rates),
      it's already filled with zeros */
   ec_enc_done(enc);
   
   if (st->signalling)
      nbCompressedBytes++;

   RESTORE_STACK;
   if (ec_get_error(enc))
      return CELT_CORRUPTED_DATA;
   else
      return nbCompressedBytes;
}

#ifdef FIXED_POINT
#ifndef DISABLE_FLOAT_API
CELT_STATIC
int celt_encode_with_ec_float(CELTEncoder * restrict st, const float * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
   int j, ret, C, N;
   VARDECL(celt_int16, in);
   ALLOC_STACK;
   SAVE_STACK;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   C = CHANNELS(st->channels);
   N = frame_size;
   ALLOC(in, C*N, celt_int16);

   for (j=0;j<C*N;j++)
     in[j] = FLOAT2INT16(pcm[j]);

   ret=celt_encode_with_ec(st,in,frame_size,compressed,nbCompressedBytes, enc);
#ifdef RESYNTH
   for (j=0;j<C*N;j++)
      ((float*)pcm)[j]=in[j]*(1.f/32768.f);
#endif
   RESTORE_STACK;
   return ret;

}
#endif /*DISABLE_FLOAT_API*/
#else
CELT_STATIC
int celt_encode_with_ec(CELTEncoder * restrict st, const celt_int16 * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
   int j, ret, C, N;
   VARDECL(celt_sig, in);
   ALLOC_STACK;
   SAVE_STACK;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   C=CHANNELS(st->channels);
   N=frame_size;
   ALLOC(in, C*N, celt_sig);
   for (j=0;j<C*N;j++) {
     in[j] = SCALEOUT(pcm[j]);
   }

   ret = celt_encode_with_ec_float(st,in,frame_size,compressed,nbCompressedBytes, enc);
#ifdef RESYNTH
   for (j=0;j<C*N;j++)
      ((celt_int16*)pcm)[j] = FLOAT2INT16(in[j]);
#endif
   RESTORE_STACK;
   return ret;
}
#endif

int celt_encode(CELTEncoder * restrict st, const celt_int16 * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes)
{
   return celt_encode_with_ec(st, pcm, frame_size, compressed, nbCompressedBytes, NULL);
}

#ifndef DISABLE_FLOAT_API
int celt_encode_float(CELTEncoder * restrict st, const float * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes)
{
   return celt_encode_with_ec_float(st, pcm, frame_size, compressed, nbCompressedBytes, NULL);
}
#endif /* DISABLE_FLOAT_API */

int celt_encoder_ctl(CELTEncoder * restrict st, int request, ...)
{
   va_list ap;
   
   va_start(ap, request);
   switch (request)
   {
      case CELT_GET_MODE_REQUEST:
      {
         const CELTMode ** value = va_arg(ap, const CELTMode**);
         if (value==0)
            goto bad_arg;
         *value=st->mode;
      }
      break;
      case CELT_SET_COMPLEXITY_REQUEST:
      {
         int value = va_arg(ap, celt_int32);
         if (value<0 || value>10)
            goto bad_arg;
         st->complexity = value;
      }
      break;
      case CELT_SET_START_BAND_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         if (value<0 || value>=st->mode->nbEBands)
            goto bad_arg;
         st->start = value;
      }
      break;
      case CELT_SET_END_BAND_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         if (value<1 || value>st->mode->nbEBands)
            goto bad_arg;
         st->end = value;
      }
      break;
      case CELT_SET_PREDICTION_REQUEST:
      {
         int value = va_arg(ap, celt_int32);
         if (value<0 || value>2)
            goto bad_arg;
         st->disable_pf = value<=1;
         st->force_intra = value==0;
      }
      break;
      case CELT_SET_VBR_CONSTRAINT_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         st->constrained_vbr = value;
      }
      break;
      case CELT_SET_VBR_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         st->vbr = value;
      }
      break;
      case CELT_SET_BITRATE_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         if (value<=500)
            goto bad_arg;
         value = IMIN(value, 260000*st->channels);
         st->bitrate = value;
      }
      break;
      case CELT_SET_CHANNELS_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         if (value<1 || value>2)
            goto bad_arg;
         st->stream_channels = value;
      }
      break;
      case CELT_SET_SIGNALLING_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         st->signalling = value;
      }
      break;
      case CELT_RESET_STATE:
      {
         CELT_MEMSET((char*)&st->ENCODER_RESET_START, 0,
               celt_encoder_get_size_custom(st->mode, st->channels)-
               ((char*)&st->ENCODER_RESET_START - (char*)st));
         st->vbr_offset = 0;
         st->delayedIntra = 1;
         st->spread_decision = SPREAD_NORMAL;
         st->tonal_average = QCONST16(1.f,8);
      }
      break;
      case CELT_SET_INPUT_CLIPPING_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         st->clip = value;
      }
      break;
      default:
         goto bad_request;
   }
   va_end(ap);
   return CELT_OK;
bad_arg:
   va_end(ap);
   return CELT_BAD_ARG;
bad_request:
   va_end(ap);
   return CELT_UNIMPLEMENTED;
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
struct CELTDecoder {
   const CELTMode *mode;
   int overlap;
   int channels;
   int stream_channels;

   int downsample;
   int start, end;
   int signalling;

   /* Everything beyond this point gets cleared on a reset */
#define DECODER_RESET_START rng

   celt_uint32 rng;
   int last_pitch_index;
   int loss_count;
   int postfilter_period;
   int postfilter_period_old;
   celt_word16 postfilter_gain;
   celt_word16 postfilter_gain_old;
   int postfilter_tapset;
   int postfilter_tapset_old;

   celt_sig preemph_memD[2];
   
   celt_sig _decode_mem[1]; /* Size = channels*(DECODE_BUFFER_SIZE+mode->overlap) */
   /* celt_word16 lpc[],  Size = channels*LPC_ORDER */
   /* celt_word16 oldEBands[], Size = channels*mode->nbEBands */
   /* celt_word16 oldLogE[], Size = channels*mode->nbEBands */
   /* celt_word16 oldLogE2[], Size = channels*mode->nbEBands */
   /* celt_word16 backgroundLogE[], Size = channels*mode->nbEBands */
};

int celt_decoder_get_size(int channels)
{
   const CELTMode *mode = celt_mode_create(48000, 960, NULL);
   return celt_decoder_get_size_custom(mode, channels);
}

int celt_decoder_get_size_custom(const CELTMode *mode, int channels)
{
   int size = sizeof(struct CELTDecoder)
            + (channels*(DECODE_BUFFER_SIZE+mode->overlap)-1)*sizeof(celt_sig)
            + channels*LPC_ORDER*sizeof(celt_word16)
            + 4*channels*mode->nbEBands*sizeof(celt_word16);
   return size;
}

CELTDecoder *celt_decoder_create(int sampling_rate, int channels, int *error)
{
   CELTDecoder *st;
   st = (CELTDecoder *)celt_alloc(celt_decoder_get_size(channels));
   if (st!=NULL && celt_decoder_init(st, sampling_rate, channels, error)==NULL)
   {
      celt_decoder_destroy(st);
      st = NULL;
   }
   return st;
}

CELTDecoder *celt_decoder_create_custom(const CELTMode *mode, int channels, int *error)
{
   CELTDecoder *st = (CELTDecoder *)celt_alloc(celt_decoder_get_size_custom(mode, channels));
   if (st!=NULL && celt_decoder_init_custom(st, mode, channels, error)==NULL)
   {
      celt_decoder_destroy(st);
      st = NULL;
   }
   return st;
}

CELTDecoder *celt_decoder_init(CELTDecoder *st, int sampling_rate, int channels, int *error)
{
   celt_decoder_init_custom(st, celt_mode_create(48000, 960, NULL), channels, error);
   st->downsample = resampling_factor(sampling_rate);
   if (st->downsample==0)
   {
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }
   return st;
}

CELTDecoder *celt_decoder_init_custom(CELTDecoder *st, const CELTMode *mode, int channels, int *error)
{
   if (channels < 0 || channels > 2)
   {
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }

   if (st==NULL)
   {
      if (error)
         *error = CELT_ALLOC_FAIL;
      return NULL;
   }

   CELT_MEMSET((char*)st, 0, celt_decoder_get_size_custom(mode, channels));

   st->mode = mode;
   st->overlap = mode->overlap;
   st->stream_channels = st->channels = channels;

   st->downsample = 1;
   st->start = 0;
   st->end = st->mode->effEBands;
   st->signalling = 1;

   st->loss_count = 0;

   if (error)
      *error = CELT_OK;
   return st;
}

void celt_decoder_destroy(CELTDecoder *st)
{
   celt_free(st);
}

static void celt_decode_lost(CELTDecoder * restrict st, celt_word16 * restrict pcm, int N, int LM)
{
   int c;
   int pitch_index;
   int overlap = st->mode->overlap;
   celt_word16 fade = Q15ONE;
   int i, len;
   const int C = CHANNELS(st->channels);
   int offset;
   celt_sig *out_mem[2];
   celt_sig *decode_mem[2];
   celt_sig *overlap_mem[2];
   celt_word16 *lpc;
   celt_word32 *out_syn[2];
   celt_word16 *oldBandE, *oldLogE2, *backgroundLogE;
   int plc=1;
   SAVE_STACK;
   
   c=0; do {
      decode_mem[c] = st->_decode_mem + c*(DECODE_BUFFER_SIZE+st->overlap);
      out_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE-MAX_PERIOD;
      overlap_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE;
   } while (++c<C);
   lpc = (celt_word16*)(st->_decode_mem+(DECODE_BUFFER_SIZE+st->overlap)*C);
   oldBandE = lpc+C*LPC_ORDER;
   oldLogE2 = oldBandE + C*st->mode->nbEBands;
   backgroundLogE = oldLogE2  + C*st->mode->nbEBands;

   out_syn[0] = out_mem[0]+MAX_PERIOD-N;
   if (C==2)
      out_syn[1] = out_mem[1]+MAX_PERIOD-N;

   len = N+st->mode->overlap;
   
   if (st->loss_count >= 5)
   {
      VARDECL(celt_sig, freq);
      VARDECL(celt_norm, X);
      VARDECL(celt_ener, bandE);
      celt_uint32 seed;

      ALLOC(freq, C*N, celt_sig); /**< Interleaved signal MDCTs */
      ALLOC(X, C*N, celt_norm);   /**< Interleaved normalised MDCTs */
      ALLOC(bandE, st->mode->nbEBands*C, celt_ener);

      log2Amp(st->mode, st->start, st->end, bandE, backgroundLogE, C);

      seed = st->rng;
      for (c=0;c<C;c++)
      {
         for (i=0;i<st->mode->effEBands;i++)
         {
            int j;
            int boffs;
            int blen;
            boffs = N*c+(st->mode->eBands[i]<<LM);
            blen = (st->mode->eBands[i+1]-st->mode->eBands[i])<<LM;
            for (j=0;j<blen;j++)
            {
               seed = lcg_rand(seed);
               X[boffs+j] = (celt_int32)(seed)>>20;
            }
            renormalise_vector(X+boffs, blen, Q15ONE);
         }
      }
      st->rng = seed;

      denormalise_bands(st->mode, X, freq, bandE, st->mode->effEBands, C, 1<<LM);

      compute_inv_mdcts(st->mode, 0, freq, out_syn, overlap_mem, C, LM);
      plc = 0;
   } else if (st->loss_count == 0)
   {
      celt_word16 pitch_buf[MAX_PERIOD>>1];
      int len2 = len;
      /* FIXME: This is a kludge */
      if (len2>MAX_PERIOD>>1)
         len2 = MAX_PERIOD>>1;
      pitch_downsample(out_mem, pitch_buf, MAX_PERIOD, C);
      pitch_search(pitch_buf+((MAX_PERIOD-len2)>>1), pitch_buf, len2,
                   MAX_PERIOD-len2-100, &pitch_index);
      pitch_index = MAX_PERIOD-len2-pitch_index;
      st->last_pitch_index = pitch_index;
   } else {
      pitch_index = st->last_pitch_index;
      fade = QCONST16(.8f,15);
   }

   if (plc)
   {
      c=0; do {
         /* FIXME: This is more memory than necessary */
         celt_word32 e[2*MAX_PERIOD];
         celt_word16 exc[2*MAX_PERIOD];
         celt_word32 ac[LPC_ORDER+1];
         celt_word16 decay = 1;
         celt_word32 S1=0;
         celt_word16 mem[LPC_ORDER]={0};

         offset = MAX_PERIOD-pitch_index;
         for (i=0;i<MAX_PERIOD;i++)
            exc[i] = ROUND16(out_mem[c][i], SIG_SHIFT);

         if (st->loss_count == 0)
         {
            _celt_autocorr(exc, ac, st->mode->window, st->mode->overlap,
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
         fir(exc, lpc+c*LPC_ORDER, exc, MAX_PERIOD, LPC_ORDER, mem);
         /*for (i=0;i<MAX_PERIOD;i++)printf("%d ", exc[i]); printf("\n");*/
         /* Check if the waveform is decaying (and if so how fast) */
         {
            celt_word32 E1=1, E2=1;
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
            decay = celt_sqrt(frac_div32(SHR(E1,1),E2));
         }

         /* Copy excitation, taking decay into account */
         for (i=0;i<len+st->mode->overlap;i++)
         {
            celt_word16 tmp;
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
         for (i=0;i<len+st->mode->overlap;i++)
            e[i] = MULT16_32_Q15(fade, e[i]);
         iir(e, lpc+c*LPC_ORDER, e, len+st->mode->overlap, LPC_ORDER, mem);

         {
            celt_word32 S2=0;
            for (i=0;i<len+overlap;i++)
            {
               celt_word16 tmp = ROUND16(e[i],SIG_SHIFT);
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
                  celt_word16 ratio = celt_sqrt(frac_div32(SHR32(S1,1)+1,S2+1));
                  for (i=0;i<len+overlap;i++)
                     e[i] = MULT16_32_Q15(ratio, e[i]);
               }
         }

#ifdef ENABLE_POSTFILTER
         /* Apply post-filter to the MDCT overlap of the previous frame */
         comb_filter(out_mem[c]+MAX_PERIOD, out_mem[c]+MAX_PERIOD, st->postfilter_period, st->postfilter_period, st->overlap,
               st->postfilter_gain, st->postfilter_gain, st->postfilter_tapset, st->postfilter_tapset,
               NULL, 0);
#endif /* ENABLE_POSTFILTER */

         for (i=0;i<MAX_PERIOD+st->mode->overlap-N;i++)
            out_mem[c][i] = out_mem[c][N+i];

         /* Apply TDAC to the concealed audio so that it blends with the
         previous and next frames */
         for (i=0;i<overlap/2;i++)
         {
            celt_word32 tmp;
            tmp = MULT16_32_Q15(st->mode->window[i],           e[N+overlap-1-i]) +
                  MULT16_32_Q15(st->mode->window[overlap-i-1], e[N+i          ]);
            out_mem[c][MAX_PERIOD+i] = MULT16_32_Q15(st->mode->window[overlap-i-1], tmp);
            out_mem[c][MAX_PERIOD+overlap-i-1] = MULT16_32_Q15(st->mode->window[i], tmp);
         }
         for (i=0;i<N;i++)
            out_mem[c][MAX_PERIOD-N+i] = e[i];

#ifdef ENABLE_POSTFILTER
         /* Apply pre-filter to the MDCT overlap for the next frame (post-filter will be applied then) */
         comb_filter(e, out_mem[c]+MAX_PERIOD, st->postfilter_period, st->postfilter_period, st->overlap,
               -st->postfilter_gain, -st->postfilter_gain, st->postfilter_tapset, st->postfilter_tapset,
               NULL, 0);
#endif /* ENABLE_POSTFILTER */
         for (i=0;i<overlap;i++)
            out_mem[c][MAX_PERIOD+i] = e[i];
      } while (++c<C);
   }

   deemphasis(out_syn, pcm, N, C, st->downsample, st->mode->preemph, st->preemph_memD);
   
   st->loss_count++;

   RESTORE_STACK;
}

#ifdef FIXED_POINT
CELT_STATIC
int celt_decode_with_ec(CELTDecoder * restrict st, const unsigned char *data, int len, celt_int16 * restrict pcm, int frame_size, ec_dec *dec)
{
#else
CELT_STATIC
int celt_decode_with_ec_float(CELTDecoder * restrict st, const unsigned char *data, int len, celt_sig * restrict pcm, int frame_size, ec_dec *dec)
{
#endif
   int c, i, N;
   int spread_decision;
   celt_int32 bits;
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
   celt_word16 *lpc;
   celt_word16 *oldBandE, *oldLogE, *oldLogE2, *backgroundLogE;

   int shortBlocks;
   int isTransient;
   int intra_ener;
   const int CC = CHANNELS(st->channels);
   int LM, M;
   int effEnd;
   int codedBands;
   int alloc_trim;
   int postfilter_pitch;
   celt_word16 postfilter_gain;
   int intensity=0;
   int dual_stereo=0;
   celt_int32 total_bits;
   celt_int32 balance;
   celt_int32 tell;
   int dynalloc_logp;
   int postfilter_tapset;
   int anti_collapse_rsv;
   int anti_collapse_on=0;
   int silence;
   int C = CHANNELS(st->stream_channels);
   ALLOC_STACK;
   SAVE_STACK;

   frame_size *= st->downsample;

   c=0; do {
      decode_mem[c] = st->_decode_mem + c*(DECODE_BUFFER_SIZE+st->overlap);
      out_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE-MAX_PERIOD;
      overlap_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE;
   } while (++c<CC);
   lpc = (celt_word16*)(st->_decode_mem+(DECODE_BUFFER_SIZE+st->overlap)*CC);
   oldBandE = lpc+CC*LPC_ORDER;
   oldLogE = oldBandE + CC*st->mode->nbEBands;
   oldLogE2 = oldLogE + CC*st->mode->nbEBands;
   backgroundLogE = oldLogE2  + CC*st->mode->nbEBands;

   if (st->signalling && data!=NULL)
   {
      st->end = IMAX(1, st->mode->effEBands-2*(data[0]>>5));
      LM = (data[0]>>3)&0x3;
      C = 1 + ((data[0]>>2)&0x1);
      data++;
      len--;
      if (LM>st->mode->maxLM)
         return CELT_CORRUPTED_DATA;
      if (frame_size < st->mode->shortMdctSize<<LM)
         return CELT_BAD_ARG;
      else
         frame_size = st->mode->shortMdctSize<<LM;
   } else {
      for (LM=0;LM<=st->mode->maxLM;LM++)
         if (st->mode->shortMdctSize<<LM==frame_size)
            break;
      if (LM>st->mode->maxLM)
         return CELT_BAD_ARG;
   }
   M=1<<LM;

   if (len<0 || len>1275 || pcm==NULL)
      return CELT_BAD_ARG;

   N = M*st->mode->shortMdctSize;

   effEnd = st->end;
   if (effEnd > st->mode->effEBands)
      effEnd = st->mode->effEBands;

   ALLOC(freq, CC*N, celt_sig); /**< Interleaved signal MDCTs */
   ALLOC(X, CC*N, celt_norm);   /**< Interleaved normalised MDCTs */
   ALLOC(bandE, st->mode->nbEBands*CC, celt_ener);
   c=0; do
      for (i=0;i<M*st->mode->eBands[st->start];i++)
         X[c*N+i] = 0;
   while (++c<CC);
   c=0; do   
      for (i=M*st->mode->eBands[effEnd];i<N;i++)
         X[c*N+i] = 0;
   while (++c<CC);

   if (data == NULL || len<=1)
   {
      celt_decode_lost(st, pcm, N, LM);
      RESTORE_STACK;
      return CELT_OK;
   }
   if (len<0) {
     RESTORE_STACK;
     return CELT_BAD_ARG;
   }

   if (dec == NULL)
   {
      ec_dec_init(&_dec,(unsigned char*)data,len);
      dec = &_dec;
   }

   if (C>CC)
   {
      RESTORE_STACK;
      return CELT_CORRUPTED_DATA;
   } else if (C<CC)
   {
      for (i=0;i<st->mode->nbEBands;i++)
         oldBandE[i]=MAX16(oldBandE[i],oldBandE[st->mode->nbEBands+i]);
   }

   total_bits = len*8;
   tell = ec_tell(dec);

   if (tell==1)
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
#ifdef ENABLE_POSTFILTER
         int qg, octave;
         octave = ec_dec_uint(dec, 6);
         postfilter_pitch = (16<<octave)+ec_dec_bits(dec, 4+octave)-1;
         qg = ec_dec_bits(dec, 3);
         if (ec_tell(dec)+2<=total_bits)
            postfilter_tapset = ec_dec_icdf(dec, tapset_icdf, 2);
         postfilter_gain = QCONST16(.09375f,15)*(qg+1);
#else /* ENABLE_POSTFILTER */
         RESTORE_STACK;
         return CELT_CORRUPTED_DATA;
#endif /* ENABLE_POSTFILTER */
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
   unquant_coarse_energy(st->mode, st->start, st->end, oldBandE,
         intra_ener, dec, C, LM);

   ALLOC(tf_res, st->mode->nbEBands, int);
   tf_decode(st->start, st->end, isTransient, tf_res, LM, dec);

   tell = ec_tell(dec);
   spread_decision = SPREAD_NORMAL;
   if (tell+4 <= total_bits)
      spread_decision = ec_dec_icdf(dec, spread_icdf, 5);

   ALLOC(pulses, st->mode->nbEBands, int);
   ALLOC(cap, st->mode->nbEBands, int);
   ALLOC(offsets, st->mode->nbEBands, int);
   ALLOC(fine_priority, st->mode->nbEBands, int);

   init_caps(st->mode,cap,LM,C);

   dynalloc_logp = 6;
   total_bits<<=BITRES;
   tell = ec_tell_frac(dec);
   for (i=st->start;i<st->end;i++)
   {
      int width, quanta;
      int dynalloc_loop_logp;
      int boost;
      width = C*(st->mode->eBands[i+1]-st->mode->eBands[i])<<LM;
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

   ALLOC(fine_quant, st->mode->nbEBands, int);
   alloc_trim = tell+(6<<BITRES) <= total_bits ?
         ec_dec_icdf(dec, trim_icdf, 7) : 5;

   bits = ((celt_int32)len*8<<BITRES) - ec_tell_frac(dec) - 1;
   anti_collapse_rsv = isTransient&&LM>=2&&bits>=(LM+2<<BITRES) ? (1<<BITRES) : 0;
   bits -= anti_collapse_rsv;
   codedBands = compute_allocation(st->mode, st->start, st->end, offsets, cap,
         alloc_trim, &intensity, &dual_stereo, bits, &balance, pulses,
         fine_quant, fine_priority, C, LM, dec, 0, 0);
   
   unquant_fine_energy(st->mode, st->start, st->end, oldBandE, fine_quant, dec, C);

   /* Decode fixed codebook */
   ALLOC(collapse_masks, C*st->mode->nbEBands, unsigned char);
   quant_all_bands(0, st->mode, st->start, st->end, X, C==2 ? X+N : NULL, collapse_masks,
         NULL, pulses, shortBlocks, spread_decision, dual_stereo, intensity, tf_res, 1,
         len*(8<<BITRES)-anti_collapse_rsv, balance, dec, LM, codedBands, &st->rng);

   if (anti_collapse_rsv > 0)
   {
      anti_collapse_on = ec_dec_bits(dec, 1);
   }

   unquant_energy_finalise(st->mode, st->start, st->end, oldBandE,
         fine_quant, fine_priority, len*8-ec_tell(dec), dec, C);

   if (anti_collapse_on)
      anti_collapse(st->mode, X, collapse_masks, LM, C, CC, N,
            st->start, st->end, oldBandE, oldLogE, oldLogE2, pulses, st->rng);

   log2Amp(st->mode, st->start, st->end, bandE, oldBandE, C);

   if (silence)
   {
      for (i=0;i<C*st->mode->nbEBands;i++)
      {
         bandE[i] = 0;
         oldBandE[i] = -QCONST16(28.f,DB_SHIFT);
      }
   }
   /* Synthesis */
   denormalise_bands(st->mode, X, freq, bandE, effEnd, C, M);

   CELT_MOVE(decode_mem[0], decode_mem[0]+N, DECODE_BUFFER_SIZE-N);
   if (CC==2)
      CELT_MOVE(decode_mem[1], decode_mem[1]+N, DECODE_BUFFER_SIZE-N);

   c=0; do
      for (i=0;i<M*st->mode->eBands[st->start];i++)
         freq[c*N+i] = 0;
   while (++c<C);
   c=0; do {
      int bound = M*st->mode->eBands[effEnd];
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

   /* Compute inverse MDCTs */
   compute_inv_mdcts(st->mode, shortBlocks, freq, out_syn, overlap_mem, CC, LM);

#ifdef ENABLE_POSTFILTER
   c=0; do {
      st->postfilter_period=IMAX(st->postfilter_period, COMBFILTER_MINPERIOD);
      st->postfilter_period_old=IMAX(st->postfilter_period_old, COMBFILTER_MINPERIOD);
      comb_filter(out_syn[c], out_syn[c], st->postfilter_period_old, st->postfilter_period, st->mode->shortMdctSize,
            st->postfilter_gain_old, st->postfilter_gain, st->postfilter_tapset_old, st->postfilter_tapset,
            st->mode->window, st->overlap);
      if (LM!=0)
         comb_filter(out_syn[c]+st->mode->shortMdctSize, out_syn[c]+st->mode->shortMdctSize, st->postfilter_period, postfilter_pitch, N-st->mode->shortMdctSize,
               st->postfilter_gain, postfilter_gain, st->postfilter_tapset, postfilter_tapset,
               st->mode->window, st->mode->overlap);

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
#endif /* ENABLE_POSTFILTER */

   if (CC==2&&C==1) {
      for (i=0;i<st->mode->nbEBands;i++)
         oldBandE[st->mode->nbEBands+i]=oldBandE[i];
   }

   /* In case start or end were to change */
   c=0; do
   {
      for (i=0;i<st->start;i++)
         oldBandE[c*st->mode->nbEBands+i]=0;
      for (i=st->end;i<st->mode->nbEBands;i++)
         oldBandE[c*st->mode->nbEBands+i]=0;
   } while (++c<CC);
   if (!isTransient)
   {
      for (i=0;i<CC*st->mode->nbEBands;i++)
         oldLogE2[i] = oldLogE[i];
      for (i=0;i<CC*st->mode->nbEBands;i++)
         oldLogE[i] = oldBandE[i];
      for (i=0;i<CC*st->mode->nbEBands;i++)
         backgroundLogE[i] = MIN16(backgroundLogE[i] + M*QCONST16(0.001f,DB_SHIFT), oldBandE[i]);
   } else {
      for (i=0;i<CC*st->mode->nbEBands;i++)
         oldLogE[i] = MIN16(oldLogE[i], oldBandE[i]);
   }
   st->rng = dec->rng;

   deemphasis(out_syn, pcm, N, CC, st->downsample, st->mode->preemph, st->preemph_memD);
   st->loss_count = 0;
   RESTORE_STACK;
   if (ec_tell(dec) > 8*len || ec_get_error(dec))
      return CELT_CORRUPTED_DATA;
   else
      return frame_size/st->downsample;
}

#ifdef FIXED_POINT
#ifndef DISABLE_FLOAT_API
CELT_STATIC
int celt_decode_with_ec_float(CELTDecoder * restrict st, const unsigned char *data, int len, float * restrict pcm, int frame_size, ec_dec *dec)
{
   int j, ret, C, N;
   VARDECL(celt_int16, out);
   ALLOC_STACK;
   SAVE_STACK;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   C = CHANNELS(st->channels);
   N = frame_size;
   
   ALLOC(out, C*N, celt_int16);
   ret=celt_decode_with_ec(st, data, len, out, frame_size, dec);
   if (ret>0)
      for (j=0;j<C*ret;j++)
         pcm[j]=out[j]*(1.f/32768.f);
     
   RESTORE_STACK;
   return ret;
}
#endif /*DISABLE_FLOAT_API*/
#else
CELT_STATIC
int celt_decode_with_ec(CELTDecoder * restrict st, const unsigned char *data, int len, celt_int16 * restrict pcm, int frame_size, ec_dec *dec)
{
   int j, ret, C, N;
   VARDECL(celt_sig, out);
   ALLOC_STACK;
   SAVE_STACK;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   C = CHANNELS(st->channels);
   N = frame_size;
   ALLOC(out, C*N, celt_sig);

   ret=celt_decode_with_ec_float(st, data, len, out, frame_size, dec);

   if (ret>0)
      for (j=0;j<C*ret;j++)
         pcm[j] = FLOAT2INT16 (out[j]);
   
   RESTORE_STACK;
   return ret;
}
#endif

int celt_decode(CELTDecoder * restrict st, const unsigned char *data, int len, celt_int16 * restrict pcm, int frame_size)
{
   return celt_decode_with_ec(st, data, len, pcm, frame_size, NULL);
}

#ifndef DISABLE_FLOAT_API
int celt_decode_float(CELTDecoder * restrict st, const unsigned char *data, int len, float * restrict pcm, int frame_size)
{
   return celt_decode_with_ec_float(st, data, len, pcm, frame_size, NULL);
}
#endif /* DISABLE_FLOAT_API */

int celt_decoder_ctl(CELTDecoder * restrict st, int request, ...)
{
   va_list ap;

   va_start(ap, request);
   switch (request)
   {
      case CELT_GET_MODE_REQUEST:
      {
         const CELTMode ** value = va_arg(ap, const CELTMode**);
         if (value==0)
            goto bad_arg;
         *value=st->mode;
      }
      break;
      case CELT_SET_START_BAND_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         if (value<0 || value>=st->mode->nbEBands)
            goto bad_arg;
         st->start = value;
      }
      break;
      case CELT_SET_END_BAND_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         if (value<1 || value>st->mode->nbEBands)
            goto bad_arg;
         st->end = value;
      }
      break;
      case CELT_SET_CHANNELS_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         if (value<1 || value>2)
            goto bad_arg;
         st->stream_channels = value;
      }
      break;
      case CELT_SET_SIGNALLING_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         st->signalling = value;
      }
      break;
      case CELT_RESET_STATE:
      {
         CELT_MEMSET((char*)&st->DECODER_RESET_START, 0,
               celt_decoder_get_size_custom(st->mode, st->channels)-
               ((char*)&st->DECODER_RESET_START - (char*)st));
      }
      break;
      default:
         goto bad_request;
   }
   va_end(ap);
   return CELT_OK;
bad_arg:
   va_end(ap);
   return CELT_BAD_ARG;
bad_request:
      va_end(ap);
  return CELT_UNIMPLEMENTED;
}

const char *celt_strerror(int error)
{
   static const char *error_strings[8] = {
      "success",
      "invalid argument",
      "invalid mode",
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

