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
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
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

static const int trim_cdf[7] = {0, 4, 10, 23, 119, 125, 128};
#define COMBFILTER_MAXPERIOD 1024
#define COMBFILTER_MINPERIOD 16

/** Encoder state 
 @brief Encoder state
 */
struct CELTEncoder {
   const CELTMode *mode;     /**< Mode used by the encoder */
   int overlap;
   int channels;
   
   int force_intra;
   int complexity;
   int start, end;

   celt_int32 vbr_rate_norm; /* Target number of 8th bits per frame */

   /* Everything beyond this point gets cleared on a reset */
#define ENCODER_RESET_START frame_max

   celt_word32 frame_max;
   int fold_decision;
   int delayedIntra;
   int tonal_average;

   int prefilter_period;
   celt_word16 prefilter_gain;

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
   /* celt_word16 oldEBands[], Size = channels*mode->nbEBands */
};

int celt_encoder_get_size(const CELTMode *mode, int channels)
{
   int size = sizeof(struct CELTEncoder)
         + (2*channels*mode->overlap-1)*sizeof(celt_sig)
         + channels*COMBFILTER_MAXPERIOD*sizeof(celt_sig)
         + channels*mode->nbEBands*sizeof(celt_word16);
   return size;
}

CELTEncoder *celt_encoder_create(const CELTMode *mode, int channels, int *error)
{
   return celt_encoder_init(
         (CELTEncoder *)celt_alloc(celt_encoder_get_size(mode, channels)),
         mode, channels, error);
}

CELTEncoder *celt_encoder_init(CELTEncoder *st, const CELTMode *mode, int channels, int *error)
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

   CELT_MEMSET((char*)st, 0, celt_encoder_get_size(mode, channels));
   
   st->mode = mode;
   st->overlap = mode->overlap;
   st->channels = channels;

   st->start = 0;
   st->end = st->mode->effEBands;

   st->vbr_rate_norm = 0;
   st->vbr_offset = -(64<<BITRES);
   st->force_intra  = 0;
   st->delayedIntra = 1;
   st->tonal_average = 256;
   st->fold_decision = 1;
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
                              celt_word32 *frame_max, int overlap)
{
   int i, n;
   celt_word32 threshold;
   VARDECL(celt_word32, begin);
   VARDECL(celt_word16, tmp);
   celt_word32 mem0=0,mem1=0;
   SAVE_STACK;
   ALLOC(tmp, len, celt_word16);
   ALLOC(begin, len+1, celt_word32);

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
      mem1 = x - .5*y;
#endif
      tmp[i] = EXTRACT16(SHR(y,2));
   }
   /* First few samples are bad because we don't propagate the memory */
   for (i=0;i<24;i++)
      tmp[i] = 0;

   begin[0] = 0;
   for (i=0;i<len;i++)
      begin[i+1] = MAX32(begin[i], ABS32(tmp[i]));

   n = -1;

   threshold = MULT16_32_Q15(QCONST16(.4f,15),begin[len]);
   /* If the following condition isn't met, there's just no way
      we'll have a transient*/
   if (*frame_max < threshold)
   {
      /* It's likely we have a transient, now find it */
      for (i=8;i<len-8;i++)
      {
         if (begin[i+1] < threshold)
            n=i;
      }
   }

   *frame_max = begin[len-overlap];
   /* Only consider the last 7.5 ms for the next transient */
   if (len>360+overlap)
   {
      *frame_max = 0;
      for (i=len-360-overlap;i<len-overlap;i++)
         *frame_max = MAX32(*frame_max, ABS32(tmp[i]));
   }
   RESTORE_STACK;
   return n>=32;
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
      for (c=0;c<C;c++)
      {
         for (b=0;b<B;b++)
         {
            int j;
            clt_mdct_forward(&mode->mdct, in+c*(B*N+overlap)+b*N, tmp, mode->window, overlap, shortBlocks ? mode->maxLM : mode->maxLM-LM);
            /* Interleaving the sub-frames */
            for (j=0;j<N;j++)
               out[(j*B+b)+c*N*B] = tmp[j];
         }
      }
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
   for (c=0;c<C;c++)
   {
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
   }
}

static void deemphasis(celt_sig *in[], celt_word16 *pcm, int N, int _C, const celt_word16 *coef, celt_sig *mem)
{
   const int C = CHANNELS(_C);
   int c;
   for (c=0;c<C;c++)
   {
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
         *y = SCALEOUT(SIG2WORD16(tmp));
         x++;
         y+=C;
      }
      mem[c] = m;
   }
}

#ifdef ENABLE_POSTFILTER
/* FIXME: Handle the case where T = maxperiod */
static void comb_filter(celt_word32 *y, celt_word32 *x, int T0, int T1, int N,
      int C, celt_word16 g0, celt_word16 g1, const celt_word16 *window, int overlap)
{
   int i;
   /* printf ("%d %d %f %f\n", T0, T1, g0, g1); */
   celt_word16 g00, g01, g02, g10, g11, g12;
   celt_word16 t0, t1, t2;
   /* zeros at theta = +/- 5*pi/6 */
   t0 = QCONST16(.26795f, 15);
   t1 = QCONST16(.46410f, 15);
   t2 = QCONST16(.26795f, 15);
   g00 = MULT16_16_Q15(g0, t0);
   g01 = MULT16_16_Q15(g0, t1);
   g02 = MULT16_16_Q15(g0, t2);
   g10 = MULT16_16_Q15(g1, t0);
   g11 = MULT16_16_Q15(g1, t1);
   g12 = MULT16_16_Q15(g1, t2);
   for (i=0;i<overlap;i++)
   {
      celt_word16 f;
      f = MULT16_16_Q15(window[i],window[i]);
      y[i] = x[i]
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g01),x[i-T0])
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g00),x[i-T0-1])
               + MULT16_32_Q15(MULT16_16_Q15((Q15ONE-f),g02),x[i-T0+1])
               + MULT16_32_Q15(MULT16_16_Q15(f,g11),x[i-T1])
               + MULT16_32_Q15(MULT16_16_Q15(f,g10),x[i-T1-1])
               + MULT16_32_Q15(MULT16_16_Q15(f,g12),x[i-T1+1]);

   }
   for (i=overlap;i<N;i++)
      y[i] = x[i]
               + MULT16_32_Q15(g11,x[i-T1])
               + MULT16_32_Q15(g10,x[i-T1-1])
               + MULT16_32_Q15(g12,x[i-T1+1]);
}
#endif /* ENABLE_POSTFILTER */

static const signed char tf_select_table[4][8] = {
      {0, -1, 0, -1,    0,-1, 0,-1},
      {0, -1, 0, -2,    1, 0, 1 -1},
      {0, -2, 0, -3,    2, 0, 1 -1},
      {0, -2, 0, -3,    2, 0, 1 -1},
};

static celt_word32 l1_metric(const celt_norm *tmp, int N, int LM, int width)
{
   int i, j;
   static const celt_word16 sqrtM_1[4] = {Q15ONE, QCONST16(0.70711f,15), QCONST16(0.5f,15), QCONST16(0.35355f,15)};
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

   /* FIXME: Should check number of bytes *left* */
   if (nbCompressedBytes<15*C)
   {
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
      /* FIXME: Do something with the right channel */
      /*if (C==2)
         for (j=0;j<N;j++)
            tmp[j] = ADD16(tmp[j],X[N0+j+(m->eBands[i]<<LM)]);*/
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

static void tf_encode(int start, int end, int isTransient, int *tf_res, int nbCompressedBytes, int LM, int tf_select, ec_enc *enc)
{
   int curr, i;
   ec_enc_bit_prob(enc, tf_res[start], isTransient ? 16384 : 4096);
   curr = tf_res[start];
   for (i=start+1;i<end;i++)
   {
      ec_enc_bit_prob(enc, tf_res[i] ^ curr, isTransient ? 4096 : 2048);
      curr = tf_res[i];
   }
   ec_enc_bits(enc, tf_select, 1);
   for (i=start;i<end;i++)
      tf_res[i] = tf_select_table[LM][4*isTransient+2*tf_select+tf_res[i]];
   /*printf("%d %d ", isTransient, tf_select); for(i=0;i<end;i++)printf("%d ", tf_res[i]);printf("\n");*/
}

static void tf_decode(int start, int end, int C, int isTransient, int *tf_res, int nbCompressedBytes, int LM, ec_dec *dec)
{
   int i, curr, tf_select;
   tf_res[start] = ec_dec_bit_prob(dec, isTransient ? 16384 : 4096);
   curr = tf_res[start];
   for (i=start+1;i<end;i++)
   {
      tf_res[i] = ec_dec_bit_prob(dec, isTransient ? 4096 : 2048) ^ curr;
      curr = tf_res[i];
   }
   tf_select = ec_dec_bits(dec, 1);
   for (i=start;i<end;i++)
      tf_res[i] = tf_select_table[LM][4*isTransient+2*tf_select+tf_res[i]];
}

static int alloc_trim_analysis(const CELTMode *m, const celt_norm *X,
      const celt_word16 *bandLogE, int nbEBands, int LM, int C, int N0)
{
   int i;
   int trim_index = 2;
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
      if (sum > QCONST16(.995,10))
         trim_index-=3;
      else if (sum > QCONST16(.92,10))
         trim_index-=2;
      else if (sum > QCONST16(.8,10))
         trim_index-=1;
      else if (sum < QCONST16(.4,10))
         trim_index+=1;
   }
#if 0
   float diff=0;
   int c;
   for (c=0;c<C;c++)
   {
      for (i=0;i<nbEBands-1;i++)
      {
         diff += bandLogE[i+c*nbEBands]*(i-.5*nbEBands);
      }
   }
   diff /= C*(nbEBands-1);
   /*printf("%f\n", diff);*/
   if (diff > 4)
      trim_index--;
   if (diff > 8)
      trim_index--;
   if (diff < -4)
      trim_index++;
#endif
   if (trim_index<0)
      trim_index = 0;
   if (trim_index>5)
      trim_index = 5;
   return trim_index;
}

#ifdef FIXED_POINT
int celt_encode_with_ec(CELTEncoder * restrict st, const celt_int16 * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
#else
int celt_encode_with_ec_float(CELTEncoder * restrict st, const celt_sig * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
#endif
   int i, c, N;
   int bits;
   int has_fold=1;
   ec_byte_buffer buf;
   ec_enc         _enc;
   VARDECL(celt_sig, in);
   VARDECL(celt_sig, freq);
   VARDECL(celt_norm, X);
   VARDECL(celt_ener, bandE);
   VARDECL(celt_word16, bandLogE);
   VARDECL(int, fine_quant);
   VARDECL(celt_word16, error);
   VARDECL(int, pulses);
   VARDECL(int, offsets);
   VARDECL(int, fine_priority);
   VARDECL(int, tf_res);
   celt_sig *_overlap_mem;
   celt_sig *prefilter_mem;
   celt_word16 *oldBandE;
   int shortBlocks=0;
   int isTransient=0;
   int resynth;
   const int C = CHANNELS(st->channels);
   int LM, M;
   int tf_select;
   int nbFilledBytes, nbAvailableBytes;
   int effEnd;
   int codedBands;
   int tf_sum;
   int alloc_trim;
   int pitch_index=0;
   celt_word16 gain1 = 0;
   SAVE_STACK;

   if (nbCompressedBytes<0 || pcm==NULL)
     return CELT_BAD_ARG;

   for (LM=0;LM<4;LM++)
      if (st->mode->shortMdctSize<<LM==frame_size)
         break;
   if (LM>=MAX_CONFIG_SIZES)
      return CELT_BAD_ARG;
   M=1<<LM;

   prefilter_mem = st->in_mem+C*(st->overlap);
   _overlap_mem = prefilter_mem+C*COMBFILTER_MAXPERIOD;
   /*_overlap_mem = st->in_mem+C*(st->overlap);*/
   oldBandE = (celt_word16*)(st->in_mem+C*(2*st->overlap+COMBFILTER_MAXPERIOD));

   if (enc==NULL)
   {
      ec_byte_writeinit_buffer(&buf, compressed, nbCompressedBytes);
      ec_enc_init(&_enc,&buf);
      enc = &_enc;
      nbFilledBytes=0;
   } else {
      nbFilledBytes=(ec_enc_tell(enc, 0)+4)>>3;
   }
   nbAvailableBytes = nbCompressedBytes - nbFilledBytes;

   effEnd = st->end;
   if (effEnd > st->mode->effEBands)
      effEnd = st->mode->effEBands;

   N = M*st->mode->shortMdctSize;
   ALLOC(in, C*(N+st->overlap), celt_sig);

   /* Find pitch period and gain */
   {
      VARDECL(celt_sig, _pre);
      celt_sig *pre[2];
      SAVE_STACK;
      c = 0;
      ALLOC(_pre, C*(N+COMBFILTER_MAXPERIOD), celt_sig);

      pre[0] = _pre;
      pre[1] = _pre + (N+COMBFILTER_MAXPERIOD);

      for (c=0;c<C;c++)
      {
         const celt_word16 * restrict pcmp = pcm+c;
         celt_sig * restrict inp = in+c*(N+st->overlap)+st->overlap;

         for (i=0;i<N;i++)
         {
            /* Apply pre-emphasis */
            celt_sig tmp = MULT16_16(st->mode->preemph[2], SCALEIN(*pcmp));
            *inp = tmp + st->preemph_memE[c];
            st->preemph_memE[c] = MULT16_32_Q15(st->mode->preemph[1], *inp)
                                   - MULT16_32_Q15(st->mode->preemph[0], tmp);
            inp++;
            pcmp+=C;
         }
         CELT_COPY(pre[c], prefilter_mem+c*COMBFILTER_MAXPERIOD, COMBFILTER_MAXPERIOD);
         CELT_COPY(pre[c]+COMBFILTER_MAXPERIOD, in+c*(N+st->overlap)+st->overlap, N);
      }

#ifdef ENABLE_POSTFILTER
      {
         VARDECL(celt_word16, pitch_buf);
         ALLOC(pitch_buf, (COMBFILTER_MAXPERIOD+N)>>1, celt_word16);
         celt_word32 tmp=0;
         celt_word32 mem0[2]={0,0};
         celt_word16 mem1[2]={0,0};

         pitch_downsample(pre, pitch_buf, COMBFILTER_MAXPERIOD+N, COMBFILTER_MAXPERIOD+N,
                          C, mem0, mem1);
         pitch_search(st->mode, pitch_buf+(COMBFILTER_MAXPERIOD>>1), pitch_buf, N,
               COMBFILTER_MAXPERIOD-COMBFILTER_MINPERIOD, &pitch_index, &tmp, 1<<LM);
         pitch_index = COMBFILTER_MAXPERIOD-pitch_index;

         gain1 = remove_doubling(pitch_buf, COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD,
               N, &pitch_index, st->prefilter_period, st->prefilter_gain);
      }
      if (pitch_index > COMBFILTER_MAXPERIOD)
         pitch_index = COMBFILTER_MAXPERIOD;
      gain1 = MULT16_16_Q15(QCONST16(.7f,15),gain1);
      if (gain1 > QCONST16(.6f,15))
         gain1 = QCONST16(.6f,15);
      if (ABS16(gain1-st->prefilter_gain)<QCONST16(.1,15))
         gain1=st->prefilter_gain;
      if (gain1<QCONST16(.2f,15))
      {
         ec_enc_bit_prob(enc, 0, 32768);
         gain1 = 0;
      } else {
         int qg;
         int octave;
#ifdef FIXED_POINT
         qg = ((gain1+2048)>>12)-2;
#else
         qg = floor(.5+gain1*8)-2;
#endif
         ec_enc_bit_prob(enc, 1, 32768);
         octave = EC_ILOG(pitch_index)-5;
         ec_enc_uint(enc, octave, 6);
         ec_enc_bits(enc, pitch_index-(16<<octave), 4+octave);
         ec_enc_bits(enc, qg, 2);
         gain1 = QCONST16(.125f,15)*(qg+2);
      }
      /*printf("%d %f\n", pitch_index, gain1);*/
#else /* ENABLE_POSTFILTER */
      ec_enc_bit_prob(enc, 0, 32768);
#endif /* ENABLE_POSTFILTER */

      for (c=0;c<C;c++)
      {
         CELT_COPY(in+c*(N+st->overlap), st->in_mem+c*(st->overlap), st->overlap);
#ifdef ENABLE_POSTFILTER
         comb_filter(in+c*(N+st->overlap)+st->overlap, pre[c]+COMBFILTER_MAXPERIOD,
               st->prefilter_period, pitch_index, N, C, -st->prefilter_gain, -gain1, st->mode->window, st->mode->overlap);
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
      }

      RESTORE_STACK;
   }

#ifdef RESYNTH
   resynth = 1;
#else
   resynth = 0;
#endif

   if (st->complexity > 1 && LM>0)
   {
      isTransient = M > 1 &&
         transient_analysis(in, N+st->overlap, C, &st->frame_max, st->overlap);
   } else {
      isTransient = 0;
   }

   if (isTransient)
      shortBlocks = M;
   else
      shortBlocks = 0;

   ALLOC(freq, C*N, celt_sig); /**< Interleaved signal MDCTs */
   ALLOC(bandE,st->mode->nbEBands*C, celt_ener);
   ALLOC(bandLogE,st->mode->nbEBands*C, celt_word16);
   /* Compute MDCTs */
   compute_mdcts(st->mode, shortBlocks, in, freq, C, LM);

   ALLOC(X, C*N, celt_norm);         /**< Interleaved normalised MDCTs */

   compute_band_energies(st->mode, freq, bandE, effEnd, C, M);

   amp2Log2(st->mode, effEnd, st->end, bandE, bandLogE, C);

   /* Band normalisation */
   normalise_bands(st->mode, freq, X, bandE, effEnd, C, M);

   ALLOC(tf_res, st->mode->nbEBands, int);
   /* Needs to be before coarse energy quantization because otherwise the energy gets modified */
   tf_select = tf_analysis(st->mode, bandLogE, oldBandE, effEnd, C, isTransient, tf_res, nbAvailableBytes, X, N, LM, &tf_sum);
   for (i=effEnd;i<st->end;i++)
      tf_res[i] = tf_res[effEnd-1];

   ALLOC(error, C*st->mode->nbEBands, celt_word16);
   quant_coarse_energy(st->mode, st->start, st->end, effEnd, bandLogE,
         oldBandE, nbCompressedBytes*8, st->mode->prob,
         error, enc, C, LM, nbAvailableBytes, st->force_intra,
         &st->delayedIntra, st->complexity >= 4);

   if (LM > 0)
      ec_enc_bit_prob(enc, shortBlocks!=0, 8192);

   tf_encode(st->start, st->end, isTransient, tf_res, nbAvailableBytes, LM, tf_select, enc);

   if (shortBlocks || st->complexity < 3)
   {
      if (st->complexity == 0)
      {
         has_fold = 0;
         st->fold_decision = 3;
      } else {
         has_fold = 1;
         st->fold_decision = 1;
      }
   } else {
      has_fold = folding_decision(st->mode, X, &st->tonal_average, &st->fold_decision, effEnd, C, M);
   }
   ec_enc_bit_prob(enc, has_fold>>1, 8192);
   ec_enc_bit_prob(enc, has_fold&1, (has_fold>>1) ? 32768 : 49152);

   ALLOC(offsets, st->mode->nbEBands, int);

   for (i=0;i<st->mode->nbEBands;i++)
      offsets[i] = 0;
   /* Dynamic allocation code */
   /* Make sure that dynamic allocation can't make us bust the budget */
   if (nbCompressedBytes > 30)
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
      for (i=1;i<st->mode->nbEBands-1;i++)
      {
         if (2*bandLogE[i]-bandLogE[i-1]-bandLogE[i+1] > SHL16(t1,DB_SHIFT))
            offsets[i] += 1;
         if (2*bandLogE[i]-bandLogE[i-1]-bandLogE[i+1] > SHL16(t2,DB_SHIFT))
            offsets[i] += 1;
      }
   }
   for (i=0;i<st->mode->nbEBands;i++)
   {
      int j;
      ec_enc_bit_prob(enc, offsets[i]!=0, 1024);
      if (offsets[i]!=0)
      {
         for (j=0;j<offsets[i]-1;j++)
            ec_enc_bit_prob(enc, 1, 32768);
         ec_enc_bit_prob(enc, 0, 32768);
      }
      offsets[i] *= (6<<BITRES);
   }
   alloc_trim = alloc_trim_analysis(st->mode, X, bandLogE, st->mode->nbEBands, LM, C, N);
   ec_encode_bin(enc, trim_cdf[alloc_trim], trim_cdf[alloc_trim+1], 7);

   /* Variable bitrate */
   if (st->vbr_rate_norm>0)
   {
     celt_word16 alpha;
     celt_int32 delta;
     /* The target rate in 8th bits per frame */
     celt_int32 vbr_rate;
     celt_int32 target;
     celt_int32 vbr_bound, max_allowed;

     target = vbr_rate = M*st->vbr_rate_norm;

     /* Shortblocks get a large boost in bitrate, but since they
        are uncommon long blocks are not greatly affected */
     if (shortBlocks || tf_sum < -2*(st->end-st->start))
        target*=2;
     else if (tf_sum < -(st->end-st->start))
        target = 3*target/2;
     else if (M > 1)
        target-=(target+14)/28;

     /* The current offset is removed from the target and the space used
        so far is added*/
     target=target+st->vbr_offset+ec_enc_tell(enc, BITRES);

     /* Computes the max bit-rate allowed in VBR more to avoid violating the target rate and buffering */
     vbr_bound = vbr_rate;
     max_allowed = IMIN(vbr_rate+vbr_bound-st->vbr_reservoir>>(BITRES+3),nbAvailableBytes);

     /* In VBR mode the frame size must not be reduced so much that it would result in the encoder running out of bits */
     nbAvailableBytes = target+(1<<(BITRES+2))>>(BITRES+3);
     nbAvailableBytes=IMAX(16,IMIN(max_allowed,nbAvailableBytes));
     target=nbAvailableBytes<<(BITRES+3);

     if (st->vbr_count < 970)
     {
        st->vbr_count++;
        alpha = celt_rcp(SHL32(EXTEND32(st->vbr_count+20),16));
     } else
        alpha = QCONST16(.001f,15);
     /* By how much did we "miss" the target on that frame */
     delta = (celt_int32)target - vbr_rate;
     /* How many bits have we used in excess of what we're allowed */
     st->vbr_reservoir += delta;
     /*printf ("%d\n", st->vbr_reservoir);*/

     /* Compute the offset we need to apply in order to reach the target */
     st->vbr_drift += (celt_int32)MULT16_32_Q15(alpha,delta-st->vbr_offset-st->vbr_drift);
     st->vbr_offset = -st->vbr_drift;
     /*printf ("%d\n", st->vbr_drift);*/

     /* We could use any multiple of vbr_rate as bound (depending on the delay) */
     if (st->vbr_reservoir < 0)
     {
        /* We're under the min value -- increase rate */
        int adjust = (-st->vbr_reservoir)/(8<<BITRES);
        nbAvailableBytes += adjust;
        st->vbr_reservoir = 0;
        /*printf ("+%d\n", adjust);*/
     }
     nbCompressedBytes = IMIN(nbCompressedBytes,nbAvailableBytes+nbFilledBytes);

     /* This moves the raw bits to take into account the new compressed size */
     ec_byte_shrink(&buf, nbCompressedBytes);
   }

   /* Bit allocation */
   ALLOC(fine_quant, st->mode->nbEBands, int);
   ALLOC(pulses, st->mode->nbEBands, int);
   ALLOC(fine_priority, st->mode->nbEBands, int);

   bits = nbCompressedBytes*8 - ec_enc_tell(enc, 0) - 1;
   codedBands = compute_allocation(st->mode, st->start, st->end, offsets, alloc_trim, bits, pulses, fine_quant, fine_priority, C, LM);

   quant_fine_energy(st->mode, st->start, st->end, bandE, oldBandE, error, fine_quant, enc, C);

#ifdef MEASURE_NORM_MSE
   float X0[3000];
   float bandE0[60];
   for (c=0;c<C;c++)
      for (i=0;i<N;i++)
         X0[i+c*N] = X[i+c*N];
   for (i=0;i<C*st->mode->nbEBands;i++)
      bandE0[i] = bandE[i];
#endif

   /* Residual quantisation */
   quant_all_bands(1, st->mode, st->start, st->end, X, C==2 ? X+N : NULL, bandE, pulses, shortBlocks, has_fold, tf_res, resynth, nbCompressedBytes*8, enc, LM, codedBands);

   quant_energy_finalise(st->mode, st->start, st->end, bandE, oldBandE, error, fine_quant, fine_priority, nbCompressedBytes*8-ec_enc_tell(enc, 0), enc, C);

#ifdef RESYNTH
   /* Re-synthesis of the coded audio if required */
   if (resynth)
   {
      celt_sig *out_mem[2];
      celt_sig *overlap_mem[2];

      log2Amp(st->mode, st->start, st->end, bandE, oldBandE, C);

#ifdef MEASURE_NORM_MSE
      measure_norm_mse(st->mode, X, X0, bandE, bandE0, M, N, C);
#endif

      /* Synthesis */
      denormalise_bands(st->mode, X, freq, bandE, effEnd, C, M);

      CELT_MOVE(st->syn_mem[0], st->syn_mem[0]+N, MAX_PERIOD);
      if (C==2)
         CELT_MOVE(st->syn_mem[1], st->syn_mem[1]+N, MAX_PERIOD);

      for (c=0;c<C;c++)
         for (i=0;i<M*st->mode->eBands[st->start];i++)
            freq[c*N+i] = 0;
      for (c=0;c<C;c++)
         for (i=M*st->mode->eBands[st->end];i<N;i++)
            freq[c*N+i] = 0;

      out_mem[0] = st->syn_mem[0]+MAX_PERIOD;
      if (C==2)
         out_mem[1] = st->syn_mem[1]+MAX_PERIOD;

      for (c=0;c<C;c++)
         overlap_mem[c] = _overlap_mem + c*st->overlap;

      compute_inv_mdcts(st->mode, shortBlocks, freq, out_mem, overlap_mem, C, LM);

#ifdef ENABLE_POSTFILTER
      for (c=0;c<C;c++)
      {
         comb_filter(out_mem[c], out_mem[c], st->prefilter_period, st->prefilter_period, st->overlap, C,
               st->prefilter_gain, st->prefilter_gain, NULL, 0);
         comb_filter(out_mem[c]+st->overlap, out_mem[c]+st->overlap, st->prefilter_period, pitch_index, N-st->overlap, C,
               st->prefilter_gain, gain1, st->mode->window, st->mode->overlap);
      }
#endif /* ENABLE_POSTFILTER */

      deemphasis(out_mem, (celt_word16*)pcm, N, C, st->mode->preemph, st->preemph_memD);
   }
#endif

   st->prefilter_period = pitch_index;
   st->prefilter_gain = gain1;

   /* If there's any room left (can only happen for very high rates),
      fill it with zeros */
   while (ec_enc_tell(enc,0) + 8 <= nbCompressedBytes*8)
      ec_enc_bits(enc, 0, 8);
   ec_enc_done(enc);
   
   RESTORE_STACK;
   if (ec_enc_get_error(enc))
      return CELT_CORRUPTED_DATA;
   else
      return nbCompressedBytes;
}

#ifdef FIXED_POINT
#ifndef DISABLE_FLOAT_API
int celt_encode_with_ec_float(CELTEncoder * restrict st, const float * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
   int j, ret, C, N, LM, M;
   VARDECL(celt_int16, in);
   SAVE_STACK;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   for (LM=0;LM<4;LM++)
      if (st->mode->shortMdctSize<<LM==frame_size)
         break;
   if (LM>=MAX_CONFIG_SIZES)
      return CELT_BAD_ARG;
   M=1<<LM;

   C = CHANNELS(st->channels);
   N = M*st->mode->shortMdctSize;
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
int celt_encode_with_ec(CELTEncoder * restrict st, const celt_int16 * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
   int j, ret, C, N, LM, M;
   VARDECL(celt_sig, in);
   SAVE_STACK;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   for (LM=0;LM<4;LM++)
      if (st->mode->shortMdctSize<<LM==frame_size)
         break;
   if (LM>=MAX_CONFIG_SIZES)
      return CELT_BAD_ARG;
   M=1<<LM;

   C=CHANNELS(st->channels);
   N=M*st->mode->shortMdctSize;
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
         if (value<0 || value>=st->mode->nbEBands)
            goto bad_arg;
         st->end = value;
      }
      break;
      case CELT_SET_PREDICTION_REQUEST:
      {
         int value = va_arg(ap, celt_int32);
         if (value<0 || value>2)
            goto bad_arg;
         if (value==0)
         {
            st->force_intra   = 1;
         } else if (value==1) {
            st->force_intra   = 0;
         } else {
            st->force_intra   = 0;
         }   
      }
      break;
      case CELT_SET_VBR_RATE_REQUEST:
      {
         celt_int32 value = va_arg(ap, celt_int32);
         int frame_rate;
         int N = st->mode->shortMdctSize;
         if (value<0)
            goto bad_arg;
         if (value>3072000)
            value = 3072000;
         frame_rate = ((st->mode->Fs<<3)+(N>>1))/N;
         st->vbr_rate_norm = ((value<<(BITRES+3))+(frame_rate>>1))/frame_rate;
      }
      break;
      case CELT_RESET_STATE:
      {
         CELT_MEMSET((char*)&st->ENCODER_RESET_START, 0,
               celt_encoder_get_size(st->mode, st->channels)-
               ((char*)&st->ENCODER_RESET_START - (char*)st));
         st->vbr_offset = -(64<<BITRES);
         st->delayedIntra = 1;
         st->fold_decision = 1;
         st->tonal_average = QCONST16(1.f,8);
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

   int start, end;

   /* Everything beyond this point gets cleared on a reset */
#define DECODER_RESET_START last_pitch_index

   int last_pitch_index;
   int loss_count;
   int postfilter_period;
   celt_word16 postfilter_gain;

   celt_sig preemph_memD[2];
   
   celt_sig _decode_mem[1]; /* Size = channels*(DECODE_BUFFER_SIZE+mode->overlap) */
   /* celt_word16 lpc[],  Size = channels*LPC_ORDER */
   /* celt_word16 oldEBands[], Size = channels*mode->nbEBands */
};

int celt_decoder_get_size(const CELTMode *mode, int channels)
{
   int size = sizeof(struct CELTDecoder)
            + (channels*(DECODE_BUFFER_SIZE+mode->overlap)-1)*sizeof(celt_sig)
            + channels*LPC_ORDER*sizeof(celt_word16)
            + channels*mode->nbEBands*sizeof(celt_word16);
   return size;
}

CELTDecoder *celt_decoder_create(const CELTMode *mode, int channels, int *error)
{
   return celt_decoder_init(
         (CELTDecoder *)celt_alloc(celt_decoder_get_size(mode, channels)),
         mode, channels, error);
}

CELTDecoder *celt_decoder_init(CELTDecoder *st, const CELTMode *mode, int channels, int *error)
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

   CELT_MEMSET((char*)st, 0, celt_decoder_get_size(mode, channels));

   st->mode = mode;
   st->overlap = mode->overlap;
   st->channels = channels;

   st->start = 0;
   st->end = st->mode->effEBands;

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
   SAVE_STACK;
   
   for (c=0;c<C;c++)
   {
      decode_mem[c] = st->_decode_mem + c*(DECODE_BUFFER_SIZE+st->overlap);
      out_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE-MAX_PERIOD;
      overlap_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE;
   }
   lpc = (celt_word16*)(st->_decode_mem+(DECODE_BUFFER_SIZE+st->overlap)*C);

   len = N+st->mode->overlap;
   
   if (st->loss_count == 0)
   {
      celt_word16 pitch_buf[MAX_PERIOD>>1];
      celt_word32 tmp=0;
      celt_word32 mem0[2]={0,0};
      celt_word16 mem1[2]={0,0};
      int len2 = len;
      /* FIXME: This is a kludge */
      if (len2>MAX_PERIOD>>1)
         len2 = MAX_PERIOD>>1;
      pitch_downsample(out_mem, pitch_buf, MAX_PERIOD, MAX_PERIOD,
                       C, mem0, mem1);
      pitch_search(st->mode, pitch_buf+((MAX_PERIOD-len2)>>1), pitch_buf, len2,
                   MAX_PERIOD-len2-100, &pitch_index, &tmp, 1<<LM);
      pitch_index = MAX_PERIOD-len2-pitch_index;
      st->last_pitch_index = pitch_index;
   } else {
      pitch_index = st->last_pitch_index;
      if (st->loss_count < 5)
         fade = QCONST16(.8f,15);
      else
         fade = 0;
   }

   for (c=0;c<C;c++)
   {
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
         if (offset+i >= MAX_PERIOD)
         {
            offset -= pitch_index;
            decay = MULT16_16_Q15(decay, decay);
         }
         e[i] = SHL32(EXTEND32(MULT16_16_Q15(decay, exc[offset+i])), SIG_SHIFT);
         S1 += SHR32(MULT16_16(out_mem[c][offset+i],out_mem[c][offset+i]),8);
      }
      for (i=0;i<LPC_ORDER;i++)
         mem[i] = out_mem[c][MAX_PERIOD-i];
      for (i=0;i<len+st->mode->overlap;i++)
         e[i] = MULT16_32_Q15(fade, e[i]);
      iir(e, lpc+c*LPC_ORDER, e, len+st->mode->overlap, LPC_ORDER, mem);

      {
         celt_word32 S2=0;
         for (i=0;i<len+overlap;i++)
            S2 += SHR32(MULT16_16(e[i],e[i]),8);
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
               e[i] = MULT16_16_Q15(ratio, e[i]);
         }
      }

#ifdef ENABLE_POSTFILTER
      /* Apply post-filter to the MDCT overlap of the previous frame */
      comb_filter(out_mem[c]+MAX_PERIOD, out_mem[c]+MAX_PERIOD, st->postfilter_period, st->postfilter_period, st->overlap, C,
                  st->postfilter_gain, st->postfilter_gain, NULL, 0);
#endif /* ENABLE_POSTFILTER */

      for (i=0;i<MAX_PERIOD+st->mode->overlap-N;i++)
         out_mem[c][i] = out_mem[c][N+i];

      /* Apply TDAC to the concealed audio so that it blends with the
         previous and next frames */
      for (i=0;i<overlap/2;i++)
      {
         celt_word32 tmp1, tmp2;
         tmp1 = MULT16_32_Q15(st->mode->window[i          ], e[i          ]) -
                MULT16_32_Q15(st->mode->window[overlap-i-1], e[overlap-i-1]);
         tmp2 = MULT16_32_Q15(st->mode->window[i],           e[N+overlap-1-i]) +
                MULT16_32_Q15(st->mode->window[overlap-i-1], e[N+i          ]);
         out_mem[c][MAX_PERIOD+i] = MULT16_32_Q15(st->mode->window[overlap-i-1], tmp2);
         out_mem[c][MAX_PERIOD+overlap-i-1] = MULT16_32_Q15(st->mode->window[i], tmp2);
         out_mem[c][MAX_PERIOD-N+i] += MULT16_32_Q15(st->mode->window[i], tmp1);
         out_mem[c][MAX_PERIOD-N+overlap-i-1] -= MULT16_32_Q15(st->mode->window[overlap-i-1], tmp1);
      }
      for (i=0;i<N-overlap;i++)
         out_mem[c][MAX_PERIOD-N+overlap+i] = e[overlap+i];

#ifdef ENABLE_POSTFILTER
      /* Apply pre-filter to the MDCT overlap for the next frame (post-filter will be applied then) */
      comb_filter(e, out_mem[c]+MAX_PERIOD, st->postfilter_period, st->postfilter_period, st->overlap, C,
                  -st->postfilter_gain, -st->postfilter_gain, NULL, 0);
#endif /* ENABLE_POSTFILTER */
      for (i=0;i<overlap;i++)
         out_mem[c][MAX_PERIOD+i] = e[i];
   }

   {
      celt_word32 *out_syn[2];
      out_syn[0] = out_mem[0]+MAX_PERIOD-N;
      if (C==2)
         out_syn[1] = out_mem[1]+MAX_PERIOD-N;
      deemphasis(out_syn, pcm, N, C, st->mode->preemph, st->preemph_memD);
   }
   
   st->loss_count++;

   RESTORE_STACK;
}

#ifdef FIXED_POINT
int celt_decode_with_ec(CELTDecoder * restrict st, const unsigned char *data, int len, celt_int16 * restrict pcm, int frame_size, ec_dec *dec)
{
#else
int celt_decode_with_ec_float(CELTDecoder * restrict st, const unsigned char *data, int len, celt_sig * restrict pcm, int frame_size, ec_dec *dec)
{
#endif
   int c, i, N;
   int has_fold;
   int bits;
   ec_dec _dec;
   ec_byte_buffer buf;
   VARDECL(celt_sig, freq);
   VARDECL(celt_norm, X);
   VARDECL(celt_ener, bandE);
   VARDECL(int, fine_quant);
   VARDECL(int, pulses);
   VARDECL(int, offsets);
   VARDECL(int, fine_priority);
   VARDECL(int, tf_res);
   celt_sig *out_mem[2];
   celt_sig *decode_mem[2];
   celt_sig *overlap_mem[2];
   celt_sig *out_syn[2];
   celt_word16 *lpc;
   celt_word16 *oldBandE;

   int shortBlocks;
   int isTransient;
   int intra_ener;
   const int C = CHANNELS(st->channels);
   int LM, M;
   int nbFilledBytes, nbAvailableBytes;
   int effEnd;
   int codedBands;
   int alloc_trim;
   int postfilter_pitch;
   celt_word16 postfilter_gain;
   SAVE_STACK;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   for (LM=0;LM<4;LM++)
      if (st->mode->shortMdctSize<<LM==frame_size)
         break;
   if (LM>=MAX_CONFIG_SIZES)
      return CELT_BAD_ARG;
   M=1<<LM;

   for (c=0;c<C;c++)
   {
      decode_mem[c] = st->_decode_mem + c*(DECODE_BUFFER_SIZE+st->overlap);
      out_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE-MAX_PERIOD;
      overlap_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE;
   }
   lpc = (celt_word16*)(st->_decode_mem+(DECODE_BUFFER_SIZE+st->overlap)*C);
   oldBandE = lpc+C*LPC_ORDER;

   N = M*st->mode->shortMdctSize;

   effEnd = st->end;
   if (effEnd > st->mode->effEBands)
      effEnd = st->mode->effEBands;

   ALLOC(freq, C*N, celt_sig); /**< Interleaved signal MDCTs */
   ALLOC(X, C*N, celt_norm);   /**< Interleaved normalised MDCTs */
   ALLOC(bandE, st->mode->nbEBands*C, celt_ener);
   for (c=0;c<C;c++)
      for (i=0;i<M*st->mode->eBands[st->start];i++)
         X[c*N+i] = 0;
   for (c=0;c<C;c++)
      for (i=M*st->mode->eBands[effEnd];i<N;i++)
         X[c*N+i] = 0;

   if (data == NULL)
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
      ec_byte_readinit(&buf,(unsigned char*)data,len);
      ec_dec_init(&_dec,&buf);
      dec = &_dec;
      nbFilledBytes = 0;
   } else {
      nbFilledBytes = (ec_dec_tell(dec, 0)+4)>>3;
   }
   nbAvailableBytes = len-nbFilledBytes;

   if (ec_dec_bit_prob(dec, 32768))
   {
#ifdef ENABLE_POSTFILTER
      int qg, octave;
      octave = ec_dec_uint(dec, 6);
      postfilter_pitch = (16<<octave)+ec_dec_bits(dec, 4+octave);
      qg = ec_dec_bits(dec, 2);
      postfilter_gain = QCONST16(.125f,15)*(qg+2);
#else /* ENABLE_POSTFILTER */
      RESTORE_STACK;
      return CELT_CORRUPTED_DATA;
#endif /* ENABLE_POSTFILTER */

   } else {
      postfilter_gain = 0;
      postfilter_pitch = 0;
   }

   /* Decode the global flags (first symbols in the stream) */
   intra_ener = ec_dec_bit_prob(dec, 8192);
   /* Get band energies */
   unquant_coarse_energy(st->mode, st->start, st->end, bandE, oldBandE,
         intra_ener, st->mode->prob, dec, C, LM);

   if (LM > 0)
      isTransient = ec_dec_bit_prob(dec, 8192);
   else
      isTransient = 0;

   if (isTransient)
      shortBlocks = M;
   else
      shortBlocks = 0;

   ALLOC(tf_res, st->mode->nbEBands, int);
   tf_decode(st->start, st->end, C, isTransient, tf_res, nbAvailableBytes, LM, dec);

   has_fold = ec_dec_bit_prob(dec, 8192)<<1;
   has_fold |= ec_dec_bit_prob(dec, (has_fold>>1) ? 32768 : 49152);

   ALLOC(pulses, st->mode->nbEBands, int);
   ALLOC(offsets, st->mode->nbEBands, int);
   ALLOC(fine_priority, st->mode->nbEBands, int);

   for (i=0;i<st->mode->nbEBands;i++)
      offsets[i] = 0;
   for (i=0;i<st->mode->nbEBands;i++)
   {
      if (ec_dec_bit_prob(dec, 1024))
      {
         while (ec_dec_bit_prob(dec, 32768))
            offsets[i]++;
         offsets[i]++;
         offsets[i] *= (6<<BITRES);
      }
   }

   ALLOC(fine_quant, st->mode->nbEBands, int);
   {
      int fl;
      alloc_trim = 0;
      fl = ec_decode_bin(dec, 7);
      while (trim_cdf[alloc_trim+1] <= fl)
         alloc_trim++;
      ec_dec_update(dec, trim_cdf[alloc_trim], trim_cdf[alloc_trim+1], 128);
   }

   bits = len*8 - ec_dec_tell(dec, 0) - 1;
   codedBands = compute_allocation(st->mode, st->start, st->end, offsets, alloc_trim, bits, pulses, fine_quant, fine_priority, C, LM);
   
   unquant_fine_energy(st->mode, st->start, st->end, bandE, oldBandE, fine_quant, dec, C);

   /* Decode fixed codebook */
   quant_all_bands(0, st->mode, st->start, st->end, X, C==2 ? X+N : NULL, NULL, pulses, shortBlocks, has_fold, tf_res, 1, len*8, dec, LM, codedBands);

   unquant_energy_finalise(st->mode, st->start, st->end, bandE, oldBandE,
         fine_quant, fine_priority, len*8-ec_dec_tell(dec, 0), dec, C);

   log2Amp(st->mode, st->start, st->end, bandE, oldBandE, C);

   /* Synthesis */
   denormalise_bands(st->mode, X, freq, bandE, effEnd, C, M);

   CELT_MOVE(decode_mem[0], decode_mem[0]+N, DECODE_BUFFER_SIZE-N);
   if (C==2)
      CELT_MOVE(decode_mem[1], decode_mem[1]+N, DECODE_BUFFER_SIZE-N);

   for (c=0;c<C;c++)
      for (i=0;i<M*st->mode->eBands[st->start];i++)
         freq[c*N+i] = 0;
   for (c=0;c<C;c++)
      for (i=M*st->mode->eBands[effEnd];i<N;i++)
         freq[c*N+i] = 0;

   out_syn[0] = out_mem[0]+MAX_PERIOD-N;
   if (C==2)
      out_syn[1] = out_mem[1]+MAX_PERIOD-N;

   /* Compute inverse MDCTs */
   compute_inv_mdcts(st->mode, shortBlocks, freq, out_syn, overlap_mem, C, LM);

#ifdef ENABLE_POSTFILTER
   for (c=0;c<C;c++)
   {
      comb_filter(out_syn[c], out_syn[c], st->postfilter_period, st->postfilter_period, st->overlap, C,
            st->postfilter_gain, st->postfilter_gain, NULL, 0);
      comb_filter(out_syn[c]+st->overlap, out_syn[c]+st->overlap, st->postfilter_period, postfilter_pitch, N-st->overlap, C,
            st->postfilter_gain, postfilter_gain, st->mode->window, st->mode->overlap);
   }
   st->postfilter_period = postfilter_pitch;
   st->postfilter_gain = postfilter_gain;
#endif /* ENABLE_POSTFILTER */

   deemphasis(out_syn, pcm, N, C, st->mode->preemph, st->preemph_memD);
   st->loss_count = 0;
   RESTORE_STACK;
   if (ec_dec_get_error(dec))
      return CELT_CORRUPTED_DATA;
   else
      return CELT_OK;
}

#ifdef FIXED_POINT
#ifndef DISABLE_FLOAT_API
int celt_decode_with_ec_float(CELTDecoder * restrict st, const unsigned char *data, int len, float * restrict pcm, int frame_size, ec_dec *dec)
{
   int j, ret, C, N, LM, M;
   VARDECL(celt_int16, out);
   SAVE_STACK;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   for (LM=0;LM<4;LM++)
      if (st->mode->shortMdctSize<<LM==frame_size)
         break;
   if (LM>=MAX_CONFIG_SIZES)
      return CELT_BAD_ARG;
   M=1<<LM;

   C = CHANNELS(st->channels);
   N = M*st->mode->shortMdctSize;
   
   ALLOC(out, C*N, celt_int16);
   ret=celt_decode_with_ec(st, data, len, out, frame_size, dec);
   if (ret==0)
      for (j=0;j<C*N;j++)
         pcm[j]=out[j]*(1.f/32768.f);
     
   RESTORE_STACK;
   return ret;
}
#endif /*DISABLE_FLOAT_API*/
#else
int celt_decode_with_ec(CELTDecoder * restrict st, const unsigned char *data, int len, celt_int16 * restrict pcm, int frame_size, ec_dec *dec)
{
   int j, ret, C, N, LM, M;
   VARDECL(celt_sig, out);
   SAVE_STACK;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   for (LM=0;LM<4;LM++)
      if (st->mode->shortMdctSize<<LM==frame_size)
         break;
   if (LM>=MAX_CONFIG_SIZES)
      return CELT_BAD_ARG;
   M=1<<LM;

   C = CHANNELS(st->channels);
   N = M*st->mode->shortMdctSize;
   ALLOC(out, C*N, celt_sig);

   ret=celt_decode_with_ec_float(st, data, len, out, frame_size, dec);

   if (ret==0)
      for (j=0;j<C*N;j++)
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
         if (value<0 || value>=st->mode->nbEBands)
            goto bad_arg;
         st->end = value;
      }
      break;
      case CELT_RESET_STATE:
      {
         CELT_MEMSET((char*)&st->DECODER_RESET_START, 0,
               celt_decoder_get_size(st->mode, st->channels)-
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

