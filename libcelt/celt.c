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

#ifdef FIXED_POINT
static const celt_word16 transientWindow[16] = {
     279,  1106,  2454,  4276,  6510,  9081, 11900, 14872,
   17896, 20868, 23687, 26258, 28492, 30314, 31662, 32489};
#else
static const float transientWindow[16] = {
   0.0085135f, 0.0337639f, 0.0748914f, 0.1304955f,
   0.1986827f, 0.2771308f, 0.3631685f, 0.4538658f,
   0.5461342f, 0.6368315f, 0.7228692f, 0.8013173f,
   0.8695045f, 0.9251086f, 0.9662361f, 0.9914865f};
#endif

/** Encoder state 
 @brief Encoder state
 */
struct CELTEncoder {
   const CELTMode *mode;     /**< Mode used by the encoder */
   int overlap;
   int channels;
   
   int force_intra;
   int start, end;

   celt_int32 vbr_rate_norm; /* Target number of 16th bits per frame */

   /* Everything beyond this point gets cleared on a reset */
#define ENCODER_RESET_START frame_max

   celt_word32 frame_max;
   int fold_decision;
   int delayedIntra;
   celt_word16 tonal_average;

   /* VBR-related parameters */
   celt_int32 vbr_reservoir;
   celt_int32 vbr_drift;
   celt_int32 vbr_offset;
   celt_int32 vbr_count;

   celt_word32 preemph_memE[2];
   celt_word32 preemph_memD[2];

   celt_sig in_mem[1]; /* Size = channels*mode->overlap */
   /* celt_sig overlap_mem[],  Size = channels*mode->overlap */
   /* celt_word16 oldEBands[], Size = channels*mode->nbEBands */
};

int celt_encoder_get_size(const CELTMode *mode, int channels)
{
   int size = sizeof(struct CELTEncoder)
         + (2*channels*mode->overlap-1)*sizeof(celt_sig)
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
   st->force_intra  = 0;
   st->delayedIntra = 1;
   st->tonal_average = QCONST16(1.f,8);
   st->fold_decision = 1;

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
                              int *transient_time, int *transient_shift,
                              celt_word32 *frame_max, int overlap)
{
   int i, n;
   celt_word32 ratio;
   celt_word32 threshold;
   VARDECL(celt_word32, begin);
   SAVE_STACK;
   ALLOC(begin, len+1, celt_word32);
   begin[0] = 0;
   if (C==1)
   {
      for (i=0;i<len;i++)
         begin[i+1] = MAX32(begin[i], ABS32(in[i]));
   } else {
      for (i=0;i<len;i++)
         begin[i+1] = MAX32(begin[i], MAX32(ABS32(in[C*i]),
                                            ABS32(in[C*i+1])));
   }
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
   if (n<32)
   {
      n = -1;
      ratio = 0;
   } else {
      ratio = DIV32(begin[len],1+MAX32(*frame_max, begin[n-16]));
   }

   if (ratio > 45)
      *transient_shift = 3;
   else
      *transient_shift = 0;
   
   *transient_time = n;
   *frame_max = begin[len-overlap];

   RESTORE_STACK;
   return ratio > 0;
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
      VARDECL(celt_word32, x);
      VARDECL(celt_word32, tmp);
      SAVE_STACK;
      if (shortBlocks)
      {
         /*lookup = &mode->mdct[0];*/
         N = mode->shortMdctSize;
         B = shortBlocks;
      }
      ALLOC(x, N+overlap, celt_word32);
      ALLOC(tmp, N, celt_word32);
      for (c=0;c<C;c++)
      {
         for (b=0;b<B;b++)
         {
            int j;
            for (j=0;j<N+overlap;j++)
               x[j] = in[C*(b*N+j)+c];
            clt_mdct_forward(&mode->mdct, x, tmp, mode->window, overlap, shortBlocks ? mode->maxLM : mode->maxLM-LM);
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
      int transient_time, int transient_shift, celt_sig * restrict out_mem[],
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

         if (transient_shift > 0)
         {
#ifdef FIXED_POINT
            for (j=0;j<16;j++)
               x[transient_time+j-16] = MULT16_32_Q15(SHR16(Q15_ONE-transientWindow[j],transient_shift)+transientWindow[j], SHL32(x[transient_time+j-16],transient_shift));
            for (j=transient_time;j<N+overlap;j++)
               x[j] = SHL32(x[j], transient_shift);
#else
            for (j=0;j<16;j++)
               x[transient_time+j-16] *= 1+transientWindow[j]*((1<<transient_shift)-1);
            for (j=transient_time;j<N+overlap;j++)
               x[j] *= 1<<transient_shift;
#endif
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

static void mdct_shape(const CELTMode *mode, celt_norm *X, int start,
                       int end, int N,
                       int mdct_weight_shift, int end_band, int _C, int renorm, int M)
{
   int m, i, c;
   const int C = CHANNELS(_C);
   for (c=0;c<C;c++)
      for (m=start;m<end;m++)
         for (i=m+c*N;i<(c+1)*N;i+=M)
#ifdef FIXED_POINT
            X[i] = SHR16(X[i], mdct_weight_shift);
#else
            X[i] = (1.f/(1<<mdct_weight_shift))*X[i];
#endif
   if (renorm)
      renormalise_bands(mode, X, end_band, C, M);
}

static signed char tf_select_table[4][8] = {
      {0, -1, 0, -1,    0,-1, 0,-1},
      {0, -1, 0, -2,    1, 0, 1 -1},
      {0, -2, 0, -3,    2, 0, 1 -1},
      {0, -2, 0, -3,    2, 0, 1 -1},
};

static int tf_analysis(celt_word16 *bandLogE, celt_word16 *oldBandE, int len, int C, int isTransient, int *tf_res, int nbCompressedBytes)
{
   int i;
   celt_word16 threshold;
   VARDECL(celt_word16, metric);
   celt_word32 average=0;
   celt_word32 cost0;
   celt_word32 cost1;
   VARDECL(int, path0);
   VARDECL(int, path1);
   celt_word16 lambda;
   int tf_select=0;
   SAVE_STACK;

   /* FIXME: Should check number of bytes *left* */
   if (nbCompressedBytes<15*C)
   {
      for (i=0;i<len;i++)
         tf_res[i] = 0;
      return 0;
   }
   if (nbCompressedBytes<40)
      lambda = QCONST16(5.f, DB_SHIFT);
   else if (nbCompressedBytes<60)
      lambda = QCONST16(2.f, DB_SHIFT);
   else if (nbCompressedBytes<100)
      lambda = QCONST16(1.f, DB_SHIFT);
   else
      lambda = QCONST16(.5f, DB_SHIFT);

   ALLOC(metric, len, celt_word16);
   ALLOC(path0, len, int);
   ALLOC(path1, len, int);
   for (i=0;i<len;i++)
   {
      metric[i] = SUB16(bandLogE[i], oldBandE[i]);
      average += metric[i];
   }
   if (C==2)
   {
      average = 0;
      for (i=0;i<len;i++)
      {
         metric[i] = HALF32(metric[i]) + HALF32(SUB16(bandLogE[i+len], oldBandE[i+len]));
         average += metric[i];
      }
   }
   average = DIV32(average, len);
   /*if (!isTransient)
      printf ("%f\n", average);*/
   if (isTransient)
   {
      threshold = QCONST16(1.f,DB_SHIFT);
      tf_select = average > QCONST16(3.f,DB_SHIFT);
   } else {
      threshold = QCONST16(.5f,DB_SHIFT);
      tf_select = average > QCONST16(1.f,DB_SHIFT);
   }
   cost0 = 0;
   cost1 = lambda;
   /* Viterbi forward pass */
   for (i=1;i<len;i++)
   {
      celt_word32 curr0, curr1;
      celt_word32 from0, from1;

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
      cost0 = curr0 + (metric[i]-threshold);
      cost1 = curr1;
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
   if (8*nbCompressedBytes - ec_enc_tell(enc, 0) < 100)
   {
      for (i=start;i<end;i++)
         tf_res[i] = isTransient;
   } else {
      ec_enc_bit_prob(enc, tf_res[start], isTransient ? 16384 : 4096);
      curr = tf_res[start];
      for (i=start+1;i<end;i++)
      {
         ec_enc_bit_prob(enc, tf_res[i] ^ curr, isTransient ? 4096 : 2048);
         curr = tf_res[i];
      }
   }
   ec_enc_bits(enc, tf_select, 1);
   for (i=start;i<end;i++)
      tf_res[i] = tf_select_table[LM][4*isTransient+2*tf_select+tf_res[i]];
}

static void tf_decode(int start, int end, int C, int isTransient, int *tf_res, int nbCompressedBytes, int LM, ec_dec *dec)
{
   int i, curr, tf_select;
   if (8*nbCompressedBytes - ec_dec_tell(dec, 0) < 100)
   {
      for (i=start;i<end;i++)
         tf_res[i] = isTransient;
   } else {
      tf_res[start] = ec_dec_bit_prob(dec, isTransient ? 16384 : 4096);
      curr = tf_res[start];
      for (i=start+1;i<end;i++)
      {
         tf_res[i] = ec_dec_bit_prob(dec, isTransient ? 4096 : 2048) ^ curr;
         curr = tf_res[i];
      }
   }
   tf_select = ec_dec_bits(dec, 1);
   for (i=start;i<end;i++)
      tf_res[i] = tf_select_table[LM][4*isTransient+2*tf_select+tf_res[i]];
}

#ifdef FIXED_POINT
int celt_encode_with_ec(CELTEncoder * restrict st, const celt_int16 * pcm, celt_int16 * optional_resynthesis, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
#else
int celt_encode_with_ec_float(CELTEncoder * restrict st, const celt_sig * pcm, celt_sig * optional_resynthesis, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
#endif
   int i, c, N, NN;
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
   celt_word16 *oldBandE;
   int shortBlocks=0;
   int isTransient=0;
   int transient_time, transient_time_quant;
   int transient_shift;
   int resynth;
   const int C = CHANNELS(st->channels);
   int mdct_weight_shift = 0;
   int mdct_weight_pos=0;
   int LM, M;
   int tf_select;
   int nbFilledBytes, nbAvailableBytes;
   int effEnd;
   SAVE_STACK;

   if (nbCompressedBytes<0 || pcm==NULL)
     return CELT_BAD_ARG;

   for (LM=0;LM<4;LM++)
      if (st->mode->shortMdctSize<<LM==frame_size)
         break;
   if (LM>=MAX_CONFIG_SIZES)
      return CELT_BAD_ARG;
   M=1<<LM;

   _overlap_mem = st->in_mem+C*(st->overlap);
   oldBandE = (celt_word16*)(st->in_mem+2*C*(st->overlap));

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

   CELT_COPY(in, st->in_mem, C*st->overlap);
   for (c=0;c<C;c++)
   {
      const celt_word16 * restrict pcmp = pcm+c;
      celt_sig * restrict inp = in+C*st->overlap+c;
      for (i=0;i<N;i++)
      {
         /* Apply pre-emphasis */
         celt_sig tmp = MULT16_16(st->mode->preemph[2], SCALEIN(*pcmp));
         *inp = tmp + st->preemph_memE[c];
         st->preemph_memE[c] = MULT16_32_Q15(st->mode->preemph[1], *inp)
                             - MULT16_32_Q15(st->mode->preemph[0], tmp);
         inp += C;
         pcmp += C;
      }
   }
   CELT_COPY(st->in_mem, in+C*N, C*st->overlap);

   /* Transient handling */
   transient_time = -1;
   transient_time_quant = -1;
   transient_shift = 0;
   isTransient = 0;

   resynth = optional_resynthesis!=NULL;

   if (M > 1 && transient_analysis(in, N+st->overlap, C, &transient_time, &transient_shift, &st->frame_max, st->overlap))
   {
#ifndef FIXED_POINT
      float gain_1;
#endif
      /* Apply the inverse shaping window */
      if (transient_shift)
      {
         transient_time_quant = transient_time*(celt_int32)8000/st->mode->Fs;
         transient_time = transient_time_quant*(celt_int32)st->mode->Fs/8000;
#ifdef FIXED_POINT
         for (c=0;c<C;c++)
            for (i=0;i<16;i++)
               in[C*(transient_time+i-16)+c] = MULT16_32_Q15(EXTRACT16(SHR32(celt_rcp(Q15ONE+MULT16_16(transientWindow[i],((1<<transient_shift)-1))),1)), in[C*(transient_time+i-16)+c]);
         for (c=0;c<C;c++)
            for (i=transient_time;i<N+st->overlap;i++)
               in[C*i+c] = SHR32(in[C*i+c], transient_shift);
#else
         for (c=0;c<C;c++)
            for (i=0;i<16;i++)
               in[C*(transient_time+i-16)+c] /= 1+transientWindow[i]*((1<<transient_shift)-1);
         gain_1 = 1.f/(1<<transient_shift);
         for (c=0;c<C;c++)
            for (i=transient_time;i<N+st->overlap;i++)
               in[C*i+c] *= gain_1;
#endif
      }
      isTransient = 1;
      has_fold = 1;
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

   NN = M*st->mode->eBands[effEnd];
   if (shortBlocks && !transient_shift)
   {
      celt_word32 sum[8]={1,1,1,1,1,1,1,1};
      int m;
      for (c=0;c<C;c++)
      {
         m=0;
         do {
            celt_word32 tmp=0;
            for (i=m+c*N;i<c*N+NN;i+=M)
               tmp += ABS32(X[i]);
            sum[m++] += tmp;
         } while (m<M);
      }
      m=0;
#ifdef FIXED_POINT
      do {
         if (SHR32(sum[m+1],3) > sum[m])
         {
            mdct_weight_shift=2;
            mdct_weight_pos = m;
         } else if (SHR32(sum[m+1],1) > sum[m] && mdct_weight_shift < 2)
         {
            mdct_weight_shift=1;
            mdct_weight_pos = m;
         }
         m++;
      } while (m<M-1);
#else
      do {
         if (sum[m+1] > 8*sum[m])
         {
            mdct_weight_shift=2;
            mdct_weight_pos = m;
         } else if (sum[m+1] > 2*sum[m] && mdct_weight_shift < 2)
         {
            mdct_weight_shift=1;
            mdct_weight_pos = m;
         }
         m++;
      } while (m<M-1);
#endif
      if (mdct_weight_shift)
         mdct_shape(st->mode, X, mdct_weight_pos+1, M, N, mdct_weight_shift, effEnd, C, 0, M);
   }

   ALLOC(tf_res, st->mode->nbEBands, int);
   /* Needs to be before coarse energy quantization because otherwise the energy gets modified */
   tf_select = tf_analysis(bandLogE, oldBandE, effEnd, C, isTransient, tf_res, nbAvailableBytes);
   for (i=effEnd;i<st->end;i++)
      tf_res[i] = tf_res[effEnd-1];

   ALLOC(error, C*st->mode->nbEBands, celt_word16);
   quant_coarse_energy(st->mode, st->start, st->end, effEnd, bandLogE,
         oldBandE, nbCompressedBytes*8, st->mode->prob,
         error, enc, C, LM, nbAvailableBytes, st->force_intra, &st->delayedIntra);

   ec_enc_bit_prob(enc, shortBlocks!=0, 8192);

   if (shortBlocks)
   {
      if (transient_shift)
      {
         int max_time = (N+st->mode->overlap)*(celt_int32)8000/st->mode->Fs;
         ec_enc_uint(enc, transient_shift, 4);
         ec_enc_uint(enc, transient_time_quant, max_time);
      } else {
         ec_enc_uint(enc, mdct_weight_shift, 4);
         if (mdct_weight_shift && M!=2)
            ec_enc_uint(enc, mdct_weight_pos, M-1);
      }
   }

   tf_encode(st->start, st->end, isTransient, tf_res, nbAvailableBytes, LM, tf_select, enc);

   if (!shortBlocks && !folding_decision(st->mode, X, &st->tonal_average, &st->fold_decision, effEnd, C, M))
      has_fold = 0;
   ec_enc_bit_prob(enc, has_fold>>1, 8192);
   ec_enc_bit_prob(enc, has_fold&1, (has_fold>>1) ? 32768 : 49152);

   /* Variable bitrate */
   if (st->vbr_rate_norm>0)
   {
     celt_word16 alpha;
     celt_int32 delta;
     /* The target rate in 16th bits per frame */
     celt_int32 vbr_rate;
     celt_int32 target;
     celt_int32 vbr_bound, max_allowed;

     vbr_rate = M*st->vbr_rate_norm;

     /* Computes the max bit-rate allowed in VBR more to avoid busting the budget */
     vbr_bound = vbr_rate;
     max_allowed = (vbr_rate + vbr_bound - st->vbr_reservoir)>>(BITRES+3);
     if (max_allowed < 4)
        max_allowed = 4;
     if (max_allowed < nbAvailableBytes)
        nbAvailableBytes = max_allowed;
     target=vbr_rate;

     /* Shortblocks get a large boost in bitrate, but since they 
        are uncommon long blocks are not greatly effected */
     if (shortBlocks)
       target*=2;
     else if (M > 1)
       target-=(target+14)/28;

     /* The average energy is removed from the target and the actual 
        energy added*/
     target=target+st->vbr_offset-588+ec_enc_tell(enc, BITRES);

     /* In VBR mode the frame size must not be reduced so much that it would result in the coarse energy busting its budget */
     target=IMIN(nbAvailableBytes,target);
     /* Make the adaptation coef (alpha) higher at the beginning */
     if (st->vbr_count < 990)
     {
        st->vbr_count++;
        alpha = celt_rcp(SHL32(EXTEND32(st->vbr_count+10),16));
        /*printf ("%d %d\n", st->vbr_count+10, alpha);*/
     } else
        alpha = QCONST16(.001f,15);

     /* By how much did we "miss" the target on that frame */
     delta = (8<<BITRES)*(celt_int32)target - vbr_rate;
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
        int adjust = 1-(st->vbr_reservoir-1)/(8<<BITRES);
        st->vbr_reservoir += adjust*(8<<BITRES);
        target += adjust;
        /*printf ("+%d\n", adjust);*/
     }
     if (target < nbAvailableBytes)
        nbAvailableBytes = target;
     nbCompressedBytes = nbAvailableBytes + nbFilledBytes;

     /* This moves the raw bits to take into account the new compressed size */
     ec_byte_shrink(&buf, nbCompressedBytes);
   }

   /* Bit allocation */
   ALLOC(fine_quant, st->mode->nbEBands, int);
   ALLOC(pulses, st->mode->nbEBands, int);
   ALLOC(offsets, st->mode->nbEBands, int);
   ALLOC(fine_priority, st->mode->nbEBands, int);

   for (i=0;i<st->mode->nbEBands;i++)
      offsets[i] = 0;
   bits = nbCompressedBytes*8 - ec_enc_tell(enc, 0) - 1;
   compute_allocation(st->mode, st->start, st->end, offsets, bits, pulses, fine_quant, fine_priority, C, LM);

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
   quant_all_bands(1, st->mode, st->start, st->end, X, C==2 ? X+N : NULL, bandE, pulses, shortBlocks, has_fold, tf_res, resynth, nbCompressedBytes*8, enc, LM);

   quant_energy_finalise(st->mode, st->start, st->end, bandE, oldBandE, error, fine_quant, fine_priority, nbCompressedBytes*8-ec_enc_tell(enc, 0), enc, C);

   /* Re-synthesis of the coded audio if required */
   if (resynth)
   {
      VARDECL(celt_sig, _out_mem);
      celt_sig *out_mem[2];
      celt_sig *overlap_mem[2];

      log2Amp(st->mode, st->start, st->end, bandE, oldBandE, C);

#ifdef MEASURE_NORM_MSE
      measure_norm_mse(st->mode, X, X0, bandE, bandE0, M, N, C);
#endif

      if (mdct_weight_shift)
      {
         mdct_shape(st->mode, X, 0, mdct_weight_pos+1, N, mdct_weight_shift, effEnd, C, 1, M);
      }

      /* Synthesis */
      denormalise_bands(st->mode, X, freq, bandE, effEnd, C, M);

      for (c=0;c<C;c++)
         for (i=0;i<M*st->mode->eBands[st->start];i++)
            freq[c*N+i] = 0;
      for (c=0;c<C;c++)
         for (i=M*st->mode->eBands[st->end];i<N;i++)
            freq[c*N+i] = 0;

      ALLOC(_out_mem, C*N, celt_sig);

      for (c=0;c<C;c++)
      {
         overlap_mem[c] = _overlap_mem + c*st->overlap;
         out_mem[c] = _out_mem+c*N;
      }

      compute_inv_mdcts(st->mode, shortBlocks, freq, transient_time,
            transient_shift, out_mem, overlap_mem, C, LM);

      /* De-emphasis and put everything back at the right place 
         in the synthesis history */
      if (optional_resynthesis != NULL) {
         deemphasis(out_mem, optional_resynthesis, N, C, st->mode->preemph, st->preemph_memD);

      }
   }

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
int celt_encode_with_ec_float(CELTEncoder * restrict st, const float * pcm, float * optional_resynthesis, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
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

   if (optional_resynthesis != NULL) {
     ret=celt_encode_with_ec(st,in,in,frame_size,compressed,nbCompressedBytes, enc);
      for (j=0;j<C*N;j++)
         optional_resynthesis[j]=in[j]*(1.f/32768.f);
   } else {
     ret=celt_encode_with_ec(st,in,NULL,frame_size,compressed,nbCompressedBytes, enc);
   }
   RESTORE_STACK;
   return ret;

}
#endif /*DISABLE_FLOAT_API*/
#else
int celt_encode_with_ec(CELTEncoder * restrict st, const celt_int16 * pcm, celt_int16 * optional_resynthesis, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
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

   if (optional_resynthesis != NULL) {
      ret = celt_encode_with_ec_float(st,in,in,frame_size,compressed,nbCompressedBytes, enc);
      for (j=0;j<C*N;j++)
         optional_resynthesis[j] = FLOAT2INT16(in[j]);
   } else {
      ret = celt_encode_with_ec_float(st,in,NULL,frame_size,compressed,nbCompressedBytes, enc);
   }
   RESTORE_STACK;
   return ret;
}
#endif

int celt_encode(CELTEncoder * restrict st, const celt_int16 * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes)
{
   return celt_encode_with_ec(st, pcm, NULL, frame_size, compressed, nbCompressedBytes, NULL);
}

#ifndef DISABLE_FLOAT_API
int celt_encode_float(CELTEncoder * restrict st, const float * pcm, int frame_size, unsigned char *compressed, int nbCompressedBytes)
{
   return celt_encode_with_ec_float(st, pcm, NULL, frame_size, compressed, nbCompressedBytes, NULL);
}
#endif /* DISABLE_FLOAT_API */

int celt_encode_resynthesis(CELTEncoder * restrict st, const celt_int16 * pcm, celt_int16 * optional_resynthesis, int frame_size, unsigned char *compressed, int nbCompressedBytes)
{
   return celt_encode_with_ec(st, pcm, optional_resynthesis, frame_size, compressed, nbCompressedBytes, NULL);
}

#ifndef DISABLE_FLOAT_API
int celt_encode_resynthesis_float(CELTEncoder * restrict st, const float * pcm, float * optional_resynthesis, int frame_size, unsigned char *compressed, int nbCompressedBytes)
{
   return celt_encode_with_ec_float(st, pcm, optional_resynthesis, frame_size, compressed, nbCompressedBytes, NULL);
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
   celt_word16 *oldBandE;
   SAVE_STACK;
   
   for (c=0;c<C;c++)
   {
      decode_mem[c] = st->_decode_mem + c*(DECODE_BUFFER_SIZE+st->overlap);
      out_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE-MAX_PERIOD;
      overlap_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE;
   }
   lpc = (celt_word16*)(st->_decode_mem+(DECODE_BUFFER_SIZE+st->overlap)*C);
   oldBandE = lpc+C*LPC_ORDER;

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
         tmp1 = MULT16_32_Q15(fade, tmp1);
         tmp2 = MULT16_32_Q15(fade, tmp2);
         out_mem[c][MAX_PERIOD+i] = MULT16_32_Q15(st->mode->window[overlap-i-1], tmp2);
         out_mem[c][MAX_PERIOD+overlap-i-1] = MULT16_32_Q15(st->mode->window[i], tmp2);
         out_mem[c][MAX_PERIOD-N+i] += MULT16_32_Q15(st->mode->window[i], tmp1);
         out_mem[c][MAX_PERIOD-N+overlap-i-1] -= MULT16_32_Q15(st->mode->window[overlap-i-1], tmp1);
      }
      for (i=0;i<N-overlap;i++)
         out_mem[c][MAX_PERIOD-N+overlap+i] = MULT16_32_Q15(fade, e[overlap+i]);
   }

   deemphasis(out_mem, pcm, N, C, st->mode->preemph, st->preemph_memD);
   
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
   int transient_time;
   int transient_shift;
   int mdct_weight_shift=0;
   const int C = CHANNELS(st->channels);
   int mdct_weight_pos=0;
   int LM, M;
   int nbFilledBytes, nbAvailableBytes;
   int effEnd;
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

   /* Decode the global flags (first symbols in the stream) */
   intra_ener = ec_dec_bit_prob(dec, 8192);
   /* Get band energies */
   unquant_coarse_energy(st->mode, st->start, st->end, bandE, oldBandE,
         intra_ener, st->mode->prob, dec, C, LM);

   isTransient = ec_dec_bit_prob(dec, 8192);

   if (isTransient)
      shortBlocks = M;
   else
      shortBlocks = 0;

   if (isTransient)
   {
      transient_shift = ec_dec_uint(dec, 4);
      if (transient_shift == 3)
      {
         int transient_time_quant;
         int max_time = (N+st->mode->overlap)*(celt_int32)8000/st->mode->Fs;
         transient_time_quant = ec_dec_uint(dec, max_time);
         transient_time = transient_time_quant*(celt_int32)st->mode->Fs/8000;
      } else {
         mdct_weight_shift = transient_shift;
         if (mdct_weight_shift && M>2)
            mdct_weight_pos = ec_dec_uint(dec, M-1);
         transient_shift = 0;
         transient_time = 0;
      }
   } else {
      transient_time = -1;
      transient_shift = 0;
   }

   ALLOC(tf_res, st->mode->nbEBands, int);
   tf_decode(st->start, st->end, C, isTransient, tf_res, nbAvailableBytes, LM, dec);

   has_fold = ec_dec_bit_prob(dec, 8192)<<1;
   has_fold |= ec_dec_bit_prob(dec, (has_fold>>1) ? 32768 : 49152);

   ALLOC(pulses, st->mode->nbEBands, int);
   ALLOC(offsets, st->mode->nbEBands, int);
   ALLOC(fine_priority, st->mode->nbEBands, int);

   for (i=0;i<st->mode->nbEBands;i++)
      offsets[i] = 0;

   bits = len*8 - ec_dec_tell(dec, 0) - 1;
   ALLOC(fine_quant, st->mode->nbEBands, int);
   compute_allocation(st->mode, st->start, st->end, offsets, bits, pulses, fine_quant, fine_priority, C, LM);
   /*bits = ec_dec_tell(dec, 0);
   compute_fine_allocation(st->mode, fine_quant, (20*C+len*8/5-(ec_dec_tell(dec, 0)-bits))/C);*/
   
   unquant_fine_energy(st->mode, st->start, st->end, bandE, oldBandE, fine_quant, dec, C);

   /* Decode fixed codebook */
   quant_all_bands(0, st->mode, st->start, st->end, X, C==2 ? X+N : NULL, NULL, pulses, shortBlocks, has_fold, tf_res, 1, len*8, dec, LM);

   unquant_energy_finalise(st->mode, st->start, st->end, bandE, oldBandE,
         fine_quant, fine_priority, len*8-ec_dec_tell(dec, 0), dec, C);

   log2Amp(st->mode, st->start, st->end, bandE, oldBandE, C);

   if (mdct_weight_shift)
   {
      mdct_shape(st->mode, X, 0, mdct_weight_pos+1, N, mdct_weight_shift, effEnd, C, 1, M);
   }

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
   compute_inv_mdcts(st->mode, shortBlocks, freq, transient_time,
         transient_shift, out_syn, overlap_mem, C, LM);

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

