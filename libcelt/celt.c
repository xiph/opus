/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
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

static const celt_word16 preemph = QCONST16(0.8f,15);

#ifdef FIXED_POINT
static const celt_word16 transientWindow[16] = {
     279,  1106,  2454,  4276,  6510,  9081, 11900, 14872,
   17896, 20868, 23687, 26258, 28492, 30314, 31662, 32489};
#else
static const float transientWindow[16] = {
   0.0085135, 0.0337639, 0.0748914, 0.1304955, 
   0.1986827, 0.2771308, 0.3631685, 0.4538658,
   0.5461342, 0.6368315, 0.7228692, 0.8013173, 
   0.8695045, 0.9251086, 0.9662361, 0.9914865};
#endif

#define ENCODERVALID   0x4c434554
#define ENCODERPARTIAL 0x5445434c
#define ENCODERFREED   0x4c004500
   
/** Encoder state 
 @brief Encoder state
 */
struct CELTEncoder {
   celt_uint32 marker;
   const CELTMode *mode;     /**< Mode used by the encoder */
   int overlap;
   int channels;
   
   int pitch_enabled;       /* Complexity level is allowed to use pitch */
   int pitch_permitted;     /*  Use of the LTP is permitted by the user */
   int pitch_available;     /*  Amount of pitch buffer available */
   int force_intra;
   int delayedIntra;
   celt_word16 tonal_average;
   int fold_decision;
   celt_word16 gain_prod;
   celt_word32 frame_max;
   int start, end;

   /* VBR-related parameters */
   celt_int32 vbr_reservoir;
   celt_int32 vbr_drift;
   celt_int32 vbr_offset;
   celt_int32 vbr_count;

   celt_int32 vbr_rate_norm; /* Target number of 16th bits per frame */
   celt_word16 * restrict preemph_memE; 
   celt_sig    * restrict preemph_memD;

   celt_sig *in_mem;
   celt_sig *out_mem;
   celt_word16 *pitch_buf;
   celt_sig xmem;

   celt_word16 *oldBandE;
};

static int check_encoder(const CELTEncoder *st) 
{
   if (st==NULL)
   {
      celt_warning("NULL passed as an encoder structure");  
      return CELT_INVALID_STATE;
   }
   if (st->marker == ENCODERVALID)
      return CELT_OK;
   if (st->marker == ENCODERFREED)
      celt_warning("Referencing an encoder that has already been freed");
   else
      celt_warning("This is not a valid CELT encoder structure");
   return CELT_INVALID_STATE;
}

CELTEncoder *celt_encoder_create(const CELTMode *mode, int channels, int *error)
{
   int C;
   CELTEncoder *st;

   if (check_mode(mode) != CELT_OK)
   {
      if (error)
         *error = CELT_INVALID_MODE;
      return NULL;
   }

   if (channels < 0 || channels > 2)
   {
      celt_warning("Only mono and stereo supported");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }

   C = channels;
   st = celt_alloc(sizeof(CELTEncoder));
   
   if (st==NULL)
   {
      if (error)
         *error = CELT_ALLOC_FAIL;
      return NULL;
   }
   st->marker = ENCODERPARTIAL;
   st->mode = mode;
   st->overlap = mode->overlap;
   st->channels = channels;

   st->start = 0;
   st->end = st->mode->nbEBands;

   st->vbr_rate_norm = 0;
   st->pitch_enabled = 1;
   st->pitch_permitted = 1;
   st->pitch_available = 1;
   st->force_intra  = 0;
   st->delayedIntra = 1;
   st->tonal_average = QCONST16(1.f,8);
   st->fold_decision = 1;

   st->in_mem = celt_alloc(st->overlap*C*sizeof(celt_sig));
   st->out_mem = celt_alloc((MAX_PERIOD+st->overlap)*C*sizeof(celt_sig));
   st->pitch_buf = celt_alloc(((MAX_PERIOD>>1)+2)*sizeof(celt_word16));

   st->oldBandE = (celt_word16*)celt_alloc(C*mode->nbEBands*sizeof(celt_word16));

   st->preemph_memE = (celt_word16*)celt_alloc(C*sizeof(celt_word16));
   st->preemph_memD = (celt_sig*)celt_alloc(C*sizeof(celt_sig));

   if ((st->in_mem!=NULL) && (st->out_mem!=NULL) && (st->oldBandE!=NULL) 
       && (st->preemph_memE!=NULL) && (st->preemph_memD!=NULL))
   {
      if (error)
         *error = CELT_OK;
      st->marker   = ENCODERVALID;
      return st;
   }
   /* If the setup fails for some reason deallocate it. */
   celt_encoder_destroy(st);  
   if (error)
      *error = CELT_ALLOC_FAIL;
   return NULL;
}

void celt_encoder_destroy(CELTEncoder *st)
{
   if (st == NULL)
   {
      celt_warning("NULL passed to celt_encoder_destroy");
      return;
   }

   if (st->marker == ENCODERFREED)
   {
      celt_warning("Freeing an encoder which has already been freed"); 
      return;
   }

   if (st->marker != ENCODERVALID && st->marker != ENCODERPARTIAL)
   {
      celt_warning("This is not a valid CELT encoder structure");
      return;
   }
   /*Check_mode is non-fatal here because we can still free
    the encoder memory even if the mode is bad, although calling
    the free functions in this order is a violation of the API.*/
   check_mode(st->mode);
   
   celt_free(st->in_mem);
   celt_free(st->out_mem);
   celt_free(st->pitch_buf);
   celt_free(st->oldBandE);
   
   celt_free(st->preemph_memE);
   celt_free(st->preemph_memD);

   st->marker = ENCODERFREED;
   
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

   threshold = MULT16_32_Q15(QCONST16(.2f,15),begin[len]);
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
   if (ratio < 0)
      ratio = 0;
   if (ratio > 1000)
      ratio = 1000;

   if (ratio > 45)
      *transient_shift = 3;
   else
      *transient_shift = 0;
   
   *transient_time = n;
   *frame_max = begin[len-overlap];

   RESTORE_STACK;
   return ratio > 4;
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
static void compute_inv_mdcts(const CELTMode *mode, int shortBlocks, celt_sig *X, int transient_time, int transient_shift, celt_sig * restrict out_mem, int _C, int LM)
{
   int c, N4;
   const int C = CHANNELS(_C);
   const int N = mode->shortMdctSize<<LM;
   const int overlap = OVERLAP(mode);
   N4 = (N-overlap)>>1;
   for (c=0;c<C;c++)
   {
      int j;
      if (transient_shift==0 && C==1 && !shortBlocks) {
         clt_mdct_backward(&mode->mdct, X, out_mem+C*(MAX_PERIOD-N-N4), mode->window, overlap, mode->maxLM-LM);
      } else {
         VARDECL(celt_word32, x);
         VARDECL(celt_word32, tmp);
         int b;
         int N2 = N;
         int B = 1;
         int n4offset=0;
         SAVE_STACK;
         
         ALLOC(x, 2*N, celt_word32);
         ALLOC(tmp, N, celt_word32);

         if (shortBlocks)
         {
            /*lookup = &mode->mdct[0];*/
            N2 = mode->shortMdctSize;
            B = shortBlocks;
            n4offset = N4;
         }
         /* Prevents problems from the imdct doing the overlap-add */
         CELT_MEMSET(x+N4, 0, N2);

         for (b=0;b<B;b++)
         {
            /* De-interleaving the sub-frames */
            for (j=0;j<N2;j++)
               tmp[j] = X[(j*B+b)+c*N2*B];
            clt_mdct_backward(&mode->mdct, tmp, x+n4offset+N2*b, mode->window, overlap, shortBlocks ? mode->maxLM : mode->maxLM-LM);
         }

         if (transient_shift > 0)
         {
#ifdef FIXED_POINT
            for (j=0;j<16;j++)
               x[N4+transient_time+j-16] = MULT16_32_Q15(SHR16(Q15_ONE-transientWindow[j],transient_shift)+transientWindow[j], SHL32(x[N4+transient_time+j-16],transient_shift));
            for (j=transient_time;j<N+overlap;j++)
               x[N4+j] = SHL32(x[N4+j], transient_shift);
#else
            for (j=0;j<16;j++)
               x[N4+transient_time+j-16] *= 1+transientWindow[j]*((1<<transient_shift)-1);
            for (j=transient_time;j<N+overlap;j++)
               x[N4+j] *= 1<<transient_shift;
#endif
         }
         /* The first and last part would need to be set to zero 
            if we actually wanted to use them. */
         for (j=0;j<overlap;j++)
            out_mem[C*(MAX_PERIOD-N)+C*j+c] += x[j+N4];
         for (j=0;j<overlap;j++)
            out_mem[C*(MAX_PERIOD)+C*(overlap-j-1)+c] = x[2*N-j-N4-1];
         for (j=0;j<2*N4;j++)
            out_mem[C*(MAX_PERIOD-N)+C*(j+overlap)+c] = x[j+N4+overlap];
         RESTORE_STACK;
      }
   }
}

#define FLAG_NONE        0
#define FLAG_INTRA       (1U<<13)
#define FLAG_PITCH       (1U<<12)
#define FLAG_SHORT       (1U<<11)
#define FLAG_FOLD        (1U<<10)
#define FLAG_MASK        (FLAG_INTRA|FLAG_PITCH|FLAG_SHORT|FLAG_FOLD)

static const int flaglist[8] = {
      0 /*00  */ | FLAG_FOLD,
      1 /*01  */ | FLAG_PITCH|FLAG_FOLD,
      8 /*1000*/ | FLAG_NONE,
      9 /*1001*/ | FLAG_SHORT|FLAG_FOLD,
     10 /*1010*/ | FLAG_PITCH,
     11 /*1011*/ | FLAG_INTRA,
      6 /*110 */ | FLAG_INTRA|FLAG_FOLD,
      7 /*111 */ | FLAG_INTRA|FLAG_SHORT|FLAG_FOLD
};

static void encode_flags(ec_enc *enc, int intra_ener, int has_pitch, int shortBlocks, int has_fold)
{
   int i;
   int flags=FLAG_NONE;
   int flag_bits;
   flags |= intra_ener   ? FLAG_INTRA : 0;
   flags |= has_pitch    ? FLAG_PITCH : 0;
   flags |= shortBlocks  ? FLAG_SHORT : 0;
   flags |= has_fold     ? FLAG_FOLD  : 0;
   for (i=0;i<8;i++)
   {
      if (flags == (flaglist[i]&FLAG_MASK))
      {
         flag_bits = flaglist[i]&0xf;
         break;
      }
   }
   celt_assert(i<8);
   /*printf ("enc %d: %d %d %d %d\n", flag_bits, intra_ener, has_pitch, shortBlocks, has_fold);*/
   if (i<2)
      ec_enc_uint(enc, flag_bits, 4);
   else if (i<6)
   {
      ec_enc_uint(enc, flag_bits>>2, 4);
      ec_enc_uint(enc, flag_bits&0x3, 4);
   } else {
      ec_enc_uint(enc, flag_bits>>1, 4);
      ec_enc_uint(enc, flag_bits&0x1, 2);
   }
}

static void decode_flags(ec_dec *dec, int *intra_ener, int *has_pitch, int *shortBlocks, int *has_fold)
{
   int i;
   int flag_bits;
   flag_bits = ec_dec_uint(dec, 4);
   /*printf ("(%d) ", flag_bits);*/
   if (flag_bits==2)
      flag_bits = (flag_bits<<2) | ec_dec_uint(dec, 4);
   else if (flag_bits==3)
      flag_bits = (flag_bits<<1) | ec_dec_uint(dec, 2);
   for (i=0;i<8;i++)
      if (flag_bits == (flaglist[i]&0xf))
         break;
   celt_assert(i<8);
   *intra_ener  = (flaglist[i]&FLAG_INTRA) != 0;
   *has_pitch   = (flaglist[i]&FLAG_PITCH) != 0;
   *shortBlocks = (flaglist[i]&FLAG_SHORT) != 0;
   *has_fold    = (flaglist[i]&FLAG_FOLD ) != 0;
   /*printf ("dec %d: %d %d %d %d\n", flag_bits, *intra_ener, *has_pitch, *shortBlocks, *has_fold);*/
}

void deemphasis(celt_sig *in, celt_word16 *pcm, int N, int _C, celt_word16 coef, celt_sig *mem)
{
   const int C = CHANNELS(_C);
   int c;
   for (c=0;c<C;c++)
   {
      int j;
      celt_sig * restrict x;
      celt_word16  * restrict y;
      celt_sig m = mem[c];
      x = &in[C*(MAX_PERIOD-N)+c];
      y = pcm+c;
      for (j=0;j<N;j++)
      {
         celt_sig tmp = MAC16_32_Q15(*x, coef,m);
         m = tmp;
         *y = SCALEOUT(SIG2WORD16(tmp));
         x+=C;
         y+=C;
      }
      mem[c] = m;
   }
}

static void mdct_shape(const CELTMode *mode, celt_norm *X, int start,
                       int end, int N,
                       int mdct_weight_shift, int _C, int renorm, int M)
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
      renormalise_bands(mode, X, C, M);
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

static void tf_encode(int len, int isTransient, int *tf_res, int nbCompressedBytes, int LM, int tf_select, ec_enc *enc)
{
   int curr, i;
   if (8*nbCompressedBytes - ec_enc_tell(enc, 0) < 100)
   {
      for (i=0;i<len;i++)
         tf_res[i] = isTransient;
   } else {
      ec_enc_bit_prob(enc, tf_res[0], isTransient ? 16384 : 4096);
      curr = tf_res[0];
      for (i=1;i<len;i++)
      {
         ec_enc_bit_prob(enc, tf_res[i] ^ curr, isTransient ? 4096 : 2048);
         curr = tf_res[i];
      }
   }
   ec_enc_bits(enc, tf_select, 1);
   for (i=0;i<len;i++)
      tf_res[i] = tf_select_table[LM][4*isTransient+2*tf_select+tf_res[i]];
}

static void tf_decode(int len, int C, int isTransient, int *tf_res, int nbCompressedBytes, int LM, ec_dec *dec)
{
   int i, curr, tf_select;
   if (8*nbCompressedBytes - ec_dec_tell(dec, 0) < 100)
   {
      for (i=0;i<len;i++)
         tf_res[i] = isTransient;
   } else {
      tf_res[0] = ec_dec_bit_prob(dec, isTransient ? 16384 : 4096);
      curr = tf_res[0];
      for (i=1;i<len;i++)
      {
         tf_res[i] = ec_dec_bit_prob(dec, isTransient ? 4096 : 2048) ^ curr;
         curr = tf_res[i];
      }
   }
   tf_select = ec_dec_bits(dec, 1);
   for (i=0;i<len;i++)
      tf_res[i] = tf_select_table[LM][4*isTransient+2*tf_select+tf_res[i]];
}

#ifdef FIXED_POINT
int celt_encode_with_ec(CELTEncoder * restrict st, const celt_int16 * pcm, celt_int16 * optional_resynthesis, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
#else
int celt_encode_with_ec_float(CELTEncoder * restrict st, const celt_sig * pcm, celt_sig * optional_resynthesis, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
#endif
   int i, c, N, NN, N4;
   int has_pitch;
   int pitch_index;
   int bits;
   int has_fold=1;
   int coarse_needed;
   ec_byte_buffer buf;
   ec_enc         _enc;
   VARDECL(celt_sig, in);
   VARDECL(celt_sig, freq);
   VARDECL(celt_sig, pitch_freq);
   VARDECL(celt_norm, X);
   VARDECL(celt_ener, bandE);
   VARDECL(celt_word16, bandLogE);
   VARDECL(int, fine_quant);
   VARDECL(celt_word16, error);
   VARDECL(int, pulses);
   VARDECL(int, offsets);
   VARDECL(int, fine_priority);
   VARDECL(int, tf_res);
   int intra_ener = 0;
   int shortBlocks=0;
   int isTransient=0;
   int transient_time;
   int transient_shift;
   int resynth;
   const int C = CHANNELS(st->channels);
   int mdct_weight_shift = 0;
   int mdct_weight_pos=0;
   int gain_id=0;
   int norm_rate;
   int LM, M;
   int tf_select;
   celt_int32 vbr_rate=0;
   celt_word16 max_decay;
   int nbFilledBytes, nbAvailableBytes;
   SAVE_STACK;

   if (check_encoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (nbCompressedBytes<0 || pcm==NULL)
     return CELT_BAD_ARG;

   for (LM=0;LM<4;LM++)
      if (st->mode->shortMdctSize<<LM==frame_size)
         break;
   if (LM>=MAX_CONFIG_SIZES)
      return CELT_BAD_ARG;
   M=1<<LM;

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

   N = M*st->mode->shortMdctSize;
   N4 = (N-st->overlap)>>1;
   ALLOC(in, 2*C*N-2*C*N4, celt_sig);

   CELT_COPY(in, st->in_mem, C*st->overlap);
   for (c=0;c<C;c++)
   {
      const celt_word16 * restrict pcmp = pcm+c;
      celt_sig * restrict inp = in+C*st->overlap+c;
      for (i=0;i<N;i++)
      {
         /* Apply pre-emphasis */
         celt_sig tmp = SCALEIN(SHL32(EXTEND32(*pcmp), SIG_SHIFT));
         *inp = SUB32(tmp, SHR32(MULT16_16(preemph,st->preemph_memE[c]),3));
         st->preemph_memE[c] = SCALEIN(*pcmp);
         inp += C;
         pcmp += C;
      }
   }
   CELT_COPY(st->in_mem, in+C*(2*N-2*N4-st->overlap), C*st->overlap);

   /* Transient handling */
   transient_time = -1;
   transient_shift = 0;
   isTransient = 0;

   resynth = st->pitch_available>0 || optional_resynthesis!=NULL;

   if (M > 1 && transient_analysis(in, N+st->overlap, C, &transient_time, &transient_shift, &st->frame_max, st->overlap))
   {
#ifndef FIXED_POINT
      float gain_1;
#endif
      /* Apply the inverse shaping window */
      if (transient_shift)
      {
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
         gain_1 = 1./(1<<transient_shift);
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


   norm_rate = (nbAvailableBytes-5)*8*(celt_uint32)st->mode->Fs/(C*N)>>10;
   /* Pitch analysis: we do it early to save on the peak stack space */
   /* Don't use pitch if there isn't enough data available yet, 
      or if we're using shortBlocks */
   has_pitch = st->pitch_enabled && st->pitch_permitted && (N <= 512) 
            && (st->pitch_available >= MAX_PERIOD) && (!shortBlocks)
            && norm_rate < 50;
   if (has_pitch)
   {
      VARDECL(celt_word16, x_lp);
      SAVE_STACK;
      ALLOC(x_lp, (2*N-2*N4)>>1, celt_word16);
      pitch_downsample(in, x_lp, 2*N-2*N4, N, C, &st->xmem, &st->pitch_buf[MAX_PERIOD>>1]);
      pitch_search(st->mode, x_lp, st->pitch_buf, 2*N-2*N4, MAX_PERIOD-(2*N-2*N4), &pitch_index, &st->xmem, M);
      RESTORE_STACK;
   }

   /* Deferred allocation after find_spectral_pitch() to reduce 
      the peak memory usage */
   ALLOC(X, C*N, celt_norm);         /**< Interleaved normalised MDCTs */

   ALLOC(pitch_freq, C*N, celt_sig); /**< Interleaved signal MDCTs */
   if (has_pitch)
   {
      compute_mdcts(st->mode, 0, st->out_mem+pitch_index*C, pitch_freq, C, LM);
      has_pitch = compute_pitch_gain(st->mode, freq, pitch_freq, norm_rate, &gain_id, C, &st->gain_prod, M);
   }
   
   if (has_pitch)
      apply_pitch(st->mode, freq, pitch_freq, gain_id, 1, C, M);

   compute_band_energies(st->mode, freq, bandE, C, M);
   for (i=0;i<st->mode->nbEBands*C;i++)
      bandLogE[i] = amp2Log(bandE[i]);

   /* Band normalisation */
   normalise_bands(st->mode, freq, X, bandE, C, M);
   if (!shortBlocks && !folding_decision(st->mode, X, &st->tonal_average, &st->fold_decision, C, M))
      has_fold = 0;

   /* Don't use intra energy when we're operating at low bit-rate */
   intra_ener = st->force_intra || (!has_pitch && st->delayedIntra && nbAvailableBytes > st->mode->nbEBands);
   if (shortBlocks || intra_decision(bandLogE, st->oldBandE, st->mode->nbEBands))
      st->delayedIntra = 1;
   else
      st->delayedIntra = 0;

   NN = M*st->mode->eBands[st->mode->nbEBands];
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
         mdct_shape(st->mode, X, mdct_weight_pos+1, M, N, mdct_weight_shift, C, 0, M);
   }


   encode_flags(enc, intra_ener, has_pitch, shortBlocks, has_fold);
   if (has_pitch)
   {
      ec_enc_uint(enc, pitch_index, MAX_PERIOD-(2*N-2*N4));
      ec_enc_uint(enc, gain_id, 16);
   }
   if (shortBlocks)
   {
      if (transient_shift)
      {
         ec_enc_uint(enc, transient_shift, 4);
         ec_enc_uint(enc, transient_time, N+st->overlap);
      } else {
         ec_enc_uint(enc, mdct_weight_shift, 4);
         if (mdct_weight_shift && M!=2)
            ec_enc_uint(enc, mdct_weight_pos, M-1);
      }
   }

   ALLOC(fine_quant, st->mode->nbEBands, int);
   ALLOC(pulses, st->mode->nbEBands, int);

   vbr_rate = M*st->vbr_rate_norm;
   /* Computes the max bit-rate allowed in VBR more to avoid busting the budget */
   if (st->vbr_rate_norm>0)
   {
      celt_int32 vbr_bound, max_allowed;

      vbr_bound = vbr_rate;
      max_allowed = (vbr_rate + vbr_bound - st->vbr_reservoir)>>(BITRES+3);
      if (max_allowed < 4)
         max_allowed = 4;
      if (max_allowed < nbAvailableBytes)
         nbAvailableBytes = max_allowed;
   }

   ALLOC(tf_res, st->mode->nbEBands, int);
   tf_select = tf_analysis(bandLogE, st->oldBandE, st->mode->nbEBands, C, isTransient, tf_res, nbAvailableBytes);

   /* Bit allocation */
   ALLOC(error, C*st->mode->nbEBands, celt_word16);

#ifdef FIXED_POINT
      max_decay = MIN32(QCONST16(16,DB_SHIFT), SHL32(EXTEND32(nbAvailableBytes),DB_SHIFT-3));
#else
   max_decay = .125*nbAvailableBytes;
#endif
   coarse_needed = quant_coarse_energy(st->mode, st->start, bandLogE, st->oldBandE, nbFilledBytes*8+nbAvailableBytes*4-8, intra_ener, st->mode->prob, error, enc, C, max_decay);
   coarse_needed -= nbFilledBytes*8;
   coarse_needed = ((coarse_needed*3-1)>>3)+1;
   if (coarse_needed > nbAvailableBytes)
      coarse_needed = nbAvailableBytes;
   /* Variable bitrate */
   if (vbr_rate>0)
   {
     celt_word16 alpha;
     celt_int32 delta;
     /* The target rate in 16th bits per frame */
     celt_int32 target=vbr_rate;
   
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
     target=IMAX(coarse_needed,(target+64)/128);
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
     st->vbr_drift += MULT16_32_Q15(alpha,delta-st->vbr_offset-st->vbr_drift);
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

   tf_encode(st->mode->nbEBands, isTransient, tf_res, nbAvailableBytes, LM, tf_select, enc);

   ALLOC(offsets, st->mode->nbEBands, int);
   ALLOC(fine_priority, st->mode->nbEBands, int);

   for (i=0;i<st->mode->nbEBands;i++)
      offsets[i] = 0;
   bits = nbCompressedBytes*8 - ec_enc_tell(enc, 0) - 1;
   compute_allocation(st->mode, st->start, offsets, bits, pulses, fine_quant, fine_priority, C, M);

   quant_fine_energy(st->mode, st->start, bandE, st->oldBandE, error, fine_quant, enc, C);

   /* Residual quantisation */
   quant_all_bands(1, st->mode, st->start, X, C==2 ? X+N : NULL, bandE, pulses, shortBlocks, has_fold, tf_res, resynth, nbCompressedBytes*8, enc, LM);

   quant_energy_finalise(st->mode, st->start, bandE, st->oldBandE, error, fine_quant, fine_priority, nbCompressedBytes*8-ec_enc_tell(enc, 0), enc, C);

   /* Re-synthesis of the coded audio if required */
   if (resynth)
   {
      if (st->pitch_available>0 && st->pitch_available<MAX_PERIOD)
        st->pitch_available+=N;

      if (mdct_weight_shift)
      {
         mdct_shape(st->mode, X, 0, mdct_weight_pos+1, N, mdct_weight_shift, C, 1, M);
      }

      /* Synthesis */
      denormalise_bands(st->mode, X, freq, bandE, C, M);

      CELT_MOVE(st->out_mem, st->out_mem+C*N, C*(MAX_PERIOD+st->overlap-N));

      if (has_pitch)
         apply_pitch(st->mode, freq, pitch_freq, gain_id, 0, C, M);
      
      compute_inv_mdcts(st->mode, shortBlocks, freq, transient_time, transient_shift, st->out_mem, C, LM);

      /* De-emphasis and put everything back at the right place 
         in the synthesis history */
      if (optional_resynthesis != NULL) {
         deemphasis(st->out_mem, optional_resynthesis, N, C, preemph, st->preemph_memD);

      }
   }

   /* If there's any room left (can only happen for very high rates),
      fill it with zeros */
   while (nbCompressedBytes*8 - ec_enc_tell(enc,0) >= 8)
      ec_enc_bits(enc, 0, 8);
   ec_enc_done(enc);
   
   RESTORE_STACK;
   return nbCompressedBytes;
}

#ifdef FIXED_POINT
#ifndef DISABLE_FLOAT_API
int celt_encode_with_ec_float(CELTEncoder * restrict st, const float * pcm, float * optional_resynthesis, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc)
{
   int j, ret, C, N, LM, M;
   VARDECL(celt_int16, in);
   SAVE_STACK;

   if (check_encoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

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
         optional_resynthesis[j]=in[j]*(1/32768.);
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

   if (check_encoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

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
   
   if (check_encoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   va_start(ap, request);
   if ((request!=CELT_GET_MODE_REQUEST) && (check_mode(st->mode) != CELT_OK))
     goto bad_mode;
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
         if (value<=2) {
            st->pitch_enabled = 0; 
            st->pitch_available = 0;
         } else {
              st->pitch_enabled = 1;
              if (st->pitch_available<1)
                st->pitch_available = 1;
         }   
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
      case CELT_SET_PREDICTION_REQUEST:
      {
         int value = va_arg(ap, celt_int32);
         if (value<0 || value>2)
            goto bad_arg;
         if (value==0)
         {
            st->force_intra   = 1;
            st->pitch_permitted = 0;
         } else if (value==1) {
            st->force_intra   = 0;
            st->pitch_permitted = 0;
         } else {
            st->force_intra   = 0;
            st->pitch_permitted = 1;
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
         const CELTMode *mode = st->mode;
         int C = st->channels;

         if (st->pitch_available > 0) st->pitch_available = 1;

         CELT_MEMSET(st->in_mem, 0, st->overlap*C);
         CELT_MEMSET(st->out_mem, 0, (MAX_PERIOD+st->overlap)*C);

         CELT_MEMSET(st->oldBandE, 0, C*mode->nbEBands);

         CELT_MEMSET(st->preemph_memE, 0, C);
         CELT_MEMSET(st->preemph_memD, 0, C);
         st->delayedIntra = 1;

         st->fold_decision = 1;
         st->tonal_average = QCONST16(1.f,8);
         st->gain_prod = 0;
         st->vbr_reservoir = 0;
         st->vbr_drift = 0;
         st->vbr_offset = 0;
         st->vbr_count = 0;
         st->xmem = 0;
         st->frame_max = 0;
         CELT_MEMSET(st->pitch_buf, 0, (MAX_PERIOD>>1)+2);
      }
      break;
      default:
         goto bad_request;
   }
   va_end(ap);
   return CELT_OK;
bad_mode:
  va_end(ap);
  return CELT_INVALID_MODE;
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

#define DECODERVALID   0x4c434454
#define DECODERPARTIAL 0x5444434c
#define DECODERFREED   0x4c004400

/** Decoder state 
 @brief Decoder state
 */
struct CELTDecoder {
   celt_uint32 marker;
   const CELTMode *mode;
   int overlap;
   int channels;

   int start, end;
   ec_byte_buffer buf;
   ec_enc         enc;

   celt_sig * restrict preemph_memD;

   celt_sig *out_mem;
   celt_sig *decode_mem;

   celt_word16 *oldBandE;
   
   celt_word16 *lpc;

   int last_pitch_index;
   int loss_count;
};

int check_decoder(const CELTDecoder *st) 
{
   if (st==NULL)
   {
      celt_warning("NULL passed a decoder structure");  
      return CELT_INVALID_STATE;
   }
   if (st->marker == DECODERVALID)
      return CELT_OK;
   if (st->marker == DECODERFREED)
      celt_warning("Referencing a decoder that has already been freed");
   else
      celt_warning("This is not a valid CELT decoder structure");
   return CELT_INVALID_STATE;
}

CELTDecoder *celt_decoder_create(const CELTMode *mode, int channels, int *error)
{
   int C;
   CELTDecoder *st;

   if (check_mode(mode) != CELT_OK)
   {
      if (error)
         *error = CELT_INVALID_MODE;
      return NULL;
   }

   if (channels < 0 || channels > 2)
   {
      celt_warning("Only mono and stereo supported");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }

   C = CHANNELS(channels);
   st = celt_alloc(sizeof(CELTDecoder));

   if (st==NULL)
   {
      if (error)
         *error = CELT_ALLOC_FAIL;
      return NULL;
   }

   st->marker = DECODERPARTIAL;
   st->mode = mode;
   st->overlap = mode->overlap;
   st->channels = channels;

   st->start = 0;
   st->end = st->mode->nbEBands;

   st->decode_mem = celt_alloc((DECODE_BUFFER_SIZE+st->overlap)*C*sizeof(celt_sig));
   st->out_mem = st->decode_mem+DECODE_BUFFER_SIZE-MAX_PERIOD;
   
   st->oldBandE = (celt_word16*)celt_alloc(C*mode->nbEBands*sizeof(celt_word16));
   
   st->preemph_memD = (celt_sig*)celt_alloc(C*sizeof(celt_sig));

   st->lpc = (celt_word16*)celt_alloc(C*LPC_ORDER*sizeof(celt_word16));

   st->loss_count = 0;

   if ((st->decode_mem!=NULL) && (st->out_mem!=NULL) && (st->oldBandE!=NULL) &&
         (st->lpc!=NULL) &&
       (st->preemph_memD!=NULL))
   {
      if (error)
         *error = CELT_OK;
      st->marker = DECODERVALID;
      return st;
   }
   /* If the setup fails for some reason deallocate it. */
   celt_decoder_destroy(st);
   if (error)
      *error = CELT_ALLOC_FAIL;
   return NULL;
}

void celt_decoder_destroy(CELTDecoder *st)
{
   if (st == NULL)
   {
      celt_warning("NULL passed to celt_decoder_destroy");
      return;
   }

   if (st->marker == DECODERFREED) 
   {
      celt_warning("Freeing a decoder which has already been freed"); 
      return;
   }
   
   if (st->marker != DECODERVALID && st->marker != DECODERPARTIAL)
   {
      celt_warning("This is not a valid CELT decoder structure");
      return;
   }
   
   /*Check_mode is non-fatal here because we can still free
     the encoder memory even if the mode is bad, although calling
     the free functions in this order is a violation of the API.*/
   check_mode(st->mode);
   
   celt_free(st->decode_mem);
   celt_free(st->oldBandE);
   celt_free(st->preemph_memD);
   celt_free(st->lpc);
   
   st->marker = DECODERFREED;
   
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
   SAVE_STACK;
   
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
      pitch_downsample(st->out_mem, pitch_buf, MAX_PERIOD, MAX_PERIOD,
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
         exc[i] = ROUND16(st->out_mem[i*C+c], SIG_SHIFT);

      if (st->loss_count == 0)
      {
         _celt_autocorr(exc, ac, st->mode->window, st->mode->overlap,
                        LPC_ORDER, MAX_PERIOD);

         /* Noise floor -40 dB */
#ifdef FIXED_POINT
         ac[0] += SHR32(ac[0],13);
#else
         ac[0] *= 1.0001;
#endif
         /* Lag windowing */
         for (i=1;i<=LPC_ORDER;i++)
         {
            /*ac[i] *= exp(-.5*(2*M_PI*.002*i)*(2*M_PI*.002*i));*/
#ifdef FIXED_POINT
            ac[i] -= MULT16_32_Q15(2*i*i, ac[i]);
#else
            ac[i] -= ac[i]*(.008*i)*(.008*i);
#endif
         }

         _celt_lpc(st->lpc+c*LPC_ORDER, ac, LPC_ORDER);
      }
      fir(exc, st->lpc+c*LPC_ORDER, exc, MAX_PERIOD, LPC_ORDER, mem);
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
         S1 += SHR32(MULT16_16(st->out_mem[offset+i],st->out_mem[offset+i]),8);
      }

      iir(e, st->lpc+c*LPC_ORDER, e, len+st->mode->overlap, LPC_ORDER, mem);

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
            celt_word16 ratio = celt_sqrt(frac_div32(SHR32(S1,1)+1,S2+1.));
            for (i=0;i<len+overlap;i++)
               e[i] = MULT16_16_Q15(ratio, e[i]);
         }
      }

      for (i=0;i<MAX_PERIOD+st->mode->overlap-N;i++)
         st->out_mem[C*i+c] = st->out_mem[C*(N+i)+c];

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
         st->out_mem[C*(MAX_PERIOD+i)+c] = MULT16_32_Q15(st->mode->window[overlap-i-1], tmp2);
         st->out_mem[C*(MAX_PERIOD+overlap-i-1)+c] = MULT16_32_Q15(st->mode->window[i], tmp2);
         st->out_mem[C*(MAX_PERIOD-N+i)+c] += MULT16_32_Q15(st->mode->window[i], tmp1);
         st->out_mem[C*(MAX_PERIOD-N+overlap-i-1)+c] -= MULT16_32_Q15(st->mode->window[overlap-i-1], tmp1);
      }
      for (i=0;i<N-overlap;i++)
         st->out_mem[C*(MAX_PERIOD-N+overlap+i)+c] = MULT16_32_Q15(fade, e[overlap+i]);
   }

   deemphasis(st->out_mem, pcm, N, C, preemph, st->preemph_memD);
   
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
   int c, i, N, N4;
   int has_pitch, has_fold;
   int pitch_index;
   int bits;
   ec_dec _dec;
   ec_byte_buffer buf;
   VARDECL(celt_sig, freq);
   VARDECL(celt_sig, pitch_freq);
   VARDECL(celt_norm, X);
   VARDECL(celt_ener, bandE);
   VARDECL(int, fine_quant);
   VARDECL(int, pulses);
   VARDECL(int, offsets);
   VARDECL(int, fine_priority);
   VARDECL(int, tf_res);

   int shortBlocks;
   int isTransient;
   int intra_ener;
   int transient_time;
   int transient_shift;
   int mdct_weight_shift=0;
   const int C = CHANNELS(st->channels);
   int mdct_weight_pos=0;
   int gain_id=0;
   int LM, M;
   int nbFilledBytes, nbAvailableBytes;
   SAVE_STACK;

   if (check_decoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   for (LM=0;LM<4;LM++)
      if (st->mode->shortMdctSize<<LM==frame_size)
         break;
   if (LM>=MAX_CONFIG_SIZES)
      return CELT_BAD_ARG;
   M=1<<LM;

   N = M*st->mode->shortMdctSize;
   N4 = (N-st->overlap)>>1;

   ALLOC(freq, C*N, celt_sig); /**< Interleaved signal MDCTs */
   ALLOC(X, C*N, celt_norm);   /**< Interleaved normalised MDCTs */
   ALLOC(bandE, st->mode->nbEBands*C, celt_ener);
   for (c=0;c<C;c++)
      for (i=0;i<M*st->mode->eBands[st->start];i++)
         X[c*N+i] = 0;

   if (data == NULL)
   {
      celt_decode_lost(st, pcm, N, LM);
      RESTORE_STACK;
      return 0;
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

   decode_flags(dec, &intra_ener, &has_pitch, &isTransient, &has_fold);
   if (isTransient)
      shortBlocks = M;
   else
      shortBlocks = 0;

   if (isTransient)
   {
      transient_shift = ec_dec_uint(dec, 4);
      if (transient_shift == 3)
      {
         transient_time = ec_dec_uint(dec, N+st->mode->overlap);
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
   
   if (has_pitch)
   {
      int maxpitch = MAX_PERIOD-(2*N-2*N4);
      if (maxpitch<0)
      {
         celt_notify("detected pitch when not allowed, bit corruption suspected");
         pitch_index = 0;
         has_pitch = 0;
      } else {
         pitch_index = ec_dec_uint(dec, maxpitch);
         gain_id = ec_dec_uint(dec, 16);
      }
   } else {
      pitch_index = 0;
   }


   ALLOC(fine_quant, st->mode->nbEBands, int);
   /* Get band energies */
   unquant_coarse_energy(st->mode, st->start, bandE, st->oldBandE, nbFilledBytes*8+nbAvailableBytes*4-8, intra_ener, st->mode->prob, dec, C);

   ALLOC(tf_res, st->mode->nbEBands, int);
   tf_decode(st->mode->nbEBands, C, isTransient, tf_res, nbAvailableBytes, LM, dec);

   ALLOC(pulses, st->mode->nbEBands, int);
   ALLOC(offsets, st->mode->nbEBands, int);
   ALLOC(fine_priority, st->mode->nbEBands, int);

   for (i=0;i<st->mode->nbEBands;i++)
      offsets[i] = 0;

   bits = len*8 - ec_dec_tell(dec, 0) - 1;
   compute_allocation(st->mode, st->start, offsets, bits, pulses, fine_quant, fine_priority, C, M);
   /*bits = ec_dec_tell(dec, 0);
   compute_fine_allocation(st->mode, fine_quant, (20*C+len*8/5-(ec_dec_tell(dec, 0)-bits))/C);*/
   
   unquant_fine_energy(st->mode, st->start, bandE, st->oldBandE, fine_quant, dec, C);

   ALLOC(pitch_freq, C*N, celt_sig); /**< Interleaved signal MDCTs */
   if (has_pitch) 
   {
      /* Pitch MDCT */
      compute_mdcts(st->mode, 0, st->out_mem+pitch_index*C, pitch_freq, C, LM);
   }

   /* Decode fixed codebook and merge with pitch */
   quant_all_bands(0, st->mode, st->start, X, C==2 ? X+N : NULL, NULL, pulses, shortBlocks, has_fold, tf_res, 1, len*8, dec, LM);

   unquant_energy_finalise(st->mode, st->start, bandE, st->oldBandE, fine_quant, fine_priority, len*8-ec_dec_tell(dec, 0), dec, C);

   if (mdct_weight_shift)
   {
      mdct_shape(st->mode, X, 0, mdct_weight_pos+1, N, mdct_weight_shift, C, 1, M);
   }

   /* Synthesis */
   denormalise_bands(st->mode, X, freq, bandE, C, M);


   CELT_MOVE(st->decode_mem, st->decode_mem+C*N, C*(DECODE_BUFFER_SIZE+st->overlap-N));

   if (has_pitch)
      apply_pitch(st->mode, freq, pitch_freq, gain_id, 0, C, M);

   for (c=0;c<C;c++)
      for (i=0;i<M*st->mode->eBands[st->start];i++)
         freq[c*N+i] = 0;

   /* Compute inverse MDCTs */
   compute_inv_mdcts(st->mode, shortBlocks, freq, transient_time, transient_shift, st->out_mem, C, LM);

   deemphasis(st->out_mem, pcm, N, C, preemph, st->preemph_memD);
   st->loss_count = 0;
   RESTORE_STACK;
   return 0;
}

#ifdef FIXED_POINT
#ifndef DISABLE_FLOAT_API
int celt_decode_with_ec_float(CELTDecoder * restrict st, const unsigned char *data, int len, float * restrict pcm, int frame_size, ec_dec *dec)
{
   int j, ret, C, N, LM, M;
   VARDECL(celt_int16, out);
   SAVE_STACK;

   if (check_decoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

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
   for (j=0;j<C*N;j++)
      pcm[j]=out[j]*(1/32768.);
     
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

   if (check_decoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

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

   if (check_decoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   va_start(ap, request);
   if ((request!=CELT_GET_MODE_REQUEST) && (check_mode(st->mode) != CELT_OK))
     goto bad_mode;
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
      case CELT_RESET_STATE:
      {
         const CELTMode *mode = st->mode;
         int C = st->channels;

         CELT_MEMSET(st->decode_mem, 0, (DECODE_BUFFER_SIZE+st->overlap)*C);
         CELT_MEMSET(st->oldBandE, 0, C*mode->nbEBands);

         CELT_MEMSET(st->preemph_memD, 0, C);

         st->loss_count = 0;

         CELT_MEMSET(st->lpc, 0, C*LPC_ORDER);
      }
      break;
      default:
         goto bad_request;
   }
   va_end(ap);
   return CELT_OK;
bad_mode:
  va_end(ap);
  return CELT_INVALID_MODE;
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

