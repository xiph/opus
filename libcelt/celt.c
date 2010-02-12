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

#define LPC_ORDER 24
/* #define NEW_PLC */
#if !defined(FIXED_POINT) || defined(NEW_PLC)
#include "plc.c"
#endif

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
   int frame_size;
   int block_size;
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

   /* VBR-related parameters */
   celt_int32 vbr_reservoir;
   celt_int32 vbr_drift;
   celt_int32 vbr_offset;
   celt_int32 vbr_count;

   celt_int32 vbr_rate; /* Target number of 16th bits per frame */
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
   int N, C;
   CELTEncoder *st;

   if (check_mode(mode) != CELT_OK)
   {
      if (error)
         *error = CELT_INVALID_MODE;
      return NULL;
   }
#ifdef DISABLE_STEREO
   if (channels > 1)
   {
      celt_warning("Stereo support was disable from this build");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }
#endif

   if (channels < 0 || channels > 2)
   {
      celt_warning("Only mono and stereo supported");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }

   N = mode->mdctSize;
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
   st->frame_size = N;
   st->block_size = N;
   st->overlap = mode->overlap;
   st->channels = channels;

   st->vbr_rate = 0;
   st->pitch_enabled = 1;
   st->pitch_permitted = 1;
   st->pitch_available = 1;
   st->force_intra  = 0;
   st->delayedIntra = 1;
   st->tonal_average = QCONST16(1.,8);
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

static int transient_analysis(celt_word32 *in, int len, int C, int *transient_time, int *transient_shift)
{
   int c, i, n;
   celt_word32 ratio;
   VARDECL(celt_word32, begin);
   SAVE_STACK;
   ALLOC(begin, len, celt_word32);
   for (i=0;i<len;i++)
      begin[i] = ABS32(SHR32(in[C*i],SIG_SHIFT));
   for (c=1;c<C;c++)
   {
      for (i=0;i<len;i++)
         begin[i] = MAX32(begin[i], ABS32(SHR32(in[C*i+c],SIG_SHIFT)));
   }
   for (i=1;i<len;i++)
      begin[i] = MAX32(begin[i-1],begin[i]);
   n = -1;
   for (i=8;i<len-8;i++)
   {
      if (begin[i] < MULT16_32_Q15(QCONST16(.2f,15),begin[len-1]))
         n=i;
   }
   if (n<32)
   {
      n = -1;
      ratio = 0;
   } else {
      ratio = DIV32(begin[len-1],1+begin[n-16]);
   }
   if (ratio < 0)
      ratio = 0;
   if (ratio > 1000)
      ratio = 1000;
   ratio *= ratio;
   
   if (ratio > 2048)
      *transient_shift = 3;
   else
      *transient_shift = 0;
   
   *transient_time = n;
   
   RESTORE_STACK;
   return ratio > 20;
}

/** Apply window and compute the MDCT for all sub-frames and 
    all channels in a frame */
static void compute_mdcts(const CELTMode *mode, int shortBlocks, celt_sig * restrict in, celt_sig * restrict out, int _C)
{
   const int C = CHANNELS(_C);
   if (C==1 && !shortBlocks)
   {
      const mdct_lookup *lookup = MDCT(mode);
      const int overlap = OVERLAP(mode);
      clt_mdct_forward(lookup, in, out, mode->window, overlap);
   } else {
      const mdct_lookup *lookup = MDCT(mode);
      const int overlap = OVERLAP(mode);
      int N = FRAMESIZE(mode);
      int B = 1;
      int b, c;
      VARDECL(celt_word32, x);
      VARDECL(celt_word32, tmp);
      SAVE_STACK;
      if (shortBlocks)
      {
         lookup = &mode->shortMdct;
         N = mode->shortMdctSize;
         B = mode->nbShortMdcts;
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
            clt_mdct_forward(lookup, x, tmp, mode->window, overlap);
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
static void compute_inv_mdcts(const CELTMode *mode, int shortBlocks, celt_sig *X, int transient_time, int transient_shift, celt_sig * restrict out_mem, int _C)
{
   int c, N4;
   const int C = CHANNELS(_C);
   const int N = FRAMESIZE(mode);
   const int overlap = OVERLAP(mode);
   N4 = (N-overlap)>>1;
   for (c=0;c<C;c++)
   {
      int j;
      if (transient_shift==0 && C==1 && !shortBlocks) {
         const mdct_lookup *lookup = MDCT(mode);
         clt_mdct_backward(lookup, X, out_mem+C*(MAX_PERIOD-N-N4), mode->window, overlap);
      } else {
         VARDECL(celt_word32, x);
         VARDECL(celt_word32, tmp);
         int b;
         int N2 = N;
         int B = 1;
         int n4offset=0;
         const mdct_lookup *lookup = MDCT(mode);
         SAVE_STACK;
         
         ALLOC(x, 2*N, celt_word32);
         ALLOC(tmp, N, celt_word32);

         if (shortBlocks)
         {
            lookup = &mode->shortMdct;
            N2 = mode->shortMdctSize;
            B = mode->nbShortMdcts;
            n4offset = N4;
         }
         /* Prevents problems from the imdct doing the overlap-add */
         CELT_MEMSET(x+N4, 0, N2);

         for (b=0;b<B;b++)
         {
            /* De-interleaving the sub-frames */
            for (j=0;j<N2;j++)
               tmp[j] = X[(j*B+b)+c*N2*B];
            clt_mdct_backward(lookup, tmp, x+n4offset+N2*b, mode->window, overlap);
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
      if (flags == (flaglist[i]&FLAG_MASK))
         break;
   celt_assert(i<8);
   flag_bits = flaglist[i]&0xf;
   /*printf ("enc %d: %d %d %d %d\n", flag_bits, intra_ener, has_pitch, shortBlocks, has_fold);*/
   if (i<2)
      ec_enc_uint(enc, flag_bits, 4);
   else if (i<6)
      ec_enc_uint(enc, flag_bits, 16);
   else
      ec_enc_uint(enc, flag_bits, 8);
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

static void deemphasis(celt_sig *in, celt_word16 *pcm, int N, int _C, celt_word16 coef, celt_sig *mem)
{
   const int C = CHANNELS(_C);
   int c;
   for (c=0;c<C;c++)
   {
      int j;
      for (j=0;j<N;j++)
      {
         celt_sig tmp = MAC16_32_Q15(in[C*(MAX_PERIOD-N)+C*j+c],
                                       coef,mem[c]);
         mem[c] = tmp;
         pcm[C*j+c] = SCALEOUT(SIG2WORD16(tmp));
      }
   }

}

static void mdct_shape(const CELTMode *mode, celt_norm *X, int start, int end, int N, int nbShortMdcts, int mdct_weight_shift, int _C)
{
   int m, i, c;
   const int C = CHANNELS(_C);
   for (c=0;c<C;c++)
      for (m=start;m<end;m++)
         for (i=m+c*N;i<(c+1)*N;i+=nbShortMdcts)
#ifdef FIXED_POINT
            X[i] = SHR16(X[i], mdct_weight_shift);
#else
            X[i] = (1.f/(1<<mdct_weight_shift))*X[i];
#endif
   renormalise_bands(mode, X, C);
}


#ifdef FIXED_POINT
int celt_encode(CELTEncoder * restrict st, const celt_int16 * pcm, celt_int16 * optional_synthesis, unsigned char *compressed, int nbCompressedBytes)
{
#else
int celt_encode_float(CELTEncoder * restrict st, const celt_sig * pcm, celt_sig * optional_synthesis, unsigned char *compressed, int nbCompressedBytes)
{
#endif
   int i, c, N, NN, N4;
   int has_pitch;
   int pitch_index;
   int bits;
   int has_fold=1;
   int coarse_needed;
   ec_byte_buffer buf;
   ec_enc         enc;
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
   int intra_ener = 0;
   int shortBlocks=0;
   int transient_time;
   int transient_shift;
   const int C = CHANNELS(st->channels);
   int mdct_weight_shift = 0;
   int mdct_weight_pos=0;
   int gain_id=0;
   int norm_rate;
   SAVE_STACK;

   if (check_encoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (nbCompressedBytes<0 || pcm==NULL)
     return CELT_BAD_ARG; 

   /* The memset is important for now in case the encoder doesn't 
      fill up all the bytes */
   CELT_MEMSET(compressed, 0, nbCompressedBytes);
   ec_byte_writeinit_buffer(&buf, compressed, nbCompressedBytes);
   ec_enc_init(&enc,&buf);

   N = st->block_size;
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
   shortBlocks = 0;

   if (st->mode->nbShortMdcts > 1 && transient_analysis(in, N+st->overlap, C, &transient_time, &transient_shift))
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
      shortBlocks = 1;
      has_fold = 1;
   }

   ALLOC(freq, C*N, celt_sig); /**< Interleaved signal MDCTs */
   ALLOC(bandE,st->mode->nbEBands*C, celt_ener);
   ALLOC(bandLogE,st->mode->nbEBands*C, celt_word16);
   /* Compute MDCTs */
   compute_mdcts(st->mode, shortBlocks, in, freq, C);


   norm_rate = (nbCompressedBytes-5)*8*(celt_uint32)st->mode->Fs/(C*N)>>10;
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
      pitch_search(st->mode, x_lp, st->pitch_buf, 2*N-2*N4, MAX_PERIOD-(2*N-2*N4), &pitch_index, &st->xmem);
      RESTORE_STACK;
   }

   /* Deferred allocation after find_spectral_pitch() to reduce 
      the peak memory usage */
   ALLOC(X, C*N, celt_norm);         /**< Interleaved normalised MDCTs */

   ALLOC(pitch_freq, C*N, celt_sig); /**< Interleaved signal MDCTs */
   if (has_pitch)
   {
      compute_mdcts(st->mode, 0, st->out_mem+pitch_index*C, pitch_freq, C);
      has_pitch = compute_pitch_gain(st->mode, freq, pitch_freq, norm_rate, &gain_id, C, &st->gain_prod);
   }
   
   if (has_pitch)
      apply_pitch(st->mode, freq, pitch_freq, gain_id, 1, C);

   compute_band_energies(st->mode, freq, bandE, C);
   for (i=0;i<st->mode->nbEBands*C;i++)
      bandLogE[i] = amp2Log(bandE[i]);

   /* Band normalisation */
   normalise_bands(st->mode, freq, X, bandE, C);
   if (!shortBlocks && !folding_decision(st->mode, X, &st->tonal_average, &st->fold_decision, C))
      has_fold = 0;

   /* Don't use intra energy when we're operating at low bit-rate */
   intra_ener = st->force_intra || (!has_pitch && st->delayedIntra && nbCompressedBytes > st->mode->nbEBands);
   if (shortBlocks || intra_decision(bandLogE, st->oldBandE, st->mode->nbEBands))
      st->delayedIntra = 1;
   else
      st->delayedIntra = 0;

   NN = st->mode->eBands[st->mode->nbEBands];
   if (shortBlocks && !transient_shift) 
   {
      celt_word32 sum[8]={1,1,1,1,1,1,1,1};
      int m;
      for (c=0;c<C;c++)
      {
         m=0;
         do {
            celt_word32 tmp=0;
            for (i=m+c*N;i<c*N+NN;i+=st->mode->nbShortMdcts)
               tmp += ABS32(X[i]);
            sum[m++] += tmp;
         } while (m<st->mode->nbShortMdcts);
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
      } while (m<st->mode->nbShortMdcts-1);
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
      } while (m<st->mode->nbShortMdcts-1);
#endif
      if (mdct_weight_shift)
      {
         mdct_shape(st->mode, X, mdct_weight_pos+1, st->mode->nbShortMdcts, N, st->mode->nbShortMdcts, mdct_weight_shift, C);
         renormalise_bands(st->mode, X, C);
      }
   }


   encode_flags(&enc, intra_ener, has_pitch, shortBlocks, has_fold);
   if (has_pitch)
   {
      ec_enc_uint(&enc, pitch_index, MAX_PERIOD-(2*N-2*N4));
      ec_enc_uint(&enc, gain_id, 16);
   }
   if (shortBlocks)
   {
      if (transient_shift)
      {
         ec_enc_uint(&enc, transient_shift, 4);
         ec_enc_uint(&enc, transient_time, N+st->overlap);
      } else {
         ec_enc_uint(&enc, mdct_weight_shift, 4);
         if (mdct_weight_shift && st->mode->nbShortMdcts!=2)
            ec_enc_uint(&enc, mdct_weight_pos, st->mode->nbShortMdcts-1);
      }
   }

   ALLOC(fine_quant, st->mode->nbEBands, int);
   ALLOC(pulses, st->mode->nbEBands, int);

   /* Computes the max bit-rate allowed in VBR more to avoid busting the budget */
   if (st->vbr_rate>0)
   {
      celt_int32 vbr_bound, max_allowed;

      vbr_bound = st->vbr_rate;
      max_allowed = (st->vbr_rate + vbr_bound - st->vbr_reservoir)>>(BITRES+3);
      if (max_allowed < 4)
         max_allowed = 4;
      if (max_allowed < nbCompressedBytes)
         nbCompressedBytes = max_allowed;
   }

   /* Bit allocation */
   ALLOC(error, C*st->mode->nbEBands, celt_word16);
   coarse_needed = quant_coarse_energy(st->mode, bandLogE, st->oldBandE, nbCompressedBytes*4-8, intra_ener, st->mode->prob, error, &enc, C);
   coarse_needed = ((coarse_needed*3-1)>>3)+1;
   if (coarse_needed > nbCompressedBytes)
      coarse_needed = nbCompressedBytes;
   /* Variable bitrate */
   if (st->vbr_rate>0)
   {
     celt_word16 alpha;
     celt_int32 delta;
     /* The target rate in 16th bits per frame */
     celt_int32 target=st->vbr_rate;
   
     /* Shortblocks get a large boost in bitrate, but since they 
        are uncommon long blocks are not greatly effected */
     if (shortBlocks)
       target*=2;
     else if (st->mode->nbShortMdcts > 1)
       target-=(target+14)/28;

     /* The average energy is removed from the target and the actual 
        energy added*/
     target=target+st->vbr_offset-588+ec_enc_tell(&enc, BITRES);

     /* In VBR mode the frame size must not be reduced so much that it would result in the coarse energy busting its budget */
     target=IMAX(coarse_needed,(target+64)/128);
     target=IMIN(nbCompressedBytes,target);
     /* Make the adaptation coef (alpha) higher at the beginning */
     if (st->vbr_count < 990)
     {
        st->vbr_count++;
        alpha = celt_rcp(SHL32(EXTEND32(st->vbr_count+10),16));
        /*printf ("%d %d\n", st->vbr_count+10, alpha);*/
     } else
        alpha = QCONST16(.001f,15);

     /* By how much did we "miss" the target on that frame */
     delta = (8<<BITRES)*(celt_int32)target - st->vbr_rate;
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
     if (target < nbCompressedBytes)
        nbCompressedBytes = target;
     /* This moves the raw bits to take into account the new compressed size */
     ec_byte_shrink(&buf, nbCompressedBytes);
   }

   ALLOC(offsets, st->mode->nbEBands, int);
   ALLOC(fine_priority, st->mode->nbEBands, int);

   for (i=0;i<st->mode->nbEBands;i++)
      offsets[i] = 0;
   bits = nbCompressedBytes*8 - ec_enc_tell(&enc, 0) - 1;
   compute_allocation(st->mode, offsets, bits, pulses, fine_quant, fine_priority, C);

   quant_fine_energy(st->mode, bandE, st->oldBandE, error, fine_quant, &enc, C);

   /* Residual quantisation */
   if (C==1)
      quant_bands(st->mode, X, bandE, pulses, shortBlocks, has_fold, nbCompressedBytes*8, 1, &enc);
#ifndef DISABLE_STEREO
   else
      quant_bands_stereo(st->mode, X, bandE, pulses, shortBlocks, has_fold, nbCompressedBytes*8, &enc);
#endif

   quant_energy_finalise(st->mode, bandE, st->oldBandE, error, fine_quant, fine_priority, nbCompressedBytes*8-ec_enc_tell(&enc, 0), &enc, C);

   /* Re-synthesis of the coded audio if required */
   if (st->pitch_available>0 || optional_synthesis!=NULL)
   {
      if (st->pitch_available>0 && st->pitch_available<MAX_PERIOD)
        st->pitch_available+=st->frame_size;

      if (mdct_weight_shift)
      {
         mdct_shape(st->mode, X, 0, mdct_weight_pos+1, N, st->mode->nbShortMdcts, mdct_weight_shift, C);
      }

      /* Synthesis */
      denormalise_bands(st->mode, X, freq, bandE, C);

      CELT_MOVE(st->out_mem, st->out_mem+C*N, C*(MAX_PERIOD+st->overlap-N));

      if (has_pitch)
         apply_pitch(st->mode, freq, pitch_freq, gain_id, 0, C);
      
      compute_inv_mdcts(st->mode, shortBlocks, freq, transient_time, transient_shift, st->out_mem, C);

      /* De-emphasis and put everything back at the right place 
         in the synthesis history */
      if (optional_synthesis != NULL) {
         deemphasis(st->out_mem, optional_synthesis, N, C, preemph, st->preemph_memD);

      }
   }

   ec_enc_done(&enc);
   
   RESTORE_STACK;
   return nbCompressedBytes;
}

#ifdef FIXED_POINT
#ifndef DISABLE_FLOAT_API
int celt_encode_float(CELTEncoder * restrict st, const float * pcm, float * optional_synthesis, unsigned char *compressed, int nbCompressedBytes)
{
   int j, ret, C, N;
   VARDECL(celt_int16, in);
   SAVE_STACK;

   if (check_encoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   C = CHANNELS(st->channels);
   N = st->block_size;
   ALLOC(in, C*N, celt_int16);

   for (j=0;j<C*N;j++)
     in[j] = FLOAT2INT16(pcm[j]);

   if (optional_synthesis != NULL) {
     ret=celt_encode(st,in,in,compressed,nbCompressedBytes);
      for (j=0;j<C*N;j++)
         optional_synthesis[j]=in[j]*(1/32768.);
   } else {
     ret=celt_encode(st,in,NULL,compressed,nbCompressedBytes);
   }
   RESTORE_STACK;
   return ret;

}
#endif /*DISABLE_FLOAT_API*/
#else
int celt_encode(CELTEncoder * restrict st, const celt_int16 * pcm, celt_int16 * optional_synthesis, unsigned char *compressed, int nbCompressedBytes)
{
   int j, ret, C, N;
   VARDECL(celt_sig, in);
   SAVE_STACK;

   if (check_encoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   C=CHANNELS(st->channels);
   N=st->block_size;
   ALLOC(in, C*N, celt_sig);
   for (j=0;j<C*N;j++) {
     in[j] = SCALEOUT(pcm[j]);
   }

   if (optional_synthesis != NULL) {
      ret = celt_encode_float(st,in,in,compressed,nbCompressedBytes);
      for (j=0;j<C*N;j++)
         optional_synthesis[j] = FLOAT2INT16(in[j]);
   } else {
      ret = celt_encode_float(st,in,NULL,compressed,nbCompressedBytes);
   }
   RESTORE_STACK;
   return ret;
}
#endif

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
         if (value<0)
            goto bad_arg;
         if (value>3072000)
            value = 3072000;
         st->vbr_rate = ((st->mode->Fs<<3)+(st->block_size>>1))/st->block_size;
         st->vbr_rate = ((value<<7)+(st->vbr_rate>>1))/st->vbr_rate;
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
         st->tonal_average = QCONST16(1.,8);
         st->gain_prod = 0;
         st->vbr_reservoir = 0;
         st->vbr_drift = 0;
         st->vbr_offset = 0;
         st->vbr_count = 0;
         st->xmem = 0;
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
#ifdef NEW_PLC
#define DECODE_BUFFER_SIZE 2048
#else
#define DECODE_BUFFER_SIZE MAX_PERIOD
#endif

#define DECODERVALID   0x4c434454
#define DECODERPARTIAL 0x5444434c
#define DECODERFREED   0x4c004400

/** Decoder state 
 @brief Decoder state
 */
struct CELTDecoder {
   celt_uint32 marker;
   const CELTMode *mode;
   int frame_size;
   int block_size;
   int overlap;
   int channels;

   ec_byte_buffer buf;
   ec_enc         enc;

   celt_sig * restrict preemph_memD;

   celt_sig *out_mem;
   celt_sig *decode_mem;

   celt_word16 *oldBandE;
   
#ifdef NEW_PLC
   celt_word16 *lpc;
#endif

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
   int N, C;
   CELTDecoder *st;

   if (check_mode(mode) != CELT_OK)
   {
      if (error)
         *error = CELT_INVALID_MODE;
      return NULL;
   }
#ifdef DISABLE_STEREO
   if (channels > 1)
   {
      celt_warning("Stereo support was disable from this build");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }
#endif

   if (channels < 0 || channels > 2)
   {
      celt_warning("Only mono and stereo supported");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }

   N = mode->mdctSize;
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
   st->frame_size = N;
   st->block_size = N;
   st->overlap = mode->overlap;
   st->channels = channels;

   st->decode_mem = celt_alloc((DECODE_BUFFER_SIZE+st->overlap)*C*sizeof(celt_sig));
   st->out_mem = st->decode_mem+DECODE_BUFFER_SIZE-MAX_PERIOD;
   
   st->oldBandE = (celt_word16*)celt_alloc(C*mode->nbEBands*sizeof(celt_word16));
   
   st->preemph_memD = (celt_sig*)celt_alloc(C*sizeof(celt_sig));

#ifdef NEW_PLC
   st->lpc = (celt_word16*)celt_alloc(C*LPC_ORDER*sizeof(celt_word16));
#endif

   st->loss_count = 0;

   if ((st->decode_mem!=NULL) && (st->out_mem!=NULL) && (st->oldBandE!=NULL) &&
#ifdef NEW_PLC
         (st->lpc!=NULL) &&
#endif
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

#ifdef NEW_PLC
   celt_free(st->lpc);
#endif
   
   st->marker = DECODERFREED;
   
   celt_free(st);
}

static void celt_decode_lost(CELTDecoder * restrict st, celt_word16 * restrict pcm)
{
   int c, N;
   int pitch_index;
   int overlap = st->mode->overlap;
   celt_word16 fade = Q15ONE;
   int i, len;
   VARDECL(celt_sig, freq);
   const int C = CHANNELS(st->channels);
   int offset;
   SAVE_STACK;
   N = st->block_size;
   
   len = N+st->mode->overlap;
   
   if (st->loss_count == 0)
   {
      celt_word16 pitch_buf[MAX_PERIOD>>1];
      celt_word32 tmp=0;
      celt_word32 mem0[2]={0,0};
      celt_word16 mem1[2]={0,0};
      pitch_downsample(st->out_mem, pitch_buf, MAX_PERIOD, MAX_PERIOD,
                       C, mem0, mem1);
      pitch_search(st->mode, pitch_buf+((MAX_PERIOD-len)>>1), pitch_buf, len,
                   MAX_PERIOD-len-100, &pitch_index, &tmp);
      pitch_index = MAX_PERIOD-len-pitch_index;
      st->last_pitch_index = pitch_index;
   } else {
      pitch_index = st->last_pitch_index;
      if (st->loss_count < 5)
         fade = QCONST16(.8f,15);
      else
         fade = 0;
   }

#ifndef NEW_PLC
   offset = MAX_PERIOD-pitch_index;
   ALLOC(freq,C*N, celt_sig); /**< Interleaved signal MDCTs */
   while (offset+len >= MAX_PERIOD)
      offset -= pitch_index;
   compute_mdcts(st->mode, 0, st->out_mem+offset*C, freq, C);
   for (i=0;i<C*N;i++)
      freq[i] = ADD32(VERY_SMALL, MULT16_32_Q15(fade,freq[i]));

   CELT_MOVE(st->out_mem, st->out_mem+C*N, C*(MAX_PERIOD+st->mode->overlap-N));
   /* Compute inverse MDCTs */
   compute_inv_mdcts(st->mode, 0, freq, -1, 0, st->out_mem, C);
#else
   for (c=0;c<C;c++)
   {
      celt_word32 e[MAX_PERIOD];
      celt_word16 exc[MAX_PERIOD];
      float ac[LPC_ORDER+1];
      float decay = 1;
      float S1=0;
      celt_word16 mem[LPC_ORDER]={0};

      offset = MAX_PERIOD-pitch_index;
      for (i=0;i<MAX_PERIOD;i++)
         exc[i] = ROUND16(st->out_mem[i*C+c], SIG_SHIFT);

      if (st->loss_count == 0)
      {
         _celt_autocorr(exc, ac, st->mode->window, st->mode->overlap,
                        LPC_ORDER, MAX_PERIOD);

         /* Noise floor -50 dB */
         ac[0] *= 1.00001;
         /* Lag windowing */
         for (i=1;i<=LPC_ORDER;i++)
         {
            /*ac[i] *= exp(-.5*(2*M_PI*.002*i)*(2*M_PI*.002*i));*/
            ac[i] -= ac[i]*(.008*i)*(.008*i);
         }

         _celt_lpc(st->lpc+c*LPC_ORDER, ac, LPC_ORDER);
      }
      fir(exc, st->lpc+c*LPC_ORDER, exc, MAX_PERIOD, LPC_ORDER, mem);
      /*for (i=0;i<MAX_PERIOD;i++)printf("%d ", exc[i]); printf("\n");*/
      /* Check if the waveform is decaying (and if so how fast) */
      {
         float E1=0, E2=0;
         int period;
         if (pitch_index <= MAX_PERIOD/2)
            period = pitch_index;
         else
            period = MAX_PERIOD/2;
         for (i=0;i<period;i++)
         {
            E1 += exc[MAX_PERIOD-period+i]*exc[MAX_PERIOD-period+i];
            E2 += exc[MAX_PERIOD-2*period+i]*exc[MAX_PERIOD-2*period+i];
         }
         decay = sqrt((E1+1)/(E2+1));
         if (decay > 1)
            decay = 1;
      }

      /* Copy excitation, taking decay into account */
      for (i=0;i<len+st->mode->overlap;i++)
      {
         if (offset+i >= MAX_PERIOD)
         {
            offset -= pitch_index;
            decay *= decay;
         }
         e[i] = decay*SHL32(EXTEND32(exc[offset+i]), SIG_SHIFT);
         S1 += st->out_mem[offset+i]*1.*st->out_mem[offset+i];
      }

      iir(e, st->lpc+c*LPC_ORDER, e, len+st->mode->overlap, LPC_ORDER, mem);

      {
         float S2=0;
         for (i=0;i<len+overlap;i++)
            S2 += e[i]*1.*e[i];
         /* This checks for an "explosion" in the synthesis (including NaNs) */
         if (!(S1 > 0.2f*S2))
         {
            for (i=0;i<len+overlap;i++)
               e[i] = 0;
         } else if (S1 < S2)
         {
            float ratio = sqrt((S1+1)/(S2+1));
            for (i=0;i<len+overlap;i++)
               e[i] *= ratio;
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
#endif

   deemphasis(st->out_mem, pcm, N, C, preemph, st->preemph_memD);
   
   st->loss_count++;

   RESTORE_STACK;
}

#ifdef FIXED_POINT
int celt_decode(CELTDecoder * restrict st, const unsigned char *data, int len, celt_int16 * restrict pcm)
{
#else
int celt_decode_float(CELTDecoder * restrict st, const unsigned char *data, int len, celt_sig * restrict pcm)
{
#endif
   int i, N, N4;
   int has_pitch, has_fold;
   int pitch_index;
   int bits;
   ec_dec dec;
   ec_byte_buffer buf;
   VARDECL(celt_sig, freq);
   VARDECL(celt_sig, pitch_freq);
   VARDECL(celt_norm, X);
   VARDECL(celt_ener, bandE);
   VARDECL(int, fine_quant);
   VARDECL(int, pulses);
   VARDECL(int, offsets);
   VARDECL(int, fine_priority);

   int shortBlocks;
   int intra_ener;
   int transient_time;
   int transient_shift;
   int mdct_weight_shift=0;
   const int C = CHANNELS(st->channels);
   int mdct_weight_pos=0;
   int gain_id=0;
   SAVE_STACK;

   if (check_decoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   N = st->block_size;
   N4 = (N-st->overlap)>>1;

   ALLOC(freq, C*N, celt_sig); /**< Interleaved signal MDCTs */
   ALLOC(X, C*N, celt_norm);   /**< Interleaved normalised MDCTs */
   ALLOC(bandE, st->mode->nbEBands*C, celt_ener);
   
   if (data == NULL)
   {
      celt_decode_lost(st, pcm);
      RESTORE_STACK;
      return 0;
   }
   if (len<0) {
     RESTORE_STACK;
     return CELT_BAD_ARG;
   }
   
   ec_byte_readinit(&buf,(unsigned char*)data,len);
   ec_dec_init(&dec,&buf);
   
   decode_flags(&dec, &intra_ener, &has_pitch, &shortBlocks, &has_fold);
   if (shortBlocks)
   {
      transient_shift = ec_dec_uint(&dec, 4);
      if (transient_shift == 3)
      {
         transient_time = ec_dec_uint(&dec, N+st->mode->overlap);
      } else {
         mdct_weight_shift = transient_shift;
         if (mdct_weight_shift && st->mode->nbShortMdcts>2)
            mdct_weight_pos = ec_dec_uint(&dec, st->mode->nbShortMdcts-1);
         transient_shift = 0;
         transient_time = 0;
      }
   } else {
      transient_time = -1;
      transient_shift = 0;
   }
   
   if (has_pitch)
   {
      pitch_index = ec_dec_uint(&dec, MAX_PERIOD-(2*N-2*N4));
      gain_id = ec_dec_uint(&dec, 16);
   } else {
      pitch_index = 0;
   }

   ALLOC(fine_quant, st->mode->nbEBands, int);
   /* Get band energies */
   unquant_coarse_energy(st->mode, bandE, st->oldBandE, len*4-8, intra_ener, st->mode->prob, &dec, C);
   
   ALLOC(pulses, st->mode->nbEBands, int);
   ALLOC(offsets, st->mode->nbEBands, int);
   ALLOC(fine_priority, st->mode->nbEBands, int);

   for (i=0;i<st->mode->nbEBands;i++)
      offsets[i] = 0;

   bits = len*8 - ec_dec_tell(&dec, 0) - 1;
   compute_allocation(st->mode, offsets, bits, pulses, fine_quant, fine_priority, C);
   /*bits = ec_dec_tell(&dec, 0);
   compute_fine_allocation(st->mode, fine_quant, (20*C+len*8/5-(ec_dec_tell(&dec, 0)-bits))/C);*/
   
   unquant_fine_energy(st->mode, bandE, st->oldBandE, fine_quant, &dec, C);

   ALLOC(pitch_freq, C*N, celt_sig); /**< Interleaved signal MDCTs */
   if (has_pitch) 
   {
      /* Pitch MDCT */
      compute_mdcts(st->mode, 0, st->out_mem+pitch_index*C, pitch_freq, C);
   }

   /* Decode fixed codebook and merge with pitch */
   if (C==1)
      quant_bands(st->mode, X, bandE, pulses, shortBlocks, has_fold, len*8, 0, &dec);
#ifndef DISABLE_STEREO
   else
      unquant_bands_stereo(st->mode, X, bandE, pulses, shortBlocks, has_fold, len*8, &dec);
#endif
   unquant_energy_finalise(st->mode, bandE, st->oldBandE, fine_quant, fine_priority, len*8-ec_dec_tell(&dec, 0), &dec, C);
   
   if (mdct_weight_shift)
   {
      mdct_shape(st->mode, X, 0, mdct_weight_pos+1, N, st->mode->nbShortMdcts, mdct_weight_shift, C);
   }

   /* Synthesis */
   denormalise_bands(st->mode, X, freq, bandE, C);


   CELT_MOVE(st->decode_mem, st->decode_mem+C*N, C*(DECODE_BUFFER_SIZE+st->overlap-N));

   if (has_pitch)
      apply_pitch(st->mode, freq, pitch_freq, gain_id, 0, C);

   /* Compute inverse MDCTs */
   compute_inv_mdcts(st->mode, shortBlocks, freq, transient_time, transient_shift, st->out_mem, C);

   deemphasis(st->out_mem, pcm, N, C, preemph, st->preemph_memD);
   st->loss_count = 0;
   RESTORE_STACK;
   return 0;
}

#ifdef FIXED_POINT
#ifndef DISABLE_FLOAT_API
int celt_decode_float(CELTDecoder * restrict st, const unsigned char *data, int len, float * restrict pcm)
{
   int j, ret, C, N;
   VARDECL(celt_int16, out);
   SAVE_STACK;

   if (check_decoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   C = CHANNELS(st->channels);
   N = st->block_size;
   
   ALLOC(out, C*N, celt_int16);
   ret=celt_decode(st, data, len, out);
   for (j=0;j<C*N;j++)
      pcm[j]=out[j]*(1/32768.);
     
   RESTORE_STACK;
   return ret;
}
#endif /*DISABLE_FLOAT_API*/
#else
int celt_decode(CELTDecoder * restrict st, const unsigned char *data, int len, celt_int16 * restrict pcm)
{
   int j, ret, C, N;
   VARDECL(celt_sig, out);
   SAVE_STACK;

   if (check_decoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   C = CHANNELS(st->channels);
   N = st->block_size;
   ALLOC(out, C*N, celt_sig);

   ret=celt_decode_float(st, data, len, out);

   for (j=0;j<C*N;j++)
      pcm[j] = FLOAT2INT16 (out[j]);
   
   RESTORE_STACK;
   return ret;
}
#endif

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
      case CELT_RESET_STATE:
      {
         const CELTMode *mode = st->mode;
         int C = st->channels;

         CELT_MEMSET(st->decode_mem, 0, (DECODE_BUFFER_SIZE+st->overlap)*C);
         CELT_MEMSET(st->oldBandE, 0, C*mode->nbEBands);

         CELT_MEMSET(st->preemph_memD, 0, C);

         st->loss_count = 0;

#ifdef NEW_PLC
         CELT_MEMSET(st->lpc, 0, C*LPC_ORDER);
#endif
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

