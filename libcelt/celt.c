/* (C) 2007-2008 Jean-Marc Valin, CSIRO
   (C) 2008 Gregory Maxwell */
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
#include "kiss_fftr.h"
#include "bands.h"
#include "modes.h"
#include "entcode.h"
#include "quant_bands.h"
#include "psy.h"
#include "rate.h"
#include "stack_alloc.h"
#include "mathops.h"
#include "float_cast.h"
#include <stdarg.h>

static const celt_word16_t preemph = QCONST16(0.8f,15);

#ifdef FIXED_POINT
static const celt_word16_t transientWindow[16] = {
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
   celt_uint32_t marker;
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
   celt_word16_t tonal_average;
   int fold_decision;

   int VBR_rate; /* Target number of 16th bits per frame */
   celt_word16_t * restrict preemph_memE; 
   celt_sig_t    * restrict preemph_memD;

   celt_sig_t *in_mem;
   celt_sig_t *out_mem;

   celt_word16_t *oldBandE;
#ifdef EXP_PSY
   celt_word16_t *psy_mem;
   struct PsyDecay psy;
#endif
};

int check_encoder(const CELTEncoder *st) 
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

CELTEncoder *celt_encoder_create(const CELTMode *mode)
{
   int N, C;
   CELTEncoder *st;

   if (check_mode(mode) != CELT_OK)
      return NULL;

   N = mode->mdctSize;
   C = mode->nbChannels;
   st = celt_alloc(sizeof(CELTEncoder));
   
   if (st==NULL) 
      return NULL;   
   st->marker = ENCODERPARTIAL;
   st->mode = mode;
   st->frame_size = N;
   st->block_size = N;
   st->overlap = mode->overlap;

   st->VBR_rate = 0;
   st->pitch_enabled = 1;
   st->pitch_permitted = 1;
   st->pitch_available = 1;
   st->force_intra  = 0;
   st->delayedIntra = 1;
   st->tonal_average = QCONST16(1.,8);
   st->fold_decision = 1;

   st->in_mem = celt_alloc(st->overlap*C*sizeof(celt_sig_t));
   st->out_mem = celt_alloc((MAX_PERIOD+st->overlap)*C*sizeof(celt_sig_t));

   st->oldBandE = (celt_word16_t*)celt_alloc(C*mode->nbEBands*sizeof(celt_word16_t));

   st->preemph_memE = (celt_word16_t*)celt_alloc(C*sizeof(celt_word16_t));
   st->preemph_memD = (celt_sig_t*)celt_alloc(C*sizeof(celt_sig_t));

#ifdef EXP_PSY
   st->psy_mem = celt_alloc(MAX_PERIOD*sizeof(celt_word16_t));
   psydecay_init(&st->psy, MAX_PERIOD/2, st->mode->Fs);
#endif

   if ((st->in_mem!=NULL) && (st->out_mem!=NULL) && (st->oldBandE!=NULL) 
#ifdef EXP_PSY
       && (st->psy_mem!=NULL) 
#endif   
       && (st->preemph_memE!=NULL) && (st->preemph_memD!=NULL))
   {
      st->marker   = ENCODERVALID;
      return st;
   }
   /* If the setup fails for some reason deallocate it. */
   celt_encoder_destroy(st);  
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
   
   celt_free(st->oldBandE);
   
   celt_free(st->preemph_memE);
   celt_free(st->preemph_memD);
   
#ifdef EXP_PSY
   celt_free (st->psy_mem);
   psydecay_clear(&st->psy);
#endif
   st->marker = ENCODERFREED;
   
   celt_free(st);
}

static inline celt_int16_t FLOAT2INT16(float x)
{
   x = x*CELT_SIG_SCALE;
   x = MAX32(x, -32768);
   x = MIN32(x, 32767);
   return (celt_int16_t)float2int(x);
}

static inline celt_word16_t SIG2WORD16(celt_sig_t x)
{
#ifdef FIXED_POINT
   x = PSHR32(x, SIG_SHIFT);
   x = MAX32(x, -32768);
   x = MIN32(x, 32767);
   return EXTRACT16(x);
#else
   return (celt_word16_t)x;
#endif
}

static int transient_analysis(celt_word32_t *in, int len, int C, int *transient_time, int *transient_shift)
{
   int c, i, n;
   celt_word32_t ratio;
   VARDECL(celt_word32_t, begin);
   SAVE_STACK;
   ALLOC(begin, len, celt_word32_t);
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
static void compute_mdcts(const CELTMode *mode, int shortBlocks, celt_sig_t * restrict in, celt_sig_t * restrict out)
{
   const int C = CHANNELS(mode);
   if (C==1 && !shortBlocks)
   {
      const mdct_lookup *lookup = MDCT(mode);
      const int overlap = OVERLAP(mode);
      mdct_forward(lookup, in, out, mode->window, overlap);
   } else if (!shortBlocks) {
      const mdct_lookup *lookup = MDCT(mode);
      const int overlap = OVERLAP(mode);
      const int N = FRAMESIZE(mode);
      int c;
      VARDECL(celt_word32_t, x);
      VARDECL(celt_word32_t, tmp);
      SAVE_STACK;
      ALLOC(x, N+overlap, celt_word32_t);
      ALLOC(tmp, N, celt_word32_t);
      for (c=0;c<C;c++)
      {
         int j;
         for (j=0;j<N+overlap;j++)
            x[j] = in[C*j+c];
         mdct_forward(lookup, x, tmp, mode->window, overlap);
         /* Interleaving the sub-frames */
         for (j=0;j<N;j++)
            out[j+c*N] = tmp[j];
      }
      RESTORE_STACK;
   } else {
      const mdct_lookup *lookup = &mode->shortMdct;
      const int overlap = mode->overlap;
      const int N = mode->shortMdctSize;
      int b, c;
      VARDECL(celt_word32_t, x);
      VARDECL(celt_word32_t, tmp);
      SAVE_STACK;
      ALLOC(x, N+overlap, celt_word32_t);
      ALLOC(tmp, N, celt_word32_t);
      for (c=0;c<C;c++)
      {
         int B = mode->nbShortMdcts;
         for (b=0;b<B;b++)
         {
            int j;
            for (j=0;j<N+overlap;j++)
               x[j] = in[C*(b*N+j)+c];
            mdct_forward(lookup, x, tmp, mode->window, overlap);
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
static void compute_inv_mdcts(const CELTMode *mode, int shortBlocks, celt_sig_t *X, int transient_time, int transient_shift, celt_sig_t * restrict out_mem)
{
   int c, N4;
   const int C = CHANNELS(mode);
   const int N = FRAMESIZE(mode);
   const int overlap = OVERLAP(mode);
   N4 = (N-overlap)>>1;
   for (c=0;c<C;c++)
   {
      int j;
      if (transient_shift==0 && C==1 && !shortBlocks) {
         const mdct_lookup *lookup = MDCT(mode);
         mdct_backward(lookup, X, out_mem+C*(MAX_PERIOD-N-N4), mode->window, overlap);
      } else if (!shortBlocks) {
         const mdct_lookup *lookup = MDCT(mode);
         VARDECL(celt_word32_t, x);
         VARDECL(celt_word32_t, tmp);
         SAVE_STACK;
         ALLOC(x, 2*N, celt_word32_t);
         ALLOC(tmp, N, celt_word32_t);
         /* De-interleaving the sub-frames */
         for (j=0;j<N;j++)
            tmp[j] = X[j+c*N];
         /* Prevents problems from the imdct doing the overlap-add */
         CELT_MEMSET(x+N4, 0, N);
         mdct_backward(lookup, tmp, x, mode->window, overlap);
         celt_assert(transient_shift == 0);
         /* The first and last part would need to be set to zero if we actually
            wanted to use them. */
         for (j=0;j<overlap;j++)
            out_mem[C*(MAX_PERIOD-N)+C*j+c] += x[j+N4];
         for (j=0;j<overlap;j++)
            out_mem[C*(MAX_PERIOD)+C*(overlap-j-1)+c] = x[2*N-j-N4-1];
         for (j=0;j<2*N4;j++)
            out_mem[C*(MAX_PERIOD-N)+C*(j+overlap)+c] = x[j+N4+overlap];
         RESTORE_STACK;
      } else {
         int b;
         const int N2 = mode->shortMdctSize;
         const int B = mode->nbShortMdcts;
         const mdct_lookup *lookup = &mode->shortMdct;
         VARDECL(celt_word32_t, x);
         VARDECL(celt_word32_t, tmp);
         SAVE_STACK;
         ALLOC(x, 2*N, celt_word32_t);
         ALLOC(tmp, N, celt_word32_t);
         /* Prevents problems from the imdct doing the overlap-add */
         CELT_MEMSET(x+N4, 0, N2);
         for (b=0;b<B;b++)
         {
            /* De-interleaving the sub-frames */
            for (j=0;j<N2;j++)
               tmp[j] = X[(j*B+b)+c*N2*B];
            mdct_backward(lookup, tmp, x+N4+N2*b, mode->window, overlap);
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
#define FLAG_INTRA       1U<<16
#define FLAG_PITCH       1U<<15
#define FLAG_SHORT       1U<<14
#define FLAG_FOLD        1U<<13
#define FLAG_MASK        (FLAG_INTRA|FLAG_PITCH|FLAG_SHORT|FLAG_FOLD)

celt_int32_t flaglist[8] = {
      0 /*00  */ | FLAG_FOLD,
      1 /*01  */ | FLAG_PITCH|FLAG_FOLD,
      8 /*1000*/ | FLAG_NONE,
      9 /*1001*/ | FLAG_SHORT|FLAG_FOLD,
     10 /*1010*/ | FLAG_PITCH,
     11 /*1011*/ | FLAG_INTRA,
      6 /*110 */ | FLAG_INTRA|FLAG_FOLD,
      7 /*111 */ | FLAG_INTRA|FLAG_SHORT|FLAG_FOLD
};

void encode_flags(ec_enc *enc, int intra_ener, int has_pitch, int shortBlocks, int has_fold)
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
      ec_enc_bits(enc, flag_bits, 2);
   else if (i<6)
      ec_enc_bits(enc, flag_bits, 4);
   else
      ec_enc_bits(enc, flag_bits, 3);
}

void decode_flags(ec_dec *dec, int *intra_ener, int *has_pitch, int *shortBlocks, int *has_fold)
{
   int i;
   int flag_bits;
   flag_bits = ec_dec_bits(dec, 2);
   /*printf ("(%d) ", flag_bits);*/
   if (flag_bits==2)
      flag_bits = (flag_bits<<2) | ec_dec_bits(dec, 2);
   else if (flag_bits==3)
      flag_bits = (flag_bits<<1) | ec_dec_bits(dec, 1);
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

#ifdef FIXED_POINT
int celt_encode(CELTEncoder * restrict st, const celt_int16_t * pcm, celt_int16_t * optional_synthesis, unsigned char *compressed, int nbCompressedBytes)
{
#else
int celt_encode_float(CELTEncoder * restrict st, const celt_sig_t * pcm, celt_sig_t * optional_synthesis, unsigned char *compressed, int nbCompressedBytes)
{
#endif
   int i, c, N, N4;
   int has_pitch;
   int pitch_index;
   int bits;
   int has_fold=1;
   unsigned coarse_needed;
   ec_byte_buffer buf;
   ec_enc         enc;
   VARDECL(celt_sig_t, in);
   VARDECL(celt_sig_t, freq);
   VARDECL(celt_norm_t, X);
   VARDECL(celt_norm_t, P);
   VARDECL(celt_ener_t, bandE);
   VARDECL(celt_word16_t, bandLogE);
   VARDECL(celt_pgain_t, gains);
   VARDECL(int, fine_quant);
   VARDECL(celt_word16_t, error);
   VARDECL(int, pulses);
   VARDECL(int, offsets);
   VARDECL(int, fine_priority);
#ifdef EXP_PSY
   VARDECL(celt_word32_t, mask);
   VARDECL(celt_word32_t, tonality);
   VARDECL(celt_word32_t, bandM);
   VARDECL(celt_ener_t, bandN);
#endif
   int intra_ener = 0;
   int shortBlocks=0;
   int transient_time;
   int transient_shift;
   const int C = CHANNELS(st->mode);
   int mdct_weight_shift = 0;
   int mdct_weight_pos=0;
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
   ALLOC(in, 2*C*N-2*C*N4, celt_sig_t);

   CELT_COPY(in, st->in_mem, C*st->overlap);
   for (c=0;c<C;c++)
   {
      const celt_word16_t * restrict pcmp = pcm+c;
      celt_sig_t * restrict inp = in+C*st->overlap+c;
      for (i=0;i<N;i++)
      {
         /* Apply pre-emphasis */
         celt_sig_t tmp = SCALEIN(SHL32(EXTEND32(*pcmp), SIG_SHIFT));
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

   ALLOC(freq, C*N, celt_sig_t); /**< Interleaved signal MDCTs */
   ALLOC(bandE,st->mode->nbEBands*C, celt_ener_t);
   ALLOC(bandLogE,st->mode->nbEBands*C, celt_word16_t);
   /* Compute MDCTs */
   compute_mdcts(st->mode, shortBlocks, in, freq);

   if (shortBlocks && !transient_shift) 
   {
      celt_word32_t sum[8]={1,1,1,1,1,1,1,1};
      int m;
      for (c=0;c<C;c++)
      {
         m=0;
         do {
            celt_word32_t tmp=0;
            for (i=m+c*N;i<(c+1)*N;i+=st->mode->nbShortMdcts)
               tmp += ABS32(freq[i]);
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
      if (mdct_weight_shift)
      {
         for (c=0;c<C;c++)
            for (m=mdct_weight_pos+1;m<st->mode->nbShortMdcts;m++)
               for (i=m+c*N;i<(c+1)*N;i+=st->mode->nbShortMdcts)
                  freq[i] = SHR32(freq[i],mdct_weight_shift);
      }
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
      if (mdct_weight_shift)
      {
         for (c=0;c<C;c++)
            for (m=mdct_weight_pos+1;m<st->mode->nbShortMdcts;m++)
               for (i=m+c*N;i<(c+1)*N;i+=st->mode->nbShortMdcts)
                  freq[i] = (1./(1<<mdct_weight_shift))*freq[i];
      }
#endif
   }

   compute_band_energies(st->mode, freq, bandE);
   for (i=0;i<st->mode->nbEBands*C;i++)
      bandLogE[i] = amp2Log(bandE[i]);

   /* Don't use intra energy when we're operating at low bit-rate */
   intra_ener = st->force_intra || (st->delayedIntra && nbCompressedBytes > st->mode->nbEBands);
   if (shortBlocks || intra_decision(bandLogE, st->oldBandE, st->mode->nbEBands))
      st->delayedIntra = 1;
   else
      st->delayedIntra = 0;

   /* Pitch analysis: we do it early to save on the peak stack space */
   /* Don't use pitch if there isn't enough data available yet, 
      or if we're using shortBlocks */
   has_pitch = st->pitch_enabled && st->pitch_permitted && (N <= 512) 
            && (st->pitch_available >= MAX_PERIOD) && (!shortBlocks)
            && !intra_ener;
#ifdef EXP_PSY
   ALLOC(tonality, MAX_PERIOD/4, celt_word16_t);
   {
      VARDECL(celt_word16_t, X);
      ALLOC(X, MAX_PERIOD/2, celt_word16_t);
      find_spectral_pitch(st->mode, st->mode->fft, &st->mode->psy, in, st->out_mem, st->mode->window, X, 2*N-2*N4, MAX_PERIOD-(2*N-2*N4), &pitch_index);
      compute_tonality(st->mode, X, st->psy_mem, MAX_PERIOD, tonality, MAX_PERIOD/4);
   }
#else
   if (has_pitch)
   {
      find_spectral_pitch(st->mode, st->mode->fft, &st->mode->psy, in, st->out_mem, st->mode->window, NULL, 2*N-2*N4, MAX_PERIOD-(2*N-2*N4), &pitch_index);
   }
#endif

#ifdef EXP_PSY
   ALLOC(mask, N, celt_sig_t);
   compute_mdct_masking(&st->psy, freq, tonality, st->psy_mem, mask, C*N);
   /*for (i=0;i<256;i++)
      printf ("%f %f %f ", freq[i], tonality[i], mask[i]);
   printf ("\n");*/
#endif

   /* Deferred allocation after find_spectral_pitch() to reduce 
      the peak memory usage */
   ALLOC(X, C*N, celt_norm_t);         /**< Interleaved normalised MDCTs */
   ALLOC(P, C*N, celt_norm_t);         /**< Interleaved normalised pitch MDCTs*/
   ALLOC(gains,st->mode->nbPBands, celt_pgain_t);


   /* Band normalisation */
   normalise_bands(st->mode, freq, X, bandE);
   if (!shortBlocks && !folding_decision(st->mode, X, &st->tonal_average, &st->fold_decision))
      has_fold = 0;
#ifdef EXP_PSY
   ALLOC(bandN,C*st->mode->nbEBands, celt_ener_t);
   ALLOC(bandM,st->mode->nbEBands, celt_ener_t);
   compute_noise_energies(st->mode, freq, tonality, bandN);

   /*for (i=0;i<st->mode->nbEBands;i++)
      printf ("%f ", (.1+bandN[i])/(.1+bandE[i]));
   printf ("\n");*/
   has_fold = 0;
   for (i=st->mode->nbPBands;i<st->mode->nbEBands;i++)
      if (bandN[i] < .4*bandE[i])
         has_fold++;
   /*printf ("%d\n", has_fold);*/
   if (has_fold>=2)
      has_fold = 0;
   else
      has_fold = 1;
   for (i=0;i<N;i++)
      mask[i] = sqrt(mask[i]);
   compute_band_energies(st->mode, mask, bandM);
   /*for (i=0;i<st->mode->nbEBands;i++)
      printf ("%f %f ", bandE[i], bandM[i]);
   printf ("\n");*/
#endif

   /* Compute MDCTs of the pitch part */
   if (has_pitch)
   {
      celt_word32_t curr_power, pitch_power=0;
      /* Normalise the pitch vector as well (discard the energies) */
      VARDECL(celt_ener_t, bandEp);
      
      compute_mdcts(st->mode, 0, st->out_mem+pitch_index*C, freq);
      ALLOC(bandEp, st->mode->nbEBands*st->mode->nbChannels, celt_ener_t);
      compute_band_energies(st->mode, freq, bandEp);
      normalise_bands(st->mode, freq, P, bandEp);
      pitch_power = bandEp[0]+bandEp[1]+bandEp[2];
      curr_power = bandE[0]+bandE[1]+bandE[2];
      if (C>1)
      {
         pitch_power += bandEp[0+st->mode->nbEBands]+bandEp[1+st->mode->nbEBands]+bandEp[2+st->mode->nbEBands];
         curr_power += bandE[0+st->mode->nbEBands]+bandE[1+st->mode->nbEBands]+bandE[2+st->mode->nbEBands];
      }
      /* Check if we can safely use the pitch (i.e. effective gain 
      isn't too high) */
      if ((MULT16_32_Q15(QCONST16(.1f, 15),curr_power) + QCONST32(10.f,ENER_SHIFT) < pitch_power))
      {
         /* Pitch prediction */
         has_pitch = compute_pitch_gain(st->mode, X, P, gains);
      } else {
         has_pitch = 0;
      }
   }
   
   encode_flags(&enc, intra_ener, has_pitch, shortBlocks, has_fold);
   if (has_pitch)
   {
      ec_enc_uint(&enc, pitch_index, MAX_PERIOD-(2*N-2*N4));
   } else {
      for (i=0;i<st->mode->nbPBands;i++)
         gains[i] = 0;
      for (i=0;i<C*N;i++)
         P[i] = 0;
   }
   if (shortBlocks)
   {
      if (transient_shift)
      {
         ec_enc_bits(&enc, transient_shift, 2);
         ec_enc_uint(&enc, transient_time, N+st->overlap);
      } else {
         ec_enc_bits(&enc, mdct_weight_shift, 2);
         if (mdct_weight_shift && st->mode->nbShortMdcts!=2)
            ec_enc_uint(&enc, mdct_weight_pos, st->mode->nbShortMdcts-1);
      }
   }

#ifdef STDIN_TUNING2
   static int fine_quant[30];
   static int pulses[30];
   static int init=0;
   if (!init)
   {
      for (i=0;i<st->mode->nbEBands;i++)
         scanf("%d ", &fine_quant[i]);
      for (i=0;i<st->mode->nbEBands;i++)
         scanf("%d ", &pulses[i]);
      init = 1;
   }
#else
   ALLOC(fine_quant, st->mode->nbEBands, int);
   ALLOC(pulses, st->mode->nbEBands, int);
#endif

   /* Bit allocation */
   ALLOC(error, C*st->mode->nbEBands, celt_word16_t);
   coarse_needed = quant_coarse_energy(st->mode, bandLogE, st->oldBandE, nbCompressedBytes*8/3, intra_ener, st->mode->prob, error, &enc);
   coarse_needed = ((coarse_needed*3-1)>>3)+1;

   /* Variable bitrate */
   if (st->VBR_rate>0)
   {
     /* The target rate in 16th bits per frame */
     int target=st->VBR_rate;
   
     /* Shortblocks get a large boost in bitrate, but since they 
        are uncommon long blocks are not greatly effected */
     if (shortBlocks)
       target*=2;
     else if (st->mode->nbShortMdcts > 1)
       target-=(target+14)/28;     

     /* The average energy is removed from the target and the actual 
        energy added*/
     target=target-588+ec_enc_tell(&enc, 4);

     /* In VBR mode the frame size must not be reduced so much that it would result in the coarse energy busting its budget */
     target=IMAX(coarse_needed,(target+64)/128);
     nbCompressedBytes=IMIN(nbCompressedBytes,target);
   }

   ALLOC(offsets, st->mode->nbEBands, int);
   ALLOC(fine_priority, st->mode->nbEBands, int);

   for (i=0;i<st->mode->nbEBands;i++)
      offsets[i] = 0;
   bits = nbCompressedBytes*8 - ec_enc_tell(&enc, 0) - 1;
   if (has_pitch)
      bits -= st->mode->nbPBands;
#ifndef STDIN_TUNING
   compute_allocation(st->mode, offsets, bits, pulses, fine_quant, fine_priority);
#endif

   quant_fine_energy(st->mode, bandE, st->oldBandE, error, fine_quant, &enc);

   /* Residual quantisation */
   if (C==1)
      quant_bands(st->mode, X, P, NULL, has_pitch, gains, bandE, pulses, shortBlocks, has_fold, nbCompressedBytes*8, &enc);
#ifndef DISABLE_STEREO
   else
      quant_bands_stereo(st->mode, X, P, NULL, has_pitch, gains, bandE, pulses, shortBlocks, has_fold, nbCompressedBytes*8, &enc);
#endif

   quant_energy_finalise(st->mode, bandE, st->oldBandE, error, fine_quant, fine_priority, nbCompressedBytes*8-ec_enc_tell(&enc, 0), &enc);

   /* Re-synthesis of the coded audio if required */
   if (st->pitch_available>0 || optional_synthesis!=NULL)
   {
      if (st->pitch_available>0 && st->pitch_available<MAX_PERIOD)
        st->pitch_available+=st->frame_size;

      /* Synthesis */
      denormalise_bands(st->mode, X, freq, bandE);
      
      
      CELT_MOVE(st->out_mem, st->out_mem+C*N, C*(MAX_PERIOD+st->overlap-N));
      
      if (mdct_weight_shift)
      {
         int m;
         for (c=0;c<C;c++)
            for (m=mdct_weight_pos+1;m<st->mode->nbShortMdcts;m++)
               for (i=m+c*N;i<(c+1)*N;i+=st->mode->nbShortMdcts)
#ifdef FIXED_POINT
                  freq[i] = SHL32(freq[i], mdct_weight_shift);
#else
                  freq[i] = (1<<mdct_weight_shift)*freq[i];
#endif
      }
      compute_inv_mdcts(st->mode, shortBlocks, freq, transient_time, transient_shift, st->out_mem);
      /* De-emphasis and put everything back at the right place 
         in the synthesis history */
      if (optional_synthesis != NULL) {
         for (c=0;c<C;c++)
         {
            int j;
            for (j=0;j<N;j++)
            {
               celt_sig_t tmp = MAC16_32_Q15(st->out_mem[C*(MAX_PERIOD-N)+C*j+c],
                                   preemph,st->preemph_memD[c]);
               st->preemph_memD[c] = tmp;
               optional_synthesis[C*j+c] = SCALEOUT(SIG2WORD16(tmp));
            }
         }
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
   VARDECL(celt_int16_t, in);

   if (check_encoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   SAVE_STACK;
   C = CHANNELS(st->mode);
   N = st->block_size;
   ALLOC(in, C*N, celt_int16_t);

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
int celt_encode(CELTEncoder * restrict st, const celt_int16_t * pcm, celt_int16_t * optional_synthesis, unsigned char *compressed, int nbCompressedBytes)
{
   int j, ret, C, N;
   VARDECL(celt_sig_t, in);

   if (check_encoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   SAVE_STACK;
   C=CHANNELS(st->mode);
   N=st->block_size;
   ALLOC(in, C*N, celt_sig_t);
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
         int value = va_arg(ap, celt_int32_t);
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
         int value = va_arg(ap, celt_int32_t);
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
         int value = va_arg(ap, celt_int32_t);
         if (value<0)
            goto bad_arg;
         if (value>3072000)
            value = 3072000;
         st->VBR_rate = ((st->mode->Fs<<3)+(st->block_size>>1))/st->block_size;
         st->VBR_rate = ((value<<7)+(st->VBR_rate>>1))/st->VBR_rate;
      }
      break;
      case CELT_RESET_STATE:
      {
         const CELTMode *mode = st->mode;
         int C = mode->nbChannels;

         if (st->pitch_available > 0) st->pitch_available = 1;

         CELT_MEMSET(st->in_mem, 0, st->overlap*C);
         CELT_MEMSET(st->out_mem, 0, (MAX_PERIOD+st->overlap)*C);

         CELT_MEMSET(st->oldBandE, 0, C*mode->nbEBands);

         CELT_MEMSET(st->preemph_memE, 0, C);
         CELT_MEMSET(st->preemph_memD, 0, C);
         st->delayedIntra = 1;
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
   celt_uint32_t marker;
   const CELTMode *mode;
   int frame_size;
   int block_size;
   int overlap;

   ec_byte_buffer buf;
   ec_enc         enc;

   celt_sig_t * restrict preemph_memD;

   celt_sig_t *out_mem;
   celt_sig_t *decode_mem;

   celt_word16_t *oldBandE;
   
   int last_pitch_index;
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

CELTDecoder *celt_decoder_create(const CELTMode *mode)
{
   int N, C;
   CELTDecoder *st;

   if (check_mode(mode) != CELT_OK)
      return NULL;

   N = mode->mdctSize;
   C = CHANNELS(mode);
   st = celt_alloc(sizeof(CELTDecoder));

   if (st==NULL)
      return NULL;
   
   st->marker = DECODERPARTIAL;
   st->mode = mode;
   st->frame_size = N;
   st->block_size = N;
   st->overlap = mode->overlap;

   st->decode_mem = celt_alloc((DECODE_BUFFER_SIZE+st->overlap)*C*sizeof(celt_sig_t));
   st->out_mem = st->decode_mem+DECODE_BUFFER_SIZE-MAX_PERIOD;
   
   st->oldBandE = (celt_word16_t*)celt_alloc(C*mode->nbEBands*sizeof(celt_word16_t));
   
   st->preemph_memD = (celt_sig_t*)celt_alloc(C*sizeof(celt_sig_t));

   st->last_pitch_index = 0;

   if ((st->decode_mem!=NULL) && (st->out_mem!=NULL) && (st->oldBandE!=NULL) &&
       (st->preemph_memD!=NULL))
   {
      st->marker = DECODERVALID;
      return st;
   }
   /* If the setup fails for some reason deallocate it. */
   celt_decoder_destroy(st);
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
   
   st->marker = DECODERFREED;
   
   celt_free(st);
}

/** Handles lost packets by just copying past data with the same
    offset as the last
    pitch period */
#ifdef NEW_PLC
#include "plc.c"
#else
static void celt_decode_lost(CELTDecoder * restrict st, celt_word16_t * restrict pcm)
{
   int c, N;
   int pitch_index;
   int i, len;
   VARDECL(celt_sig_t, freq);
   const int C = CHANNELS(st->mode);
   int offset;
   SAVE_STACK;
   N = st->block_size;
   ALLOC(freq,C*N, celt_sig_t); /**< Interleaved signal MDCTs */
   
   len = N+st->mode->overlap;
#if 0
   pitch_index = st->last_pitch_index;
   
   /* Use the pitch MDCT as the "guessed" signal */
   compute_mdcts(st->mode, st->mode->window, st->out_mem+pitch_index*C, freq);

#else
   find_spectral_pitch(st->mode, st->mode->fft, &st->mode->psy, st->out_mem+MAX_PERIOD-len, st->out_mem, st->mode->window, NULL, len, MAX_PERIOD-len-100, &pitch_index);
   pitch_index = MAX_PERIOD-len-pitch_index;
   offset = MAX_PERIOD-pitch_index;
   while (offset+len >= MAX_PERIOD)
      offset -= pitch_index;
   compute_mdcts(st->mode, 0, st->out_mem+offset*C, freq);
   for (i=0;i<C*N;i++)
      freq[i] = ADD32(EPSILON, MULT16_32_Q15(QCONST16(.9f,15),freq[i]));
#endif
   
   
   
   CELT_MOVE(st->out_mem, st->out_mem+C*N, C*(MAX_PERIOD+st->mode->overlap-N));
   /* Compute inverse MDCTs */
   compute_inv_mdcts(st->mode, 0, freq, -1, 0, st->out_mem);

   for (c=0;c<C;c++)
   {
      int j;
      for (j=0;j<N;j++)
      {
         celt_sig_t tmp = MAC16_32_Q15(st->out_mem[C*(MAX_PERIOD-N)+C*j+c],
                                preemph,st->preemph_memD[c]);
         st->preemph_memD[c] = tmp;
         pcm[C*j+c] = SCALEOUT(SIG2WORD16(tmp));
      }
   }
   RESTORE_STACK;
}
#endif

#ifdef FIXED_POINT
int celt_decode(CELTDecoder * restrict st, const unsigned char *data, int len, celt_int16_t * restrict pcm)
{
#else
int celt_decode_float(CELTDecoder * restrict st, const unsigned char *data, int len, celt_sig_t * restrict pcm)
{
#endif
   int i, c, N, N4;
   int has_pitch, has_fold;
   int pitch_index;
   int bits;
   ec_dec dec;
   ec_byte_buffer buf;
   VARDECL(celt_sig_t, freq);
   VARDECL(celt_norm_t, X);
   VARDECL(celt_norm_t, P);
   VARDECL(celt_ener_t, bandE);
   VARDECL(celt_pgain_t, gains);
   VARDECL(int, fine_quant);
   VARDECL(int, pulses);
   VARDECL(int, offsets);
   VARDECL(int, fine_priority);

   int shortBlocks;
   int intra_ener;
   int transient_time;
   int transient_shift;
   int mdct_weight_shift=0;
   const int C = CHANNELS(st->mode);
   int mdct_weight_pos=0;
   SAVE_STACK;

   if (check_decoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   N = st->block_size;
   N4 = (N-st->overlap)>>1;

   ALLOC(freq, C*N, celt_sig_t); /**< Interleaved signal MDCTs */
   ALLOC(X, C*N, celt_norm_t);   /**< Interleaved normalised MDCTs */
   ALLOC(P, C*N, celt_norm_t);   /**< Interleaved normalised pitch MDCTs*/
   ALLOC(bandE, st->mode->nbEBands*C, celt_ener_t);
   ALLOC(gains, st->mode->nbPBands, celt_pgain_t);
   
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
      transient_shift = ec_dec_bits(&dec, 2);
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
      st->last_pitch_index = pitch_index;
   } else {
      pitch_index = 0;
      for (i=0;i<st->mode->nbPBands;i++)
         gains[i] = 0;
   }

   ALLOC(fine_quant, st->mode->nbEBands, int);
   /* Get band energies */
   unquant_coarse_energy(st->mode, bandE, st->oldBandE, len*8/3, intra_ener, st->mode->prob, &dec);
   
   ALLOC(pulses, st->mode->nbEBands, int);
   ALLOC(offsets, st->mode->nbEBands, int);
   ALLOC(fine_priority, st->mode->nbEBands, int);

   for (i=0;i<st->mode->nbEBands;i++)
      offsets[i] = 0;

   bits = len*8 - ec_dec_tell(&dec, 0) - 1;
   if (has_pitch)
      bits -= st->mode->nbPBands;
   compute_allocation(st->mode, offsets, bits, pulses, fine_quant, fine_priority);
   /*bits = ec_dec_tell(&dec, 0);
   compute_fine_allocation(st->mode, fine_quant, (20*C+len*8/5-(ec_dec_tell(&dec, 0)-bits))/C);*/
   
   unquant_fine_energy(st->mode, bandE, st->oldBandE, fine_quant, &dec);


   if (has_pitch) 
   {
      VARDECL(celt_ener_t, bandEp);
      
      /* Pitch MDCT */
      compute_mdcts(st->mode, 0, st->out_mem+pitch_index*C, freq);
      ALLOC(bandEp, st->mode->nbEBands*C, celt_ener_t);
      compute_band_energies(st->mode, freq, bandEp);
      normalise_bands(st->mode, freq, P, bandEp);
      /* Apply pitch gains */
   } else {
      for (i=0;i<C*N;i++)
         P[i] = 0;
   }

   /* Decode fixed codebook and merge with pitch */
   if (C==1)
      unquant_bands(st->mode, X, P, has_pitch, gains, bandE, pulses, shortBlocks, has_fold, len*8, &dec);
#ifndef DISABLE_STEREO
   else
      unquant_bands_stereo(st->mode, X, P, has_pitch, gains, bandE, pulses, shortBlocks, has_fold, len*8, &dec);
#endif
   unquant_energy_finalise(st->mode, bandE, st->oldBandE, fine_quant, fine_priority, len*8-ec_dec_tell(&dec, 0), &dec);
   
   /* Synthesis */
   denormalise_bands(st->mode, X, freq, bandE);


   CELT_MOVE(st->decode_mem, st->decode_mem+C*N, C*(DECODE_BUFFER_SIZE+st->overlap-N));
   if (mdct_weight_shift)
   {
      int m;
      for (c=0;c<C;c++)
         for (m=mdct_weight_pos+1;m<st->mode->nbShortMdcts;m++)
            for (i=m+c*N;i<(c+1)*N;i+=st->mode->nbShortMdcts)
#ifdef FIXED_POINT
               freq[i] = SHL32(freq[i], mdct_weight_shift);
#else
               freq[i] = (1<<mdct_weight_shift)*freq[i];
#endif
   }
   /* Compute inverse MDCTs */
   compute_inv_mdcts(st->mode, shortBlocks, freq, transient_time, transient_shift, st->out_mem);

   for (c=0;c<C;c++)
   {
      int j;
      for (j=0;j<N;j++)
      {
         celt_sig_t tmp = MAC16_32_Q15(st->out_mem[C*(MAX_PERIOD-N)+C*j+c],
                                preemph,st->preemph_memD[c]);
         st->preemph_memD[c] = tmp;
         pcm[C*j+c] = SCALEOUT(SIG2WORD16(tmp));
      }
   }

   RESTORE_STACK;
   return 0;
}

#ifdef FIXED_POINT
#ifndef DISABLE_FLOAT_API
int celt_decode_float(CELTDecoder * restrict st, const unsigned char *data, int len, float * restrict pcm)
{
   int j, ret, C, N;
   VARDECL(celt_int16_t, out);

   if (check_decoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   SAVE_STACK;
   C = CHANNELS(st->mode);
   N = st->block_size;
   
   ALLOC(out, C*N, celt_int16_t);
   ret=celt_decode(st, data, len, out);
   for (j=0;j<C*N;j++)
      pcm[j]=out[j]*(1/32768.);
     
   RESTORE_STACK;
   return ret;
}
#endif /*DISABLE_FLOAT_API*/
#else
int celt_decode(CELTDecoder * restrict st, const unsigned char *data, int len, celt_int16_t * restrict pcm)
{
   int j, ret, C, N;
   VARDECL(celt_sig_t, out);

   if (check_decoder(st) != CELT_OK)
      return CELT_INVALID_STATE;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   if (pcm==NULL)
      return CELT_BAD_ARG;

   SAVE_STACK;
   C = CHANNELS(st->mode);
   N = st->block_size;
   ALLOC(out, C*N, celt_sig_t);

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
         int C = mode->nbChannels;

         CELT_MEMSET(st->decode_mem, 0, (DECODE_BUFFER_SIZE+st->overlap)*C);
         CELT_MEMSET(st->oldBandE, 0, C*mode->nbEBands);

         CELT_MEMSET(st->preemph_memD, 0, C);

         st->last_pitch_index = 0;
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
