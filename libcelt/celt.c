/* (C) 2007-2008 Jean-Marc Valin, CSIRO
*/
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
#include "quant_pitch.h"
#include "quant_bands.h"
#include "psy.h"
#include "rate.h"
#include "stack_alloc.h"
#include "mathops.h"

static const celt_word16_t preemph = QCONST16(0.8f,15);

#ifdef FIXED_POINT
static const celt_word16_t transientWindow[16] = {
     279,  1106,  2454,  4276,  6510,  9081, 11900, 14872,
   17896, 20868, 23687, 26258, 28492, 30314, 31662, 32489};
#else
static const float transientWindow[16] = {
   0.0085135, 0.0337639, 0.0748914, 0.1304955, 0.1986827, 0.2771308, 0.3631685, 0.4538658,
   0.5461342, 0.6368315, 0.7228692, 0.8013173, 0.8695045, 0.9251086, 0.9662361, 0.9914865};
#endif

/** Encoder state 
 @brief Encoder state
 */
struct CELTEncoder {
   const CELTMode *mode;     /**< Mode used by the encoder */
   int frame_size;
   int block_size;
   int overlap;
   int channels;
   
   ec_byte_buffer buf;
   ec_enc         enc;

   celt_word16_t * restrict preemph_memE; /* Input is 16-bit, so why bother with 32 */
   celt_sig_t    * restrict preemph_memD;

   celt_sig_t *in_mem;
   celt_sig_t *out_mem;

   celt_word16_t *oldBandE;
#ifdef EXP_PSY
   celt_word16_t *psy_mem;
   struct PsyDecay psy;
#endif
};

CELTEncoder *celt_encoder_create(const CELTMode *mode)
{
   int N, C;
   CELTEncoder *st;

   if (check_mode(mode) != CELT_OK)
      return NULL;

   N = mode->mdctSize;
   C = mode->nbChannels;
   st = celt_alloc(sizeof(CELTEncoder));
   
   st->mode = mode;
   st->frame_size = N;
   st->block_size = N;
   st->overlap = mode->overlap;

   ec_byte_writeinit(&st->buf);
   ec_enc_init(&st->enc,&st->buf);

   st->in_mem = celt_alloc(st->overlap*C*sizeof(celt_sig_t));
   st->out_mem = celt_alloc((MAX_PERIOD+st->overlap)*C*sizeof(celt_sig_t));

   st->oldBandE = (celt_word16_t*)celt_alloc(C*mode->nbEBands*sizeof(celt_word16_t));

   st->preemph_memE = (celt_word16_t*)celt_alloc(C*sizeof(celt_word16_t));;
   st->preemph_memD = (celt_sig_t*)celt_alloc(C*sizeof(celt_sig_t));;

#ifdef EXP_PSY
   st->psy_mem = celt_alloc(MAX_PERIOD*sizeof(celt_word16_t));
   psydecay_init(&st->psy, MAX_PERIOD/2, st->mode->Fs);
#endif

   return st;
}

void celt_encoder_destroy(CELTEncoder *st)
{
   if (st == NULL)
   {
      celt_warning("NULL passed to celt_encoder_destroy");
      return;
   }
   if (check_mode(st->mode) != CELT_OK)
      return;

   ec_byte_writeclear(&st->buf);

   celt_free(st->in_mem);
   celt_free(st->out_mem);
   
   celt_free(st->oldBandE);
   
   celt_free(st->preemph_memE);
   celt_free(st->preemph_memD);
   
#ifdef EXP_PSY
   celt_free (st->psy_mem);
   psydecay_clear(&st->psy);
#endif
   
   celt_free(st);
}

static inline celt_int16_t SIG2INT16(celt_sig_t x)
{
   x = PSHR32(x, SIG_SHIFT);
   x = MAX32(x, -32768);
   x = MIN32(x, 32767);
#ifdef FIXED_POINT
   return EXTRACT16(x);
#else
   return (celt_int16_t)floor(.5+x);
#endif
}
#ifdef FIXED_POINT
static int ratio_compare(celt_word32_t num1, celt_word32_t den1, celt_word32_t num2, celt_word32_t den2)
{
   int shift = celt_zlog2(MAX32(num1, num2));
   if (shift > 14)
   {
      num1 = SHR32(num1, shift-14);
      num2 = SHR32(num2, shift-14);
   }
   shift = celt_zlog2(MAX32(den1, den2));
   if (shift > 14)
   {
      den1 = SHR32(den1, shift-14);
      den2 = SHR32(den2, shift-14);
   }
   return MULT16_16(EXTRACT16(num1),EXTRACT16(den2)) > MULT16_16(EXTRACT16(den1),EXTRACT16(num2));
}
#else
static int ratio_compare(celt_word32_t num1, celt_word32_t den1, celt_word32_t num2, celt_word32_t den2)
{
   return num1*den2 > den1*num2;
}
#endif

static int transient_analysis(celt_word32_t *in, int len, int C, celt_word32_t *r)
{
   int c, i, n;
   celt_word32_t ratio;
   /* FIXME: Remove the floats here */
   celt_word32_t maxN, maxD;
   VARDECL(celt_word32_t, begin);
   SAVE_STACK;
   ALLOC(begin, len, celt_word32_t);
   
   for (i=0;i<len;i++)
      begin[i] = EXTEND32(ABS16(SHR32(in[C*i],SIG_SHIFT)));
   for (c=1;c<C;c++)
   {
      for (i=0;i<len;i++)
         begin[i] = ADD32(begin[i], EXTEND32(ABS16(SHR32(in[C*i+c],SIG_SHIFT))));
   }
   for (i=1;i<len;i++)
      begin[i] = begin[i-1]+begin[i];

   maxD = VERY_LARGE32;
   maxN = 0;
   n = -1;
   for (i=8;i<len-8;i++)
   {
      celt_word32_t endi;
      celt_word32_t num, den;
      endi = begin[len-1]-begin[i];
      num = endi*i;
      den = (30+begin[i])*(len-i)+MULT16_32_Q15(QCONST16(.1f,15),endi)*len;
      if (ratio_compare(num, den, maxN, maxD) && (endi > MULT16_32_Q15(QCONST16(.05f,15),begin[i])))
      {
         maxN = num;
         maxD = den;
         n = i;
      }
   }
   ratio = DIV32((begin[len-1]-begin[n])*n,(10+begin[n])*(len-n));
   if (n<32)
   {
      n = -1;
      ratio = 0;
   }
   if (ratio < 0)
      ratio = 0;
   if (ratio > 1000)
      ratio = 1000;
   *r = ratio*ratio;
   RESTORE_STACK;
   return n;
}

/** Apply window and compute the MDCT for all sub-frames and all channels in a frame */
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
            out[C*j+c] = tmp[j];
      }
      RESTORE_STACK;
   } else {
      const mdct_lookup *lookup = &mode->shortMdct;
      const int overlap = mode->shortMdctSize;
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
               out[C*(j*B+b)+c] = tmp[j];
         }
      }
      RESTORE_STACK;
   }
}

/** Compute the IMDCT and apply window for all sub-frames and all channels in a frame */
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
            tmp[j] = X[C*j+c];
         /* Prevents problems from the imdct doing the overlap-add */
         CELT_MEMSET(x+N4, 0, overlap);
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
         CELT_MEMSET(x+N4, 0, overlap);
         for (b=0;b<B;b++)
         {
            /* De-interleaving the sub-frames */
            for (j=0;j<N2;j++)
               tmp[j] = X[C*(j*B+b)+c];
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
         /* The first and last part would need to be set to zero if we actually
         wanted to use them. */
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

int celt_encode(CELTEncoder * restrict st, celt_int16_t * restrict pcm, unsigned char *compressed, int nbCompressedBytes)
{
   int i, c, N, N4;
   int has_pitch;
   int pitch_index;
   celt_word32_t curr_power, pitch_power;
   VARDECL(celt_sig_t, in);
   VARDECL(celt_sig_t, freq);
   VARDECL(celt_norm_t, X);
   VARDECL(celt_norm_t, P);
   VARDECL(celt_ener_t, bandE);
   VARDECL(celt_pgain_t, gains);
   VARDECL(int, stereo_mode);
#ifdef EXP_PSY
   VARDECL(celt_word32_t, mask);
#endif
   int shortBlocks=0;
   int transient_time;
   int transient_shift;
   celt_word32_t maxR;
   const int C = CHANNELS(st->mode);
   SAVE_STACK;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   N = st->block_size;
   N4 = (N-st->overlap)>>1;
   ALLOC(in, 2*C*N-2*C*N4, celt_sig_t);

   CELT_COPY(in, st->in_mem, C*st->overlap);
   for (c=0;c<C;c++)
   {
      const celt_int16_t * restrict pcmp = pcm+c;
      celt_sig_t * restrict inp = in+C*st->overlap+c;
      for (i=0;i<N;i++)
      {
         /* Apply pre-emphasis */
         celt_sig_t tmp = SHL32(EXTEND32(*pcmp), SIG_SHIFT);
         *inp = SUB32(tmp, SHR32(MULT16_16(preemph,st->preemph_memE[c]),1));
         st->preemph_memE[c] = *pcmp;
         inp += C;
         pcmp += C;
      }
   }
   CELT_COPY(st->in_mem, in+C*(2*N-2*N4-st->overlap), C*st->overlap);
   
   transient_time = transient_analysis(in, N+st->overlap, C, &maxR);
   if (maxR > 30)
   {
#ifndef FIXED_POINT
      float gain_1;
#endif
      ec_enc_bits(&st->enc, 1, 1);
      if (maxR < 30)
      {
         transient_shift = 0;
      } else if (maxR < 100)
      {
         transient_shift = 1;
      } else if (maxR < 500)
      {
         transient_shift = 2;
      } else
      {
         transient_shift = 3;
      }
      ec_enc_bits(&st->enc, transient_shift, 2);
      if (transient_shift)
         ec_enc_uint(&st->enc, transient_time, N+st->overlap);
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
   } else {
      ec_enc_bits(&st->enc, 0, 1);
      transient_time = -1;
      transient_shift = 0;
      shortBlocks = 0;
   }
   /* Pitch analysis: we do it early to save on the peak stack space */
   if (!shortBlocks)
      find_spectral_pitch(st->mode, st->mode->fft, &st->mode->psy, in, st->out_mem, st->mode->window, 2*N-2*N4, MAX_PERIOD-(2*N-2*N4), &pitch_index);

   ALLOC(freq, C*N, celt_sig_t); /**< Interleaved signal MDCTs */
   
   /*for (i=0;i<(B+1)*C*N;i++) printf ("%f(%d) ", in[i], i); printf ("\n");*/
   /* Compute MDCTs */
   compute_mdcts(st->mode, shortBlocks, in, freq);

#ifdef EXP_PSY
   CELT_MOVE(st->psy_mem, st->out_mem+N, MAX_PERIOD+st->overlap-N);
   for (i=0;i<N;i++)
      st->psy_mem[MAX_PERIOD+st->overlap-N+i] = in[C*(st->overlap+i)];
   for (c=1;c<C;c++)
      for (i=0;i<N;i++)
         st->psy_mem[MAX_PERIOD+st->overlap-N+i] += in[C*(st->overlap+i)+c];

   ALLOC(mask, N, celt_sig_t);
   compute_mdct_masking(&st->psy, freq, st->psy_mem, mask, C*N);

   /* Invert and stretch the mask to length of X 
      For some reason, I get better results by using the sqrt instead,
      although there's no valid reason to. Must investigate further */
   for (i=0;i<C*N;i++)
      mask[i] = 1/(.1+mask[i]);
#endif
   
   /* Deferred allocation after find_spectral_pitch() to reduce the peak memory usage */
   ALLOC(X, C*N, celt_norm_t);         /**< Interleaved normalised MDCTs */
   ALLOC(P, C*N, celt_norm_t);         /**< Interleaved normalised pitch MDCTs*/
   ALLOC(bandE,st->mode->nbEBands*C, celt_ener_t);
   ALLOC(gains,st->mode->nbPBands, celt_pgain_t);

   /*printf ("%f %f\n", curr_power, pitch_power);*/
   /*int j;
   for (j=0;j<B*N;j++)
      printf ("%f ", X[j]);
   for (j=0;j<B*N;j++)
      printf ("%f ", P[j]);
   printf ("\n");*/

   /* Band normalisation */
   compute_band_energies(st->mode, freq, bandE);
   normalise_bands(st->mode, freq, X, bandE);
   /*for (i=0;i<st->mode->nbEBands;i++)printf("%f ", bandE[i]);printf("\n");*/
   /*for (i=0;i<N*B*C;i++)printf("%f ", X[i]);printf("\n");*/

   /* Compute MDCTs of the pitch part */
   if (!shortBlocks)
      compute_mdcts(st->mode, 0, st->out_mem+pitch_index*C, freq);

   {
      /* Normalise the pitch vector as well (discard the energies) */
      VARDECL(celt_ener_t, bandEp);
      ALLOC(bandEp, st->mode->nbEBands*st->mode->nbChannels, celt_ener_t);
      compute_band_energies(st->mode, freq, bandEp);
      normalise_bands(st->mode, freq, P, bandEp);
      pitch_power = bandEp[0]+bandEp[1]+bandEp[2];
   }
   curr_power = bandE[0]+bandE[1]+bandE[2];
   /* Check if we can safely use the pitch (i.e. effective gain isn't too high) */
   if (!shortBlocks && (MULT16_32_Q15(QCONST16(.1f, 15),curr_power) + QCONST32(10.f,ENER_SHIFT) < pitch_power))
   {
      /* Simulates intensity stereo */
      /*for (i=30;i<N*B;i++)
         X[i*C+1] = P[i*C+1] = 0;*/

      /* Pitch prediction */
      compute_pitch_gain(st->mode, X, P, gains);
      has_pitch = quant_pitch(gains, st->mode->nbPBands, &st->enc);
      if (has_pitch)
         ec_enc_uint(&st->enc, pitch_index, MAX_PERIOD-(2*N-2*N4));
   } else {
      /* No pitch, so we just pretend we found a gain of zero */
      for (i=0;i<st->mode->nbPBands;i++)
         gains[i] = 0;
      ec_enc_bits(&st->enc, 0, 7);
      for (i=0;i<C*N;i++)
         P[i] = 0;
   }
   quant_energy(st->mode, bandE, st->oldBandE, 20*C+nbCompressedBytes*8/5, st->mode->prob, &st->enc);

   ALLOC(stereo_mode, st->mode->nbEBands, int);
   stereo_decision(st->mode, X, stereo_mode, st->mode->nbEBands);

   pitch_quant_bands(st->mode, P, gains);

   /*for (i=0;i<B*N;i++) printf("%f ",P[i]);printf("\n");*/

   /* Residual quantisation */
   quant_bands(st->mode, X, P, NULL, bandE, stereo_mode, nbCompressedBytes*8, shortBlocks, &st->enc);
   
   if (C==2)
   {
      renormalise_bands(st->mode, X);
   }
   /* Synthesis */
   denormalise_bands(st->mode, X, freq, bandE);


   CELT_MOVE(st->out_mem, st->out_mem+C*N, C*(MAX_PERIOD+st->overlap-N));

   compute_inv_mdcts(st->mode, shortBlocks, freq, transient_time, transient_shift, st->out_mem);
   /* De-emphasis and put everything back at the right place in the synthesis history */
#ifndef SHORTCUTS
   for (c=0;c<C;c++)
   {
      int j;
      celt_sig_t * restrict outp=st->out_mem+C*(MAX_PERIOD-N)+c;
      celt_int16_t * restrict pcmp = pcm+c;
      for (j=0;j<N;j++)
      {
         celt_sig_t tmp = ADD32(*outp, MULT16_32_Q15(preemph,st->preemph_memD[c]));
         st->preemph_memD[c] = tmp;
         *pcmp = SIG2INT16(tmp);
         pcmp += C;
         outp += C;
      }
   }
#endif
   if (ec_enc_tell(&st->enc, 0) < nbCompressedBytes*8 - 7)
      celt_warning_int ("many unused bits: ", nbCompressedBytes*8-ec_enc_tell(&st->enc, 0));
   /*printf ("%d\n", ec_enc_tell(&st->enc, 0)-8*nbCompressedBytes);*/
   /* Finishing the stream with a 0101... pattern so that the decoder can check is everything's right */
   {
      int val = 0;
      while (ec_enc_tell(&st->enc, 0) < nbCompressedBytes*8)
      {
         ec_enc_uint(&st->enc, val, 2);
         val = 1-val;
      }
   }
   ec_enc_done(&st->enc);
   {
      unsigned char *data;
      int nbBytes = ec_byte_bytes(&st->buf);
      if (nbBytes > nbCompressedBytes)
      {
         celt_warning_int ("got too many bytes:", nbBytes);
         RESTORE_STACK;
         return CELT_INTERNAL_ERROR;
      }
      /*printf ("%d\n", *nbBytes);*/
      data = ec_byte_get_buffer(&st->buf);
      for (i=0;i<nbBytes;i++)
         compressed[i] = data[i];
      for (;i<nbCompressedBytes;i++)
         compressed[i] = 0;
   }
   /* Reset the packing for the next encoding */
   ec_byte_reset(&st->buf);
   ec_enc_init(&st->enc,&st->buf);

   RESTORE_STACK;
   return nbCompressedBytes;
}


/****************************************************************************/
/*                                                                          */
/*                                DECODER                                   */
/*                                                                          */
/****************************************************************************/


/** Decoder state 
 @brief Decoder state
 */
struct CELTDecoder {
   const CELTMode *mode;
   int frame_size;
   int block_size;
   int overlap;

   ec_byte_buffer buf;
   ec_enc         enc;

   celt_sig_t * restrict preemph_memD;

   celt_sig_t *out_mem;

   celt_word16_t *oldBandE;
   
   int last_pitch_index;
};

CELTDecoder *celt_decoder_create(const CELTMode *mode)
{
   int N, C;
   CELTDecoder *st;

   if (check_mode(mode) != CELT_OK)
      return NULL;

   N = mode->mdctSize;
   C = CHANNELS(mode);
   st = celt_alloc(sizeof(CELTDecoder));
   
   st->mode = mode;
   st->frame_size = N;
   st->block_size = N;
   st->overlap = mode->overlap;

   st->out_mem = celt_alloc((MAX_PERIOD+st->overlap)*C*sizeof(celt_sig_t));
   
   st->oldBandE = (celt_word16_t*)celt_alloc(C*mode->nbEBands*sizeof(celt_word16_t));

   st->preemph_memD = (celt_sig_t*)celt_alloc(C*sizeof(celt_sig_t));;

   st->last_pitch_index = 0;
   return st;
}

void celt_decoder_destroy(CELTDecoder *st)
{
   if (st == NULL)
   {
      celt_warning("NULL passed to celt_encoder_destroy");
      return;
   }
   if (check_mode(st->mode) != CELT_OK)
      return;


   celt_free(st->out_mem);
   
   celt_free(st->oldBandE);
   
   celt_free(st->preemph_memD);

   celt_free(st);
}

/** Handles lost packets by just copying past data with the same offset as the last
    pitch period */
static void celt_decode_lost(CELTDecoder * restrict st, short * restrict pcm)
{
   int c, N;
   int pitch_index;
   int i, len;
   VARDECL(celt_sig_t, freq);
   const int C = CHANNELS(st->mode);
   int offset;
   SAVE_STACK;
   N = st->block_size;
   ALLOC(freq,C*N, celt_sig_t);         /**< Interleaved signal MDCTs */
   
   len = N+st->mode->overlap;
#if 0
   pitch_index = st->last_pitch_index;
   
   /* Use the pitch MDCT as the "guessed" signal */
   compute_mdcts(st->mode, st->mode->window, st->out_mem+pitch_index*C, freq);

#else
   find_spectral_pitch(st->mode, st->mode->fft, &st->mode->psy, st->out_mem+MAX_PERIOD-len, st->out_mem, st->mode->window, len, MAX_PERIOD-len-100, &pitch_index);
   pitch_index = MAX_PERIOD-len-pitch_index;
   offset = MAX_PERIOD-pitch_index;
   while (offset+len >= MAX_PERIOD)
      offset -= pitch_index;
   compute_mdcts(st->mode, 0, st->out_mem+offset*C, freq);
   for (i=0;i<N;i++)
      freq[i] = MULT16_32_Q15(QCONST16(.9f,15),freq[i]);
#endif
   
   
   
   CELT_MOVE(st->out_mem, st->out_mem+C*N, C*(MAX_PERIOD+st->mode->overlap-N));
   /* Compute inverse MDCTs */
   compute_inv_mdcts(st->mode, 0, freq, -1, 1, st->out_mem);

   for (c=0;c<C;c++)
   {
      int j;
      for (j=0;j<N;j++)
      {
         celt_sig_t tmp = ADD32(st->out_mem[C*(MAX_PERIOD-N)+C*j+c],
                                MULT16_32_Q15(preemph,st->preemph_memD[c]));
         st->preemph_memD[c] = tmp;
         pcm[C*j+c] = SIG2INT16(tmp);
      }
   }
   RESTORE_STACK;
}

int celt_decode(CELTDecoder * restrict st, unsigned char *data, int len, celt_int16_t * restrict pcm)
{
   int c, N, N4;
   int has_pitch;
   int pitch_index;
   ec_dec dec;
   ec_byte_buffer buf;
   VARDECL(celt_sig_t, freq);
   VARDECL(celt_norm_t, X);
   VARDECL(celt_norm_t, P);
   VARDECL(celt_ener_t, bandE);
   VARDECL(celt_pgain_t, gains);
   VARDECL(int, stereo_mode);
   int shortBlocks;
   int transient_time;
   int transient_shift;
   const int C = CHANNELS(st->mode);
   SAVE_STACK;

   if (check_mode(st->mode) != CELT_OK)
      return CELT_INVALID_MODE;

   N = st->block_size;
   N4 = (N-st->overlap)>>1;

   ALLOC(freq, C*N, celt_sig_t); /**< Interleaved signal MDCTs */
   ALLOC(X, C*N, celt_norm_t);         /**< Interleaved normalised MDCTs */
   ALLOC(P, C*N, celt_norm_t);         /**< Interleaved normalised pitch MDCTs*/
   ALLOC(bandE, st->mode->nbEBands*C, celt_ener_t);
   ALLOC(gains, st->mode->nbPBands, celt_pgain_t);
   
   if (check_mode(st->mode) != CELT_OK)
   {
      RESTORE_STACK;
      return CELT_INVALID_MODE;
   }
   if (data == NULL)
   {
      celt_decode_lost(st, pcm);
      RESTORE_STACK;
      return 0;
   }
   
   ec_byte_readinit(&buf,data,len);
   ec_dec_init(&dec,&buf);
   
   shortBlocks = ec_dec_bits(&dec, 1);
   if (shortBlocks)
   {
      transient_shift = ec_dec_bits(&dec, 2);
      if (transient_shift)
         transient_time = ec_dec_uint(&dec, N+st->mode->overlap);
      else
         transient_time = 0;
   } else {
      transient_time = -1;
      transient_shift = 0;
   }
   /* Get the pitch gains */
   has_pitch = unquant_pitch(gains, st->mode->nbPBands, &dec);
   
   /* Get the pitch index */
   if (has_pitch)
   {
      pitch_index = ec_dec_uint(&dec, MAX_PERIOD-(2*N-2*N4));
      st->last_pitch_index = pitch_index;
   } else {
      /* FIXME: We could be more intelligent here and just not compute the MDCT */
      pitch_index = 0;
   }

   /* Get band energies */
   unquant_energy(st->mode, bandE, st->oldBandE, 20*C+len*8/5, st->mode->prob, &dec);

   /* Pitch MDCT */
   compute_mdcts(st->mode, 0, st->out_mem+pitch_index*C, freq);

   {
      VARDECL(celt_ener_t, bandEp);
      ALLOC(bandEp, st->mode->nbEBands*C, celt_ener_t);
      compute_band_energies(st->mode, freq, bandEp);
      normalise_bands(st->mode, freq, P, bandEp);
   }

   ALLOC(stereo_mode, st->mode->nbEBands, int);
   stereo_decision(st->mode, X, stereo_mode, st->mode->nbEBands);
   /* Apply pitch gains */
   pitch_quant_bands(st->mode, P, gains);

   /* Decode fixed codebook and merge with pitch */
   unquant_bands(st->mode, X, P, bandE, stereo_mode, len*8, shortBlocks, &dec);

   if (C==2)
   {
      renormalise_bands(st->mode, X);
   }
   /* Synthesis */
   denormalise_bands(st->mode, X, freq, bandE);


   CELT_MOVE(st->out_mem, st->out_mem+C*N, C*(MAX_PERIOD+st->overlap-N));
   /* Compute inverse MDCTs */
   compute_inv_mdcts(st->mode, shortBlocks, freq, transient_time, transient_shift, st->out_mem);

   for (c=0;c<C;c++)
   {
      int j;
      const celt_sig_t * restrict outp=st->out_mem+C*(MAX_PERIOD-N)+c;
      celt_int16_t * restrict pcmp = pcm+c;
      for (j=0;j<N;j++)
      {
         celt_sig_t tmp = ADD32(*outp, MULT16_32_Q15(preemph,st->preemph_memD[c]));
         st->preemph_memD[c] = tmp;
         *pcmp = SIG2INT16(tmp);
         pcmp += C;
         outp += C;
      }
   }

   {
      unsigned int val = 0;
      while (ec_dec_tell(&dec, 0) < len*8)
      {
         if (ec_dec_uint(&dec, 2) != val)
         {
            celt_warning("decode error");
            RESTORE_STACK;
            return CELT_CORRUPTED_DATA;
         }
         val = 1-val;
      }
   }

   RESTORE_STACK;
   return 0;
   /*printf ("\n");*/
}

