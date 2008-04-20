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

static const celt_word16_t preemph = QCONST16(0.8f,15);


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

   kiss_fftr_cfg fft;

   celt_sig_t *in_mem;
   celt_sig_t *mdct_overlap;
   celt_sig_t *out_mem;

   celt_word16_t *oldBandE;
};

CELTEncoder EXPORT *celt_encoder_create(const CELTMode *mode)
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

   st->fft = pitch_state_alloc(MAX_PERIOD);
   
   st->in_mem = celt_alloc(N*C*sizeof(celt_sig_t));
   st->mdct_overlap = celt_alloc(N*C*sizeof(celt_sig_t));
   st->out_mem = celt_alloc(MAX_PERIOD*C*sizeof(celt_sig_t));

   st->oldBandE = (celt_word16_t*)celt_alloc(C*mode->nbEBands*sizeof(celt_word16_t));

   st->preemph_memE = (celt_word16_t*)celt_alloc(C*sizeof(celt_word16_t));;
   st->preemph_memD = (celt_sig_t*)celt_alloc(C*sizeof(celt_sig_t));;

   return st;
}

void EXPORT celt_encoder_destroy(CELTEncoder *st)
{
   if (st == NULL)
   {
      celt_warning("NULL passed to celt_encoder_destroy");
      return;
   }
   if (check_mode(st->mode) != CELT_OK)
      return;

   ec_byte_writeclear(&st->buf);

   pitch_state_free(st->fft);

   celt_free(st->in_mem);
   celt_free(st->mdct_overlap);
   celt_free(st->out_mem);
   
   celt_free(st->oldBandE);
   
   celt_free(st->preemph_memE);
   celt_free(st->preemph_memD);
   
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

/** Apply window and compute the MDCT for all sub-frames and all channels in a frame */
static void compute_mdcts(const CELTMode *mode, const celt_word16_t * restrict window, celt_sig_t * restrict in, celt_sig_t * restrict out)
{
   int c, N4;
   VARDECL(celt_word32_t, x);
   VARDECL(celt_word32_t, tmp);
   const int C = CHANNELS(mode);
   const mdct_lookup *lookup = MDCT(mode);
   const int N = FRAMESIZE(mode);
   const int overlap = OVERLAP(mode);
   SAVE_STACK;
   N4 = (N-overlap)>>1;
   ALLOC(x, 2*N, celt_word32_t);
   ALLOC(tmp, N, celt_word32_t);
   for (c=0;c<C;c++)
   {
      int j;
      celt_word32_t * restrict x1, * restrict x2;
      for (j=0;j<2*N-2*N4;j++)
         x[j+N4] = in[C*j+c];
      x1 = x+N4;
      x2 = x+2*N-N4-1;
      for (j=0;j<overlap;j++)
      {
         *x1 = MULT16_32_Q15(window[j],*x1);
         *x2 = MULT16_32_Q15(window[j],*x2);
         x1++;
         x2--;
      }
      CELT_MEMSET(x, 0, N4);
      CELT_MEMSET(x+2*N-N4, 0, N4);
      mdct_forward(lookup, x, tmp);
      /* Interleaving the sub-frames */
      for (j=0;j<N;j++)
         out[C*j+c] = tmp[j];
   }
   RESTORE_STACK;
}

/** Compute the IMDCT and apply window for all sub-frames and all channels in a frame */
static void compute_inv_mdcts(const CELTMode *mode, const celt_word16_t * restrict window, celt_sig_t *X, celt_sig_t * restrict out_mem, celt_sig_t * restrict mdct_overlap)
{
   int c, N4;
   VARDECL(celt_word32_t, x);
   VARDECL(celt_word32_t, tmp);
   const int C = CHANNELS(mode);
   const mdct_lookup *lookup = MDCT(mode);
   const int N = FRAMESIZE(mode);
   const int overlap = OVERLAP(mode);
   SAVE_STACK;
   ALLOC(x, 2*N, celt_word32_t);
   ALLOC(tmp, N, celt_word32_t);
   N4 = (N-overlap)>>1;
   for (c=0;c<C;c++)
   {
      int j;
      /* De-interleaving the sub-frames */
      for (j=0;j<N;j++)
         tmp[j] = X[C*j+c];
      mdct_backward(lookup, tmp, x);
         /* The first and last part would need to be set to zero if we actually
      wanted to use them. */
      for (j=0;j<overlap;j++)
         out_mem[C*(MAX_PERIOD-N)+C*j+c] = 2*(mdct_overlap[C*j+c]+MULT16_32_Q15(window[j],x[j+N4]));
      for (j=0;j<overlap;j++)
         mdct_overlap[C*(overlap-j-1)+c] = MULT16_32_Q15(window[j],x[2*N-j-N4-1]);
      for (j=0;j<2*N4;j++)
         out_mem[C*(MAX_PERIOD-N)+C*(j+overlap)+c] = 2*x[j+N4+overlap];
   }
   RESTORE_STACK;
}

int EXPORT celt_encode(CELTEncoder * restrict st, celt_int16_t * restrict pcm, unsigned char *compressed, int nbCompressedBytes)
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
      for (i=0;i<N;i++)
      {
         celt_sig_t tmp = SHL32(EXTEND32(pcm[C*i+c]), SIG_SHIFT);
         in[C*(i+st->overlap)+c] = SUB32(tmp, SHR32(MULT16_16(preemph,st->preemph_memE[c]),1));
         st->preemph_memE[c] = pcm[C*i+c];
      }
   }
   CELT_COPY(st->in_mem, in+C*(2*N-2*N4-st->overlap), C*st->overlap);

   /* Pitch analysis: we do it early to save on the peak stack space */
   find_spectral_pitch(st->mode, st->fft, &st->mode->psy, in, st->out_mem, st->mode->window, 2*N-2*N4, &pitch_index);

   ALLOC(freq, C*N, celt_sig_t); /**< Interleaved signal MDCTs */
   
   /*for (i=0;i<(B+1)*C*N;i++) printf ("%f(%d) ", in[i], i); printf ("\n");*/
   /* Compute MDCTs */
   compute_mdcts(st->mode, st->mode->window, in, freq);

#if 0 /* Mask disabled until it can be made to do something useful */
   compute_mdct_masking(X, mask, B*C*N, st->Fs);

   /* Invert and stretch the mask to length of X 
      For some reason, I get better results by using the sqrt instead,
      although there's no valid reason to. Must investigate further */
   for (i=0;i<B*C*N;i++)
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
   compute_mdcts(st->mode, st->mode->window, st->out_mem+pitch_index*C, freq);

   quant_energy(st->mode, bandE, st->oldBandE, nbCompressedBytes*8/3, &st->enc);

   if (C==2)
   {
      stereo_mix(st->mode, X, bandE, 1);
   }

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
   if (MULT16_32_Q15(QCONST16(.1f, 15),curr_power) + QCONST32(10.f,ENER_SHIFT) < pitch_power)
   {
      if (C==2)
         stereo_mix(st->mode, P, bandE, 1);
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
      ec_enc_uint(&st->enc, 0, 128);
      for (i=0;i<C*N;i++)
         P[i] = 0;
   }
   

   pitch_quant_bands(st->mode, P, gains);

   /*for (i=0;i<B*N;i++) printf("%f ",P[i]);printf("\n");*/
   /* Compute residual that we're going to encode */
   for (i=0;i<C*N;i++)
      X[i] -= P[i];

   /* Residual quantisation */
   quant_bands(st->mode, X, P, NULL, nbCompressedBytes*8, &st->enc);
   
   if (C==2)
   {
      stereo_mix(st->mode, X, bandE, -1);
      renormalise_bands(st->mode, X);
   }
   /* Synthesis */
   denormalise_bands(st->mode, X, freq, bandE);


   CELT_MOVE(st->out_mem, st->out_mem+C*N, C*(MAX_PERIOD-N));

   compute_inv_mdcts(st->mode, st->mode->window, freq, st->out_mem, st->mdct_overlap);
   /* De-emphasis and put everything back at the right place in the synthesis history */
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

   celt_sig_t *mdct_overlap;
   celt_sig_t *out_mem;

   celt_word16_t *oldBandE;
   
   int last_pitch_index;
};

CELTDecoder EXPORT *celt_decoder_create(const CELTMode *mode)
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

   st->mdct_overlap = celt_alloc(N*C*sizeof(celt_sig_t));
   st->out_mem = celt_alloc(MAX_PERIOD*C*sizeof(celt_sig_t));
   
   st->oldBandE = (celt_word16_t*)celt_alloc(C*mode->nbEBands*sizeof(celt_word16_t));

   st->preemph_memD = (celt_sig_t*)celt_alloc(C*sizeof(celt_sig_t));;

   st->last_pitch_index = 0;
   return st;
}

void EXPORT celt_decoder_destroy(CELTDecoder *st)
{
   if (st == NULL)
   {
      celt_warning("NULL passed to celt_encoder_destroy");
      return;
   }
   if (check_mode(st->mode) != CELT_OK)
      return;


   celt_free(st->mdct_overlap);
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
   VARDECL(celt_sig_t, freq);
   const int C = CHANNELS(st->mode);
   SAVE_STACK;
   N = st->block_size;
   ALLOC(freq,C*N, celt_sig_t);         /**< Interleaved signal MDCTs */
   
   pitch_index = st->last_pitch_index;
   
   /* Use the pitch MDCT as the "guessed" signal */
   compute_mdcts(st->mode, st->mode->window, st->out_mem+pitch_index*C, freq);

   CELT_MOVE(st->out_mem, st->out_mem+C*N, C*(MAX_PERIOD-N));
   /* Compute inverse MDCTs */
   compute_inv_mdcts(st->mode, st->mode->window, freq, st->out_mem, st->mdct_overlap);

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

int EXPORT celt_decode(CELTDecoder * restrict st, unsigned char *data, int len, celt_int16_t * restrict pcm)
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
   
   /* Get band energies */
   unquant_energy(st->mode, bandE, st->oldBandE, len*8/3, &dec);
   
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
   
   /* Pitch MDCT */
   compute_mdcts(st->mode, st->mode->window, st->out_mem+pitch_index*C, freq);

   {
      VARDECL(celt_ener_t, bandEp);
      ALLOC(bandEp, st->mode->nbEBands*C, celt_ener_t);
      compute_band_energies(st->mode, freq, bandEp);
      normalise_bands(st->mode, freq, P, bandEp);
   }

   if (C==2)
      stereo_mix(st->mode, P, bandE, 1);

   /* Apply pitch gains */
   pitch_quant_bands(st->mode, P, gains);

   /* Decode fixed codebook and merge with pitch */
   unquant_bands(st->mode, X, P, len*8, &dec);

   if (C==2)
   {
      stereo_mix(st->mode, X, bandE, -1);
      renormalise_bands(st->mode, X);
   }
   /* Synthesis */
   denormalise_bands(st->mode, X, freq, bandE);


   CELT_MOVE(st->out_mem, st->out_mem+C*N, C*(MAX_PERIOD-N));
   /* Compute inverse MDCTs */
   compute_inv_mdcts(st->mode, st->mode->window, freq, st->out_mem, st->mdct_overlap);

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

