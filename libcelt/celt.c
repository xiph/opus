/* (C) 2007 Jean-Marc Valin, CSIRO
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

#include "os_support.h"
#include "mdct.h"
#include <math.h>
#include "celt.h"
#include "pitch.h"
#include "fftwrap.h"
#include "bands.h"
#include "modes.h"
#include "probenc.h"
#include "quant_pitch.h"
#include "quant_bands.h"

#define MAX_PERIOD 1024


struct CELTEncoder {
   const CELTMode *mode;
   int frame_size;
   int block_size;
   int nb_blocks;
   int channels;
   
   ec_byte_buffer buf;
   ec_enc         enc;

   float preemph;
   float *preemph_memE;
   float *preemph_memD;
   
   mdct_lookup mdct_lookup;
   void *fft;
   
   float *window;
   float *in_mem;
   float *mdct_overlap;
   float *out_mem;

   float *oldBandE;
};



CELTEncoder *celt_encoder_new(const CELTMode *mode)
{
   int i, N, B, C;
   N = mode->mdctSize;
   B = mode->nbMdctBlocks;
   C = mode->nbChannels;
   CELTEncoder *st = celt_alloc(sizeof(CELTEncoder));
   
   st->mode = mode;
   st->frame_size = B*N;
   st->block_size = N;
   st->nb_blocks  = B;
   
   ec_byte_writeinit(&st->buf);
   ec_enc_init(&st->enc,&st->buf);

   mdct_init(&st->mdct_lookup, 2*N);
   st->fft = spx_fft_init(MAX_PERIOD*C);
   
   st->window = celt_alloc(2*N*sizeof(float));
   st->in_mem = celt_alloc(N*C*sizeof(float));
   st->mdct_overlap = celt_alloc(N*C*sizeof(float));
   st->out_mem = celt_alloc(MAX_PERIOD*C*sizeof(float));
   for (i=0;i<N;i++)
      st->window[i] = st->window[2*N-i-1] = sin(.5*M_PI* sin(.5*M_PI*(i+.5)/N) * sin(.5*M_PI*(i+.5)/N));
   
   st->oldBandE = celt_alloc(mode->nbEBands*sizeof(float));

   st->preemph = 0.8;
   st->preemph_memE = celt_alloc(C*sizeof(float));;
   st->preemph_memD = celt_alloc(C*sizeof(float));;

   return st;
}

void celt_encoder_destroy(CELTEncoder *st)
{
   if (st == NULL)
   {
      celt_warning("NULL passed to celt_encoder_destroy");
      return;
   }
   ec_byte_writeclear(&st->buf);

   mdct_clear(&st->mdct_lookup);
   spx_fft_destroy(st->fft);

   celt_free(st->window);
   celt_free(st->in_mem);
   celt_free(st->mdct_overlap);
   celt_free(st->out_mem);
   
   celt_free(st->oldBandE);
   celt_free(st);
}

static void haar1(float *X, int N)
{
   int i;
   for (i=0;i<N;i+=2)
   {
      float a, b;
      a = X[i];
      b = X[i+1];
      X[i] = .707107f*(a+b);
      X[i+1] = .707107f*(a-b);
   }
}

static void inv_haar1(float *X, int N)
{
   int i;
   for (i=0;i<N;i+=2)
   {
      float a, b;
      a = X[i];
      b = X[i+1];
      X[i] = .707107f*(a+b);
      X[i+1] = .707107f*(a-b);
   }
}

static void compute_mdcts(mdct_lookup *mdct_lookup, float *window, float *in, float *out, int N, int B, int C)
{
   int i, c;
   for (c=0;c<C;c++)
   {
      for (i=0;i<B;i++)
      {
         int j;
         float x[2*N];
         float tmp[N];
         for (j=0;j<2*N;j++)
            x[j] = window[j]*in[C*i*N+C*j+c];
         mdct_forward(mdct_lookup, x, tmp);
         /* Interleaving the sub-frames */
         for (j=0;j<N;j++)
            out[C*B*j+C*i+c] = tmp[j];
      }
   }
}

static void compute_inv_mdcts(mdct_lookup *mdct_lookup, float *window, float *X, float *out_mem, float *mdct_overlap, int N, int B, int C)
{
   int i, c;
   for (c=0;c<C;c++)
   {
      for (i=0;i<B;i++)
      {
         int j;
         float x[2*N];
         float tmp[N];
         /* De-interleaving the sub-frames */
         for (j=0;j<N;j++)
            tmp[j] = X[C*B*j+C*i+c];
         mdct_backward(mdct_lookup, tmp, x);
         for (j=0;j<2*N;j++)
            x[j] = window[j]*x[j];
         for (j=0;j<N;j++)
            out_mem[C*(MAX_PERIOD+(i-B)*N)+C*j+c] = x[j]+mdct_overlap[C*j+c];
         for (j=0;j<N;j++)
            mdct_overlap[C*j+c] = x[N+j];
      }
   }
}

int celt_encode(CELTEncoder *st, short *pcm)
{
   int i, c, N, B, C;
   N = st->block_size;
   B = st->nb_blocks;
   C = st->mode->nbChannels;
   float in[(B+1)*C*N];
   
   float X[B*C*N];         /**< Interleaved signal MDCTs */
   float P[B*C*N];         /**< Interleaved pitch MDCTs*/
   float bandE[st->mode->nbEBands];
   float gains[st->mode->nbPBands];
   int pitch_index;
   
   for (c=0;c<C;c++)
   {
      for (i=0;i<N;i++)
         in[C*i+c] = st->in_mem[C*i+c];
      for (;i<(B+1)*N;i++)
      {
         float tmp = pcm[C*(i-N)+c];
         in[C*i+c] = tmp - st->preemph*st->preemph_memE[c];
         st->preemph_memE[c] = tmp;
      }
      for (i=0;i<N;i++)
         st->in_mem[C*i+c] = in[C*(B*N+i)+c];
   }
   //for (i=0;i<(B+1)*C*N;i++) printf ("%f(%d) ", in[i], i); printf ("\n");
   /* Compute MDCTs */
   compute_mdcts(&st->mdct_lookup, st->window, in, X, N, B, C);
   
   /* Pitch analysis */
   for (c=0;c<C;c++)
   {
      for (i=0;i<N;i++)
      {
         in[C*i+c] *= st->window[i];
         in[C*(B*N+i)+c] *= st->window[N+i];
      }
   }
   find_spectral_pitch(st->fft, in, st->out_mem, MAX_PERIOD, (B+1)*N, C, &pitch_index);
   ec_enc_uint(&st->enc, pitch_index, MAX_PERIOD-(B+1)*N);
   
   /* Compute MDCTs of the pitch part */
   compute_mdcts(&st->mdct_lookup, st->window, st->out_mem+pitch_index*C, P, N, B, C);
   
   /*int j;
   for (j=0;j<B*N;j++)
      printf ("%f ", X[j]);
   for (j=0;j<B*N;j++)
      printf ("%f ", P[j]);
   printf ("\n");*/
   if (C==2)
   {
      haar1(X, B*N);
      haar1(P, B*N);
   }
   
   /* Band normalisation */
   compute_band_energies(st->mode, X, bandE);
   normalise_bands(st->mode, X, bandE);
   //for (i=0;i<st->mode->nbEBands;i++)printf("%f ", bandE[i]);printf("\n");
   
   {
      float bandEp[st->mode->nbEBands];
      compute_band_energies(st->mode, P, bandEp);
      normalise_bands(st->mode, P, bandEp);
   }
   
   quant_energy(st->mode, bandE, st->oldBandE, &st->enc);
   
   /* Pitch prediction */
   compute_pitch_gain(st->mode, X, P, gains, bandE);
   quant_pitch(gains, st->mode->nbPBands, &st->enc);
   pitch_quant_bands(st->mode, X, P, gains);

   //for (i=0;i<B*N;i++) printf("%f ",P[i]);printf("\n");
   /* Subtract the pitch prediction from the signal to encode */
   for (i=0;i<B*C*N;i++)
      X[i] -= P[i];

   /*float sum=0;
   for (i=0;i<B*N;i++)
      sum += X[i]*X[i];
   printf ("%f\n", sum);*/
   /* Residual quantisation */
   quant_bands(st->mode, X, P, &st->enc);
   
   if (0) {//This is just for debugging
      ec_enc_done(&st->enc);
      ec_dec dec;
      ec_byte_readinit(&st->buf,ec_byte_get_buffer(&st->buf),ec_byte_bytes(&st->buf));
      ec_dec_init(&dec,&st->buf);

      unquant_bands(st->mode, X, P, &dec);
      //printf ("\n");
   }
   
   /* Synthesis */
   denormalise_bands(st->mode, X, bandE);

   if (C==2)
      inv_haar1(X, B*N);

   CELT_MOVE(st->out_mem, st->out_mem+C*B*N, C*(MAX_PERIOD-B*N));
   /* Compute inverse MDCTs */
   compute_inv_mdcts(&st->mdct_lookup, st->window, X, st->out_mem, st->mdct_overlap, N, B, C);

   for (c=0;c<C;c++)
   {
      for (i=0;i<B;i++)
      {
         int j;
         for (j=0;j<N;j++)
         {
            float tmp = st->out_mem[C*(MAX_PERIOD+(i-B)*N)+C*j+c] + st->preemph*st->preemph_memD[c];
            st->preemph_memD[c] = tmp;
            if (tmp > 32767) tmp = 32767;
            if (tmp < -32767) tmp = -32767;
            pcm[C*i*N+C*j+c] = (short)floor(.5+tmp);
         }
      }
   }
   return 0;
}

char *celt_encoder_get_bytes(CELTEncoder *st, int *nbBytes)
{
   char *data;
   ec_enc_done(&st->enc);
   *nbBytes = ec_byte_bytes(&st->buf);
   data = ec_byte_get_buffer(&st->buf);
   //printf ("%d\n", *nbBytes);
   
   /* Reset the packing for the next encoding */
   ec_byte_reset(&st->buf);
   ec_enc_init(&st->enc,&st->buf);

   return data;
}


/****************************************************************************/
/*                                                                          */
/*                                DECODER                                   */
/*                                                                          */
/****************************************************************************/



struct CELTDecoder {
   const CELTMode *mode;
   int frame_size;
   int block_size;
   int nb_blocks;
   
   ec_byte_buffer buf;
   ec_enc         enc;

   float preemph;
   float *preemph_memD;
   
   mdct_lookup mdct_lookup;
   
   float *window;
   float *mdct_overlap;
   float *out_mem;

   float *oldBandE;
   
   int last_pitch_index;
};

CELTDecoder *celt_decoder_new(const CELTMode *mode)
{
   int i, N, B, C;
   N = mode->mdctSize;
   B = mode->nbMdctBlocks;
   C = mode->nbChannels;
   CELTDecoder *st = celt_alloc(sizeof(CELTDecoder));
   
   st->mode = mode;
   st->frame_size = B*N;
   st->block_size = N;
   st->nb_blocks  = B;
   
   mdct_init(&st->mdct_lookup, 2*N);
   
   st->window = celt_alloc(2*N*sizeof(float));
   st->mdct_overlap = celt_alloc(N*C*sizeof(float));
   st->out_mem = celt_alloc(MAX_PERIOD*C*sizeof(float));
   for (i=0;i<N;i++)
      st->window[i] = st->window[2*N-i-1] = sin(.5*M_PI* sin(.5*M_PI*(i+.5)/N) * sin(.5*M_PI*(i+.5)/N));
   
   st->oldBandE = celt_alloc(mode->nbEBands*sizeof(float));

   st->preemph = 0.8;
   st->preemph_memD = celt_alloc(C*sizeof(float));;

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

   mdct_clear(&st->mdct_lookup);

   celt_free(st->window);
   celt_free(st->mdct_overlap);
   celt_free(st->out_mem);
   
   celt_free(st->oldBandE);
   celt_free(st);
}

static void celt_decode_lost(CELTDecoder *st, short *pcm)
{
   int i, c, N, B, C;
   N = st->block_size;
   B = st->nb_blocks;
   C = st->mode->nbChannels;
   float X[C*B*N];         /**< Interleaved signal MDCTs */
   int pitch_index;
   
   pitch_index = st->last_pitch_index;
   
   /* Use the pitch MDCT as the "guessed" signal */
   compute_mdcts(&st->mdct_lookup, st->window, st->out_mem+pitch_index*C, X, N, B, C);

   CELT_MOVE(st->out_mem, st->out_mem+C*B*N, C*(MAX_PERIOD-B*N));
   /* Compute inverse MDCTs */
   compute_inv_mdcts(&st->mdct_lookup, st->window, X, st->out_mem, st->mdct_overlap, N, B, C);

   for (c=0;c<C;c++)
   {
      for (i=0;i<B;i++)
      {
         int j;
         for (j=0;j<N;j++)
         {
            float tmp = st->out_mem[C*(MAX_PERIOD+(i-B)*N)+C*j+c] + st->preemph*st->preemph_memD[c];
            st->preemph_memD[c] = tmp;
            if (tmp > 32767) tmp = 32767;
            if (tmp < -32767) tmp = -32767;
            pcm[C*i*N+C*j+c] = (short)floor(.5+tmp);
         }
      }
   }
}

int celt_decode(CELTDecoder *st, char *data, int len, short *pcm)
{
   int i, c, N, B, C;
   N = st->block_size;
   B = st->nb_blocks;
   C = st->mode->nbChannels;
   
   float X[C*B*N];         /**< Interleaved signal MDCTs */
   float P[C*B*N];         /**< Interleaved pitch MDCTs*/
   float bandE[st->mode->nbEBands];
   float gains[st->mode->nbPBands];
   int pitch_index;
   ec_dec dec;
   ec_byte_buffer buf;
   
   if (data == NULL)
   {
      celt_decode_lost(st, pcm);
      return 0;
   }
   
   ec_byte_readinit(&buf,data,len);
   ec_dec_init(&dec,&buf);
   
   /* Get the pitch index */
   pitch_index = ec_dec_uint(&dec, MAX_PERIOD-(B+1)*N);
   st->last_pitch_index = pitch_index;
   
   /* Get band energies */
   unquant_energy(st->mode, bandE, st->oldBandE, &dec);
   
   /* Pitch MDCT */
   compute_mdcts(&st->mdct_lookup, st->window, st->out_mem+pitch_index*C, P, N, B, C);

   if (C==2)
      haar1(P, B*N);

   {
      float bandEp[st->mode->nbEBands];
      compute_band_energies(st->mode, P, bandEp);
      normalise_bands(st->mode, P, bandEp);
   }

   /* Get the pitch gains */
   unquant_pitch(gains, st->mode->nbPBands, &dec);

   /* Apply pitch gains */
   pitch_quant_bands(st->mode, X, P, gains);

   /* Decode fixed codebook and merge with pitch */
   unquant_bands(st->mode, X, P, &dec);

   /* Synthesis */
   denormalise_bands(st->mode, X, bandE);

   if (C==2)
      inv_haar1(X, B*N);

   CELT_MOVE(st->out_mem, st->out_mem+C*B*N, C*(MAX_PERIOD-B*N));
   /* Compute inverse MDCTs */
   compute_inv_mdcts(&st->mdct_lookup, st->window, X, st->out_mem, st->mdct_overlap, N, B, C);

   for (c=0;c<C;c++)
   {
      for (i=0;i<B;i++)
      {
         int j;
         for (j=0;j<N;j++)
         {
            float tmp = st->out_mem[C*(MAX_PERIOD+(i-B)*N)+C*j+c] + st->preemph*st->preemph_memD[c];
            st->preemph_memD[c] = tmp;
            if (tmp > 32767) tmp = 32767;
            if (tmp < -32767) tmp = -32767;
            pcm[C*i*N+C*j+c] = (short)floor(.5+tmp);
         }
      }
   }
   return 0;
   //printf ("\n");
}

