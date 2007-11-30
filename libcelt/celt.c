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

#define MAX_PERIOD 1024

struct CELTState_ {
   int frame_size;
   int block_size;
   int nb_blocks;
      
   float preemph;
   float preemph_mem;
   
   mdct_lookup mdct_lookup;
   
   float *window;
   float *in_mem;
   float *mdct_overlap;
   float *out_mem;
};



CELTState *celt_encoder_new(int blockSize, int blocksPerFrame)
{
   int i, N;
   N = blockSize;
   CELTState *st = celt_alloc(sizeof(CELTState));
   
   st->frame_size = blockSize * blocksPerFrame;
   st->block_size = blockSize;
   st->nb_blocks  = blocksPerFrame;
   
   mdct_init(&st->mdct_lookup, 2*N);
   
   st->window = celt_alloc(2*N*sizeof(float));
   st->in_mem = celt_alloc(N*sizeof(float));
   st->mdct_overlap = celt_alloc(N*sizeof(float));
   st->out_mem = celt_alloc(MAX_PERIOD*sizeof(float));

   for (i=0;i<N;i++)
      st->window[i] = st->window[2*N-i-1] = sin(.5*M_PI* sin(.5*M_PI*(i+.5)/N) * sin(.5*M_PI*(i+.5)/N));
   return st;
}

void celt_encoder_destroy(CELTState *st)
{
   if (st == NULL)
   {
      celt_warning("NULL passed to celt_encoder_destroy");
      return;
   }
   celt_free(st->window);
   celt_free(st->in_mem);
   celt_free(st->mdct_overlap);
   celt_free(st->out_mem);
   celt_free(st);
}

int celt_encode(CELTState *st, short *pcm)
{
   int i, N, B;
   N = st->block_size;
   B = st->nb_blocks;
   float in[(B+1)*N];
   float X[B*N];
   int pitch_index;
   
   /* FIXME: Add preemphasis */
   for (i=0;i<N;i++)
      in[i] = st->in_mem[i];
   for (;i<(B+1)*N;i++)
      in[i] = pcm[i-N];
   
   for (i=0;i<N;i++)
      st->in_mem[i] = pcm[(B-1)*N+i];

   /* Compute MDCTs */
   for (i=0;i<B;i++)
   {
      int j;
      float x[2*N];
      float tmp[N];
      for (j=0;j<2*N;j++)
         x[j] = st->window[j]*in[i*N+j];
      mdct_forward(&st->mdct_lookup, x, tmp);
      /* Interleaving the sub-frames */
      for (j=0;j<N;j++)
         X[B*j+i] = tmp[j];
   }
   
   
   /* Pitch analysis */
   find_spectral_pitch(in, st->out_mem, MAX_PERIOD, (B+1)*N, &pitch_index, NULL);
   
   /* Band normalisation */

   /* Pitch prediction */

   /* Residual quantisation */
   
   /* Synthesis */

   CELT_MOVE(st->out_mem, st->out_mem+B*N, MAX_PERIOD-B*N);
   /* Compute inverse MDCTs */
   for (i=0;i<B;i++)
   {
      int j;
      float x[2*N];
      float tmp[N];
      /* De-interleaving the sub-frames */
      for (j=0;j<N;j++)
         tmp[j] = X[B*j+i];
      mdct_backward(&st->mdct_lookup, tmp, x);
      for (j=0;j<2*N;j++)
         x[j] = st->window[j]*x[j];
      for (j=0;j<N;j++)
         st->out_mem[MAX_PERIOD+(i-B)*N+j] = x[j]+st->mdct_overlap[j];
      for (j=0;j<N;j++)
         st->mdct_overlap[j] = x[N+j];
      
      for (j=0;j<N;j++)
         pcm[i*N+j] = (short)floor(.5+st->out_mem[MAX_PERIOD+(i-B)*N+j]);
   }

   return 0;
}

