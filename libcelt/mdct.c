/* (C) 2008 Jean-Marc Valin, CSIRO
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

/* This is a simple MDCT implementation that uses a N/4 complex FFT
   to do most of the work. It should be relatively straightforward to
   plug in pretty much and FFT here.
   
   This replaces the Vorbis FFT (and uses the exact same API), which 
   was a bit too messy and that was ending up duplicating code 
   (might as well use the same FFT everywhere).
   
   The algorithm is similar to (and inspired from) Fabrice Bellard's
   MDCT implementation in FFMPEG, but has differences in signs, ordering
   and scaling in many places. 
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mdct.h"
#include "kiss_fft.h"
#include <math.h>
#include "os_support.h"

#ifndef M_PI
#define M_PI 3.14159263
#endif

void mdct_init(mdct_lookup *l,int N)
{
   int i;
   int N2, N4;
   l->n = N;
   N2 = N/2;
   N4 = N/4;
   l->kfft = kiss_fft_alloc(N4, NULL, NULL);
   l->trig = celt_alloc(N2*sizeof(float));
   /* We have enough points that sine isn't necessary */
   for (i=0;i<N2;i++)
      l->trig[i] = cos(2*M_PI*(i+1./8.)/N);
   l->scale = 1./N4;
}

void mdct_clear(mdct_lookup *l)
{
   kiss_fft_free(l->kfft);
   celt_free(l->trig);
}

void mdct_forward(mdct_lookup *l, celt_sig_t *in, celt_sig_t *out)
{
   int i;
   int N, N2, N4, N8;
   VARDECL(celt_sig_t *f);
   N = l->n;
   N2 = N/2;
   N4 = N/4;
   N8 = N/8;
   ALLOC(f, N2, celt_sig_t);
   
   /* Consider the input to be compused of four blocks: [a, b, c, d] */
   /* Shuffle, fold, pre-rotate (part 1) */
   for(i=0;i<N8;i++)
   {
      float re, im;
      /* Real part arranged as -d-cR, Imag part arranged as -b+aR*/
      re = -.5*(in[N2+N4+2*i] + in[N2+N4-2*i-1]);
      im = -.5*(in[N4+2*i]    - in[N4-2*i-1]);
      out[2*i]   = re*l->trig[i]  -  im*l->trig[i+N4];
      out[2*i+1] = im*l->trig[i]  +  re*l->trig[i+N4];
   }
   for(;i<N4;i++)
   {
      float re, im;
      /* Real part arranged as a-bR, Imag part arranged as -c-dR */
      re =  .5*(in[2*i-N4] - in[N2+N4-2*i-1]);
      im = -.5*(in[N4+2*i] + in[N+N4-2*i-1]);
      out[2*i]   = re*l->trig[i]  -  im*l->trig[i+N4];
      out[2*i+1] = im*l->trig[i]  +  re*l->trig[i+N4];
   }

   /* N/4 complex FFT, which should normally down-scale by 4/N (but doesn't now) */
   kiss_fft(l->kfft, (const kiss_fft_cpx *)out, (kiss_fft_cpx *)f);

   /* Post-rotate and apply the scaling if the FFT doesn't to it itself */
   for(i=0;i<N4;i++)
   {
      out[2*i]      = l->scale * (-f[2*i+1]*l->trig[i+N4] + f[2*i]  *l->trig[i]);
      out[N2-1-2*i] = l->scale * (-f[2*i]  *l->trig[i+N4] - f[2*i+1]*l->trig[i]);
   }
}


void mdct_backward(mdct_lookup *l, celt_sig_t *in, celt_sig_t *out)
{
   int i;
   int N, N2, N4, N8;
   VARDECL(celt_sig_t *f);
   N = l->n;
   N2 = N/2;
   N4 = N/4;
   N8 = N/8;
   ALLOC(f, N2, celt_sig_t);
   
   /* Pre-rotate */
   for(i=0;i<N4;i++) 
   {
      out[2*i]   = -in[N2-2*i-1] * l->trig[i]    - in[2*i]*l->trig[i+N4];
      out[2*i+1] =  in[N2-2*i-1] * l->trig[i+N4] - in[2*i]*l->trig[i];
   }

   /* Inverse N/4 complex FFT. This one should *not* downscale even in fixed-point */
   kiss_ifft(l->kfft, (const kiss_fft_cpx *)out, (kiss_fft_cpx *)f);
   
   /* Post-rotate */
   for(i=0;i<N4;i++)
   {
      float re, im;
      re = f[2*i];
      im = f[2*i+1];
      /* We'd scale up by 2 here, but instead it's done when mixing the windows */
      f[2*i]   = re*l->trig[i] + im*l->trig[i+N4];
      f[2*i+1] = im*l->trig[i] - re*l->trig[i+N4];
   }
   /* De-shuffle the components for the middle of the window only */
   for(i = 0; i < N4; i++)
   {
      out[N4+2*i]   =-f[2*i];
      out[N4+2*i+1] = f[N2-2*i-1];
   }

   /* Mirror on both sides for TDAC */
   for(i = 0; i < N4; i++)
   {
      out[i]     =-out[N2-i-1];
      out[N-i-1] = out[N2+i];
   }
}


