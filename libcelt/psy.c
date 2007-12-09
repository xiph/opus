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

#include "psy.h"
#include <math.h>

/* Psychoacoustic spreading function. The idea here is compute a first order 
   recursive smoothing. The filter coefficient is frequency dependent and 
   chosen such that we have a -10dB/Bark slope on the right side and a -25dB/Bark
   slope on the left side. */
static void spreading_func(float *psd, float *mask, int len, int Fs)
{
   int i;
   float decayL[len], decayR[len];
   float mem;
   //for (i=0;i<len;i++) printf ("%f ", psd[i]);
   /* This can easily be tabulated, which makes the function very fast. */
   for (i=0;i<len;i++)
   {
      float f;
      float deriv;
      f = Fs*i*(1/(2.f*len));
      deriv = (8.288e-8 * f)/(3.4225e-16 *f*f*f*f + 1) +  .009694/(5.476e-7 *f*f + 1) + 1e-4;
      deriv *= Fs*(1/(2.f*len));
      decayR[i] = pow(.1f, deriv);
      decayL[i] = pow(0.0031623f, deriv);
   }
   /* Compute right slope (-10 dB/Bark) */
   mem=psd[0];
   for (i=0;i<len;i++)
   {
      mask[i] = (1-decayR[i])*psd[i] + decayR[i]*mem;
      mem = mask[i];
   }
   /* Compute left slope (-25 dB/Bark) */
   mem=mask[len-1];
   for (i=len-1;i>=0;i--)
   {
      mask[i] = (1-decayR[i])*mask[i] + decayL[i]*mem;
      mem = mask[i];
   }
   //for (i=0;i<len;i++) printf ("%f ", mask[i]); printf ("\n");
}

/* Compute a marking threshold from the spectrum X. */
void compute_masking(float *X, float *mask, int len, int Fs)
{
   int i;
   int N=len/2;
   float psd[N];
   psd[0] = X[0]*X[0];
   for (i=1;i<N;i++)
      psd[i] = X[i*2-1]*X[i*2-1] + X[i*2]*X[i*2];
   /* TODO: Do tone masking */
   /* Noise masking */
   spreading_func(psd, mask, N, Fs);
   
}

