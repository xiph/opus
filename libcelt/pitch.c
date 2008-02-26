/* (C) 2007-2008 Jean-Marc Valin, CSIRO
*/
/**
   @file pitch.c
   @brief Pitch analysis
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

#include <stdio.h>
#include <math.h>
#include "pitch.h"
#include "psy.h"

void find_spectral_pitch(kiss_fftr_cfg fft, struct PsyDecay *decay, celt_sig_t *x, celt_sig_t *y, int lag, int len, int C, int *pitch)
{
   int c, i;
   float max_corr;
   VARDECL(celt_word32_t *xx);
   VARDECL(celt_word32_t *yy);
   VARDECL(celt_word32_t *X);
   VARDECL(celt_word32_t *Y);
   VARDECL(float *curve);
   int n2 = lag/2;
   ALLOC(xx, lag*C, celt_word32_t);
   ALLOC(yy, lag*C, celt_word32_t);
   ALLOC(X, lag*C, celt_word32_t);
   ALLOC(Y, lag*C, celt_word32_t);
   ALLOC(curve, n2*C, float);
   
   for (i=0;i<C*lag;i++)
      xx[i] = 0;
   for (c=0;c<C;c++)
   {
      for (i=0;i<len;i++)
         xx[c*lag+i] = x[C*i+c];
      for (i=0;i<lag;i++)
         yy[c*lag+i] = y[C*i+c];
      
   }
   
   kiss_fftr(fft, xx, X);
   kiss_fftr(fft, yy, Y);
   
   compute_masking(decay, X, curve, lag*C);
   
   for (i=1;i<C*n2;i++)
   {
      float n, tmp;
      /*n = 1.f/(1e1+sqrt(sqrt((X[2*i-1]*X[2*i-1] + X[2*i  ]*X[2*i  ])*(Y[2*i-1]*Y[2*i-1] + Y[2*i  ]*Y[2*i  ]))));*/
      /*n = 1;*/
      n = 1.f/sqrt(1+curve[i]);
      /*printf ("%f ", n);*/
      /*n = 1.f/(1+curve[i]);*/
      tmp = X[2*i];
      X[2*i] = (1.f*X[2*i  ]*Y[2*i  ] + 1.f*X[2*i+1]*Y[2*i+1])*n;
      X[2*i+1] = (- 1.f*X[2*i+1]*Y[2*i  ] + 1.f*tmp*Y[2*i+1])*n;
   }
   /*printf ("\n");*/
   X[0] = X[1] = 0;
   kiss_fftri(fft, X, xx);
   /*for (i=0;i<C*lag;i++)
      printf ("%d %d\n", X[i], xx[i]);*/
   
   max_corr=-1e10;
   *pitch = 0;
   for (i=0;i<lag-len;i++)
   {
      /*printf ("%f ", xx[i]);*/
      if (xx[i] > max_corr)
      {
         *pitch = i;
         max_corr = xx[i];
      }
   }
   /*printf ("%f\n", max_corr);*/
   /*printf ("\n");
   printf ("%d %f\n", *pitch, max_corr);
   printf ("%d\n", *pitch);*/
}
