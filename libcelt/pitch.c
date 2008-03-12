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

/*#include "_kiss_fft_guts.h"
#include "kiss_fftr.h"*/
#include "kfft_single.h"

#include <stdio.h>
#include <math.h>
#include "pitch.h"
#include "psy.h"
#include "os_support.h"
#include "mathops.h"

kiss_fftr_cfg pitch_state_alloc(int max_lag)
{
   return kiss_fftr_alloc_celt_single(max_lag, 0, 0);
}

void pitch_state_free(kiss_fftr_cfg st)
{
   kiss_fft_free(st);
}

#ifdef FIXED_POINT
static void normalise16(celt_word16_t *x, int len, celt_word16_t val)
{
   int i;
   celt_word16_t maxval = 0;
   for (i=0;i<len;i++)
      maxval = MAX16(maxval, ABS16(x[i]));
   if (maxval > val)
   {
      int shift = 0;
      while (maxval > val)
      {
         maxval >>= 1;
         shift++;
      }
      if (shift==0)
         return;
      for (i=0;i<len;i++)
         x[i] = SHR16(x[i], shift);
   } else {
      int shift=0;
      if (maxval == 0)
         return;
      val >>= 1;
      while (maxval < val)
      {
         val >>= 1;
         shift++;
      }
      if (shift==0)
         return;
      for (i=0;i<len;i++)
         x[i] = SHL16(x[i], shift);
   }
}
#else
#define normalise16(x,len,val)
#endif

#define INPUT_SHIFT 15

void find_spectral_pitch(kiss_fftr_cfg fft, const struct PsyDecay *decay, const celt_sig_t *x, const celt_sig_t *y, const celt_word16_t *window, int overlap, int lag, int len, int C, int *pitch)
{
   int c, i;
   celt_word32_t max_corr;
   VARDECL(celt_word16_t, X);
   VARDECL(celt_word16_t, Y);
   VARDECL(celt_mask_t, curve);
   int n2;
   int L2;
   const int *bitrev;
   SAVE_STACK;
   n2 = lag/2;
   L2 = len/2;
   ALLOC(X, lag, celt_word16_t);
   ALLOC(curve, n2, celt_mask_t);

   bitrev = fft->substate->bitrev;
   for (i=0;i<lag;i++)
      X[i] = 0;
   /* Sum all channels of the current frame and copy into X in bit-reverse order */
   for (c=0;c<C;c++)
   {
      for (i=0;i<L2;i++)
      {
         X[2*bitrev[i]] += SHR32(x[C*(2*i)+c],INPUT_SHIFT);
         X[2*bitrev[i]+1] += SHR32(x[C*(2*i+1)+c],INPUT_SHIFT);
      }
   }
   /* Applying the window in the bit-reverse domain. It's a bit weird, but it
      can help save memory */
   for (i=0;i<overlap/2;i++)
   {
      X[2*bitrev[i]] = MULT16_32_Q15(window[2*i], X[2*bitrev[i]]);
      X[2*bitrev[i]+1] = MULT16_32_Q15(window[2*i+1], X[2*bitrev[i]+1]);
      X[2*bitrev[L2-i-1]] = MULT16_32_Q15(window[2*i+1], X[2*bitrev[L2-i-1]]);
      X[2*bitrev[L2-i-1]+1] = MULT16_32_Q15(window[2*i], X[2*bitrev[L2-i-1]+1]);
   }
   normalise16(X, lag, 8192);
   /*for (i=0;i<lag;i++) printf ("%d ", X[i]);printf ("\n");*/
   /* Forward real FFT (in-place) */
   kf_work((kiss_fft_cpx*)X, NULL, 1,1, fft->substate->factors,fft->substate, 1, 1, 1);
   kiss_fftr_twiddles(fft,X);

   compute_masking(decay, X, curve, lag);

   /* Deferred allocation to reduce peak stack usage */
   ALLOC(Y, lag, celt_word16_t);
   for (i=0;i<lag;i++)
      Y[i] = 0;
   /* Sum all channels of the past audio and copy into Y in bit-reverse order */
   for (c=0;c<C;c++)
   {
      for (i=0;i<n2;i++)
      {
         Y[2*bitrev[i]] += SHR32(y[C*(2*i)+c],INPUT_SHIFT);
         Y[2*bitrev[i]+1] += SHR32(y[C*(2*i+1)+c],INPUT_SHIFT);
      }
   }
   normalise16(Y, lag, 8192);
   /* Forward real FFT (in-place) */
   kf_work((kiss_fft_cpx*)Y, NULL, 1,1, fft->substate->factors,fft->substate, 1, 1, 1);
   kiss_fftr_twiddles(fft,Y);

   /* Compute cross-spectrum using the inverse masking curve as weighting */
   for (i=1;i<n2;i++)
   {
      celt_word16_t n;
      celt_word32_t tmp;
      /*printf ("%d %d ", X[2*i]*X[2*i]+X[2*i+1]*X[2*i+1], Y[2*i]*Y[2*i]+Y[2*i+1]*Y[2*i+1]);*/
      n = DIV32_16(Q15ONE,celt_sqrt(EPSILON+curve[i]));
      /*printf ("%f ", n);*/
      tmp = X[2*i];
      X[2*i] = MULT16_32_Q15(n, ADD32(MULT16_16(X[2*i  ],Y[2*i  ]), MULT16_16(X[2*i+1],Y[2*i+1])));
      X[2*i+1] = MULT16_32_Q15(n, SUB32(MULT16_16(tmp,Y[2*i+1]), MULT16_16(X[2*i+1],Y[2*i  ])));
   }
   /*printf ("\n");*/
   X[0] = X[1] = 0;
   /*for (i=0;i<lag;i++) printf ("%d ", X[i]);printf ("\n");*/
   normalise16(X, lag, 50);
   /* Inverse half-complex to real FFT gives us the correlation */
   kiss_fftri(fft, X, Y);
   
   /* The peak in the correlation gives us the pitch */
   max_corr=-VERY_LARGE32;
   *pitch = 0;
   for (i=0;i<lag-len;i++)
   {
      /*printf ("%f ", xx[i]);*/
      if (Y[i] > max_corr)
      {
         *pitch = i;
         max_corr = Y[i];
      }
   }
   RESTORE_STACK;
}
