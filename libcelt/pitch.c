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

#include "pitch.h"
#include "psy.h"
#include "os_support.h"
#include "mathops.h"
#include "stack_alloc.h"

kiss_fftr_cfg pitch_state_alloc(int max_lag)
{
   return real16_fft_alloc(max_lag);
}

void pitch_state_free(kiss_fftr_cfg st)
{
   real16_fft_free(st);
}

#ifdef FIXED_POINT
static void normalise16(celt_word16_t *x, int len, celt_word16_t val)
{
   int i;
   celt_word16_t maxabs;
   maxabs = celt_maxabs16(x,len);
   if (maxabs > val)
   {
      int shift = 0;
      while (maxabs > val)
      {
         maxabs >>= 1;
         shift++;
      }
      if (shift==0)
         return;
      i=0;
      do{
         x[i] = SHR16(x[i], shift);
      } while (++i<len);
   } else {
      int shift=0;
      if (maxabs == 0)
         return;
      val >>= 1;
      while (maxabs < val)
      {
         val >>= 1;
         shift++;
      }
      if (shift==0)
         return;
      i=0;
      do{
         x[i] = SHL16(x[i], shift);
      } while (++i<len);
   }
}
#else
#define normalise16(x,len,val)
#endif

#define INPUT_SHIFT 15

void find_spectral_pitch(kiss_fftr_cfg fft, const struct PsyDecay *decay, const celt_sig_t * restrict x, const celt_sig_t * restrict y, const celt_word16_t * restrict window, int overlap, int lag, int len, int C, int *pitch)
{
   int c, i;
   VARDECL(celt_word16_t, _X);
   VARDECL(celt_word16_t, _Y);
   VARDECL(celt_mask_t, curve);
   celt_word16_t * restrict X, * restrict Y;
   int n2;
   int L2;
   SAVE_STACK;
   n2 = lag>>1;
   L2 = len>>1;
   ALLOC(_X, lag, celt_word16_t);
   X = _X;
   ALLOC(curve, n2, celt_mask_t);

   CELT_MEMSET(X,0,lag);
   /* Sum all channels of the current frame and copy into X in bit-reverse order */
   for (c=0;c<C;c++)
   {
      for (i=0;i<L2;i++)
      {
         X[2*BITREV(fft,i)] += SHR32(x[C*(2*i)+c],INPUT_SHIFT);
         X[2*BITREV(fft,i)+1] += SHR32(x[C*(2*i+1)+c],INPUT_SHIFT);
      }
   }
   /* Applying the window in the bit-reverse domain. It's a bit weird, but it
      can help save memory */
   for (i=0;i<overlap>>1;i++)
   {
      X[2*BITREV(fft,i)] = MULT16_16_Q15(window[2*i], X[2*BITREV(fft,i)]);
      X[2*BITREV(fft,i)+1] = MULT16_16_Q15(window[2*i+1], X[2*BITREV(fft,i)+1]);
      X[2*BITREV(fft,L2-i-1)] = MULT16_16_Q15(window[2*i+1], X[2*BITREV(fft,L2-i-1)]);
      X[2*BITREV(fft,L2-i-1)+1] = MULT16_16_Q15(window[2*i], X[2*BITREV(fft,L2-i-1)+1]);
   }
   normalise16(X, lag, 8192);
   /*for (i=0;i<lag;i++) printf ("%d ", X[i]);printf ("\n");*/
   /* Forward real FFT (in-place) */
   real16_fft_inplace(fft, X, lag);

   compute_masking(decay, X, curve, lag);

   /* Deferred allocation to reduce peak stack usage */
   ALLOC(_Y, lag, celt_word16_t);
   Y = _Y;
   CELT_MEMSET(Y,0,lag);
   /* Sum all channels of the past audio and copy into Y in bit-reverse order */
   for (c=0;c<C;c++)
   {
      for (i=0;i<n2;i++)
      {
         Y[2*BITREV(fft,i)] += SHR32(y[C*(2*i)+c],INPUT_SHIFT);
         Y[2*BITREV(fft,i)+1] += SHR32(y[C*(2*i+1)+c],INPUT_SHIFT);
      }
   }
   normalise16(Y, lag, 8192);
   /* Forward real FFT (in-place) */
   real16_fft_inplace(fft, Y, lag);

   /* Compute cross-spectrum using the inverse masking curve as weighting */
   for (i=1;i<n2;i++)
   {
      celt_word16_t Xr, Xi, n;
      /* weight = 1/sqrt(curve) */
      n = celt_rsqrt(EPSILON+curve[i]);
      /* Pre-multiply X by n, so we can keep everything in 16 bits */
      Xr = SHR32(MULT16_16(n, X[2*i  ]),3);
      Xi = SHR32(MULT16_16(n, X[2*i+1]),3);
      /* Cross-spectrum between X and conj(Y) */
      X[2*i]   = ADD16(MULT16_16_Q15(Xr, Y[2*i  ]), MULT16_16_Q15(Xi,Y[2*i+1]));
      X[2*i+1] = SUB16(MULT16_16_Q15(Xr, Y[2*i+1]), MULT16_16_Q15(Xi,Y[2*i  ]));
   }
   /*printf ("\n");*/
   X[0] = X[1] = 0;
   /*for (i=0;i<lag;i++) printf ("%d ", X[i]);printf ("\n");*/
   normalise16(X, lag, 50);
   /* Inverse half-complex to real FFT gives us the correlation */
   real16_ifft(fft, X, Y, lag);
   
   /* The peak in the correlation gives us the pitch */
   *pitch = find_max16(Y, lag-len);
   RESTORE_STACK;
}
