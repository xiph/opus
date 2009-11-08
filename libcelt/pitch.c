/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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
#include "modes.h"
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
static void normalise16(celt_word16 *x, int len, celt_word16 val)
{
   int i;
   celt_word16 maxabs;
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

void find_spectral_pitch(const CELTMode *m, kiss_fftr_cfg fft, const struct PsyDecay *decay, const celt_sig * restrict x, const celt_sig * restrict y, const celt_word16 * restrict window, celt_word16 * restrict spectrum, int len, int max_pitch, int *pitch, int _C)
{
   int c, i;
   VARDECL(celt_word16, _X);
   VARDECL(celt_word16, _Y);
   const celt_word16 * restrict wptr;
#ifndef SHORTCUTS
   VARDECL(celt_mask, curve);
#endif
   celt_word16 * restrict X, * restrict Y;
   celt_word16 * restrict Xptr, * restrict Yptr;
   const celt_sig * restrict yptr;
   int n2;
   int L2;
   const int C = CHANNELS(_C);
   const int overlap = OVERLAP(m);
   const int lag = MAX_PERIOD;
   SAVE_STACK;
   n2 = lag>>1;
   L2 = len>>1;
   ALLOC(_X, lag, celt_word16);
   X = _X;
#ifndef SHORTCUTS
   ALLOC(curve, n2, celt_mask);
#endif
   CELT_MEMSET(X,0,lag);
   /* Sum all channels of the current frame and copy into X in bit-reverse order */
   for (c=0;c<C;c++)
   {
      const celt_sig * restrict xptr = &x[c];
      for (i=0;i<L2;i++)
      {
         X[2*BITREV(fft,i)] += SHR32(*xptr,INPUT_SHIFT);
         xptr += C;
         X[2*BITREV(fft,i)+1] += SHR32(*xptr,INPUT_SHIFT);
         xptr += C;
      }
   }
   /* Applying the window in the bit-reverse domain. It's a bit weird, but it
      can help save memory */
   wptr = window;
   for (i=0;i<overlap>>1;i++)
   {
      X[2*BITREV(fft,i)]        = MULT16_16_Q15(wptr[0], X[2*BITREV(fft,i)]);
      X[2*BITREV(fft,i)+1]      = MULT16_16_Q15(wptr[1], X[2*BITREV(fft,i)+1]);
      X[2*BITREV(fft,L2-i-1)]   = MULT16_16_Q15(wptr[1], X[2*BITREV(fft,L2-i-1)]);
      X[2*BITREV(fft,L2-i-1)+1] = MULT16_16_Q15(wptr[0], X[2*BITREV(fft,L2-i-1)+1]);
      wptr += 2;
   }
   normalise16(X, lag, 8192);
   /*for (i=0;i<lag;i++) printf ("%d ", X[i]);printf ("\n");*/
   /* Forward real FFT (in-place) */
   real16_fft_inplace(fft, X, lag);

   if (spectrum)
   {
      for (i=0;i<lag/4;i++)
      {
         spectrum[2*i] = X[4*i];
         spectrum[2*i+1] = X[4*i+1];
      }
   }
#ifndef SHORTCUTS
   compute_masking(decay, X, curve, lag);
#endif
   
   /* Deferred allocation to reduce peak stack usage */
   ALLOC(_Y, lag, celt_word16);
   Y = _Y;
   yptr = &y[0];
   /* Copy first channel of the past audio into Y in bit-reverse order */
   for (i=0;i<n2;i++)
   {
      Y[2*BITREV(fft,i)] = SHR32(*yptr,INPUT_SHIFT);
      yptr += C;
      Y[2*BITREV(fft,i)+1] = SHR32(*yptr,INPUT_SHIFT);
      yptr += C;
   }
   /* Add remaining channels into Y in bit-reverse order */
   for (c=1;c<C;c++)
   {
      yptr = &y[c];
      for (i=0;i<n2;i++)
      {
         Y[2*BITREV(fft,i)] += SHR32(*yptr,INPUT_SHIFT);
         yptr += C;
         Y[2*BITREV(fft,i)+1] += SHR32(*yptr,INPUT_SHIFT);
         yptr += C;
      }
   }
   normalise16(Y, lag, 8192);
   /* Forward real FFT (in-place) */
   real16_fft_inplace(fft, Y, lag);

   /* Compute cross-spectrum using the inverse masking curve as weighting */
   Xptr = &X[2];
   Yptr = &Y[2];
   for (i=1;i<n2;i++)
   {
      celt_word16 Xr, Xi, n;
      /* weight = 1/sqrt(curve) */
      Xr = Xptr[0];
      Xi = Xptr[1];
#ifdef SHORTCUTS
      /*n = SHR32(32767,(celt_ilog2(EPSILON+curve[i])>>1));*/
      n = 1+(8192>>(celt_ilog2(1+MULT16_16(Xr,Xr)+MULT16_16(Xi,Xi))>>1));
      /* Pre-multiply X by n, so we can keep everything in 16 bits */
      Xr = MULT16_16_16(n, Xr);
      Xi = MULT16_16_16(n, Xi);
#else
      {
         celt_word32 t;
#ifdef FIXED_POINT
         int k;
#endif
         t = EPSILON+curve[i];
#ifdef FIXED_POINT
         k = celt_ilog2(t)>>1;
#endif
         t = VSHR32(t, (k-7)<<1);
         n = celt_rsqrt_norm(t);
         /* Pre-multiply X by n, so we can keep everything in 16 bits */
         Xr = EXTRACT16(PSHR32(MULT16_16(n, Xr),3+k));
         Xi = EXTRACT16(PSHR32(MULT16_16(n, Xi),3+k));
      }
#endif
      /* Cross-spectrum between X and conj(Y) */
      *Xptr++ = ADD16(MULT16_16_Q15(Xr, Yptr[0]), MULT16_16_Q15(Xi,Yptr[1]));
      *Xptr++ = SUB16(MULT16_16_Q15(Xr, Yptr[1]), MULT16_16_Q15(Xi,Yptr[0]));
      Yptr += 2;
   }
   /*printf ("\n");*/
   X[0] = X[1] = 0;
   /*for (i=0;i<lag;i++) printf ("%d ", X[i]);printf ("\n");*/
   normalise16(X, lag, 50);
   /* Inverse half-complex to real FFT gives us the correlation */
   real16_ifft(fft, X, Y, lag);
   
   /* The peak in the correlation gives us the pitch */
   *pitch = find_max16(Y, max_pitch);
   /*printf ("%d ", *pitch);*/
   RESTORE_STACK;
}

void find_best_pitch(celt_word32 *xcorr, celt_word32 maxcorr, celt_word16 *y, int yshift, int len, int max_pitch, int best_pitch[2])
{
   int i, j;
   celt_word32 Syy=1;
   celt_word16 best_num[2];
   celt_word32 best_den[2];
#ifdef FIXED_POINT
   int xshift;

   xshift = celt_ilog2(maxcorr)-14;
#endif

   best_num[0] = -1;
   best_num[1] = -1;
   best_den[0] = 0;
   best_den[1] = 0;
   best_pitch[0] = 0;
   best_pitch[1] = 1;
   for (j=0;j<len;j++)
      Syy = MAC16_16(Syy, y[j],y[j]);
   for (i=0;i<max_pitch;i++)
   {
      float score;
      if (xcorr[i]>0)
      {
         celt_word16 num;
         celt_word32 xcorr16;
         xcorr16 = EXTRACT16(VSHR32(xcorr[i], xshift));
         num = MULT16_16_Q15(xcorr16,xcorr16);
         score = num*1./Syy;
         if (MULT16_32_Q15(num,best_den[1]) > MULT16_32_Q15(best_num[1],Syy))
         {
            if (MULT16_32_Q15(num,best_den[0]) > MULT16_32_Q15(best_num[0],Syy))
            {
               best_num[1] = best_num[0];
               best_den[1] = best_den[0];
               best_pitch[1] = best_pitch[0];
               best_num[0] = num;
               best_den[0] = Syy;
               best_pitch[0] = i;
            } else {
               best_num[1] = num;
               best_den[1] = Syy;
               best_pitch[1] = i;
            }
         }
      }
      Syy += SHR32(MULT16_16(y[i+len],y[i+len]),yshift) - SHR32(MULT16_16(y[i],y[i]),yshift);
      Syy = MAX32(1, Syy);
   }
}

void find_temporal_pitch(const CELTMode *m, const celt_sig * restrict x, celt_word16 * restrict y, int len, int max_pitch, int *pitch, int _C, celt_sig *xmem)
{
   int i, j;
   const int C = CHANNELS(_C);
   const int lag = MAX_PERIOD;
   const int N = FRAMESIZE(m);
   int best_pitch[2]={0};
   celt_word16 x_lp[len>>1];
   celt_word16 x_lp4[len>>2];
   celt_word16 y_lp4[lag>>2];
   celt_word32 xcorr[max_pitch>>1];
   celt_word32 maxcorr=1;
   int offset;
   int shift=0;

   /* Down-sample by two and downmix to mono */
   for (i=1;i<len>>1;i++)
      x_lp[i] = SHR32(HALF32(HALF32(x[(2*i-1)*C]+x[(2*i+1)*C])+x[2*i*C]), SIG_SHIFT);
   x_lp[0] = SHR32(HALF32(HALF32(*xmem+x[C])+x[0]), SIG_SHIFT);
   *xmem = x[N-C];
   if (C==2)
   {
      for (i=1;i<len>>1;i++)
      x_lp[i] = SHR32(HALF32(HALF32(x[(2*i-1)*C+1]+x[(2*i+1)*C+1])+x[2*i*C+1]), SIG_SHIFT);
      x_lp[0] += SHR32(HALF32(HALF32(x[C+1])+x[1]), SIG_SHIFT);
      *xmem += x[N-C+1];
   }

   /* Downsample by 2 again */
   for (j=0;j<len>>2;j++)
      x_lp4[j] = x_lp[2*j];
   for (j=0;j<lag>>2;j++)
      y_lp4[j] = y[2*j];

#ifdef FIXED_POINT
   shift = celt_ilog2(MAX16(celt_maxabs16(x_lp4, len>>2), celt_maxabs16(y_lp4, lag>>2)))-11;
   if (shift>0)
   {
      for (j=0;j<len>>2;j++)
         x_lp4[j] = SHR16(x_lp4[j], shift);
      for (j=0;j<lag>>2;j++)
         y_lp4[j] = SHR16(y_lp4[j], shift);
      /* Use double the shift for a MAC */
      shift *= 2;
   } else {
      shift = 0;
   }
#endif

   /* Coarse search with 4x decimation */

   for (i=0;i<max_pitch>>2;i++)
   {
      celt_word32 sum = 0;
      for (j=0;j<len>>2;j++)
         sum = MAC16_16(sum, x_lp4[j],y_lp4[i+j]);
      xcorr[i] = MAX32(-1, sum);
      maxcorr = MAX32(maxcorr, sum);
   }
   find_best_pitch(xcorr, maxcorr, y_lp4, 0, len>>2, max_pitch>>2, best_pitch);

   /* Finer search with 2x decimation */
   maxcorr=1;
   for (i=0;i<max_pitch>>1;i++)
   {
      celt_word32 sum=0;
      xcorr[i] = 0;
      if (abs(i-2*best_pitch[0])>2 && abs(i-2*best_pitch[1])>2)
         continue;
      for (j=0;j<len>>1;j++)
         sum += SHR32(MULT16_16(x_lp[j],y[i+j]), shift);
      xcorr[i] = MAX32(-1, sum);
      maxcorr = MAX32(maxcorr, sum);
   }
   find_best_pitch(xcorr, maxcorr, y, shift, len>>1, max_pitch>>1, best_pitch);

   /* Refine by pseudo-interpolation */
   if (best_pitch[0]>0 && best_pitch[0]<(max_pitch>>1)-1)
   {
      celt_word32 a, b, c;
      a = xcorr[best_pitch[0]-1];
      b = xcorr[best_pitch[0]];
      c = xcorr[best_pitch[0]+1];
      if ((c-a) > MULT16_32_Q15(QCONST16(.7f,15),b-a))
         offset = 1;
      else if ((a-c) > MULT16_32_Q15(QCONST16(.7f,15),b-c))
         offset = -1;
      else 
         offset = 0;
   } else {
      offset = 0;
   }
   *pitch = 2*best_pitch[0]-offset;

   CELT_COPY(y, y+(N>>1), (lag-N)>>1);
   CELT_COPY(y+((lag-N)>>1), x_lp, N>>1);

   /*printf ("%d\n", *pitch);*/
}
