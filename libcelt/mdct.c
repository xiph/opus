/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2008 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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
#include "kfft_double.h"
#include <math.h>
#include "os_support.h"
#include "mathops.h"
#include "stack_alloc.h"

#ifndef M_PI
#define M_PI 3.141592653
#endif

void clt_mdct_init(mdct_lookup *l,int N)
{
   int i;
   int N2;
   l->n = N;
   N2 = N>>1;
   l->kfft = cpx32_fft_alloc(N>>2);
#ifndef ENABLE_TI_DSPLIB55
   if (l->kfft==NULL)
     return;
#endif
   l->trig = (kiss_twiddle_scalar*)celt_alloc(N2*sizeof(kiss_twiddle_scalar));
   if (l->trig==NULL)
     return;
   /* We have enough points that sine isn't necessary */
#if defined(FIXED_POINT)
#if defined(DOUBLE_PRECISION) & !defined(MIXED_PRECISION)
   for (i=0;i<N2;i++)
      l->trig[i] = SAMP_MAX*cos(2*M_PI*(i+1./8.)/N);
#else
   for (i=0;i<N2;i++)
      l->trig[i] = TRIG_UPSCALE*celt_cos_norm(DIV32(ADD32(SHL32(EXTEND32(i),17),16386),N));
#endif
#else
   for (i=0;i<N2;i++)
      l->trig[i] = cos(2*M_PI*(i+1./8.)/N);
#endif
}

void clt_mdct_clear(mdct_lookup *l)
{
   cpx32_fft_free(l->kfft);
   celt_free(l->trig);
}

void clt_mdct_forward(const mdct_lookup *l, kiss_fft_scalar *in, kiss_fft_scalar * restrict out, const celt_word16 *window, int overlap)
{
   int i;
   int N, N2, N4;
   VARDECL(kiss_fft_scalar, f);
   SAVE_STACK;
   N = l->n;
   N2 = N>>1;
   N4 = N>>2;
   ALLOC(f, N2, kiss_fft_scalar);
   
   /* Consider the input to be compused of four blocks: [a, b, c, d] */
   /* Window, shuffle, fold */
   {
      /* Temp pointers to make it really clear to the compiler what we're doing */
      const kiss_fft_scalar * restrict xp1 = in+(overlap>>1);
      const kiss_fft_scalar * restrict xp2 = in+N2-1+(overlap>>1);
      kiss_fft_scalar * restrict yp = out;
      const celt_word16 * restrict wp1 = window+(overlap>>1);
      const celt_word16 * restrict wp2 = window+(overlap>>1)-1;
      for(i=0;i<(overlap>>2);i++)
      {
         /* Real part arranged as -d-cR, Imag part arranged as -b+aR*/
         *yp++ = MULT16_32_Q15(*wp2, xp1[N2]) + MULT16_32_Q15(*wp1,*xp2);
         *yp++ = MULT16_32_Q15(*wp1, *xp1)    - MULT16_32_Q15(*wp2, xp2[-N2]);
         xp1+=2;
         xp2-=2;
         wp1+=2;
         wp2-=2;
      }
      wp1 = window;
      wp2 = window+overlap-1;
      for(;i<N4-(overlap>>2);i++)
      {
         /* Real part arranged as a-bR, Imag part arranged as -c-dR */
         *yp++ = *xp2;
         *yp++ = *xp1;
         xp1+=2;
         xp2-=2;
      }
      for(;i<N4;i++)
      {
         /* Real part arranged as a-bR, Imag part arranged as -c-dR */
         *yp++ =  -MULT16_32_Q15(*wp1, xp1[-N2]) + MULT16_32_Q15(*wp2, *xp2);
         *yp++ = MULT16_32_Q15(*wp2, *xp1)     + MULT16_32_Q15(*wp1, xp2[N2]);
         xp1+=2;
         xp2-=2;
         wp1+=2;
         wp2-=2;
      }
   }
   /* Pre-rotation */
   {
      kiss_fft_scalar * restrict yp = out;
      kiss_fft_scalar *t = &l->trig[0];
      for(i=0;i<N4;i++)
      {
         kiss_fft_scalar re, im;
         re = yp[0];
         im = yp[1];
         *yp++ = -S_MUL(re,t[0])  +  S_MUL(im,t[N4]);
         *yp++ = -S_MUL(im,t[0])  -  S_MUL(re,t[N4]);
         t++;
      }
   }

   /* N/4 complex FFT, down-scales by 4/N */
   cpx32_fft(l->kfft, out, f, N4);

   /* Post-rotate */
   {
      /* Temp pointers to make it really clear to the compiler what we're doing */
      const kiss_fft_scalar * restrict fp = f;
      kiss_fft_scalar * restrict yp1 = out;
      kiss_fft_scalar * restrict yp2 = out+N2-1;
      kiss_fft_scalar *t = &l->trig[0];
      /* Temp pointers to make it really clear to the compiler what we're doing */
      for(i=0;i<N4;i++)
      {
         *yp1 = -S_MUL(fp[1],t[N4]) + S_MUL(fp[0],t[0]);
         *yp2 = -S_MUL(fp[0],t[N4]) - S_MUL(fp[1],t[0]);
         fp += 2;
         yp1 += 2;
         yp2 -= 2;
         t++;
      }
   }
   RESTORE_STACK;
}


void clt_mdct_backward(const mdct_lookup *l, kiss_fft_scalar *in, kiss_fft_scalar * restrict out, const celt_word16 * restrict window, int overlap)
{
   int i;
   int N, N2, N4;
   VARDECL(kiss_fft_scalar, f);
   VARDECL(kiss_fft_scalar, f2);
   SAVE_STACK;
   N = l->n;
   N2 = N>>1;
   N4 = N>>2;
   ALLOC(f, N2, kiss_fft_scalar);
   ALLOC(f2, N2, kiss_fft_scalar);
   
   /* Pre-rotate */
   {
      /* Temp pointers to make it really clear to the compiler what we're doing */
      const kiss_fft_scalar * restrict xp1 = in;
      const kiss_fft_scalar * restrict xp2 = in+N2-1;
      kiss_fft_scalar * restrict yp = f2;
      kiss_fft_scalar *t = &l->trig[0];
      for(i=0;i<N4;i++) 
      {
         *yp++ = -S_MUL(*xp2, t[0])  - S_MUL(*xp1,t[N4]);
         *yp++ =  S_MUL(*xp2, t[N4]) - S_MUL(*xp1,t[0]);
         xp1+=2;
         xp2-=2;
         t++;
      }
   }

   /* Inverse N/4 complex FFT. This one should *not* downscale even in fixed-point */
   cpx32_ifft(l->kfft, f2, f, N4);
   
   /* Post-rotate */
   {
      kiss_fft_scalar * restrict fp = f;
      kiss_fft_scalar *t = &l->trig[0];

      for(i=0;i<N4;i++)
      {
         kiss_fft_scalar re, im;
         re = fp[0];
         im = fp[1];
         /* We'd scale up by 2 here, but instead it's done when mixing the windows */
         *fp++ = S_MUL(re,*t) + S_MUL(im,t[N4]);
         *fp++ = S_MUL(im,*t) - S_MUL(re,t[N4]);
         t++;
      }
   }
   /* De-shuffle the components for the middle of the window only */
   {
      const kiss_fft_scalar * restrict fp1 = f;
      const kiss_fft_scalar * restrict fp2 = f+N2-1;
      kiss_fft_scalar * restrict yp = f2;
      for(i = 0; i < N4; i++)
      {
         *yp++ =-*fp1;
         *yp++ = *fp2;
         fp1 += 2;
         fp2 -= 2;
      }
   }

   /* Mirror on both sides for TDAC */
   {
      kiss_fft_scalar * restrict fp1 = f2+N4-1;
      kiss_fft_scalar * restrict xp1 = out+N2-1;
      kiss_fft_scalar * restrict yp1 = out+N4-overlap/2;
      const celt_word16 * restrict wp1 = window;
      const celt_word16 * restrict wp2 = window+overlap-1;
      for(i = 0; i< N4-overlap/2; i++)
      {
         *xp1 = *fp1;
         xp1--;
         fp1--;
      }
      for(; i < N4; i++)
      {
         kiss_fft_scalar x1;
         x1 = *fp1--;
         *yp1++ +=-MULT16_32_Q15(*wp1, x1);
         *xp1-- += MULT16_32_Q15(*wp2, x1);
         wp1++;
         wp2--;
      }
   }
   {
      kiss_fft_scalar * restrict fp2 = f2+N4;
      kiss_fft_scalar * restrict xp2 = out+N2;
      kiss_fft_scalar * restrict yp2 = out+N-1-(N4-overlap/2);
      const celt_word16 * restrict wp1 = window;
      const celt_word16 * restrict wp2 = window+overlap-1;
      for(i = 0; i< N4-overlap/2; i++)
      {
         *xp2 = *fp2;
         xp2++;
         fp2++;
      }
      for(; i < N4; i++)
      {
         kiss_fft_scalar x2;
         x2 = *fp2++;
         *yp2--  = MULT16_32_Q15(*wp1, x2);
         *xp2++  = MULT16_32_Q15(*wp2, x2);
         wp1++;
         wp2--;
      }
   }
   RESTORE_STACK;
}


