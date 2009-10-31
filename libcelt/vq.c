/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mathops.h"
#include "cwrs.h"
#include "vq.h"
#include "arch.h"
#include "os_support.h"
#include "rate.h"

#ifndef M_PI
#define M_PI 3.141592653
#endif

static void exp_rotation(celt_norm *X, int len, int dir, int stride, int K)
{
   int i, k, iter;
   celt_word16 c, s;
   celt_word16 gain, theta;
   celt_norm *Xptr;
   gain = celt_div((celt_word32)MULT16_16(Q15_ONE,len),(celt_word32)(3+len+6*K));
   /* FIXME: Make that HALF16 instead of HALF32 */
   theta = SUB16(Q15ONE, HALF32(MULT16_16_Q15(gain,gain)));
   /*if (len==30)
   {
   for (i=0;i<len;i++)
   X[i] = 0;
   X[14] = 1;
}*/ 
   c = celt_cos_norm(EXTEND32(theta));
   s = dir*celt_cos_norm(EXTEND32(SUB16(Q15ONE,theta))); /*  sin(theta) */
   if (len > 8*stride)
      stride *= len/(8*stride);
   iter = 1;
   for (k=0;k<iter;k++)
   {
      /* We could use MULT16_16_P15 instead of MULT16_16_Q15 for more accuracy, 
      but at this point, I really don't think it's necessary */
      Xptr = X;
      for (i=0;i<len-stride;i++)
      {
         celt_norm x1, x2;
         x1 = Xptr[0];
         x2 = Xptr[stride];
         Xptr[stride] = MULT16_16_Q15(c,x2) + MULT16_16_Q15(s,x1);
         *Xptr++      = MULT16_16_Q15(c,x1) - MULT16_16_Q15(s,x2);
      }
      Xptr = &X[len-2*stride-1];
      for (i=len-2*stride-1;i>=0;i--)
      {
         celt_norm x1, x2;
         x1 = Xptr[0];
         x2 = Xptr[stride];
         Xptr[stride] = MULT16_16_Q15(c,x2) + MULT16_16_Q15(s,x1);
         *Xptr--      = MULT16_16_Q15(c,x1) - MULT16_16_Q15(s,x2);
      }
   }
   /*if (len==30)
   {
   for (i=0;i<len;i++)
   printf ("%f ", X[i]);
   printf ("\n");
   exit(0);
}*/
}


/** Takes the pitch vector and the decoded residual vector, computes the gain
    that will give ||p+g*y||=1 and mixes the residual with the pitch. */
static void normalise_residual(int * restrict iy, celt_norm * restrict X, int N, int K, celt_word32 Ryy)
{
   int i;
#ifdef FIXED_POINT
   int k;
#endif
   celt_word32 t;
   celt_word16 g;

#ifdef FIXED_POINT
   k = celt_ilog2(Ryy)>>1;
#endif
   t = VSHR32(Ryy, (k-7)<<1);
   g = celt_rsqrt_norm(t);

   i=0;
   do
      X[i] = EXTRACT16(PSHR32(MULT16_16(g, iy[i]), k+1));
   while (++i < N);
}

void alg_quant(celt_norm *X, int N, int K, int spread, ec_enc *enc)
{
   VARDECL(celt_norm, y);
   VARDECL(int, iy);
   VARDECL(celt_word16, signx);
   int j, is;
   celt_word16 s;
   int pulsesLeft;
   celt_word32 sum;
   celt_word32 xy, yy;
   int N_1; /* Inverse of N, in Q14 format (even for float) */
#ifdef FIXED_POINT
   int yshift;
#endif
   SAVE_STACK;

   K = get_pulses(K);
#ifdef FIXED_POINT
   yshift = 13-celt_ilog2(K);
#endif

   ALLOC(y, N, celt_norm);
   ALLOC(iy, N, int);
   ALLOC(signx, N, celt_word16);
   N_1 = 512/N;
   
   if (spread)
      exp_rotation(X, N, 1, spread, K);

   sum = 0;
   j=0; do {
      if (X[j]>0)
         signx[j]=1;
      else {
         signx[j]=-1;
         X[j]=-X[j];
      }
      iy[j] = 0;
      y[j] = 0;
   } while (++j<N);

   xy = yy = 0;

   pulsesLeft = K;

   /* Do a pre-search by projecting on the pyramid */
   if (K > (N>>1))
   {
      celt_word16 rcp;
      j=0; do {
         sum += X[j];
      }  while (++j<N);

#ifdef FIXED_POINT
      if (sum <= K)
#else
      if (sum <= EPSILON)
#endif
      {
         X[0] = QCONST16(1.f,14);
         j=1; do
            X[j]=0;
         while (++j<N);
         sum = QCONST16(1.f,14);
      }
      /* Do we have sufficient accuracy here? */
      rcp = EXTRACT16(MULT16_32_Q16(K-1, celt_rcp(sum)));
      j=0; do {
#ifdef FIXED_POINT
         /* It's really important to round *towards zero* here */
         iy[j] = MULT16_16_Q15(X[j],rcp);
#else
         iy[j] = floor(rcp*X[j]);
#endif
         y[j] = SHL16(iy[j],yshift);
         yy = MAC16_16(yy, y[j],y[j]);
         xy = MAC16_16(xy, X[j],y[j]);
         y[j] *= 2;
         pulsesLeft -= iy[j];
      }  while (++j<N);
   }
   celt_assert2(pulsesLeft>=1, "Allocated too many pulses in the quick pass");

   while (pulsesLeft > 0)
   {
      int pulsesAtOnce=1;
      int best_id;
      celt_word16 magnitude;
      celt_word32 best_num = -VERY_LARGE16;
      celt_word16 best_den = 0;
#ifdef FIXED_POINT
      int rshift;
#endif
      /* Decide on how many pulses to find at once */
      pulsesAtOnce = (pulsesLeft*N_1)>>9; /* pulsesLeft/N */
      if (pulsesAtOnce<1)
         pulsesAtOnce = 1;
#ifdef FIXED_POINT
      rshift = yshift+1+celt_ilog2(K-pulsesLeft+pulsesAtOnce);
#endif
      magnitude = SHL16(pulsesAtOnce, yshift);

      best_id = 0;
      /* The squared magnitude term gets added anyway, so we might as well 
         add it outside the loop */
      yy = MAC16_16(yy, magnitude,magnitude);
      /* Choose between fast and accurate strategy depending on where we are in the search */
         /* This should ensure that anything we can process will have a better score */
      j=0;
      do {
         celt_word16 Rxy, Ryy;
         /* Select sign based on X[j] alone */
         s = magnitude;
         /* Temporary sums of the new pulse(s) */
         Rxy = EXTRACT16(SHR32(MAC16_16(xy, s,X[j]),rshift));
         /* We're multiplying y[j] by two so we don't have to do it here */
         Ryy = EXTRACT16(SHR32(MAC16_16(yy, s,y[j]),rshift));
            
            /* Approximate score: we maximise Rxy/sqrt(Ryy) (we're guaranteed that 
         Rxy is positive because the sign is pre-computed) */
         Rxy = MULT16_16_Q15(Rxy,Rxy);
            /* The idea is to check for num/den >= best_num/best_den, but that way
         we can do it without any division */
         /* OPT: Make sure to use conditional moves here */
         if (MULT16_16(best_den, Rxy) > MULT16_16(Ryy, best_num))
         {
            best_den = Ryy;
            best_num = Rxy;
            best_id = j;
         }
      } while (++j<N);
      
      j = best_id;
      is = pulsesAtOnce;
      s = SHL16(is, yshift);

      /* Updating the sums of the new pulse(s) */
      xy = xy + MULT16_16(s,X[j]);
      /* We're multiplying y[j] by two so we don't have to do it here */
      yy = yy + MULT16_16(s,y[j]);

      /* Only now that we've made the final choice, update y/iy */
      /* Multiplying y[j] by 2 so we don't have to do it everywhere else */
      y[j] += 2*s;
      iy[j] += is;
      pulsesLeft -= pulsesAtOnce;
   }
   j=0;
   do {
      X[j] = MULT16_16(signx[j],X[j]);
      if (signx[j] < 0)
         iy[j] = -iy[j];
   } while (++j<N);
   encode_pulses(iy, N, K, enc);
   
   /* Recompute the gain in one pass to reduce the encoder-decoder mismatch
   due to the recursive computation used in quantisation. */
   normalise_residual(iy, X, N, K, EXTRACT16(SHR32(yy,2*yshift)));
   if (spread)
      exp_rotation(X, N, -1, spread, K);
   RESTORE_STACK;
}


/** Decode pulse vector and combine the result with the pitch vector to produce
    the final normalised signal in the current band. */
void alg_unquant(celt_norm *X, int N, int K, int spread, ec_dec *dec)
{
   int i;
   celt_word32 Ryy;
   VARDECL(int, iy);
   SAVE_STACK;
   K = get_pulses(K);
   ALLOC(iy, N, int);
   decode_pulses(iy, N, K, dec);
   Ryy = 0;
   i=0;
   do {
      Ryy = MAC16_16(Ryy, iy[i], iy[i]);
   } while (++i < N);
   normalise_residual(iy, X, N, K, Ryy);
   if (spread)
      exp_rotation(X, N, -1, spread, K);
   RESTORE_STACK;
}

celt_word16 renormalise_vector(celt_norm *X, celt_word16 value, int N, int stride)
{
   int i;
   celt_word32 E = EPSILON;
   celt_word16 rE;
   celt_word16 g;
   celt_norm *xptr = X;
   for (i=0;i<N;i++)
   {
      E = MAC16_16(E, *xptr, *xptr);
      xptr += stride;
   }

   rE = celt_sqrt(E);
#ifdef FIXED_POINT
   if (rE <= 128)
      g = Q15ONE;
   else
#endif
      g = MULT16_16_Q15(value,celt_rcp(SHL32(rE,9)));
   xptr = X;
   for (i=0;i<N;i++)
   {
      *xptr = PSHR32(MULT16_16(g, *xptr),8);
      xptr += stride;
   }
   return rE;
}

static void fold(const CELTMode *m, int N, const celt_norm * restrict Y, celt_norm * restrict P, int N0, int B)
{
   int j;
   int id = N0 % B;
   /* Here, we assume that id will never be greater than N0, i.e. that 
      no band is wider than N0. In the unlikely case it happens, we set
      everything to zero */
   /*{
	   int offset = (N0*C - (id+C*N))/2;
	   if (offset > C*N0/16)
		   offset = C*N0/16;
	   offset -= offset % (C*B);
	   if (offset < 0)
		   offset = 0;
	   //printf ("%d\n", offset);
	   id += offset;
   }*/
   if (id+N>N0)
      for (j=0;j<N;j++)
         P[j] = 0;
   else
      for (j=0;j<N;j++)
         P[j] = Y[id++];
}

void intra_fold(const CELTMode *m, int N, const celt_norm * restrict Y, celt_norm * restrict P, int N0, int B)
{
   fold(m, N, Y, P, N0, B);
   renormalise_vector(P, Q15ONE, N, 1);
}

