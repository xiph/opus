/* (C) 2007-2008 Jean-Marc Valin, CSIRO
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

#include "mathops.h"
#include "cwrs.h"
#include "vq.h"
#include "arch.h"
#include "os_support.h"

/** Takes the pitch vector and the decoded residual vector, computes the gain
    that will give ||p+g*y||=1 and mixes the residual with the pitch. */
static void mix_pitch_and_residual(int * restrict iy, celt_norm_t * restrict X, int N, int K, const celt_norm_t * restrict P)
{
   int i;
   celt_word32_t Ryp, Ryy, Rpp;
   celt_word16_t ryp, ryy, rpp;
   celt_word32_t g;
   VARDECL(celt_norm_t, y);
#ifdef FIXED_POINT
   int yshift;
#endif
   SAVE_STACK;
#ifdef FIXED_POINT
   yshift = 13-celt_ilog2(K);
#endif
   ALLOC(y, N, celt_norm_t);

   /*for (i=0;i<N;i++)
   printf ("%d ", iy[i]);*/
   Rpp = 0;
   i=0;
   do {
      Rpp = MAC16_16(Rpp,P[i],P[i]);
      y[i] = SHL16(iy[i],yshift);
   } while (++i < N);

   Ryp = 0;
   Ryy = 0;
   /* If this doesn't generate a dual MAC (on supported archs), fire the compiler guy */
   i=0;
   do {
      Ryp = MAC16_16(Ryp, y[i], P[i]);
      Ryy = MAC16_16(Ryy, y[i], y[i]);
   } while (++i < N);

   ryp = ROUND16(Ryp,14);
   ryy = ROUND16(Ryy,14);
   rpp = ROUND16(Rpp,14);
   /* g = (sqrt(Ryp^2 + Ryy - Rpp*Ryy)-Ryp)/Ryy */
   g = MULT16_32_Q15(celt_sqrt(MAC16_16(Ryy, ryp,ryp) - MULT16_16(ryy,rpp)) - ryp,
                     celt_rcp(SHR32(Ryy,9)));

   i=0;
   do 
      X[i] = ADD16(P[i], ROUND16(MULT16_16(y[i], g),11));
   while (++i < N);

   RESTORE_STACK;
}


void alg_quant(celt_norm_t *X, celt_mask_t *W, int N, int K, celt_norm_t *P, ec_enc *enc)
{
   VARDECL(celt_norm_t, y);
   VARDECL(int, iy);
   VARDECL(celt_word16_t, signx);
   int j, is;
   celt_word16_t s;
   int pulsesLeft;
   celt_word32_t sum;
   celt_word32_t xy, yy, yp;
   celt_word16_t Rpp;
   int N_1; /* Inverse of N, in Q14 format (even for float) */
#ifdef FIXED_POINT
   int yshift;
#endif
   SAVE_STACK;

#ifdef FIXED_POINT
   yshift = 13-celt_ilog2(K);
#endif

   ALLOC(y, N, celt_norm_t);
   ALLOC(iy, N, int);
   ALLOC(signx, N, celt_word16_t);
   N_1 = 512/N;

   sum = 0;
   j=0; do {
      X[j] -= P[j];
      if (X[j]>0)
         signx[j]=1;
      else {
         signx[j]=-1;
         X[j]=-X[j];
         P[j]=-P[j];
      }
      iy[j] = 0;
      y[j] = 0;
      sum = MAC16_16(sum, P[j],P[j]);
   } while (++j<N);
   Rpp = ROUND16(sum, NORM_SHIFT);

   celt_assert2(Rpp<=NORM_SCALING, "Rpp should never have a norm greater than unity");

   xy = yy = yp = 0;

   pulsesLeft = K;

   /* Do a pre-search by projecting on the pyramid */
   if (K > (N>>1))
   {
      celt_word16_t rcp;
      sum=0;
      j=0; do {
         sum += X[j];
      }  while (++j<N);
      if (sum == 0)
      {
         X[0] = 16384;
         sum = 16384;
      }
      /* Do we have sufficient accuracy here? */
      rcp = EXTRACT16(MULT16_32_Q16(K-1, celt_rcp(sum)));
      /*rcp = DIV32(SHL32(EXTEND32(K-1),15),EPSILON+sum);*/
      /*printf ("%d (%d %d)\n", rcp, N, K);*/
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
         yp += P[j]*y[j];
         y[j] *= 2;
         pulsesLeft -= iy[j];
      }  while (++j<N);
   }
   /*if (pulsesLeft > N+2)
      printf ("%d / %d (%d)\n", pulsesLeft, K, N);*/
   celt_assert2(pulsesLeft>=1, "Allocated too many pulses in the quick pass");

   while (pulsesLeft > 1)
   {
      int pulsesAtOnce=1;
      int best_id;
      celt_word16_t magnitude;
      celt_word32_t best_num = -VERY_LARGE16;
      celt_word16_t best_den = 0;
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
         celt_word16_t Rxy, Ryy;
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
      yp = yp + MULT16_16(s, P[j]);

      /* Only now that we've made the final choice, update y/iy */
      /* Multiplying y[j] by 2 so we don't have to do it everywhere else */
      y[j] += 2*s;
      iy[j] += is;
      pulsesLeft -= pulsesAtOnce;
   }
   
   if (pulsesLeft > 0)
   {
      celt_word16_t g;
      celt_word16_t best_num = -VERY_LARGE16;
      celt_word16_t best_den = 0;
      int best_id = 0;
      celt_word16_t magnitude = SHL16(1, yshift);

      /* The squared magnitude term gets added anyway, so we might as well 
      add it outside the loop */
      yy = MAC16_16(yy, magnitude,magnitude);
      j=0;
      do {
         celt_word16_t Rxy, Ryy, Ryp;
         celt_word16_t num;
         /* Select sign based on X[j] alone */
         s = magnitude;
         /* Temporary sums of the new pulse(s) */
         Rxy = ROUND16(MAC16_16(xy, s,X[j]), 14);
         /* We're multiplying y[j] by two so we don't have to do it here */
         Ryy = ROUND16(MAC16_16(yy, s,y[j]), 14);
         Ryp = ROUND16(MAC16_16(yp, s,P[j]), 14);

            /* Compute the gain such that ||p + g*y|| = 1 
         ...but instead, we compute g*Ryy to avoid dividing */
         g = celt_psqrt(MULT16_16(Ryp,Ryp) + MULT16_16(Ryy,QCONST16(1.f,14)-Rpp)) - Ryp;
            /* Knowing that gain, what's the error: (x-g*y)^2 
         (result is negated and we discard x^2 because it's constant) */
         /* score = 2*g*Rxy - g*g*Ryy;*/
#ifdef FIXED_POINT
         /* No need to multiply Rxy by 2 because we did it earlier */
         num = MULT16_16_Q15(ADD16(SUB16(Rxy,g),Rxy),g);
#else
         num = g*(2*Rxy-g);
#endif
         if (MULT16_16(best_den, num) > MULT16_16(Ryy, best_num))
         {
            best_den = Ryy;
            best_num = num;
            best_id = j;
         }
      } while (++j<N);
      iy[best_id] += 1;
   }
   j=0;
   do {
      P[j] = MULT16_16(signx[j],P[j]);
      X[j] = MULT16_16(signx[j],X[j]);
      if (signx[j] < 0)
         iy[j] = -iy[j];
   } while (++j<N);
   encode_pulses(iy, N, K, enc);
   
   /* Recompute the gain in one pass to reduce the encoder-decoder mismatch
   due to the recursive computation used in quantisation. */
   mix_pitch_and_residual(iy, X, N, K, P);
   RESTORE_STACK;
}


/** Decode pulse vector and combine the result with the pitch vector to produce
    the final normalised signal in the current band. */
void alg_unquant(celt_norm_t *X, int N, int K, celt_norm_t *P, ec_dec *dec)
{
   VARDECL(int, iy);
   SAVE_STACK;
   ALLOC(iy, N, int);
   decode_pulses(iy, N, K, dec);
   mix_pitch_and_residual(iy, X, N, K, P);
   RESTORE_STACK;
}

void renormalise_vector(celt_norm_t *X, celt_word16_t value, int N, int stride)
{
   int i;
   celt_word32_t E = EPSILON;
   celt_word16_t g;
   celt_norm_t *xptr = X;
   for (i=0;i<N;i++)
   {
      E = MAC16_16(E, *xptr, *xptr);
      xptr += stride;
   }

   g = MULT16_16_Q15(value,celt_rcp(SHL32(celt_sqrt(E),9)));
   xptr = X;
   for (i=0;i<N;i++)
   {
      *xptr = PSHR32(MULT16_16(g, *xptr),8);
      xptr += stride;
   }
}

static void fold(const CELTMode *m, int N, celt_norm_t *Y, celt_norm_t * restrict P, int N0, int B)
{
   int j;
   const int C = CHANNELS(m);
   int id = N0 % (C*B);
   /* Here, we assume that id will never be greater than N0, i.e. that 
      no band is wider than N0. In the unlikely case it happens, we set
      everything to zero */
   if (id+C*N>N0)
      for (j=0;j<C*N;j++)
         P[j] = 0;
   else
      for (j=0;j<C*N;j++)
         P[j] = Y[id++];
}

#define KGAIN 6

void intra_fold(const CELTMode *m, celt_norm_t * restrict x, int N, int K, celt_norm_t *Y, celt_norm_t * restrict P, int N0, int B)
{
   celt_word16_t pred_gain;
   const int C = CHANNELS(m);

   if (K==0)
      pred_gain = Q15ONE;
   else
      pred_gain = celt_div((celt_word32_t)MULT16_16(Q15_ONE,N),(celt_word32_t)(N+KGAIN*K));

   fold(m, N, Y, P, N0, B);

   renormalise_vector(P, pred_gain, C*N, 1);
}

