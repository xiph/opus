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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mathops.h"
#include "cwrs.h"
#include "vq.h"
#include "arch.h"
#include "os_support.h"

/** Takes the pitch vector and the decoded residual vector (non-compressed), 
   applies the compression in the pitch direction, computes the gain that will
   give ||p+g*y||=1 and mixes the residual with the pitch. */
static void mix_pitch_and_residual(int *iy, celt_norm_t *X, int N, int K, const celt_norm_t *P)
{
   int i;
   celt_word32_t Ryp, Ryy, Rpp;
   celt_word32_t g;
   VARDECL(celt_norm_t, y);
#ifdef FIXED_POINT
   int yshift;
#endif
   SAVE_STACK;
#ifdef FIXED_POINT
   yshift = 14-EC_ILOG(K);
#endif
   ALLOC(y, N, celt_norm_t);

   /*for (i=0;i<N;i++)
   printf ("%d ", iy[i]);*/
   Rpp = 0;
   for (i=0;i<N;i++)
      Rpp = MAC16_16(Rpp,P[i],P[i]);

   Ryp = 0;
   for (i=0;i<N;i++)
      Ryp = MAC16_16(Ryp,SHL16(iy[i],yshift),P[i]);

   /* Remove part of the pitch component to compute the real residual from
   the encoded (int) one */
   for (i=0;i<N;i++)
      y[i] = SHL16(iy[i],yshift);

   /* Recompute after the projection (I think it's right) */
   Ryp = 0;
   for (i=0;i<N;i++)
      Ryp = MAC16_16(Ryp,y[i],P[i]);

   Ryy = 0;
   for (i=0;i<N;i++)
      Ryy = MAC16_16(Ryy, y[i],y[i]);

   /* g = (sqrt(Ryp^2 + Ryy - Rpp*Ryy)-Ryp)/Ryy */
   g = MULT16_32_Q15(
            celt_sqrt(MULT16_16(ROUND16(Ryp,14),ROUND16(Ryp,14)) + Ryy -
                      MULT16_16(ROUND16(Ryy,14),ROUND16(Rpp,14)))
            - ROUND16(Ryp,14),
       celt_rcp(SHR32(Ryy,9)));

   for (i=0;i<N;i++)
      X[i] = P[i] + ROUND16(MULT16_16(y[i], g),11);
   RESTORE_STACK;
}

/** All the info necessary to keep track of a hypothesis during the search */
struct NBest {
   celt_word32_t score;
   int sign;
   int pos;
   int orig;
   celt_word32_t xy;
   celt_word32_t yy;
   celt_word32_t yp;
};

void alg_quant(celt_norm_t *X, celt_mask_t *W, int N, int K, const celt_norm_t *P, ec_enc *enc)
{
   VARDECL(celt_norm_t, _y);
   VARDECL(celt_norm_t, _ny);
   VARDECL(int, _iy);
   VARDECL(int, _iny);
   celt_norm_t *y, *ny;
   int *iy, *iny;
   int i, j;
   int pulsesLeft;
   celt_word32_t xy, yy, yp;
   struct NBest nbest;
   celt_word32_t Rpp=0, Rxp=0;
#ifdef FIXED_POINT
   int yshift;
#endif
   SAVE_STACK;

#ifdef FIXED_POINT
   yshift = 14-EC_ILOG(K);
#endif

   ALLOC(_y, N, celt_norm_t);
   ALLOC(_ny, N, celt_norm_t);
   ALLOC(_iy, N, int);
   ALLOC(_iny, N, int);
   y = _y;
   ny = _ny;
   iy = _iy;
   iny = _iny;
   
   for (j=0;j<N;j++)
   {
      Rpp = MAC16_16(Rpp, P[j],P[j]);
      Rxp = MAC16_16(Rxp, X[j],P[j]);
   }
   Rpp = ROUND16(Rpp, NORM_SHIFT);
   Rxp = ROUND16(Rxp, NORM_SHIFT);

   celt_assert2(Rpp<=NORM_SCALING, "Rpp should never have a norm greater than unity");

   for (i=0;i<N;i++)
      y[i] = 0;
   for (i=0;i<N;i++)
      iy[i] = 0;
   xy = yy = yp = 0;

   pulsesLeft = K;
   while (pulsesLeft > 0)
   {
      int pulsesAtOnce=1;
      
      /* Decide on how many pulses to find at once */
      pulsesAtOnce = pulsesLeft/N;
      if (pulsesAtOnce<1)
         pulsesAtOnce = 1;
      /*printf ("%d %d %d/%d %d\n", Lupdate, pulsesAtOnce, pulsesLeft, K, N);*/

      nbest.score = -VERY_LARGE32;

      for (j=0;j<N;j++)
      {
         int sign;
         /*fprintf (stderr, "%d/%d %d/%d %d/%d\n", i, K, m, L2, j, N);*/
         celt_word32_t Rxy, Ryy, Ryp;
         celt_word32_t score;
         celt_word32_t g;
         celt_word16_t s;
         
         /* Select sign based on X[j] alone */
         if (X[j]>0) sign=1; else sign=-1;
         s = SHL16(sign*pulsesAtOnce, yshift);

         /* Updating the sums of the new pulse(s) */
         Rxy = xy + MULT16_16(s,X[j]);
         Ryy = yy + 2*MULT16_16(s,y[j]) + MULT16_16(s,s);
         Ryp = yp + MULT16_16(s, P[j]);
         
         if (pulsesLeft>1)
         {
            score = MULT32_32_Q31(MULT16_16(ROUND16(Rxy,14),ABS16(ROUND16(Rxy,14))), celt_rcp(SHR32(Ryy,12)));
         } else
         {
            /* Compute the gain such that ||p + g*y|| = 1 */
            g = MULT16_32_Q15(
                     celt_sqrt(MULT16_16(ROUND16(Ryp,14),ROUND16(Ryp,14)) + Ryy -
                               MULT16_16(ROUND16(Ryy,14),Rpp))
                     - ROUND16(Ryp,14),
                celt_rcp(SHR32(Ryy,12)));
            /* Knowing that gain, what's the error: (x-g*y)^2 
               (result is negated and we discard x^2 because it's constant) */
            /* score = 2.f*g*Rxy - 1.f*g*g*Ryy*NORM_SCALING_1;*/
            score = 2*MULT16_32_Q14(ROUND16(Rxy,14),g)
                    - MULT16_32_Q14(EXTRACT16(MULT16_32_Q14(ROUND16(Ryy,14),g)),g);
         }
         
         if (score>nbest.score)
         {
            nbest.score = score;
            nbest.pos = j;
            nbest.orig = 0;
            nbest.sign = sign;
            nbest.xy = Rxy;
            nbest.yy = Ryy;
            nbest.yp = Ryp;
         }
      }

      celt_assert2(nbest[0]->score > -VERY_LARGE32, "Could not find any match in VQ codebook. Something got corrupted somewhere.");

      /* Only now that we've made the final choice, update ny/iny and others */
      {
         int n;
         int is;
         celt_norm_t s;
         is = nbest.sign*pulsesAtOnce;
         s = SHL16(is, yshift);
         for (n=0;n<N;n++)
            ny[n] = y[n];
         ny[nbest.pos] += s;

         for (n=0;n<N;n++)
            iny[n] = iy[n];
         iny[nbest.pos] += is;

         xy = nbest.xy;
         yy = nbest.yy;
         yp = nbest.yp;
      }
      /* Swap ny/iny with y/iy */
      {
         celt_norm_t *tmp_ny;
         int *tmp_iny;

         tmp_ny = ny;
         ny = y;
         y = tmp_ny;
         tmp_iny = iny;
         iny = iy;
         iy = tmp_iny;
      }
      pulsesLeft -= pulsesAtOnce;
   }
   
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

#ifdef FIXED_POINT
static const celt_word16_t pg[11] = {32767, 24576, 21299, 19661, 19661, 19661, 18022, 18022, 16384, 16384, 16384};
#else
static const celt_word16_t pg[11] = {1.f, .75f, .65f, 0.6f, 0.6f, .6f, .55f, .55f, .5f, .5f, .5f};
#endif

void intra_prediction(celt_norm_t *x, celt_mask_t *W, int N, int K, celt_norm_t *Y, celt_norm_t *P, int B, int N0, ec_enc *enc)
{
   int i,j;
   int best=0;
   celt_word32_t best_score=0;
   celt_word16_t s = 1;
   int sign;
   celt_word32_t E;
   celt_word16_t pred_gain;
   int max_pos = N0-N/B;
   if (max_pos > 32)
      max_pos = 32;

   for (i=0;i<max_pos*B;i+=B)
   {
      celt_word32_t xy=0, yy=0;
      celt_word32_t score;
      for (j=0;j<N;j++)
      {
         xy = MAC16_16(xy, x[j], Y[i+N-j-1]);
         yy = MAC16_16(yy, Y[i+N-j-1], Y[i+N-j-1]);
      }
      score = celt_div(MULT16_16(ROUND16(xy,14),ROUND16(xy,14)), ROUND16(yy,14));
      if (score > best_score)
      {
         best_score = score;
         best = i;
         if (xy>0)
            s = 1;
         else
            s = -1;
      }
   }
   if (s<0)
      sign = 1;
   else
      sign = 0;
   /*printf ("%d %d ", sign, best);*/
   ec_enc_uint(enc,sign,2);
   ec_enc_uint(enc,best/B,max_pos);
   /*printf ("%d %f\n", best, best_score);*/
   
   if (K>10)
      pred_gain = pg[10];
   else
      pred_gain = pg[K];
   E = EPSILON;
   for (j=0;j<N;j++)
   {
      P[j] = s*Y[best+N-j-1];
      E = MAC16_16(E, P[j],P[j]);
   }
   /*pred_gain = pred_gain/sqrt(E);*/
   pred_gain = MULT16_16_Q15(pred_gain,celt_rcp(SHL32(celt_sqrt(E),9)));
   for (j=0;j<N;j++)
      P[j] = PSHR32(MULT16_16(pred_gain, P[j]),8);
   if (K>0)
   {
      for (j=0;j<N;j++)
         x[j] -= P[j];
   } else {
      for (j=0;j<N;j++)
         x[j] = P[j];
   }
   /*printf ("quant ");*/
   /*for (j=0;j<N;j++) printf ("%f ", P[j]);*/

}

void intra_unquant(celt_norm_t *x, int N, int K, celt_norm_t *Y, celt_norm_t *P, int B, int N0, ec_dec *dec)
{
   int j;
   int sign;
   celt_word16_t s;
   int best;
   celt_word32_t E;
   celt_word16_t pred_gain;
   int max_pos = N0-N/B;
   if (max_pos > 32)
      max_pos = 32;
   
   sign = ec_dec_uint(dec, 2);
   if (sign == 0)
      s = 1;
   else
      s = -1;
   
   best = B*ec_dec_uint(dec, max_pos);
   /*printf ("%d %d ", sign, best);*/

   if (K>10)
      pred_gain = pg[10];
   else
      pred_gain = pg[K];
   E = EPSILON;
   for (j=0;j<N;j++)
   {
      P[j] = s*Y[best+N-j-1];
      E = MAC16_16(E, P[j],P[j]);
   }
   /*pred_gain = pred_gain/sqrt(E);*/
   pred_gain = MULT16_16_Q15(pred_gain,celt_rcp(SHL32(celt_sqrt(E),9)));
   for (j=0;j<N;j++)
      P[j] = PSHR32(MULT16_16(pred_gain, P[j]),8);
   if (K==0)
   {
      for (j=0;j<N;j++)
         x[j] = P[j];
   }
}

void intra_fold(celt_norm_t *x, int N, celt_norm_t *Y, celt_norm_t *P, int B, int N0, int Nmax)
{
   int i, j;
   celt_word32_t E;
   celt_word16_t g;
   
   E = EPSILON;
   if (N0 >= (Nmax>>1))
   {
      for (i=0;i<B;i++)
      {
         for (j=0;j<N/B;j++)
         {
            P[j*B+i] = Y[(Nmax-N0-j-1)*B+i];
            E += P[j*B+i]*P[j*B+i];
         }
      }
   } else {
      for (j=0;j<N;j++)
      {
         P[j] = Y[j];
         E = MAC16_16(E, P[j],P[j]);
      }
   }
   g = celt_rcp(SHL32(celt_sqrt(E),9));
   for (j=0;j<N;j++)
      P[j] = PSHR32(MULT16_16(g, P[j]),8);
   for (j=0;j<N;j++)
      x[j] = P[j];
}

