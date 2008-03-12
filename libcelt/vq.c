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

#include <math.h>
#include <stdlib.h>
#include "mathops.h"
#include "cwrs.h"
#include "vq.h"
#include "arch.h"
#include "os_support.h"

/** Takes the pitch vector and the decoded residual vector (non-compressed), 
   applies the compression in the pitch direction, computes the gain that will
   give ||p+g*y||=1 and mixes the residual with the pitch. */
static void mix_pitch_and_residual(int *iy, celt_norm_t *X, int N, int K, const celt_norm_t *P, celt_word16_t alpha)
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
      y[i] = SUB16(SHL16(iy[i],yshift),
                   MULT16_16_Q15(alpha,MULT16_16_Q14(ROUND(Ryp,14),P[i])));

   /* Recompute after the projection (I think it's right) */
   Ryp = 0;
   for (i=0;i<N;i++)
      Ryp = MAC16_16(Ryp,y[i],P[i]);

   Ryy = 0;
   for (i=0;i<N;i++)
      Ryy = MAC16_16(Ryy, y[i],y[i]);

   /* g = (sqrt(Ryp^2 + Ryy - Rpp*Ryy)-Ryp)/Ryy */
   g = DIV32(SHL32(celt_sqrt(MULT16_16(ROUND(Ryp,14),ROUND(Ryp,14)) + Ryy - MULT16_16(ROUND(Ryy,14),ROUND(Rpp,14))) - ROUND(Ryp,14),14),ROUND(Ryy,14));

   for (i=0;i<N;i++)
      X[i] = P[i] + MULT16_32_Q14(y[i], g);
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

void alg_quant(celt_norm_t *X, celt_mask_t *W, int N, int K, const celt_norm_t *P, celt_word16_t alpha, ec_enc *enc)
{
   int L = 3;
   VARDECL(celt_norm_t, _y);
   VARDECL(celt_norm_t, _ny);
   VARDECL(int, _iy);
   VARDECL(int, _iny);
   VARDECL(celt_norm_t *, y);
   VARDECL(celt_norm_t *, ny);
   VARDECL(int *, iy);
   VARDECL(int *, iny);
   int i, j, k, m;
   int pulsesLeft;
   VARDECL(celt_word32_t, xy);
   VARDECL(celt_word32_t, yy);
   VARDECL(celt_word32_t, yp);
   VARDECL(struct NBest, _nbest);
   VARDECL(struct NBest *, nbest);
   celt_word32_t Rpp=0, Rxp=0;
   int maxL = 1;
#ifdef FIXED_POINT
   int yshift;
#endif
   SAVE_STACK;

#ifdef FIXED_POINT
   yshift = 14-EC_ILOG(K);
#endif

   ALLOC(_y, L*N, celt_norm_t);
   ALLOC(_ny, L*N, celt_norm_t);
   ALLOC(_iy, L*N, int);
   ALLOC(_iny, L*N, int);
   ALLOC(y, L, celt_norm_t*);
   ALLOC(ny, L, celt_norm_t*);
   ALLOC(iy, L, int*);
   ALLOC(iny, L, int*);
   
   ALLOC(xy, L, celt_word32_t);
   ALLOC(yy, L, celt_word32_t);
   ALLOC(yp, L, celt_word32_t);
   ALLOC(_nbest, L, struct NBest);
   ALLOC(nbest, L, struct NBest *);
   
   for (m=0;m<L;m++)
      nbest[m] = &_nbest[m];
   
   for (m=0;m<L;m++)
   {
      ny[m] = &_ny[m*N];
      iny[m] = &_iny[m*N];
      y[m] = &_y[m*N];
      iy[m] = &_iy[m*N];
   }
   
   for (j=0;j<N;j++)
   {
      Rpp = MAC16_16(Rpp, P[j],P[j]);
      Rxp = MAC16_16(Rxp, X[j],P[j]);
   }
   Rpp = ROUND(Rpp, NORM_SHIFT);
   Rxp = ROUND(Rxp, NORM_SHIFT);
   if (Rpp>NORM_SCALING)
      celt_fatal("Rpp > 1");

   /* We only need to initialise the zero because the first iteration only uses that */
   for (i=0;i<N;i++)
      y[0][i] = 0;
   for (i=0;i<N;i++)
      iy[0][i] = 0;
   xy[0] = yy[0] = yp[0] = 0;

   pulsesLeft = K;
   while (pulsesLeft > 0)
   {
      int pulsesAtOnce=1;
      int Lupdate = L;
      int L2 = L;
      
      /* Decide on complexity strategy */
      pulsesAtOnce = pulsesLeft/N;
      if (pulsesAtOnce<1)
         pulsesAtOnce = 1;
      if (pulsesLeft-pulsesAtOnce > 3 || N > 30)
         Lupdate = 1;
      /*printf ("%d %d %d/%d %d\n", Lupdate, pulsesAtOnce, pulsesLeft, K, N);*/
      L2 = Lupdate;
      if (L2>maxL)
      {
         L2 = maxL;
         maxL *= N;
      }

      for (m=0;m<Lupdate;m++)
         nbest[m]->score = -VERY_LARGE32;

      for (m=0;m<L2;m++)
      {
         for (j=0;j<N;j++)
         {
            int sign;
            /*if (x[j]>0) sign=1; else sign=-1;*/
            for (sign=-1;sign<=1;sign+=2)
            {
               /*fprintf (stderr, "%d/%d %d/%d %d/%d\n", i, K, m, L2, j, N);*/
               celt_word32_t Rxy, Ryy, Ryp;
               celt_word16_t spj, aspj; /* Intermediate results */
               celt_word32_t score;
               celt_word32_t g;
               celt_word16_t s = SHL16(sign*pulsesAtOnce, yshift);
               
               /* All pulses at one location must have the same sign. */
               if (iy[m][j]*sign < 0)
                  continue;

               spj = MULT16_16_P14(s, P[j]);
               aspj = MULT16_16_P15(alpha, spj);
               /* Updating the sums of the new pulse(s) */
               Rxy = xy[m] + MULT16_16(s,X[j])     - MULT16_16(MULT16_16_P15(alpha,spj),Rxp);
               Ryy = yy[m] + 2*MULT16_16(s,y[m][j]) + MULT16_16(s,s)   +MULT16_16(aspj,MULT16_16_Q14(aspj,Rpp)) - 2*MULT16_32_Q14(aspj,yp[m]) - 2*MULT16_16(s,MULT16_16_Q14(aspj,P[j]));
               Ryp = yp[m] + MULT16_16(spj, SUB16(QCONST16(1.f,14),MULT16_16_Q15(alpha,Rpp)));
               
               /* Compute the gain such that ||p + g*y|| = 1 */
               g = MULT32_32_Q31(
                     SHL32(celt_sqrt(MULT16_16(ROUND(Ryp,14),ROUND(Ryp,14)) + Ryy -
                                     MULT16_16(ROUND(Ryy,14),Rpp))
                           - ROUND(Ryp,14), 14),
                     celt_rcp(ROUND(Ryy,14)));
               /* Knowing that gain, what's the error: (x-g*y)^2 
                  (result is negated and we discard x^2 because it's constant) */
               /*score = 2.f*g*Rxy - 1.f*g*g*Ryy*NORM_SCALING_1;*/
               score = 2*MULT16_32_Q14(ROUND(Rxy,14),g) -
                     MULT16_32_Q14(EXTRACT16(MULT16_32_Q14(ROUND(Ryy,14),g)),g);

               if (score>nbest[Lupdate-1]->score)
               {
                  int k;
                  int id = Lupdate-1;
                  struct NBest *tmp_best;

                  /* Save some pointers that would be deleted and use them for the current entry*/
                  tmp_best = nbest[Lupdate-1];
                  while (id > 0 && score > nbest[id-1]->score)
                     id--;
               
                  for (k=Lupdate-1;k>id;k--)
                     nbest[k] = nbest[k-1];

                  nbest[id] = tmp_best;
                  nbest[id]->score = score;
                  nbest[id]->pos = j;
                  nbest[id]->orig = m;
                  nbest[id]->sign = sign;
                  nbest[id]->xy = Rxy;
                  nbest[id]->yy = Ryy;
                  nbest[id]->yp = Ryp;
               }
            }
         }

      }
      
      if (!(nbest[0]->score > -VERY_LARGE32))
         celt_fatal("Could not find any match in VQ codebook. Something got corrupted somewhere.");
      /* Only now that we've made the final choice, update ny/iny and others */
      for (k=0;k<Lupdate;k++)
      {
         int n;
         int is;
         celt_norm_t s;
         is = nbest[k]->sign*pulsesAtOnce;
         s = SHL16(is, yshift);
         for (n=0;n<N;n++)
            ny[k][n] = y[nbest[k]->orig][n] - MULT16_16_Q15(alpha,MULT16_16_Q14(s,MULT16_16_Q14(P[nbest[k]->pos],P[n])));
         ny[k][nbest[k]->pos] += s;

         for (n=0;n<N;n++)
            iny[k][n] = iy[nbest[k]->orig][n];
         iny[k][nbest[k]->pos] += is;

         xy[k] = nbest[k]->xy;
         yy[k] = nbest[k]->yy;
         yp[k] = nbest[k]->yp;
      }
      /* Swap ny/iny with y/iy */
      for (k=0;k<Lupdate;k++)
      {
         celt_norm_t *tmp_ny;
         int *tmp_iny;

         tmp_ny = ny[k];
         ny[k] = y[k];
         y[k] = tmp_ny;
         tmp_iny = iny[k];
         iny[k] = iy[k];
         iy[k] = tmp_iny;
      }
      pulsesLeft -= pulsesAtOnce;
   }
   
#if 0
   if (0) {
      celt_word32_t err=0;
      for (i=0;i<N;i++)
         err += (x[i]-nbest[0]->gain*y[0][i])*(x[i]-nbest[0]->gain*y[0][i]);
      /*if (N<=10)
        printf ("%f %d %d\n", err, K, N);*/
   }
   /* Sanity checks, don't bother */
   if (0) {
      for (i=0;i<N;i++)
         x[i] = p[i]+nbest[0]->gain*y[0][i];
      celt_word32_t E=1e-15;
      int ABS = 0;
      for (i=0;i<N;i++)
         ABS += abs(iy[0][i]);
      /*if (K != ABS)
         printf ("%d %d\n", K, ABS);*/
      for (i=0;i<N;i++)
         E += x[i]*x[i];
      /*printf ("%f\n", E);*/
      E = 1/sqrt(E);
      for (i=0;i<N;i++)
         x[i] *= E;
   }
#endif
   
   encode_pulses(iy[0], N, K, enc);
   
   /* Recompute the gain in one pass to reduce the encoder-decoder mismatch
      due to the recursive computation used in quantisation.
      Not quite sure whether we need that or not */
   mix_pitch_and_residual(iy[0], X, N, K, P, alpha);
   RESTORE_STACK;
}

/** Decode pulse vector and combine the result with the pitch vector to produce
    the final normalised signal in the current band. */
void alg_unquant(celt_norm_t *X, int N, int K, celt_norm_t *P, celt_word16_t alpha, ec_dec *dec)
{
   VARDECL(int, iy);
   SAVE_STACK;
   ALLOC(iy, N, int);
   decode_pulses(iy, N, K, dec);
   mix_pitch_and_residual(iy, X, N, K, P, alpha);
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
      int j;
      celt_word32_t xy=0, yy=0;
      celt_word32_t score;
      for (j=0;j<N;j++)
      {
         xy = MAC16_16(xy, x[j], Y[i+N-j-1]);
         yy = MAC16_16(yy, Y[i+N-j-1], Y[i+N-j-1]);
      }
      score = DIV32(MULT16_16(ROUND(xy,14),ROUND(xy,14)), ROUND(yy,14));
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
   pred_gain = MULT16_16_Q15(pred_gain,DIV32_16(SHL32(EXTEND32(1),14+8),celt_sqrt(E)));
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
   pred_gain = MULT16_16_Q15(pred_gain,DIV32_16(SHL32(EXTEND32(1),14+8),celt_sqrt(E)));
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
   if (N0 >= Nmax/2)
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
   g = DIV32_16(SHL32(EXTEND32(1),14+8),celt_sqrt(E));
   for (j=0;j<N;j++)
      P[j] = PSHR32(MULT16_16(g, P[j]),8);
   for (j=0;j<N;j++)
      x[j] = P[j];
}

