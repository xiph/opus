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

#include <math.h>
#include <stdlib.h>
#include "cwrs.h"
#include "vq.h"


/* Improved algebraic pulse-base quantiser. The signal x is replaced by the sum of the pitch 
   a combination of pulses such that its norm is still equal to 1. The only difference with 
   the quantiser above is that the search is more complete. */
void alg_quant(float *x, int N, int K, float *p, ec_enc *enc)
{
   int L = 5;
   //float tata[200];
   float y[L][N];
   int iy[L][N];
   //float tata2[200];
   float ny[L][N];
   int iny[L][N];
   int i, j, m;
   float xy[L], nxy[L];
   float yy[L], nyy[L];
   float yp[L], nyp[L];
   float best_scores[L];
   float Rpp=0, Rxp=0;
   float gain[L];
   int maxL = 1;
   float alpha = .9;
   
   for (j=0;j<N;j++)
      Rpp += p[j]*p[j];
   //if (Rpp>.01)
   //   alpha = (1-sqrt(1-Rpp))/Rpp;
   for (j=0;j<N;j++)
      Rxp += x[j]*p[j];
   for (m=0;m<L;m++)
      for (i=0;i<N;i++)
         y[m][i] = 0;
      
   for (m=0;m<L;m++)
      for (i=0;i<N;i++)
         ny[m][i] = 0;

   for (m=0;m<L;m++)
      for (i=0;i<N;i++)
         iy[m][i] = iny[m][i] = 0;

   for (m=0;m<L;m++)
      xy[m] = yy[m] = yp[m] = gain[m] = 0;
   
   for (i=0;i<K;i++)
   {
      int L2 = L;
      if (L>maxL)
      {
         L2 = maxL;
         maxL *= N;
      }
      for (m=0;m<L;m++)
         best_scores[m] = -1e10;

      for (m=0;m<L2;m++)
      {
         for (j=0;j<N;j++)
         {
            int sign;
            for (sign=-1;sign<=1;sign+=2)
            {
               if (iy[m][j]*sign < 0)
                  continue;
               //fprintf (stderr, "%d/%d %d/%d %d/%d\n", i, K, m, L2, j, N);
               float tmp_xy, tmp_yy, tmp_yp;
               float score;
               float g;
               float s = sign;
               tmp_xy = xy[m] + s*x[j]               - alpha*s*p[j]*Rxp;
               tmp_yy = yy[m] + 2*s*y[m][j] + 1      +alpha*alpha*p[j]*p[j]*Rpp - 2*alpha*s*p[j]*yp[m] - 2*alpha*p[j]*p[j];
               tmp_yp = yp[m] + s*p[j]               *(1-alpha*Rpp);
               g = (sqrt(tmp_yp*tmp_yp + tmp_yy - tmp_yy*Rpp) - tmp_yp)/tmp_yy;
               score = 2*g*tmp_xy - g*g*tmp_yy;

               if (score>best_scores[L-1])
               {
                  int k, n;
                  int id = L-1;
                  while (id > 0 && score > best_scores[id-1])
                     id--;
               
                  for (k=L-1;k>id;k--)
                  {
                     nxy[k] = nxy[k-1];
                     nyy[k] = nyy[k-1];
                     nyp[k] = nyp[k-1];
                     //fprintf(stderr, "%d %d \n", N, k);
                     for (n=0;n<N;n++)
                        ny[k][n] = ny[k-1][n];
                     for (n=0;n<N;n++)
                        iny[k][n] = iny[k-1][n];
                     gain[k] = gain[k-1];
                     best_scores[k] = best_scores[k-1];
                  }

                  nxy[id] = tmp_xy;
                  nyy[id] = tmp_yy;
                  nyp[id] = tmp_yp;
                  gain[id] = g;
                  for (n=0;n<N;n++)
                     ny[id][n] = y[m][n];
                  ny[id][j] += s;
                  for (n=0;n<N;n++)
                     ny[id][n] -= alpha*s*p[j]*p[n];
               
                  for (n=0;n<N;n++)
                     iny[id][n] = iy[m][n];
                  if (s>0)
                     iny[id][j] += 1;
                  else
                     iny[id][j] -= 1;
                  best_scores[id] = score;
               }
            }   
         }
         
      }
      int k,n;
      for (k=0;k<L;k++)
      {
         xy[k] = nxy[k];
         yy[k] = nyy[k];
         yp[k] = nyp[k];
         for (n=0;n<N;n++)
            y[k][n] = ny[k][n];
         for (n=0;n<N;n++)
            iy[k][n] = iny[k][n];
      }

   }
   
   for (i=0;i<N;i++)
      x[i] = p[i]+gain[0]*y[0][i];
   if (0) {
      float E=1e-15;
      int ABS = 0;
      for (i=0;i<N;i++)
         ABS += abs(iy[0][i]);
      //if (K != ABS)
      //   printf ("%d %d\n", K, ABS);
      for (i=0;i<N;i++)
         E += x[i]*x[i];
      //printf ("%f\n", E);
      E = 1/sqrt(E);
      for (i=0;i<N;i++)
         x[i] *= E;
   }
   int comb[K];
   int signs[K];
   //for (i=0;i<N;i++)
   //   printf ("%d ", iy[0][i]);
   pulse2comb(N, K, comb, signs, iy[0]); 
   ec_enc_uint(enc,icwrs(N, K, comb, signs),ncwrs(N, K));
}

static const float pg[5] = {1.f, .82f, .75f, 0.7f, 0.6f};

/* Finds the right offset into Y and copy it */
void copy_quant(float *x, int N, int K, float *Y, int B, int N0, ec_enc *enc)
{
   int i,j;
   int best=0;
   float best_score=0;
   float s = 1;
   int sign;
   float E;
   for (i=0;i<N0*B-N;i+=B)
   {
      int j;
      float xy=0, yy=0;
      float score;
      for (j=0;j<N;j++)
      {
         xy += x[j]*Y[i+j];
         yy += Y[i+j]*Y[i+j];
      }
      score = xy*xy/(.001+yy);
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
   //printf ("%d %d ", sign, best);
   ec_enc_uint(enc,sign,2);
   ec_enc_uint(enc,best/B,N0-N/B);
   //printf ("%d %f\n", best, best_score);
   if (K==0)
   {
      E = 1e-10;
      for (j=0;j<N;j++)
      {
         x[j] = s*Y[best+j];
         E += x[j]*x[j];
      }
      E = 1/sqrt(E);
      for (j=0;j<N;j++)
         x[j] *= E;
   } else {
      float P[N];
      float pred_gain;
      if (K>4)
         pred_gain = .5;
      else
         pred_gain = pg[K];
      E = 1e-10;
      for (j=0;j<N;j++)
      {
         P[j] = s*Y[best+j];
         E += P[j]*P[j];
      }
      E = .8/sqrt(E);
      for (j=0;j<N;j++)
         P[j] *= E;
      alg_quant(x, N, K, P, enc);
   }
}

void alg_unquant(float *x, int N, int K, float *p, ec_dec *dec)
{
   int i;
   unsigned int id;
   int comb[K];
   int signs[K];
   int iy[N];
   float y[N];
   float alpha = .9;
   float Rpp=0, Ryp=0, Ryy=0;
   float g;
   
   id = ec_dec_uint(dec, ncwrs(N, K));
   cwrsi(N, K, id, comb, signs);
   comb2pulse(N, K, iy, comb, signs);
   //for (i=0;i<N;i++)
   //   printf ("%d ", iy[i]);
   for (i=0;i<N;i++)
      Rpp += p[i]*p[i];

   for (i=0;i<N;i++)
      Ryp += iy[i]*p[i];

   for (i=0;i<N;i++)
      y[i] = iy[i] - alpha*Ryp*p[i];
   
   /* Recompute after the projection (I think it's right) */
   Ryp = 0;
   for (i=0;i<N;i++)
      Ryp += y[i]*p[i];
   
   for (i=0;i<N;i++)
      Ryy += y[i]*y[i];

   g = (sqrt(Ryp*Ryp + Ryy - Ryy*Rpp) - Ryp)/Ryy;
   
   for (i=0;i<N;i++)
      x[i] = p[i] + g*y[i];
}

void copy_unquant(float *x, int N, int K, float *Y, int B, int N0, ec_dec *dec)
{
   int j;
   int sign;
   float s;
   int best;
   float E;
   sign = ec_dec_uint(dec, 2);
   if (sign == 0)
      s = 1;
   else
      s = -1;
   
   best = B*ec_dec_uint(dec, N0-N/B);
   //printf ("%d %d ", sign, best);

   if (K==0)
   {
      E = 1e-10;
      for (j=0;j<N;j++)
      {
         x[j] = s*Y[best+j];
         E += x[j]*x[j];
      }
      E = 1/sqrt(E);
      for (j=0;j<N;j++)
         x[j] *= E;
   } else {
      float P[N];
      float pred_gain;
      if (K>4)
         pred_gain = .5;
      else
         pred_gain = pg[K];
      E = 1e-10;
      for (j=0;j<N;j++)
      {
         P[j] = s*Y[best+j];
         E += P[j]*P[j];
      }
      E = .8/sqrt(E);
      for (j=0;j<N;j++)
         P[j] *= E;
      alg_unquant(x, N, K, P, dec);
   }
}
