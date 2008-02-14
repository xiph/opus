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

struct NBest {
   float score;
   float gain;
   int sign;
   int pos;
   int orig;
   float xy;
   float yy;
   float yp;
};

/* Improved algebraic pulse-base quantiser. The signal x is replaced by the sum of the pitch 
   a combination of pulses such that its norm is still equal to 1. The only difference with 
   the quantiser above is that the search is more complete. */
void alg_quant(float *x, float *W, int N, int K, float *p, float alpha, ec_enc *enc)
{
   int L = 3;
   //float tata[200];
   float _y[L][N];
   int _iy[L][N];
   //float tata2[200];
   float _ny[L][N];
   int _iny[L][N];
   float *(ny[L]), *(y[L]);
   int *(iny[L]), *(iy[L]);
   int i, j, m;
   int pulsesLeft;
   float xy[L];
   float yy[L];
   float yp[L];
   struct NBest _nbest[L];
   struct NBest *(nbest[L]);
   float Rpp=0, Rxp=0;
   int maxL = 1;
   
   for (m=0;m<L;m++)
      nbest[m] = &_nbest[m];
   
   for (m=0;m<L;m++)
   {
      ny[m] = _ny[m];
      iny[m] = _iny[m];
      y[m] = _y[m];
      iy[m] = _iy[m];
   }
   
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
         iy[m][i] = 0;

   for (m=0;m<L;m++)
      xy[m] = yy[m] = yp[m] = 0;
   
   pulsesLeft = K;
   while (pulsesLeft > 0)
   {
      int pulsesAtOnce=1;
      int Lupdate = L;
      int L2 = L;
      pulsesAtOnce = pulsesLeft/N;
      if (pulsesAtOnce<1)
         pulsesAtOnce = 1;
      if (pulsesLeft-pulsesAtOnce > 3 || N > 30)
         Lupdate = 1;
      //printf ("%d %d %d/%d %d\n", Lupdate, pulsesAtOnce, pulsesLeft, K, N);
      L2 = Lupdate;
      if (L2>maxL)
      {
         L2 = maxL;
         maxL *= N;
      }

      for (m=0;m<L;m++)
         nbest[m]->score = -1e10;

      for (m=0;m<L2;m++)
      {
         for (j=0;j<N;j++)
         {
            int sign;
            //if (x[j]>0) sign=1; else sign=-1;
            for (sign=-1;sign<=1;sign+=2)
            {
               /* All pulses at one location must have the same sign. Also,
                  only consider sign in the same direction as x[j], except for the
                  last pulses */
               if (iy[m][j]*sign < 0 || (x[j]*sign<0 && pulsesLeft>((K+1)>>1)))
                  continue;
               //fprintf (stderr, "%d/%d %d/%d %d/%d\n", i, K, m, L2, j, N);
               float tmp_xy, tmp_yy, tmp_yp;
               float score;
               float g;
               float s = sign*pulsesAtOnce;
               
               /* Updating the sums of the new pulse(s) */
               tmp_xy = xy[m] + s*x[j]               - alpha*s*p[j]*Rxp;
               tmp_yy = yy[m] + 2*s*y[m][j] + s*s      +s*s*alpha*alpha*p[j]*p[j]*Rpp - 2*alpha*s*p[j]*yp[m] - 2*s*s*alpha*p[j]*p[j];
               tmp_yp = yp[m] + s*p[j]               *(1-alpha*Rpp);
               
               /* Compute the gain such that ||p + g*y|| = 1 */
               g = (sqrt(tmp_yp*tmp_yp + tmp_yy - tmp_yy*Rpp) - tmp_yp)/tmp_yy;
               /* Knowing that gain, what the error: (x-g*y)^2 
                  (result is negated and we discard x^2 because it's constant) */
               score = 2*g*tmp_xy - g*g*tmp_yy;

               if (score>nbest[Lupdate-1]->score)
               {
                  int k, n;
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
                  nbest[id]->gain = g;
                  nbest[id]->xy = tmp_xy;
                  nbest[id]->yy = tmp_yy;
                  nbest[id]->yp = tmp_yp;
               }
            }
         }

      }
      int k;
      for (k=0;k<Lupdate;k++)
      {
         int n;
         int is;
         float s;
         is = nbest[k]->sign*pulsesAtOnce;
         s = is;
         for (n=0;n<N;n++)
            ny[k][n] = y[nbest[k]->orig][n] - alpha*s*p[nbest[k]->pos]*p[n];
         ny[k][nbest[k]->pos] += s;
      
         for (n=0;n<N;n++)
            iny[k][n] = iy[nbest[k]->orig][n];
         iny[k][nbest[k]->pos] += is;
         
         xy[k] = nbest[k]->xy;
         yy[k] = nbest[k]->yy;
         yp[k] = nbest[k]->yp;
      }
         
      for (k=0;k<Lupdate;k++)
      {
         float *tmp_ny;
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
   
   if (0) {
      float err=0;
      for (i=0;i<N;i++)
         err += (x[i]-nbest[0]->gain*y[0][i])*(x[i]-nbest[0]->gain*y[0][i]);
      //if (N<=10)
      //printf ("%f %d %d\n", err, K, N);
   }
   for (i=0;i<N;i++)
      x[i] = p[i]+nbest[0]->gain*y[0][i];
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
   
   encode_pulses(iy[0], N, K, enc);
   
   //printf ("%llu ", icwrs64(N, K, comb, signs));
   /* Recompute the gain in one pass to reduce the encoder-decoder mismatch
      due to the recursive computation used in quantisation.
      Not quite sure whether we need that or not */
   if (1) {
      float Ryp=0;
      float Rpp=0;
      float Ryy=0;
      float g=0;
      for (i=0;i<N;i++)
         Rpp += p[i]*p[i];
      
      for (i=0;i<N;i++)
         Ryp += iy[0][i]*p[i];
      
      for (i=0;i<N;i++)
         y[0][i] = iy[0][i] - alpha*Ryp*p[i];
      
      Ryp = 0;
      for (i=0;i<N;i++)
         Ryp += y[0][i]*p[i];
      
      for (i=0;i<N;i++)
         Ryy += y[0][i]*y[0][i];
      
      g = (sqrt(Ryp*Ryp + Ryy - Ryy*Rpp) - Ryp)/Ryy;
        
      for (i=0;i<N;i++)
         x[i] = p[i] + g*y[0][i];
      
   }

}

void alg_unquant(float *x, int N, int K, float *p, float alpha, ec_dec *dec)
{
   int i;
   int iy[N];
   float y[N];
   float Rpp=0, Ryp=0, Ryy=0;
   float g;

   decode_pulses(iy, N, K, dec);

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


static const float pg[11] = {1.f, .75f, .65f, 0.6f, 0.6f, .6f, .55f, .55f, .5f, .5f, .5f};

void intra_prediction(float *x, float *W, int N, int K, float *Y, float *P, int B, int N0, ec_enc *enc)
{
   int i,j;
   int best=0;
   float best_score=0;
   float s = 1;
   int sign;
   float E;
   int max_pos = N0-N/B;
   if (max_pos > 32)
      max_pos = 32;

   for (i=0;i<max_pos*B;i+=B)
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
   ec_enc_uint(enc,best/B,max_pos);
   //printf ("%d %f\n", best, best_score);
   
   float pred_gain;
   if (K>10)
      pred_gain = pg[10];
   else
      pred_gain = pg[K];
   E = 1e-10;
   for (j=0;j<N;j++)
   {
      P[j] = s*Y[best+j];
      E += P[j]*P[j];
   }
   E = pred_gain/sqrt(E);
   for (j=0;j<N;j++)
      P[j] *= E;
   if (K>0)
   {
      for (j=0;j<N;j++)
         x[j] -= P[j];
   } else {
      for (j=0;j<N;j++)
         x[j] = P[j];
   }
   //printf ("quant ");
   //for (j=0;j<N;j++) printf ("%f ", P[j]);

}

void intra_unquant(float *x, int N, int K, float *Y, float *P, int B, int N0, ec_dec *dec)
{
   int j;
   int sign;
   float s;
   int best;
   float E;
   int max_pos = N0-N/B;
   if (max_pos > 32)
      max_pos = 32;
   
   sign = ec_dec_uint(dec, 2);
   if (sign == 0)
      s = 1;
   else
      s = -1;
   
   best = B*ec_dec_uint(dec, max_pos);
   //printf ("%d %d ", sign, best);

   float pred_gain;
   if (K>10)
      pred_gain = pg[10];
   else
      pred_gain = pg[K];
   E = 1e-10;
   for (j=0;j<N;j++)
   {
      P[j] = s*Y[best+j];
      E += P[j]*P[j];
   }
   E = pred_gain/sqrt(E);
   for (j=0;j<N;j++)
      P[j] *= E;
   if (K==0)
   {
      for (j=0;j<N;j++)
         x[j] = P[j];
   }
}

void intra_fold(float *x, int N, float *Y, float *P, int B, int N0, int Nmax)
{
   int i, j;
   float E;
   
   E = 1e-10;
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
         E += P[j]*P[j];
      }
   }
   E = 1.f/sqrt(E);
   for (j=0;j<N;j++)
      P[j] *= E;
   for (j=0;j<N;j++)
      x[j] = P[j];
}

