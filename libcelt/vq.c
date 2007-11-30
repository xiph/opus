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

/* Algebraic pulse-base quantiser. The signal x is replaced by the sum of the pitch 
   a combination of pulses such that its norm is still equal to 1 */
void alg_quant(float *x, int N, int K, float *p)
{
   float y[N];
   int i,j;
   float xy = 0;
   float yy = 0;
   float yp = 0;
   float Rpp=0;
   float gain=0;
   for (j=0;j<N;j++)
      Rpp += p[j]*p[j];
   for (i=0;i<N;i++)
      y[i] = 0;
      
   for (i=0;i<K;i++)
   {
      int best_id=0;
      float max_val=-1e10;
      float best_xy=0, best_yy=0, best_yp = 0;
      for (j=0;j<N;j++)
      {
         float tmp_xy, tmp_yy, tmp_yp;
         float score;
         float g;
         tmp_xy = xy + fabs(x[j]);
         tmp_yy = yy + 2*fabs(y[j]) + 1;
         if (x[j]>0)
            tmp_yp = yp + p[j];
         else
            tmp_yp = yp - p[j];
         g = (sqrt(tmp_yp*tmp_yp + tmp_yy - tmp_yy*Rpp) - tmp_yp)/tmp_yy;
         score = 2*g*tmp_xy - g*g*tmp_yy;
         if (score>max_val)
         {
            max_val = score;
            best_id = j;
            best_xy = tmp_xy;
            best_yy = tmp_yy;
            best_yp = tmp_yp;
            gain = g;
         }
      }

      xy = best_xy;
      yy = best_yy;
      yp = best_yp;
      if (x[best_id]>0)
         y[best_id] += 1;
      else
         y[best_id] -= 1;
   }
   
   for (i=0;i<N;i++)
      x[i] = p[i]+gain*y[i];
   
}

/* Improved algebraic pulse-base quantiser. The signal x is replaced by the sum of the pitch 
   a combination of pulses such that its norm is still equal to 1. The only difference with 
   the quantiser above is that the search is more complete. */
void alg_quant2(float *x, int N, int K, float *p)
{
   int L = 5;
   //float tata[200];
   float y[L][N];
   //float tata2[200];
   float ny[L][N];
   int i, j, m;
   float xy[L], nxy[L];
   float yy[L], nyy[L];
   float yp[L], nyp[L];
   float best_scores[L];
   float Rpp=0;
   float gain[L];
   int maxL = 1;
   for (j=0;j<N;j++)
      Rpp += p[j]*p[j];
   for (m=0;m<L;m++)
      for (i=0;i<N;i++)
         y[m][i] = 0;
      
   for (m=0;m<L;m++)
      for (i=0;i<N;i++)
         ny[m][i] = 0;

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
            //fprintf (stderr, "%d/%d %d/%d %d/%d\n", i, K, m, L2, j, N);
            float tmp_xy, tmp_yy, tmp_yp;
            float score;
            float g;
            tmp_xy = xy[m] + fabs(x[j]);
            tmp_yy = yy[m] + 2*fabs(y[m][j]) + 1;
            if (x[j]>0)
               tmp_yp = yp[m] + p[j];
            else
               tmp_yp = yp[m] - p[j];
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
                  gain[k] = gain[k-1];
                  best_scores[k] = best_scores[k-1];
               }

               nxy[id] = tmp_xy;
               nyy[id] = tmp_yy;
               nyp[id] = tmp_yp;
               gain[id] = g;
               for (n=0;n<N;n++)
                  ny[id][n] = y[m][n];
               if (x[j]>0)
                  ny[id][j] += 1;
               else
                  ny[id][j] -= 1;
               best_scores[id] = score;
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
      }

   }
   
   for (i=0;i<N;i++)
      x[i] = p[i]+gain[0]*y[0][i];
   
}

/* Just replace the band with noise of unit energy */
void noise_quant(float *x, int N, int K, float *p)
{
   int i;
   float E = 1e-10;
   for (i=0;i<N;i++)
   {
      x[i] = (rand()%1000)/500.-1;
      E += x[i]*x[i];
   }
   E = 1./sqrt(E);
   for (i=0;i<N;i++)
   {
      x[i] *= E;
   }
}
