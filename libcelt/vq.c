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
#include "cwrs.h"
#include "vq.h"
#include "arch.h"
#include "os_support.h"

/* Enable this or define your own implementation if you want to speed up the
   VQ search (used in inner loop only) */
#if 0
#include <xmmintrin.h>
static inline float approx_sqrt(float x)
{
   _mm_store_ss(&x, _mm_sqrt_ss(_mm_set_ss(x)));
   return x;
}
static inline float approx_inv(float x)
{
   _mm_store_ss(&x, _mm_rcp_ss(_mm_set_ss(x)));
   return x;
}
#else
#define approx_sqrt(x) (sqrt(x))
#define approx_inv(x) (1.f/(x))
#endif

/** All the info necessary to keep track of a hypothesis during the search */
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

void alg_quant(celt_norm_t *X, celt_mask_t *W, int N, int K, celt_norm_t *P, float alpha, ec_enc *enc)
{
   int L = 3;
   VARDECL(float *x);
   VARDECL(float *p);
   VARDECL(float *_y);
   VARDECL(float *_ny);
   VARDECL(int *_iy);
   VARDECL(int *_iny);
   VARDECL(float **y);
   VARDECL(float **ny);
   VARDECL(int **iy);
   VARDECL(int **iny);
   int i, j, k, m;
   int pulsesLeft;
   VARDECL(float *xy);
   VARDECL(float *yy);
   VARDECL(float *yp);
   VARDECL(struct NBest *_nbest);
   VARDECL(struct NBest **nbest);
   float Rpp=0, Rxp=0;
   int maxL = 1;
   
   ALLOC(x, N, float);
   ALLOC(p, N, float);
   ALLOC(_y, L*N, float);
   ALLOC(_ny, L*N, float);
   ALLOC(_iy, L*N, int);
   ALLOC(_iny, L*N, int);
   ALLOC(y, L*N, float*);
   ALLOC(ny, L*N, float*);
   ALLOC(iy, L*N, int*);
   ALLOC(iny, L*N, int*);
   
   ALLOC(xy, L, float);
   ALLOC(yy, L, float);
   ALLOC(yp, L, float);
   ALLOC(_nbest, L, struct NBest);
   ALLOC(nbest, L, struct NBest *);

   for (j=0;j<N;j++)
   {
      x[j] = X[j]*NORM_SCALING_1;
      p[j] = P[j]*NORM_SCALING_1;
   }
   
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
      Rpp += p[j]*p[j];
      Rxp += x[j]*p[j];
   }
   if (Rpp>1)
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
         nbest[m]->score = -1e10f;

      for (m=0;m<L2;m++)
      {
         for (j=0;j<N;j++)
         {
            int sign;
            /*if (x[j]>0) sign=1; else sign=-1;*/
            for (sign=-1;sign<=1;sign+=2)
            {
               /*fprintf (stderr, "%d/%d %d/%d %d/%d\n", i, K, m, L2, j, N);*/
               float tmp_xy, tmp_yy, tmp_yp;
               float score;
               float g;
               float s = sign*pulsesAtOnce;
               
               /* All pulses at one location must have the same sign. */
               if (iy[m][j]*sign < 0)
                  continue;

               /* Updating the sums of the new pulse(s) */
               tmp_xy = xy[m] + s*x[j]               - alpha*s*p[j]*Rxp;
               tmp_yy = yy[m] + 2.f*s*y[m][j] + s*s      +s*s*alpha*alpha*p[j]*p[j]*Rpp - 2.f*alpha*s*p[j]*yp[m] - 2.f*s*s*alpha*p[j]*p[j];
               tmp_yp = yp[m] + s*p[j]               *(1.f-alpha*Rpp);
               
               /* Compute the gain such that ||p + g*y|| = 1 */
               g = (approx_sqrt(tmp_yp*tmp_yp + tmp_yy - tmp_yy*Rpp) - tmp_yp)*approx_inv(tmp_yy);
               /* Knowing that gain, what the error: (x-g*y)^2 
                  (result is negated and we discard x^2 because it's constant) */
               score = 2.f*g*tmp_xy - g*g*tmp_yy;

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
                  nbest[id]->gain = g;
                  nbest[id]->xy = tmp_xy;
                  nbest[id]->yy = tmp_yy;
                  nbest[id]->yp = tmp_yp;
               }
            }
         }

      }
      
      if (!(nbest[0]->score > -1e10f))
         celt_fatal("Could not find any match in VQ codebook. Something got corrupted somewhere.");
      /* Only now that we've made the final choice, update ny/iny and others */
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
      /* Swap ny/iny with y/iy */
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
      /*if (N<=10)
        printf ("%f %d %d\n", err, K, N);*/
   }
   for (i=0;i<N;i++)
      x[i] = p[i]+nbest[0]->gain*y[0][i];
   /* Sanity checks, don't bother */
   if (0) {
      float E=1e-15;
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
   
   encode_pulses(iy[0], N, K, enc);
   
   /* Recompute the gain in one pass to reduce the encoder-decoder mismatch
      due to the recursive computation used in quantisation.
      Not quite sure whether we need that or not */
   if (1) {
      float Ryp=0;
      float Ryy=0;
      float g=0;
      
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
   for (j=0;j<N;j++)
   {
      X[j] = x[j] * NORM_SCALING;
      P[j] = p[j] * NORM_SCALING;
   }

}

/** Decode pulse vector and combine the result with the pitch vector to produce
    the final normalised signal in the current band. */
void alg_unquant(celt_norm_t *X, int N, int K, celt_norm_t *P, float alpha, ec_dec *dec)
{
   int i;
   float Rpp=0, Ryp=0, Ryy=0;
   float g;
   VARDECL(int *iy);
   VARDECL(float *y);
   VARDECL(float *x);
   VARDECL(float *p);
   
   ALLOC(iy, N, int);
   ALLOC(y, N, float);
   ALLOC(x, N, float);
   ALLOC(p, N, float);

   decode_pulses(iy, N, K, dec);
   for (i=0;i<N;i++)
   {
      x[i] = X[i]*NORM_SCALING_1;
      p[i] = P[i]*NORM_SCALING_1;
   }

   /*for (i=0;i<N;i++)
      printf ("%d ", iy[i]);*/
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
   for (i=0;i<N;i++)
   {
      X[i] = x[i] * NORM_SCALING;
      P[i] = p[i] * NORM_SCALING;
   }

}


static const float pg[11] = {1.f, .75f, .65f, 0.6f, 0.6f, .6f, .55f, .55f, .5f, .5f, .5f};

void intra_prediction(celt_norm_t *x, celt_mask_t *W, int N, int K, celt_norm_t *Y, celt_norm_t *P, int B, int N0, ec_enc *enc)
{
   int i,j;
   int best=0;
   float best_score=0;
   float s = 1;
   int sign;
   float E;
   float pred_gain;
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
         xy += 1.f*x[j]*Y[i+N-j-1];
         yy += 1.f*Y[i+N-j-1]*Y[i+N-j-1];
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
   /*printf ("%d %d ", sign, best);*/
   ec_enc_uint(enc,sign,2);
   ec_enc_uint(enc,best/B,max_pos);
   /*printf ("%d %f\n", best, best_score);*/
   
   if (K>10)
      pred_gain = pg[10];
   else
      pred_gain = pg[K];
   E = 1e-10;
   for (j=0;j<N;j++)
   {
      P[j] = s*Y[best+N-j-1];
      E += NORM_SCALING_1*NORM_SCALING_1*P[j]*P[j];
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
   /*printf ("quant ");*/
   /*for (j=0;j<N;j++) printf ("%f ", P[j]);*/

}

void intra_unquant(celt_norm_t *x, int N, int K, celt_norm_t *Y, celt_norm_t *P, int B, int N0, ec_dec *dec)
{
   int j;
   int sign;
   float s;
   int best;
   float E;
   float pred_gain;
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
   E = 1e-10;
   for (j=0;j<N;j++)
   {
      P[j] = s*Y[best+N-j-1];
      E += NORM_SCALING_1*NORM_SCALING_1*P[j]*P[j];
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

void intra_fold(celt_norm_t *x, int N, celt_norm_t *Y, celt_norm_t *P, int B, int N0, int Nmax)
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
            E += NORM_SCALING_1*NORM_SCALING_1*P[j*B+i]*P[j*B+i];
         }
      }
   } else {
      for (j=0;j<N;j++)
      {
         P[j] = Y[j];
         E += NORM_SCALING_1*NORM_SCALING_1*P[j]*P[j];
      }
   }
   E = 1.f/sqrt(E);
   for (j=0;j<N;j++)
      P[j] *= E;
   for (j=0;j<N;j++)
      x[j] = P[j];
}

