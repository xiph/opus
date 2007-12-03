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
#include "bands.h"
#include "vq.h"
#include "cwrs.h"

const int qbank[NBANDS+2] =   {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 36, 44, 52, 68, 84, 116, 128};

const int qpulses[NBANDS  ] = {7, 5, 4, 4, 3,  3,  3,  4,  4,  4, -2, -1, -1, -1,  0};
//const int qpulses[NBANDS  ] = {17,15,14,14,13, 13, 13, 13, 13, 14, 14, 14, 10, 10, 10};

#define WAVEFORM_END 36

/* Start frequency of each band */
int pbank[] = {0, 4, 8, 12, 20, WAVEFORM_END, 128};

/* Compute the energy in each of the bands */
void compute_bands(float *X, int B, float *bank)
{
   int i;
   for (i=0;i<NBANDS;i++)
   {
      int j;
      bank[i] = 1e-10;
      for (j=B*qbank[i];j<B*qbank[i+1];j++)
         bank[i] += X[j]*X[j];
      bank[i] = sqrt(bank[i]);
   }
}

/* Normalise each band such that the energy is one. */
void normalise_bands(float *X, int B, float *bank)
{
   int i;
   for (i=0;i<NBANDS;i++)
   {
      int j;
      float x = 1.f/(1e-10+bank[i]);
      for (j=B*qbank[i];j<B*qbank[i+1];j++)
         X[j] *= x;
   }
   for (i=B*qbank[NBANDS];i<B*qbank[NBANDS+1];i++)
      X[i] = 0;
}

/* De-normalise the energy to produce the synthesis from the unit-energy bands */
void denormalise_bands(float *X, int B, float *bank)
{
   int i;
   for (i=0;i<NBANDS;i++)
   {
      int j;
      float x = bank[i];
      for (j=B*qbank[i];j<B*qbank[i+1];j++)
         X[j] *= x;
   }
   for (i=B*qbank[NBANDS];i<B*qbank[NBANDS+1];i++)
      X[i] = 0;
}


/* Compute the best gain for each "pitch band" */
void compute_pitch_gain(float *X, int B, float *P, float *gains, float *bank)
{
   int i;
   float w[B*qbank[NBANDS]];
   for (i=0;i<NBANDS;i++)
   {
      int j;
      for (j=B*qbank[i];j<B*qbank[i+1];j++)
         w[j] = bank[i];
   }

   
   for (i=0;i<PBANDS;i++)
   {
      float Sxy=0;
      float Sxx = 0;
      int j;
      float gain;
      for (j=B*pbank[i];j<B*pbank[i+1];j++)
      {
         Sxy += X[j]*P[j]*w[j];
         Sxx += X[j]*X[j]*w[j];
      }
      gain = Sxy/(1e-10+Sxx);
      //gain = Sxy/(2*(pbank[i+1]-pbank[i]));
      //if (i<3)
      //gain *= 1+.02*gain;
      if (gain > .90)
         gain = .90;
      if (gain < 0.0)
         gain = 0.0;
      
      gains[i] = gain;
   }
   for (i=B*pbank[PBANDS];i<B*pbank[PBANDS+1];i++)
      P[i] = 0;
}

/* Apply the (quantised) gain to each "pitch band" */
void pitch_quant_bands(float *X, int B, float *P, float *gains)
{
   int i;
   for (i=0;i<PBANDS;i++)
   {
      int j;
      for (j=B*pbank[i];j<B*pbank[i+1];j++)
         P[j] *= gains[i];
      //printf ("%f ", gain);
   }
   for (i=B*pbank[PBANDS];i<B*pbank[PBANDS+1];i++)
      P[i] = 0;
}

void quant_bands(float *X, int B, float *P)
{
   int i, j;
   float norm[B*qbank[NBANDS+1]];
   //float bits = 0;
   
   for (i=0;i<NBANDS;i++)
   {
      int q;
      q =qpulses[i];
      if (q>0) {
         float n = sqrt(B*(qbank[i+1]-qbank[i]));
         alg_quant2(X+B*qbank[i], B*(qbank[i+1]-qbank[i]), q, P+B*qbank[i]);
         for (j=B*qbank[i];j<B*qbank[i+1];j++)
            norm[j] = X[j] * n;
         //bits += log2(ncwrs(B*(qbank[i+1]-qbank[i]), q));
      } else {
         float n = sqrt(B*(qbank[i+1]-qbank[i]));
         copy_quant(X+B*qbank[i], B*(qbank[i+1]-qbank[i]), -q, norm, B, qbank[i]);
         for (j=B*qbank[i];j<B*qbank[i+1];j++)
            norm[j] = X[j] * n;
         //bits += 1+log2(qbank[i])+log2(ncwrs(B*(qbank[i+1]-qbank[i]), -q));
         //noise_quant(X+B*qbank[i], B*(qbank[i+1]-qbank[i]), q, P+B*qbank[i]);
      }
   }
   //printf ("%f\n", bits);
   for (i=B*qbank[NBANDS];i<B*qbank[NBANDS+1];i++)
      X[i] = 0;
}


/* Scales the pulse-codebook entry in each band such that unit-energy is conserved when 
   adding the pitch */
void pitch_renormalise_bands(float *X, int B, float *P)
{
   int i;
   for (i=0;i<NBANDS;i++)
   {
      int j;
      float Rpp=0;
      float Rxp=0;
      float Rxx=0;
      float gain1;
      for (j=B*qbank[i];j<B*qbank[i+1];j++)
      {
         Rxp += X[j]*P[j];
         Rpp += P[j]*P[j];
         Rxx += X[j]*X[j];
      }
      float arg = Rxp*Rxp + 1 - Rpp;
      if (arg < 0)
         arg = 0;
      gain1 = sqrt(arg)-Rxp;
      Rxx = 0;
      for (j=B*qbank[i];j<B*qbank[i+1];j++)
      {
         X[j] = P[j]+gain1*X[j];
         Rxx += X[j]*X[j];
      }
   }
   for (i=B*qbank[NBANDS];i<B*qbank[NBANDS+1];i++)
      X[i] = 0;
}

