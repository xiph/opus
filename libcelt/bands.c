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

const int qbank[NBANDS+2] =   {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 36, 44, 52, 68, 84, 116, 128};


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
      float x = 1.f/bank[i];
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
      gain1 = sqrt(arg)-Rxp;
      if (Rpp>.9999)
         Rpp = .9999;
      Rxx = 0;
      for (j=B*qbank[i];j<B*qbank[i+1];j++)
      {
         X[j*2-1] = P[j]+gain1*X[j];
         Rxx += X[j]*X[j];
      }
   }
   for (i=B*qbank[NBANDS];i<B*qbank[NBANDS+1];i++)
      X[i] = 0;
}

