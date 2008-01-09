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


#include "quant_bands.h"
#include "laplace.h"
#include <math.h>
#include "os_support.h"

void quant_energy(const CELTMode *m, float *eBands, float *oldEBands, ec_enc *enc)
{
   int C;
   
   C = m->nbChannels;

   if (C==1)
      quant_energy_mono(m, eBands, oldEBands, enc);
   else if (C==2)
   {
      int i;
      int NB = m->nbEBands;
      float mid[NB];
      float side[NB];
      float left;
      float right;
      for (i=0;i<NB;i++)
      {
         //left = eBands[C*i];
         //right = eBands[C*i+1];
         mid[i] = sqrt(eBands[C*i]*eBands[C*i] + eBands[C*i+1]*eBands[C*i+1]);
         side[i] = 20*log10((eBands[2*i]+.3)/(eBands[2*i+1]+.3));
         //printf ("%f %f ", mid[i], side[i]);
      }
      //printf ("\n");
      quant_energy_mono(m, mid, oldEBands, enc);
      for (i=0;i<NB;i++)
         side[i] = pow(10.f,floor(.5f+side[i])/10.f);
         
      //quant_energy_side(m, side, oldEBands+NB, enc);
      for (i=0;i<NB;i++)
      {
         eBands[C*i] = mid[i]*sqrt(side[i]/(1.f+side[i]));
         eBands[C*i+1] = mid[i]*sqrt(1.f/(1.f+side[i]));
         //printf ("%f %f ", mid[i], side[i]);
      }

   } else {
      celt_fatal("more than 2 channels not supported");
   }
}

void quant_energy_mono(const CELTMode *m, float *eBands, float *oldEBands, ec_enc *enc)
{
   int i;
   float prev = 0;
   for (i=0;i<m->nbEBands;i++)
   {
      int qi;
      float q;
      float res;
      float x;
      float pred = m->ePredCoef*oldEBands[i]+m->eMeans[i];
      
      x = 20*log10(.3+eBands[i]);
      res = .25f*(i+3.f);
      //res = 1;
      qi = (int)floor(.5+(x-pred-prev)/res);
      ec_laplace_encode(enc, qi, m->eDecay[i]);
      q = qi*res;
      
      //printf("%d ", qi);
      //printf("%f %f ", pred+prev+q, x);
      //printf("%f ", x-pred);
      
      oldEBands[i] = pred+prev+q;
      eBands[i] = pow(10, .05*oldEBands[i])-.3;
      if (eBands[i] < 0)
         eBands[i] = 0;
      prev = (prev + .5*q);
   }
   //printf ("\n");
}

void unquant_energy(const CELTMode *m, float *eBands, float *oldEBands, ec_dec *dec)
{
   int i;
   float prev = 0;
   for (i=0;i<m->nbEBands;i++)
   {
      int qi;
      float q;
      float res;
      float pred = m->ePredCoef*oldEBands[i]+m->eMeans[i];
      
      res = .25f*(i+3.f);
      qi = ec_laplace_decode(dec, m->eDecay[i]);
      q = qi*res;
      //printf("%f %f ", pred+prev+q, x);
      //printf("%d ", qi);
      //printf("%f ", x-pred-prev);
      
      oldEBands[i] = pred+prev+q;
      eBands[i] = pow(10, .05*oldEBands[i])-.3;
      if (eBands[i] < 0)
         eBands[i] = 0;
      prev = (prev + .5*q);
   }
   //printf ("\n");
}
