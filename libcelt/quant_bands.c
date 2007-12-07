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
#include <math.h>

int dummy_qi[100];

void quant_energy(CELTMode *m, float *eBands, float *oldEBands, ec_enc *enc)
{
   int i;
   float prev = 0;
   for (i=0;i<m->nbEBands;i++)
   {
      int qi;
      float q;
      float res;
      float x;
      float pred = .7*oldEBands[i];
      
      x = 20*log10(.3+eBands[i]);
      res = .25f*(i+3.f);
      //res = 1;
      qi = (int)floor(.5+(x-pred-prev)/res);
      dummy_qi[i] = qi;
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

void unquant_energy(CELTMode *m, float *eBands, float *oldEBands, ec_dec *dec)
{
   int i;
   float prev = 0;
   for (i=0;i<m->nbEBands;i++)
   {
      int qi;
      float q;
      float res;
      float pred = .7*oldEBands[i];
      
      res = .25f*(i+3.f);
      qi = dummy_qi[i];
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
}
