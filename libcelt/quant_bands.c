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

const float eMeans[24] = {45.f, -8.f, -12.f, -2.5f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

const int frac[24] = {8, 6, 5, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

static void quant_energy_mono(const CELTMode *m, float *eBands, float *oldEBands, ec_enc *enc)
{
   int i;
   float prev = 0;
   float coef = m->ePredCoef;
   float error[m->nbEBands];
   /* The .7 is a heuristic */
   float beta = .7*coef;
   for (i=0;i<m->nbEBands;i++)
   {
      int qi;
      float q;
      float res;
      float x;
      float f;
      float mean = (1-coef)*eMeans[i];
      x = 20*log10(.3+eBands[i]);
      res = 6.;
      //res = 1;
      f = (x-mean-coef*oldEBands[i]-prev)/res;
      qi = (int)floor(.5+f);
      //if (i> 4 && qi<-2)
      //   qi = -2;
      //ec_laplace_encode(enc, qi, i==0?11192:6192);
      //ec_laplace_encode(enc, qi, 8500-i*200);
      ec_laplace_encode(enc, qi, 6000-i*200);
      q = qi*res;
      error[i] = f - qi;
      
      //printf("%d ", qi);
      //printf("%f %f ", pred+prev+q, x);
      //printf("%f ", x-pred);
      
      oldEBands[i] = mean+coef*oldEBands[i]+prev+q;
      
      prev = mean+prev+(1-beta)*q;
   }
   for (i=0;i<m->nbEBands;i++)
   {
      int q2;
      float offset = (error[i]+.5)*frac[i];
      q2 = (int)floor(offset);
      if (q2 > frac[i]-1)
         q2 = frac[i]-1;
      ec_enc_uint(enc, q2, frac[i]);
      offset = ((q2+.5)/frac[i])-.5;
      oldEBands[i] += 6.*offset;
      //printf ("%f ", error[i] - offset);
      eBands[i] = pow(10, .05*oldEBands[i])-.3;
      if (eBands[i] < 0)
         eBands[i] = 0;
   }
   //printf ("%d\n", ec_enc_tell(enc, 0)-9);

   //printf ("\n");
}

static void unquant_energy_mono(const CELTMode *m, float *eBands, float *oldEBands, ec_dec *dec)
{
   int i;
   float prev = 0;
   float coef = m->ePredCoef;
   float error[m->nbEBands];
   /* The .7 is a heuristic */
   float beta = .7*coef;
   for (i=0;i<m->nbEBands;i++)
   {
      int qi;
      float q;
      float res;
      float mean = (1-coef)*eMeans[i];
      res = 6.;
      qi = ec_laplace_decode(dec, 6000-i*200);
      q = qi*res;
      
      oldEBands[i] = mean+coef*oldEBands[i]+prev+q;
      
      prev = mean+prev+(1-beta)*q;
   }
   for (i=0;i<m->nbEBands;i++)
   {
      int q2;
      float offset;
      q2 = ec_dec_uint(dec, frac[i]);
      offset = ((q2+.5)/frac[i])-.5;
      oldEBands[i] += 6.*offset;
      //printf ("%f ", error[i] - offset);
      eBands[i] = pow(10, .05*oldEBands[i])-.3;
      if (eBands[i] < 0)
         eBands[i] = 0;
   }
   //printf ("\n");
}



void quant_energy(const CELTMode *m, float *eBands, float *oldEBands, ec_enc *enc)
{
   int C;
   
   C = m->nbChannels;

   if (C==1)
      quant_energy_mono(m, eBands, oldEBands, enc);
   else 
#if 1
   {
      int c;
      for (c=0;c<C;c++)
      {
         int i;
         float E[m->nbEBands];
         for (i=0;i<m->nbEBands;i++)
            E[i] = eBands[C*i+c];
         quant_energy_mono(m, E, oldEBands+c*m->nbEBands, enc);
         for (i=0;i<m->nbEBands;i++)
            eBands[C*i+c] = E[i];
      }
   }
#else
      if (C==2)
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
#endif
}



void unquant_energy(const CELTMode *m, float *eBands, float *oldEBands, ec_dec *dec)
{
   int C;   
   C = m->nbChannels;

   if (C==1)
      unquant_energy_mono(m, eBands, oldEBands, dec);
   else {
      int c;
      for (c=0;c<C;c++)
      {
         int i;
         float E[m->nbEBands];
         unquant_energy_mono(m, E, oldEBands+c*m->nbEBands, dec);
         for (i=0;i<m->nbEBands;i++)
            eBands[C*i+c] = E[i];
      }
   }
}
