/* (C) 2007-2008 Jean-Marc Valin, CSIRO
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

#include "quant_bands.h"
#include "laplace.h"
#include <math.h>
#include "os_support.h"
#include "arch.h"
#include "mathops.h"
#include "stack_alloc.h"

#ifdef FIXED_POINT
const celt_word16_t eMeans[24] = {11520, -2048, -3072, -640, 256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#else
const celt_word16_t eMeans[24] = {45.f, -8.f, -12.f, -2.5f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#endif

/*const int frac[24] = {4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};*/
/*const int frac[24] = {8, 6, 5, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};*/

static void compute_fine_allocation(const CELTMode *m, celt_int16_t *bits, int budget)
{
   int i,j;
   int len;
   len = m->nbEBands;
   for (i=0;i<m->nbAllocVectors;i++)
   {
      if (m->energy_alloc[i*(len+1)+len] > budget)
         break;
   }
   if (i==0)
   {
      for (j=0;j<len;j++)
         bits[j] = 0;
   } else {
      for (j=0;j<len;j++)
         bits[j] = m->energy_alloc[(i-1)*(len+1)+j];
      budget -= m->energy_alloc[(i-1)*(len+1)+len];
   }
   if (i<m->nbAllocVectors)
   {
      j=0;
      while (budget>0)
      {
         if (m->energy_alloc[i*(len+1)+j]>bits[j])
         {
            bits[j]++;
            budget--;
         }
         j++;
         if (j>=len)
            j=0;
      }
   }
   
   /*for (j=0;j<len;j++)
      printf ("%d ", bits[j]);
   printf ("\n");*/
}

#ifdef FIXED_POINT
static inline celt_ener_t dB2Amp(celt_ener_t dB)
{
   celt_ener_t amp;
   amp = PSHR32(celt_exp2(MULT16_16_Q14(21771,dB)),2)-QCONST16(.3f, 14);
   if (amp < 0)
      amp = 0;
   return amp;
}

#define DBofTWO 24661
static inline celt_word16_t amp2dB(celt_ener_t amp)
{
   /* equivalent to return 6.0207*log2(.3+amp) */
   return ROUND16(MULT16_16(24661,celt_log2(ADD32(QCONST32(.3f,14),amp))),12);
   /* return DB_SCALING*20*log10(.3+ENER_SCALING_1*amp); */
}
#else
static inline celt_ener_t dB2Amp(celt_ener_t dB)
{
   celt_ener_t amp;
   amp = pow(10, .05*dB)-.3;
   if (amp < 0)
      amp = 0;
   return amp;
}
static inline celt_word16_t amp2dB(celt_ener_t amp)
{
   return 20*log10(.3+amp);
}
#endif

/* We could use ec_enc_uint directly, but this saves some divisions */
static inline void enc_frac(ec_enc *enc, int val, int ft)
{
   switch(ft)
   {
      case 1:
         break;
      case 2:
         ec_enc_bits(enc, val, 1);
         break;
      case 4:
         ec_enc_bits(enc, val, 2);
         break;
      case 8:
         ec_enc_bits(enc, val, 3);
         break;
      default:
         ec_enc_uint(enc, val, ft);
         break;
   }
}

/* We could use ec_enc_uint directly, but this saves some divisions */
static inline int dec_frac(ec_dec *dec, int ft)
{
   switch(ft)
   {
      case 1:
         return 0;
         break;
      case 2:
         return ec_dec_bits(dec, 1);
         break;
      case 4:
         return ec_dec_bits(dec, 2);
         break;
      case 8:
         return ec_dec_bits(dec, 3);
         break;
      default:
         return ec_dec_uint(dec, ft);
         break;
   }
}

static const celt_word16_t base_resolution = QCONST16(6.f,8);

int *quant_prob_alloc(const CELTMode *m)
{
   int i;
   int *prob;
   prob = celt_alloc(2*m->nbEBands*sizeof(int));
   for (i=0;i<m->nbEBands;i++)
   {
      prob[2*i] = 6000-i*200;
      prob[2*i+1] = ec_laplace_get_start_freq(prob[2*i]);
   }
   return prob;
}

void quant_prob_free(int *freq)
{
   celt_free(freq);
}

static void quant_energy_mono(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, unsigned budget, int *prob, ec_enc *enc)
{
   int i;
   unsigned bits;
   celt_word16_t prev = 0;
   celt_word16_t coef = m->ePredCoef;
   celt_word16_t beta;
   VARDECL(celt_word16_t, error);
   VARDECL(celt_int16_t, fine_quant);
   SAVE_STACK;
   /* The .7 is a heuristic */
   beta = MULT16_16_Q15(QCONST16(.8f,15),coef);
   
   ALLOC(error, m->nbEBands, celt_word16_t);
   ALLOC(fine_quant, m->nbEBands, celt_int16_t);
   bits = ec_enc_tell(enc, 0);
   /* Encode at a fixed coarse resolution */
   for (i=0;i<m->nbEBands;i++)
   {
      int qi;
      celt_word16_t q;   /* dB */
      celt_word16_t x;   /* dB */
      celt_word16_t f;   /* Q8 */
      celt_word16_t mean = MULT16_16_Q15(Q15ONE-coef,eMeans[i]);
      x = amp2dB(eBands[i]);
      f = EXTRACT16(celt_div(SHL32(EXTEND32(x-mean-MULT16_16_Q15(coef,oldEBands[i])-prev),8),base_resolution));
#ifdef FIXED_POINT
      /* Rounding to nearest integer here is really important! */
      qi = (f+128)>>8;
#else
      qi = (int)floor(.5+f);
#endif
      /* If we don't have enough bits to encode all the energy, just assume something safe.
         We allow slightly busting the budget here */
      if (ec_enc_tell(enc, 0) - bits > budget+16)
         qi = -1;
      else
         ec_laplace_encode_start(enc, qi, prob[2*i], prob[2*i+1]);
      q = qi*base_resolution;
      error[i] = f - SHL16(qi,8);
      
      oldEBands[i] = mean+MULT16_16_Q15(coef,oldEBands[i])+prev+q;
      
      prev = mean+prev+MULT16_16_Q15(Q15ONE-beta,q);
   }
   
   compute_fine_allocation(m, fine_quant, budget-(ec_enc_tell(enc, 0)-bits));

   /* Encode finer resolution */
   for (i=0;i<m->nbEBands;i++)
   {
      int q2;
      celt_int16_t frac = 1<<fine_quant[i];
      celt_word16_t offset = (error[i]+QCONST16(.5f,8))*frac;
#ifdef FIXED_POINT
      /* Has to be without rounding */
      q2 = offset>>8;
#else
      q2 = (int)floor(offset);
#endif
      if (q2 > frac-1)
         q2 = frac-1;
      enc_frac(enc, q2, frac);
      offset = EXTRACT16(celt_div(SHL16(q2,8)+QCONST16(.5,8),frac)-QCONST16(.5f,8));
      oldEBands[i] += PSHR32(MULT16_16(DB_SCALING*6,offset),8);
      /*printf ("%f ", error[i] - offset);*/
   }
   for (i=0;i<m->nbEBands;i++)
   {
      eBands[i] = dB2Amp(oldEBands[i]);
   }
   /*printf ("%d\n", ec_enc_tell(enc, 0)-9);*/

   /*printf ("\n");*/
   RESTORE_STACK;
}

static void unquant_energy_mono(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, unsigned budget, int *prob, ec_dec *dec)
{
   int i;
   unsigned bits;
   celt_word16_t prev = 0;
   celt_word16_t coef = m->ePredCoef;
   /* The .7 is a heuristic */
   VARDECL(celt_int16_t, fine_quant);
   celt_word16_t beta = MULT16_16_Q15(QCONST16(.8f,15),coef);
   SAVE_STACK;
   
   ALLOC(fine_quant, m->nbEBands, celt_int16_t);
   bits = ec_dec_tell(dec, 0);
   
   /* Decode at a fixed coarse resolution */
   for (i=0;i<m->nbEBands;i++)
   {
      int qi;
      celt_word16_t q;
      celt_word16_t mean = MULT16_16_Q15(Q15ONE-coef,eMeans[i]);
      /* If we didn't have enough bits to encode all the energy, just assume something safe.
         We allow slightly busting the budget here */
      if (ec_dec_tell(dec, 0) - bits > budget+16)
         qi = -1;
      else
         qi = ec_laplace_decode_start(dec, prob[2*i], prob[2*i+1]);
      q = qi*base_resolution;
      
      oldEBands[i] = mean+MULT16_16_Q15(coef,oldEBands[i])+prev+q;
      
      prev = mean+prev+MULT16_16_Q15(Q15ONE-beta,q);
   }
   
   compute_fine_allocation(m, fine_quant, budget-(ec_dec_tell(dec, 0)-bits));

   /* Decode finer resolution */
   for (i=0;i<m->nbEBands;i++)
   {
      int q2;
      celt_int16_t frac = 1<<fine_quant[i];
      celt_word16_t offset;
      q2 = dec_frac(dec, frac);
      offset = EXTRACT16(celt_div(SHL16(q2,8)+QCONST16(.5,8),frac)-QCONST16(.5f,8));
      oldEBands[i] += PSHR32(MULT16_16(DB_SCALING*6,offset),8);
   }
   for (i=0;i<m->nbEBands;i++)
   {
      eBands[i] = dB2Amp(oldEBands[i]);
   }
   RESTORE_STACK;
   /*printf ("\n");*/
}



void quant_energy(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, int budget, int *prob, ec_enc *enc)
{
   int C;
   SAVE_STACK;
   
   C = m->nbChannels;

   if (C==1)
      quant_energy_mono(m, eBands, oldEBands, budget, prob, enc);
   else 
#if 1
   {
      int c;
      VARDECL(celt_ener_t, E);
      ALLOC(E, m->nbEBands, celt_ener_t);
      for (c=0;c<C;c++)
      {
         int i;
         for (i=0;i<m->nbEBands;i++)
            E[i] = eBands[C*i+c];
         quant_energy_mono(m, E, oldEBands+c*m->nbEBands, budget/C, prob, enc);
         for (i=0;i<m->nbEBands;i++)
            eBands[C*i+c] = E[i];
      }
   }
#else
      if (C==2)
   {
      int i;
      int NB = m->nbEBands;
      celt_ener_t mid[NB];
      celt_ener_t side[NB];
      for (i=0;i<NB;i++)
      {
         //left = eBands[C*i];
         //right = eBands[C*i+1];
         mid[i] = ENER_SCALING_1*sqrt(eBands[C*i]*eBands[C*i] + eBands[C*i+1]*eBands[C*i+1]);
         side[i] = 20*log10((ENER_SCALING_1*eBands[2*i]+.3)/(ENER_SCALING_1*eBands[2*i+1]+.3));
         //printf ("%f %f ", mid[i], side[i]);
      }
      //printf ("\n");
      quant_energy_mono(m, mid, oldEBands, enc);
      for (i=0;i<NB;i++)
         side[i] = pow(10.f,floor(.5f+side[i])/10.f);
         
      //quant_energy_side(m, side, oldEBands+NB, enc);
      for (i=0;i<NB;i++)
      {
         eBands[C*i] = ENER_SCALING*mid[i]*sqrt(side[i]/(1.f+side[i]));
         eBands[C*i+1] = ENER_SCALING*mid[i]*sqrt(1.f/(1.f+side[i]));
         //printf ("%f %f ", mid[i], side[i]);
      }

   } else {
      celt_fatal("more than 2 channels not supported");
   }
#endif
   RESTORE_STACK;
}



void unquant_energy(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, int budget, int *prob, ec_dec *dec)
{
   int C;   
   SAVE_STACK;
   C = m->nbChannels;

   if (C==1)
      unquant_energy_mono(m, eBands, oldEBands, budget, prob, dec);
   else {
      int c;
      VARDECL(celt_ener_t, E);
      ALLOC(E, m->nbEBands, celt_ener_t);
      for (c=0;c<C;c++)
      {
         int i;
         unquant_energy_mono(m, E, oldEBands+c*m->nbEBands, budget/C, prob, dec);
         for (i=0;i<m->nbEBands;i++)
            eBands[C*i+c] = E[i];
      }
   }
   RESTORE_STACK;
}
