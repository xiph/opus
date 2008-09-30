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


#ifdef FIXED_POINT
static inline celt_ener_t dB2Amp(celt_ener_t dB)
{
   celt_ener_t amp;
   if (dB>24659)
      dB=24659;
   amp = PSHR32(celt_exp2(MULT16_16_Q14(21771,dB)),2)-QCONST16(.3f, 14);
   if (amp < 0)
      amp = 0;
   return PSHR32(amp,2);
}

#define DBofTWO 24661
static inline celt_word16_t amp2dB(celt_ener_t amp)
{
   /* equivalent to return 6.0207*log2(.3+amp) */
   return ROUND16(MULT16_16(24661,celt_log2(ADD32(QCONST32(.3f,14),SHL32(amp,2)))),12);
   /* return DB_SCALING*20*log10(.3+ENER_SCALING_1*amp); */
}
#else
static inline celt_ener_t dB2Amp(celt_ener_t dB)
{
   celt_ener_t amp;
   /*amp = pow(10, .05*dB)-.3;*/
   amp = exp(0.115129f*dB)-.3f;
   if (amp < 0)
      amp = 0;
   return amp;
}
static inline celt_word16_t amp2dB(celt_ener_t amp)
{
   /*return 20*log10(.3+amp);*/
   return 8.68589f*log(.3f+amp);
}
#endif

static const celt_word16_t base_resolution = QCONST16(6.f,8);
static const celt_word16_t base_resolution_1 = QCONST16(0.1666667f,15);

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

static void quant_coarse_energy_mono(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, unsigned budget, int *prob, celt_word16_t *error, ec_enc *enc)
{
   int i;
   unsigned bits;
   celt_word16_t prev = 0;
   celt_word16_t coef = m->ePredCoef;
   celt_word16_t beta;
   /* The .7 is a heuristic */
   beta = MULT16_16_Q15(QCONST16(.8f,15),coef);
   
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
#ifdef FIXED_POINT
      f = MULT16_16_Q15(x-mean-MULT16_16_Q15(coef,oldEBands[i])-prev,base_resolution_1);
      /* Rounding to nearest integer here is really important! */
      qi = (f+128)>>8;
#else
      f = (x-mean-coef*oldEBands[i]-prev)*base_resolution_1;
      /* Rounding to nearest integer here is really important! */
      qi = (int)floor(.5+f);
#endif
      /* If we don't have enough bits to encode all the energy, just assume something safe.
         We allow slightly busting the budget here */
      if (ec_enc_tell(enc, 0) - bits > budget)
      {
         qi = -1;
         error[i] = 128;
      } else {
         ec_laplace_encode_start(enc, &qi, prob[2*i], prob[2*i+1]);
         error[i] = f - SHL16(qi,8);
      }
      q = qi*base_resolution;
      
      oldEBands[i] = mean+MULT16_16_Q15(coef,oldEBands[i])+prev+q;
      if (oldEBands[i] < -QCONST16(12.f,8))
         oldEBands[i] = -QCONST16(12.f,8);
      prev = mean+prev+MULT16_16_Q15(Q15ONE-beta,q);
   }
}

static void quant_fine_energy_mono(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, celt_word16_t *error, int *fine_quant, ec_enc *enc)
{
   int i;
   /* Encode finer resolution */
   for (i=0;i<m->nbEBands;i++)
   {
      int q2;
      celt_int16_t frac = 1<<fine_quant[i];
      celt_word16_t offset = (error[i]+QCONST16(.5f,8))*frac;
      if (fine_quant[i] <= 0)
         continue;
#ifdef FIXED_POINT
      /* Has to be without rounding */
      q2 = offset>>8;
#else
      q2 = (int)floor(offset);
#endif
      if (q2 > frac-1)
         q2 = frac-1;
      ec_enc_bits(enc, q2, fine_quant[i]);
#ifdef FIXED_POINT
      offset = SUB16(SHR16(SHL16(q2,8)+QCONST16(.5,8),fine_quant[i]),QCONST16(.5f,8));
#else
      offset = (q2+.5f)*(1<<(14-fine_quant[i]))*(1.f/16384) - .5f;
#endif
      oldEBands[i] += PSHR32(MULT16_16(DB_SCALING*6,offset),8);
      /*printf ("%f ", error[i] - offset);*/
   }
   for (i=0;i<m->nbEBands;i++)
   {
      eBands[i] = dB2Amp(oldEBands[i]);
   }
   /*printf ("%d\n", ec_enc_tell(enc, 0)-9);*/

   /*printf ("\n");*/
}

static void unquant_coarse_energy_mono(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, unsigned budget, int *prob, ec_dec *dec)
{
   int i;
   unsigned bits;
   celt_word16_t prev = 0;
   celt_word16_t coef = m->ePredCoef;
   /* The .7 is a heuristic */
   celt_word16_t beta = MULT16_16_Q15(QCONST16(.8f,15),coef);
   
   bits = ec_dec_tell(dec, 0);
   /* Decode at a fixed coarse resolution */
   for (i=0;i<m->nbEBands;i++)
   {
      int qi;
      celt_word16_t q;
      celt_word16_t mean = MULT16_16_Q15(Q15ONE-coef,eMeans[i]);
      /* If we didn't have enough bits to encode all the energy, just assume something safe.
         We allow slightly busting the budget here */
      if (ec_dec_tell(dec, 0) - bits > budget)
         qi = -1;
      else
         qi = ec_laplace_decode_start(dec, prob[2*i], prob[2*i+1]);
      q = qi*base_resolution;
      
      oldEBands[i] = mean+MULT16_16_Q15(coef,oldEBands[i])+prev+q;
      if (oldEBands[i] < -QCONST16(12.f,8))
         oldEBands[i] = -QCONST16(12.f,8);
      
      prev = mean+prev+MULT16_16_Q15(Q15ONE-beta,q);
   }
}

static void unquant_fine_energy_mono(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, int *fine_quant, ec_dec *dec)
{
   int i;
   /* Decode finer resolution */
   for (i=0;i<m->nbEBands;i++)
   {
      int q2;
      celt_word16_t offset;
      if (fine_quant[i] <= 0)
         continue;
      q2 = ec_dec_bits(dec, fine_quant[i]);
#ifdef FIXED_POINT
      offset = SUB16(SHR16(SHL16(q2,8)+QCONST16(.5,8),fine_quant[i]),QCONST16(.5f,8));
#else
      offset = (q2+.5f)*(1<<(14-fine_quant[i]))*(1.f/16384) - .5f;
#endif
      oldEBands[i] += PSHR32(MULT16_16(DB_SCALING*6,offset),8);
   }
   for (i=0;i<m->nbEBands;i++)
   {
      eBands[i] = dB2Amp(oldEBands[i]);
   }
   /*printf ("\n");*/
}



void quant_coarse_energy(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, int budget, int *prob, celt_word16_t *error, ec_enc *enc)
{
   int C;
   C = m->nbChannels;

   if (C==1)
   {
      quant_coarse_energy_mono(m, eBands, oldEBands, budget, prob, error, enc);

   } else {
      int c;
      for (c=0;c<C;c++)
      {
         int i;
         VARDECL(celt_ener_t, E);
         SAVE_STACK;
         ALLOC(E, m->nbEBands, celt_ener_t);
         for (i=0;i<m->nbEBands;i++)
            E[i] = eBands[C*i+c];
         quant_coarse_energy_mono(m, E, oldEBands+c*m->nbEBands, budget/C, prob, error+c*m->nbEBands, enc);
         RESTORE_STACK;
      }
   }
}

void quant_fine_energy(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, celt_word16_t *error, int *fine_quant, ec_enc *enc)
{
   int C;
   C = m->nbChannels;

   if (C==1)
   {
      quant_fine_energy_mono(m, eBands, oldEBands, error, fine_quant, enc);

   } else {
      int c;
      VARDECL(celt_ener_t, E);
      ALLOC(E, m->nbEBands, celt_ener_t);
      for (c=0;c<C;c++)
      {
         int i;
         SAVE_STACK;
         quant_fine_energy_mono(m, E, oldEBands+c*m->nbEBands, error+c*m->nbEBands, fine_quant, enc);
         for (i=0;i<m->nbEBands;i++)
            eBands[C*i+c] = E[i];
         RESTORE_STACK;
      }
   }
}


void unquant_coarse_energy(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, int budget, int *prob, ec_dec *dec)
{
   int C;   

   C = m->nbChannels;
   if (C==1)
   {
      unquant_coarse_energy_mono(m, eBands, oldEBands, budget, prob, dec);
   }
   else {
      int c;
      VARDECL(celt_ener_t, E);
      SAVE_STACK;
      ALLOC(E, m->nbEBands, celt_ener_t);
      for (c=0;c<C;c++)
      {
         unquant_coarse_energy_mono(m, E, oldEBands+c*m->nbEBands, budget/C, prob, dec);
      }
      RESTORE_STACK;
   }
}

void unquant_fine_energy(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, int *fine_quant, ec_dec *dec)
{
   int C;   

   C = m->nbChannels;

   if (C==1)
   {
      unquant_fine_energy_mono(m, eBands, oldEBands, fine_quant, dec);
   }
   else {
      int c;
      VARDECL(celt_ener_t, E);
      SAVE_STACK;
      ALLOC(E, m->nbEBands, celt_ener_t);
      for (c=0;c<C;c++)
      {
         int i;
         unquant_fine_energy_mono(m, E, oldEBands+c*m->nbEBands, fine_quant, dec);
         for (i=0;i<m->nbEBands;i++)
            eBands[C*i+c] = E[i];
      }
      RESTORE_STACK;
   }
}
