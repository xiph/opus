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
const celt_word16_t eMeans[24] = {1920, -341, -512, -107, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#else
const celt_word16_t eMeans[24] = {7.5f, -1.33f, -2.f, -0.42f, 0.17f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#endif

#define amp2Log(amp) celt_log2(MAX32(QCONST32(.001f,14),SHL32(amp,2)))

#define log2Amp(lg) PSHR32(celt_exp2(SHL16(lg,3)),4)

int intra_decision(celt_ener_t *eBands, celt_word16_t *oldEBands, int len)
{
   int i;
   celt_word32_t dist = 0;
   for (i=0;i<len;i++)
   {
      celt_word16_t d = SUB16(amp2Log(eBands[i]), oldEBands[i]);
      dist = MAC16_16(dist, d,d);
   }
   return SHR32(dist,16) > 2*len;
}

int *quant_prob_alloc(const CELTMode *m)
{
   int i;
   int *prob;
   prob = celt_alloc(4*m->nbEBands*sizeof(int));
   if (prob==NULL)
     return NULL;
   for (i=0;i<m->nbEBands;i++)
   {
      prob[2*i] = 6000-i*200;
      prob[2*i+1] = ec_laplace_get_start_freq(prob[2*i]);
   }
   for (i=0;i<m->nbEBands;i++)
   {
      prob[2*m->nbEBands+2*i] = 9000-i*240;
      prob[2*m->nbEBands+2*i+1] = ec_laplace_get_start_freq(prob[2*m->nbEBands+2*i]);
   }
   return prob;
}

void quant_prob_free(int *freq)
{
   celt_free(freq);
}

static unsigned quant_coarse_energy_mono(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, unsigned budget, int intra, int *prob, celt_word16_t *error, ec_enc *enc)
{
   int i;
   unsigned bits;
   unsigned bits_used = 0;
   celt_word16_t prev = 0;
   celt_word16_t coef = m->ePredCoef;
   celt_word16_t beta;
   
   if (intra)
   {
      coef = 0;
      prob += 2*m->nbEBands;
   }
   /* The .8 is a heuristic */
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
      x = amp2Log(eBands[i]);
#ifdef FIXED_POINT
      f = x-mean -MULT16_16_Q15(coef,oldEBands[i])-prev;
      /* Rounding to nearest integer here is really important! */
      qi = (f+128)>>8;
#else
      f = x-mean-coef*oldEBands[i]-prev;
      /* Rounding to nearest integer here is really important! */
      qi = (int)floor(.5+f);
#endif
      /* If we don't have enough bits to encode all the energy, just assume something safe.
         We allow slightly busting the budget here */
      bits_used=ec_enc_tell(enc, 0) - bits;
      if (bits_used > budget)
      {
         qi = -1;
         error[i] = 128;
      } else {
         ec_laplace_encode_start(enc, &qi, prob[2*i], prob[2*i+1]);
         error[i] = f - SHL16(qi,8);
      }
      q = qi*DB_SCALING;

      oldEBands[i] = MULT16_16_Q15(coef,oldEBands[i])+(mean+prev+q);
      prev = mean+prev+MULT16_16_Q15(Q15ONE-beta,q);
   }
   return bits_used;
}

static void quant_fine_energy_mono(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, celt_word16_t *error, int *fine_quant, ec_enc *enc)
{
   int i;
   /* Encode finer resolution */
   for (i=0;i<m->nbEBands;i++)
   {
      int q2;
      celt_int16_t frac = 1<<fine_quant[i];
      celt_word16_t offset;
      if (fine_quant[i] <= 0)
         continue;
#ifdef FIXED_POINT
      /* Has to be without rounding */
      q2 = (error[i]+QCONST16(.5f,8))>>(8-fine_quant[i]);
#else
      q2 = (int)floor((error[i]+.5)*frac);
#endif
      if (q2 > frac-1)
         q2 = frac-1;
      ec_enc_bits(enc, q2, fine_quant[i]);
#ifdef FIXED_POINT
      offset = SUB16(SHR16(SHL16(q2,8)+QCONST16(.5,8),fine_quant[i]),QCONST16(.5f,8));
#else
      offset = (q2+.5f)*(1<<(14-fine_quant[i]))*(1.f/16384) - .5f;
#endif
      oldEBands[i] += offset;
      /*printf ("%f ", error[i] - offset);*/
   }
   for (i=0;i<m->nbEBands;i++)
   {
      eBands[i] = log2Amp(oldEBands[i]);
      if (oldEBands[i] < -QCONST16(7.f,8))
         oldEBands[i] = -QCONST16(7.f,8);
   }
   /*printf ("%d\n", ec_enc_tell(enc, 0)-9);*/

   /*printf ("\n");*/
}

static void unquant_coarse_energy_mono(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, unsigned budget, int intra, int *prob, ec_dec *dec)
{
   int i;
   unsigned bits;
   celt_word16_t prev = 0;
   celt_word16_t coef = m->ePredCoef;
   celt_word16_t beta;
   
   if (intra)
   {
      coef = 0;
      prob += 2*m->nbEBands;
   }
   /* The .8 is a heuristic */
   beta = MULT16_16_Q15(QCONST16(.8f,15),coef);
   
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
      q = qi*DB_SCALING;

      oldEBands[i] = MULT16_16_Q15(coef,oldEBands[i])+(mean+prev+q);
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
      oldEBands[i] += offset;
   }
   for (i=0;i<m->nbEBands;i++)
   {
      eBands[i] = log2Amp(oldEBands[i]);
      if (oldEBands[i] < -QCONST16(7.f,8))
         oldEBands[i] = -QCONST16(7.f,8);
   }
   /*printf ("\n");*/
}



unsigned quant_coarse_energy(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, int budget, int intra, int *prob, celt_word16_t *error, ec_enc *enc)
{
   int C;
   C = m->nbChannels;

   if (C==1)
   {
      return quant_coarse_energy_mono(m, eBands, oldEBands, budget, intra, prob, error, enc);
   } else {
      int c;
      unsigned maxBudget=0;
      for (c=0;c<C;c++)
      {
         int i;
         unsigned coarse_needed;
         VARDECL(celt_ener_t, E);
         SAVE_STACK;
         ALLOC(E, m->nbEBands, celt_ener_t);
         for (i=0;i<m->nbEBands;i++)
            E[i] = eBands[C*i+c];
         coarse_needed=quant_coarse_energy_mono(m, E, oldEBands+c*m->nbEBands, budget/C, intra, prob, error+c*m->nbEBands, enc);
         maxBudget=IMAX(maxBudget,coarse_needed);
         RESTORE_STACK;
      }
      return maxBudget*C;
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


void unquant_coarse_energy(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, int budget, int intra, int *prob, ec_dec *dec)
{
   int C;   

   C = m->nbChannels;
   if (C==1)
   {
      unquant_coarse_energy_mono(m, eBands, oldEBands, budget, intra, prob, dec);
   }
   else {
      int c;
      VARDECL(celt_ener_t, E);
      SAVE_STACK;
      ALLOC(E, m->nbEBands, celt_ener_t);
      for (c=0;c<C;c++)
      {
         unquant_coarse_energy_mono(m, E, oldEBands+c*m->nbEBands, budget/C, intra, prob, dec);
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
