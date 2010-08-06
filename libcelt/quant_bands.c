/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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
const celt_word16 eMeans[25] = {
      7941, 7777, 7344, 6791, 6397,
      6076, 5825, 5773, 6305, 6151,
      6030, 5922, 6290, 5842, 5525,
      5733, 5604, 5659, 5732, 5445,
      4082, 4082, 4082, 4082, 4082};
#else
const celt_word16 eMeans[25] = {
      7.755326, 7.594506, 7.172360, 6.632112, 6.247387,
      5.933998, 5.688906, 5.637953, 6.157458, 6.006739,
      5.889151, 5.783105, 6.142725, 5.704652, 5.395896,
      5.598698, 5.472708, 5.526389, 5.597547, 5.317134,
      3.986353, 3.986353, 3.986353, 3.986353, 3.986353};
#endif
/* prediction coefficients: 0.9, 0.8, 0.65, 0.5 */
#ifdef FIXED_POINT
static const celt_word16 pred_coef[4] = {29440, 26112, 21248, 16384};
#else
static const celt_word16 pred_coef[4] = {29440/32768., 26112/32768., 21248/32768., 16384/32768.};
#endif

int intra_decision(celt_word16 *eBands, celt_word16 *oldEBands, int start, int end, int len, int C)
{
   int c, i;
   celt_word32 dist = 0;
   for (c=0;c<C;c++)
   {
      for (i=start;i<end;i++)
      {
         celt_word16 d = SUB16(eBands[i+c*len], oldEBands[i+c*len]);
         dist = MAC16_16(dist, d,d);
      }
   }
   return SHR32(dist,2*DB_SHIFT) > 2*C*(end-start);
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
      prob[2*i] = 7000-i*200;
      prob[2*i+1] = ec_laplace_get_start_freq(prob[2*i]);
   }
   for (i=0;i<m->nbEBands;i++)
   {
      prob[2*m->nbEBands+2*i] = 9000-i*220;
      prob[2*m->nbEBands+2*i+1] = ec_laplace_get_start_freq(prob[2*m->nbEBands+2*i]);
   }
   return prob;
}

void quant_prob_free(int *freq)
{
   celt_free(freq);
}

void quant_coarse_energy(const CELTMode *m, int start, int end, const celt_word16 *eBands, celt_word16 *oldEBands, int budget, int intra, int *prob, celt_word16 *error, ec_enc *enc, int _C, int LM, celt_word16 max_decay)
{
   int i, c;
   celt_word32 prev[2] = {0,0};
   celt_word16 coef;
   celt_word16 beta;
   const int C = CHANNELS(_C);

   coef = pred_coef[LM];

   if (intra)
   {
      coef = 0;
      prob += 2*m->nbEBands;
   }
   /* No theoretical justification for this, it just works */
   beta = MULT16_16_P15(coef,coef);
   /* Encode at a fixed coarse resolution */
   for (i=start;i<end;i++)
   {
      c=0;
      do {
         int bits_left;
         int qi;
         celt_word16 q;
         celt_word16 x;
         celt_word32 f;
         x = eBands[i+c*m->nbEBands];
#ifdef FIXED_POINT
         f = SHL32(EXTEND32(x),15) -MULT16_16(coef,oldEBands[i+c*m->nbEBands])-prev[c];
         /* Rounding to nearest integer here is really important! */
         qi = (f+QCONST32(.5,DB_SHIFT+15))>>(DB_SHIFT+15);
#else
         f = x-coef*oldEBands[i+c*m->nbEBands]-prev[c];
         /* Rounding to nearest integer here is really important! */
         qi = (int)floor(.5f+f);
#endif
         /* Prevent the energy from going down too quickly (e.g. for bands
            that have just one bin) */
         if (qi < 0 && x < oldEBands[i+c*m->nbEBands]-max_decay)
         {
            qi += SHR16(oldEBands[i+c*m->nbEBands]-max_decay-x, DB_SHIFT);
            if (qi > 0)
               qi = 0;
         }
         /* If we don't have enough bits to encode all the energy, just assume something safe.
            We allow slightly busting the budget here */
         bits_left = budget-(int)ec_enc_tell(enc, 0)-2*C*(end-i);
         if (bits_left < 24)
         {
            if (qi > 1)
               qi = 1;
            if (qi < -1)
               qi = -1;
            if (bits_left<8)
               qi = 0;
         }
         ec_laplace_encode_start(enc, &qi, prob[2*i], prob[2*i+1]);
         error[i+c*m->nbEBands] = PSHR32(f,15) - SHL16(qi,DB_SHIFT);
         q = SHL16(qi,DB_SHIFT);
         
         oldEBands[i+c*m->nbEBands] = PSHR32(MULT16_16(coef,oldEBands[i+c*m->nbEBands]) + prev[c] + SHL32(EXTEND32(q),15), 15);
         prev[c] = prev[c] + SHL32(EXTEND32(q),15) - MULT16_16(beta,q);
      } while (++c < C);
   }
}

void quant_fine_energy(const CELTMode *m, int start, int end, celt_ener *eBands, celt_word16 *oldEBands, celt_word16 *error, int *fine_quant, ec_enc *enc, int _C)
{
   int i, c;
   const int C = CHANNELS(_C);

   /* Encode finer resolution */
   for (i=start;i<end;i++)
   {
      celt_int16 frac = 1<<fine_quant[i];
      if (fine_quant[i] <= 0)
         continue;
      c=0;
      do {
         int q2;
         celt_word16 offset;
#ifdef FIXED_POINT
         /* Has to be without rounding */
         q2 = (error[i+c*m->nbEBands]+QCONST16(.5f,DB_SHIFT))>>(DB_SHIFT-fine_quant[i]);
#else
         q2 = (int)floor((error[i+c*m->nbEBands]+.5f)*frac);
#endif
         if (q2 > frac-1)
            q2 = frac-1;
         if (q2<0)
            q2 = 0;
         ec_enc_bits(enc, q2, fine_quant[i]);
#ifdef FIXED_POINT
         offset = SUB16(SHR16(SHL16(q2,DB_SHIFT)+QCONST16(.5,DB_SHIFT),fine_quant[i]),QCONST16(.5f,DB_SHIFT));
#else
         offset = (q2+.5f)*(1<<(14-fine_quant[i]))*(1.f/16384) - .5f;
#endif
         oldEBands[i+c*m->nbEBands] += offset;
         error[i+c*m->nbEBands] -= offset;
         /*printf ("%f ", error[i] - offset);*/
      } while (++c < C);
   }
}

void quant_energy_finalise(const CELTMode *m, int start, int end, celt_ener *eBands, celt_word16 *oldEBands, celt_word16 *error, int *fine_quant, int *fine_priority, int bits_left, ec_enc *enc, int _C)
{
   int i, prio, c;
   const int C = CHANNELS(_C);

   /* Use up the remaining bits */
   for (prio=0;prio<2;prio++)
   {
      for (i=start;i<end && bits_left>=C ;i++)
      {
         if (fine_quant[i] >= 7 || fine_priority[i]!=prio)
            continue;
         c=0;
         do {
            int q2;
            celt_word16 offset;
            q2 = error[i+c*m->nbEBands]<0 ? 0 : 1;
            ec_enc_bits(enc, q2, 1);
#ifdef FIXED_POINT
            offset = SHR16(SHL16(q2,DB_SHIFT)-QCONST16(.5,DB_SHIFT),fine_quant[i]+1);
#else
            offset = (q2-.5f)*(1<<(14-fine_quant[i]-1))*(1.f/16384);
#endif
            oldEBands[i+c*m->nbEBands] += offset;
            bits_left--;
         } while (++c < C);
      }
   }
}

void unquant_coarse_energy(const CELTMode *m, int start, int end, celt_ener *eBands, celt_word16 *oldEBands, int intra, int *prob, ec_dec *dec, int _C, int LM)
{
   int i, c;
   celt_word32 prev[2] = {0, 0};
   celt_word16 coef;
   celt_word16 beta;
   const int C = CHANNELS(_C);

   coef = pred_coef[LM];

   if (intra)
   {
      coef = 0;
      prob += 2*m->nbEBands;
   }
   /* No theoretical justification for this, it just works */
   beta = MULT16_16_P15(coef,coef);

   /* Decode at a fixed coarse resolution */
   for (i=start;i<end;i++)
   {
      c=0;
      do {
         int qi;
         celt_word16 q;
         qi = ec_laplace_decode_start(dec, prob[2*i], prob[2*i+1]);
         q = SHL16(qi,DB_SHIFT);

         oldEBands[i+c*m->nbEBands] = PSHR32(MULT16_16(coef,oldEBands[i+c*m->nbEBands]) + prev[c] + SHL32(EXTEND32(q),15), 15);
         prev[c] = prev[c] + SHL32(EXTEND32(q),15) - MULT16_16(beta,q);
      } while (++c < C);
   }
}

void unquant_fine_energy(const CELTMode *m, int start, int end, celt_ener *eBands, celt_word16 *oldEBands, int *fine_quant, ec_dec *dec, int _C)
{
   int i, c;
   const int C = CHANNELS(_C);
   /* Decode finer resolution */
   for (i=start;i<end;i++)
   {
      if (fine_quant[i] <= 0)
         continue;
      c=0; 
      do {
         int q2;
         celt_word16 offset;
         q2 = ec_dec_bits(dec, fine_quant[i]);
#ifdef FIXED_POINT
         offset = SUB16(SHR16(SHL16(q2,DB_SHIFT)+QCONST16(.5,DB_SHIFT),fine_quant[i]),QCONST16(.5f,DB_SHIFT));
#else
         offset = (q2+.5f)*(1<<(14-fine_quant[i]))*(1.f/16384) - .5f;
#endif
         oldEBands[i+c*m->nbEBands] += offset;
      } while (++c < C);
   }
}

void unquant_energy_finalise(const CELTMode *m, int start, int end, celt_ener *eBands, celt_word16 *oldEBands, int *fine_quant,  int *fine_priority, int bits_left, ec_dec *dec, int _C)
{
   int i, prio, c;
   const int C = CHANNELS(_C);

   /* Use up the remaining bits */
   for (prio=0;prio<2;prio++)
   {
      for (i=start;i<end && bits_left>=C ;i++)
      {
         if (fine_quant[i] >= 7 || fine_priority[i]!=prio)
            continue;
         c=0;
         do {
            int q2;
            celt_word16 offset;
            q2 = ec_dec_bits(dec, 1);
#ifdef FIXED_POINT
            offset = SHR16(SHL16(q2,DB_SHIFT)-QCONST16(.5,DB_SHIFT),fine_quant[i]+1);
#else
            offset = (q2-.5f)*(1<<(14-fine_quant[i]-1))*(1.f/16384);
#endif
            oldEBands[i+c*m->nbEBands] += offset;
            bits_left--;
         } while (++c < C);
      }
   }
}

void log2Amp(const CELTMode *m, int start, int end,
      celt_ener *eBands, celt_word16 *oldEBands, int _C)
{
   int c, i;
   const int C = CHANNELS(_C);
   c=0;
   do {
      for (i=start;i<m->nbEBands;i++)
      {
         celt_word16 lg = oldEBands[i+c*m->nbEBands]+eMeans[i];
         eBands[i+c*m->nbEBands] = PSHR32(celt_exp2(SHL16(lg,11-DB_SHIFT)),4);
         if (oldEBands[i+c*m->nbEBands] < -QCONST16(14.f,DB_SHIFT))
            oldEBands[i+c*m->nbEBands] = -QCONST16(14.f,DB_SHIFT);
      }
   } while (++c < C);
}

void amp2Log2(const CELTMode *m, int effEnd, int end,
      celt_ener *bandE, celt_word16 *bandLogE, int _C)
{
   int c, i;
   const int C = CHANNELS(_C);
   c=0;
   do {
      for (i=0;i<effEnd;i++)
         bandLogE[i+c*m->nbEBands] =
               celt_log2(MAX32(QCONST32(.001f,14),SHL32(bandE[i+c*m->nbEBands],2)))
               - eMeans[i];
      for (i=effEnd;i<end;i++)
         bandLogE[c*m->nbEBands+i] = -QCONST16(14.f,DB_SHIFT);
   } while (++c < C);
}
