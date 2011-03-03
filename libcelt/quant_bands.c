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
#include "rate.h"

#ifdef FIXED_POINT
/* Mean energy in each band quantized in Q6 */
static const signed char eMeans[25] = {
      103,100, 92, 85, 81,
       77, 72, 70, 78, 75,
       73, 71, 78, 74, 69,
       72, 70, 74, 76, 71,
       60, 60, 60, 60, 60
};
#else
/* Mean energy in each band quantized in Q6 and converted back to float */
static const celt_word16 eMeans[25] = {
      6.437500f, 6.250000f, 5.750000f, 5.312500f, 5.062500f,
      4.812500f, 4.500000f, 4.375000f, 4.875000f, 4.687500f,
      4.562500f, 4.437500f, 4.875000f, 4.625000f, 4.312500f,
      4.500000f, 4.375000f, 4.625000f, 4.750000f, 4.437500f,
      3.750000f, 3.750000f, 3.750000f, 3.750000f, 3.750000f
};
#endif
/* prediction coefficients: 0.9, 0.8, 0.65, 0.5 */
#ifdef FIXED_POINT
static const celt_word16 pred_coef[4] = {29440, 26112, 21248, 16384};
static const celt_word16 beta_coef[4] = {30147, 22282, 12124, 6554};
static const celt_word16 beta_intra = 4915;
#else
static const celt_word16 pred_coef[4] = {29440/32768., 26112/32768., 21248/32768., 16384/32768.};
static const celt_word16 beta_coef[4] = {30147/32768., 22282/32768., 12124/32768., 6554/32768.};
static const celt_word16 beta_intra = 4915/32768.;
#endif

/*Parameters of the Laplace-like probability models used for the coarse energy.
  There is one pair of parameters for each frame size, prediction type
   (inter/intra), and band number.
  The first number of each pair is the probability of 0, and the second is the
   decay rate, both in Q8 precision.*/
static const unsigned char e_prob_model[4][2][42] = {
   /*120 sample frames.*/
   {
      /*Inter*/
      {
          72, 127,  65, 129,  66, 128,  65, 128,  64, 128,  62, 128,  64, 128,
          64, 128,  92,  78,  92,  79,  92,  78,  90,  79, 116,  41, 115,  40,
         114,  40, 132,  26, 132,  26, 145,  17, 161,  12, 176,  10, 177,  11
      },
      /*Intra*/
      {
          24, 179,  48, 138,  54, 135,  54, 132,  53, 134,  56, 133,  55, 132,
          55, 132,  61, 114,  70,  96,  74,  88,  75,  88,  87,  74,  89,  66,
          91,  67, 100,  59, 108,  50, 120,  40, 122,  37,  97,  43,  78,  50
      }
   },
   /*240 sample frames.*/
   {
      /*Inter*/
      {
          83,  78,  84,  81,  88,  75,  86,  74,  87,  71,  90,  73,  93,  74,
          93,  74, 109,  40, 114,  36, 117,  34, 117,  34, 143,  17, 145,  18,
         146,  19, 162,  12, 165,  10, 178,   7, 189,   6, 190,   8, 177,   9
      },
      /*Intra*/
      {
          23, 178,  54, 115,  63, 102,  66,  98,  69,  99,  74,  89,  71,  91,
          73,  91,  78,  89,  86,  80,  92,  66,  93,  64, 102,  59, 103,  60,
         104,  60, 117,  52, 123,  44, 138,  35, 133,  31,  97,  38,  77,  45
      }
   },
   /*480 sample frames.*/
   {
      /*Inter*/
      {
          61,  90,  93,  60, 105,  42, 107,  41, 110,  45, 116,  38, 113,  38,
         112,  38, 124,  26, 132,  27, 136,  19, 140,  20, 155,  14, 159,  16,
         158,  18, 170,  13, 177,  10, 187,   8, 192,   6, 175,   9, 159,  10
      },
      /*Intra*/
      {
          21, 178,  59, 110,  71,  86,  75,  85,  84,  83,  91,  66,  88,  73,
          87,  72,  92,  75,  98,  72, 105,  58, 107,  54, 115,  52, 114,  55,
         112,  56, 129,  51, 132,  40, 150,  33, 140,  29,  98,  35,  77,  42
      }
   },
   /*960 sample frames.*/
   {
      /*Inter*/
      {
          42, 121,  96,  66, 108,  43, 111,  40, 117,  44, 123,  32, 120,  36,
         119,  33, 127,  33, 134,  34, 139,  21, 147,  23, 152,  20, 158,  25,
         154,  26, 166,  21, 173,  16, 184,  13, 184,  10, 150,  13, 139,  15
      },
      /*Intra*/
      {
          22, 178,  63, 114,  74,  82,  84,  83,  92,  82, 103,  62,  96,  72,
          96,  67, 101,  73, 107,  72, 113,  55, 118,  52, 125,  52, 118,  52,
         117,  55, 135,  49, 137,  39, 157,  32, 145,  29,  97,  33,  77,  40
      }
   }
};

static const unsigned char small_energy_icdf[3]={2,1,0};

static int intra_decision(const celt_word16 *eBands, celt_word16 *oldEBands, int start, int end, int len, int C)
{
   int c, i;
   celt_word32 dist = 0;
   c=0; do {
      for (i=start;i<end;i++)
      {
         celt_word16 d = SHR16(SUB16(eBands[i+c*len], oldEBands[i+c*len]),2);
         dist = MAC16_16(dist, d,d);
      }
   } while (++c<C);
   return SHR32(dist,2*DB_SHIFT-4) > 2*C*(end-start);
}

static int quant_coarse_energy_impl(const CELTMode *m, int start, int end,
      const celt_word16 *eBands, celt_word16 *oldEBands,
      celt_int32 budget, celt_int32 tell,
      const unsigned char *prob_model, celt_word16 *error, ec_enc *enc,
      int _C, int LM, int intra, celt_word16 max_decay)
{
   const int C = CHANNELS(_C);
   int i, c;
   int badness = 0;
   celt_word32 prev[2] = {0,0};
   celt_word16 coef;
   celt_word16 beta;

   if (tell+3 <= budget)
      ec_enc_bit_logp(enc, intra, 3);
   if (intra)
   {
      coef = 0;
      beta = beta_intra;
   } else {
      beta = beta_coef[LM];
      coef = pred_coef[LM];
   }

   /* Encode at a fixed coarse resolution */
   for (i=start;i<end;i++)
   {
      c=0;
      do {
         int bits_left;
         int qi, qi0;
         celt_word32 q;
         celt_word16 x;
         celt_word32 f, tmp;
         celt_word16 oldE;
         celt_word16 decay_bound;
         x = eBands[i+c*m->nbEBands];
         oldE = MAX16(-QCONST16(9.f,DB_SHIFT), oldEBands[i+c*m->nbEBands]);
#ifdef FIXED_POINT
         f = SHL32(EXTEND32(x),7) - PSHR32(MULT16_16(coef,oldE), 8) - prev[c];
         /* Rounding to nearest integer here is really important! */
         qi = (f+QCONST32(.5f,DB_SHIFT+7))>>(DB_SHIFT+7);
         decay_bound = EXTRACT16(MAX32(-QCONST16(28.f,DB_SHIFT),
               SUB32((celt_word32)oldEBands[i+c*m->nbEBands],max_decay)));
#else
         f = x-coef*oldE-prev[c];
         /* Rounding to nearest integer here is really important! */
         qi = (int)floor(.5f+f);
         decay_bound = MAX16(-QCONST16(28.f,DB_SHIFT), oldEBands[i+c*m->nbEBands]) - max_decay;
#endif
         /* Prevent the energy from going down too quickly (e.g. for bands
            that have just one bin) */
         if (qi < 0 && x < decay_bound)
         {
            qi += (int)SHR16(SUB16(decay_bound,x), DB_SHIFT);
            if (qi > 0)
               qi = 0;
         }
         qi0 = qi;
         /* If we don't have enough bits to encode all the energy, just assume
             something safe. */
         tell = ec_tell(enc);
         bits_left = budget-tell-3*C*(end-i);
         if (i!=start && bits_left < 30)
         {
            if (bits_left < 24)
               qi = IMIN(1, qi);
            if (bits_left < 16)
               qi = IMAX(-1, qi);
         }
         if (budget-tell >= 15)
         {
            int pi;
            pi = 2*IMIN(i,20);
            ec_laplace_encode(enc, &qi,
                  prob_model[pi]<<7, prob_model[pi+1]<<6);
         }
         else if(budget-tell >= 2)
         {
            qi = IMAX(-1, IMIN(qi, 1));
            ec_enc_icdf(enc, 2*qi^-(qi<0), small_energy_icdf, 2);
         }
         else if(budget-tell >= 1)
         {
            qi = IMIN(0, qi);
            ec_enc_bit_logp(enc, -qi, 1);
         }
         else
            qi = -1;
         error[i+c*m->nbEBands] = PSHR32(f,7) - SHL16(qi,DB_SHIFT);
         badness += abs(qi0-qi);
         q = SHL32(EXTEND32(qi),DB_SHIFT);
         
         tmp = PSHR32(MULT16_16(coef,oldE),8) + prev[c] + SHL32(q,7);
#ifdef FIXED_POINT
         tmp = MAX32(-QCONST32(28.f, DB_SHIFT+7), tmp);
#endif
         oldEBands[i+c*m->nbEBands] = PSHR32(tmp, 7);
         prev[c] = prev[c] + SHL32(q,7) - MULT16_16(beta,PSHR32(q,8));
      } while (++c < C);
   }
   return badness;
}

void quant_coarse_energy(const CELTMode *m, int start, int end, int effEnd,
      const celt_word16 *eBands, celt_word16 *oldEBands, celt_uint32 budget,
      celt_word16 *error, ec_enc *enc, int _C, int LM, int nbAvailableBytes,
      int force_intra, int *delayedIntra, int two_pass)
{
   const int C = CHANNELS(_C);
   int intra;
   celt_word16 max_decay;
   VARDECL(celt_word16, oldEBands_intra);
   VARDECL(celt_word16, error_intra);
   ec_enc enc_start_state;
   celt_uint32 tell;
   int badness1=0;
   SAVE_STACK;

   intra = force_intra || (*delayedIntra && nbAvailableBytes > (end-start)*C);
   if (/*shortBlocks || */intra_decision(eBands, oldEBands, start, effEnd, m->nbEBands, C))
      *delayedIntra = 1;
   else
      *delayedIntra = 0;

   tell = ec_tell(enc);
   if (tell+3 > budget)
      two_pass = intra = 0;

   /* Encode the global flags using a simple probability model
      (first symbols in the stream) */

#ifdef FIXED_POINT
      max_decay = MIN32(QCONST16(16.f,DB_SHIFT), SHL32(EXTEND32(nbAvailableBytes),DB_SHIFT-3));
#else
   max_decay = MIN32(16.f, .125f*nbAvailableBytes);
#endif

   enc_start_state = *enc;

   ALLOC(oldEBands_intra, C*m->nbEBands, celt_word16);
   ALLOC(error_intra, C*m->nbEBands, celt_word16);
   CELT_COPY(oldEBands_intra, oldEBands, C*m->nbEBands);

   if (two_pass || intra)
   {
      badness1 = quant_coarse_energy_impl(m, start, end, eBands, oldEBands_intra, budget,
            tell, e_prob_model[LM][1], error_intra, enc, C, LM, 1, max_decay);
   }

   if (!intra)
   {
      ec_enc enc_intra_state;
      int tell_intra;
      celt_uint32 nstart_bytes;
      celt_uint32 nintra_bytes;
      int badness2;
      VARDECL(unsigned char, intra_bits);

      tell_intra = ec_tell_frac(enc);

      enc_intra_state = *enc;

      nstart_bytes = ec_range_bytes(&enc_start_state);
      nintra_bytes = ec_range_bytes(&enc_intra_state);
      ALLOC(intra_bits, nintra_bytes-nstart_bytes, unsigned char);
      /* Copy bits from intra bit-stream */
      CELT_COPY(intra_bits,
            ec_get_buffer(&enc_intra_state) + nstart_bytes,
            nintra_bytes - nstart_bytes);

      *enc = enc_start_state;

      badness2 = quant_coarse_energy_impl(m, start, end, eBands, oldEBands, budget,
            tell, e_prob_model[LM][intra], error, enc, C, LM, 0, max_decay);

      if (two_pass && (badness1 < badness2 || (badness1 == badness2 && ec_tell_frac(enc) > tell_intra)))
      {
         *enc = enc_intra_state;
         /* Copy intra bits to bit-stream */
         CELT_COPY(ec_get_buffer(&enc_intra_state) + nstart_bytes,
               intra_bits, nintra_bytes - nstart_bytes);
         CELT_COPY(oldEBands, oldEBands_intra, C*m->nbEBands);
         CELT_COPY(error, error_intra, C*m->nbEBands);
      }
   } else {
      CELT_COPY(oldEBands, oldEBands_intra, C*m->nbEBands);
      CELT_COPY(error, error_intra, C*m->nbEBands);
   }
   RESTORE_STACK;
}

void quant_fine_energy(const CELTMode *m, int start, int end, celt_word16 *oldEBands, celt_word16 *error, int *fine_quant, ec_enc *enc, int _C)
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
         offset = SUB16(SHR32(SHL32(EXTEND32(q2),DB_SHIFT)+QCONST16(.5f,DB_SHIFT),fine_quant[i]),QCONST16(.5f,DB_SHIFT));
#else
         offset = (q2+.5f)*(1<<(14-fine_quant[i]))*(1.f/16384) - .5f;
#endif
         oldEBands[i+c*m->nbEBands] += offset;
         error[i+c*m->nbEBands] -= offset;
         /*printf ("%f ", error[i] - offset);*/
      } while (++c < C);
   }
}

void quant_energy_finalise(const CELTMode *m, int start, int end, celt_word16 *oldEBands, celt_word16 *error, int *fine_quant, int *fine_priority, int bits_left, ec_enc *enc, int _C)
{
   int i, prio, c;
   const int C = CHANNELS(_C);

   /* Use up the remaining bits */
   for (prio=0;prio<2;prio++)
   {
      for (i=start;i<end && bits_left>=C ;i++)
      {
         if (fine_quant[i] >= MAX_FINE_BITS || fine_priority[i]!=prio)
            continue;
         c=0;
         do {
            int q2;
            celt_word16 offset;
            q2 = error[i+c*m->nbEBands]<0 ? 0 : 1;
            ec_enc_bits(enc, q2, 1);
#ifdef FIXED_POINT
            offset = SHR16(SHL16(q2,DB_SHIFT)-QCONST16(.5f,DB_SHIFT),fine_quant[i]+1);
#else
            offset = (q2-.5f)*(1<<(14-fine_quant[i]-1))*(1.f/16384);
#endif
            oldEBands[i+c*m->nbEBands] += offset;
            bits_left--;
         } while (++c < C);
      }
   }
}

void unquant_coarse_energy(const CELTMode *m, int start, int end, celt_word16 *oldEBands, int intra, ec_dec *dec, int _C, int LM)
{
   const unsigned char *prob_model = e_prob_model[LM][intra];
   int i, c;
   celt_word32 prev[2] = {0, 0};
   celt_word16 coef;
   celt_word16 beta;
   const int C = CHANNELS(_C);
   celt_int32 budget;
   celt_int32 tell;


   if (intra)
   {
      coef = 0;
      beta = beta_intra;
   } else {
      beta = beta_coef[LM];
      coef = pred_coef[LM];
   }

   budget = dec->storage*8;

   /* Decode at a fixed coarse resolution */
   for (i=start;i<end;i++)
   {
      c=0;
      do {
         int qi;
         celt_word32 q;
         celt_word32 tmp;
         tell = ec_tell(dec);
         if(budget-tell>=15)
         {
            int pi;
            pi = 2*IMIN(i,20);
            qi = ec_laplace_decode(dec,
                  prob_model[pi]<<7, prob_model[pi+1]<<6);
         }
         else if(budget-tell>=2)
         {
            qi = ec_dec_icdf(dec, small_energy_icdf, 2);
            qi = (qi>>1)^-(qi&1);
         }
         else if(budget-tell>=1)
         {
            qi = -ec_dec_bit_logp(dec, 1);
         }
         else
            qi = -1;
         q = SHL32(EXTEND32(qi),DB_SHIFT);

         oldEBands[i+c*m->nbEBands] = MAX16(-QCONST16(9.f,DB_SHIFT), oldEBands[i+c*m->nbEBands]);
         tmp = PSHR32(MULT16_16(coef,oldEBands[i+c*m->nbEBands]),8) + prev[c] + SHL32(q,7);
#ifdef FIXED_POINT
         tmp = MAX32(-QCONST32(28.f, DB_SHIFT+7), tmp);
#endif
         oldEBands[i+c*m->nbEBands] = PSHR32(tmp, 7);
         prev[c] = prev[c] + SHL32(q,7) - MULT16_16(beta,PSHR32(q,8));
      } while (++c < C);
   }
}

void unquant_fine_energy(const CELTMode *m, int start, int end, celt_word16 *oldEBands, int *fine_quant, ec_dec *dec, int _C)
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
         offset = SUB16(SHR32(SHL32(EXTEND32(q2),DB_SHIFT)+QCONST16(.5f,DB_SHIFT),fine_quant[i]),QCONST16(.5f,DB_SHIFT));
#else
         offset = (q2+.5f)*(1<<(14-fine_quant[i]))*(1.f/16384) - .5f;
#endif
         oldEBands[i+c*m->nbEBands] += offset;
      } while (++c < C);
   }
}

void unquant_energy_finalise(const CELTMode *m, int start, int end, celt_word16 *oldEBands, int *fine_quant,  int *fine_priority, int bits_left, ec_dec *dec, int _C)
{
   int i, prio, c;
   const int C = CHANNELS(_C);

   /* Use up the remaining bits */
   for (prio=0;prio<2;prio++)
   {
      for (i=start;i<end && bits_left>=C ;i++)
      {
         if (fine_quant[i] >= MAX_FINE_BITS || fine_priority[i]!=prio)
            continue;
         c=0;
         do {
            int q2;
            celt_word16 offset;
            q2 = ec_dec_bits(dec, 1);
#ifdef FIXED_POINT
            offset = SHR16(SHL16(q2,DB_SHIFT)-QCONST16(.5f,DB_SHIFT),fine_quant[i]+1);
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
      for (i=0;i<start;i++)
         eBands[i+c*m->nbEBands] = 0;
      for (;i<end;i++)
      {
         celt_word16 lg = ADD16(oldEBands[i+c*m->nbEBands],
                         SHL16((celt_word16)eMeans[i],6));
         eBands[i+c*m->nbEBands] = PSHR32(celt_exp2(lg),4);
      }
      for (;i<m->nbEBands;i++)
         eBands[i+c*m->nbEBands] = 0;
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
               celt_log2(SHL32(bandE[i+c*m->nbEBands],2))
               - SHL16((celt_word16)eMeans[i],6);
      for (i=effEnd;i<end;i++)
         bandLogE[c*m->nbEBands+i] = -QCONST16(14.f,DB_SHIFT);
   } while (++c < C);
}
