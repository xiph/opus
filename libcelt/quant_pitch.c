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

#include "quant_pitch.h"
#include <math.h>
#include "pgain_table.h"
#include "arch.h"
#include "mathops.h"

#ifdef FIXED_POINT
#define PGAIN_ODD(codebook, i) ((celt_word16_t)(((codebook)[(i)]&0x00ffU)<<7))
#define PGAIN_EVEN(codebook, i) ((celt_word16_t)(((codebook)[(i)]&0xff00U)>>1))

#else
#define PGAIN_ODD(codebook, i) ((1.f/32768.f)*(celt_word16_t)(((codebook)[(i)]&0x00ffU)<<7))
#define PGAIN_EVEN(codebook, i) ((1.f/32768.f)*(celt_word16_t)(((codebook)[(i)]&0xff00U)>>1) )
#endif

#define PGAIN(codebook, i) ((i)&1 ? PGAIN_ODD(codebook, (i)>>1) : PGAIN_EVEN(codebook, (i)>>1))


#define Q1515ONE MULT16_16(Q15ONE,Q15ONE)

/** Taken from Speex.Finds the index of the entry in a codebook that best matches the input*/
int vq_index(celt_pgain_t *in, const celt_uint16_t *codebook, int len, int entries)
{
   int i,j;
   int ind = 0;
   celt_word32_t min_dist=0;
   int best_index=0;
   for (i=0;i<entries;i++)
   {
      celt_word32_t dist=0;
      for (j=0;j<len>>1;j++)
      {
         celt_pgain_t tmp1 = SHR16(SUB16(in[2*j],PGAIN_EVEN(codebook, ind)),1);
         celt_pgain_t tmp2 = SHR16(SUB16(in[2*j+1],PGAIN_ODD(codebook, ind)),1);
         ind++;
         dist = MAC16_16(dist, tmp1, tmp1);
         dist = MAC16_16(dist, tmp2, tmp2);
      }
      if (i==0 || dist<min_dist)
      {
         min_dist=dist;
         best_index=i;
      }
   }
   return best_index;
}

/** Returns the pitch gain vector corresponding to a certain id */
static void id2gains(int id, celt_pgain_t *gains, int len)
{
   int i;
   for (i=0;i<len;i++)
      gains[i] = celt_sqrt(Q1515ONE-MULT16_16(Q15ONE-PGAIN(pgain_table,id*len+i),Q15ONE-PGAIN(pgain_table,id*len+i)));
}

int quant_pitch(celt_pgain_t *gains, int len, ec_enc *enc)
{
   int i, id;
   /*for (i=0;i<len;i++) printf ("%f ", gains[i]);printf ("\n");*/
   /* Convert to a representation where the MSE criterion should be near-optimal */
   for (i=0;i<len;i++)
      gains[i] = Q15ONE-celt_sqrt(Q1515ONE-MULT16_16(gains[i],gains[i]));
   id = vq_index(gains, pgain_table, len, 128);
   ec_enc_uint(enc, id, 128);
   /*for (i=0;i<len;i++) printf ("%f ", pgain_table[id*len+i]);printf ("\n");*/
   id2gains(id, gains, len);
   return id!=0;
}

int unquant_pitch(celt_pgain_t *gains, int len, ec_dec *dec)
{
   int id;
   id = ec_dec_uint(dec, 128);
   id2gains(id, gains, len);
   return id!=0;
}
