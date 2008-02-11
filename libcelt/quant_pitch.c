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

#include "quant_pitch.h"
#include <math.h>
#include "pgain_table.h"

/* Taken from Speex.
   Finds the index of the entry in a codebook that best matches the input*/
int vq_index(float *in, const float *codebook, int len, int entries)
{
   int i,j;
   float min_dist=0;
   int best_index=0;
   for (i=0;i<entries;i++)
   {
      float dist=0;
      for (j=0;j<len;j++)
      {
         float tmp = in[j]-*codebook++;
         dist += tmp*tmp;
      }
      if (i==0 || dist<min_dist)
      {
         min_dist=dist;
         best_index=i;
      }
   }
   return best_index;
}

int quant_pitch(float *gains, int len, ec_enc *enc)
{
   int i, id;
   float g2[len];
   //for (i=0;i<len;i++) printf ("%f ", gains[i]);printf ("\n");
   for (i=0;i<len;i++)
      g2[i] = 1-sqrt(1-gains[i]*gains[i]);
   id = vq_index(g2, pgain_table, len, 128);
   ec_enc_uint(enc, id, 128);
   //for (i=0;i<len;i++) printf ("%f ", pgain_table[id*len+i]);printf ("\n");   
   for (i=0;i<len;i++)
      gains[i] = (sqrt(1-(1-pgain_table[id*len+i])*(1-pgain_table[id*len+i])));
   return id!=0;
}

int unquant_pitch(float *gains, int len, ec_dec *dec)
{
   int i, id;
   id = ec_dec_uint(dec, 128);
   for (i=0;i<len;i++)
      gains[i] = (sqrt(1-(1-pgain_table[id*len+i])*(1-pgain_table[id*len+i])));
   return id!=0;
}
