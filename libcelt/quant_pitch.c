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

static const float cdbk_pitch[]={     0.00826816, 0.00646836, 0.00520978, 0.00632398, 0.0108199 ,
   0.723347, 0.0985132, 0.630212, 0.0546661, 0.0246779 ,
   0.802152, 0.759963, 0.453441, 0.384415, 0.0625198 ,
   0.461809, 0.313537, 0.0465707, 0.0484357, 0.0196977 ,
   0.704463, 0.117481, 0.0612584, 0.576791, 0.0273508 ,
   0.798109, 0.743359, 0.706289, 0.0697711, 0.0386172 ,
   0.228351, 0.0379557, 0.0285191, 0.0236265, 0.0248726 ,
   0.114495, 0.279541, 0.038657, 0.0342054, 0.0319817 ,
   0.609812, 0.0397131, 0.0266344, 0.0239864, 0.0212439 ,
   0.781349, 0.493848, 0.0353301, 0.0267205, 0.0196015 ,
   0.0819145, 0.0543806, 0.301274, 0.0507369, 0.0456495 ,
   0.785118, 0.7315, 0.213724, 0.675786, 0.117822 ,
   0.791656, 0.0322449, 0.0200075, 0.0203656, 0.0191236 ,
   0.768495, 0.416117, 0.386172, 0.0510886, 0.022891 ,
   0.802694, 0.790402, 0.755665, 0.71349, 0.401332 ,
   0.142781, 0.122736, 0.195102, 0.587634, 0.0490036 ,
   0.104903, 0.611318, 0.0587345, 0.0822444, 0.028738 ,
   0.182943, 0.541788, 0.518271, 0.0920779, 0.0338024 ,
   0.76004, 0.0553314, 0.293129, 0.0392962, 0.0191814 ,
   0.776575, 0.257797, 0.0323301, 0.0290356, 0.0185467 ,
   0.798177, 0.759494, 0.368838, 0.0497087, 0.0262797 ,
   0.724, 0.0643148, 0.0435992, 0.275742, 0.0232963 ,
   0.803454, 0.783768, 0.741939, 0.711135, 0.0950338 ,
   0.138766, 0.0770751, 0.649913, 0.129772, 0.0342524 ,
   0.795955, 0.754889, 0.044966, 0.0279729, 0.0199437 ,
   0.784278, 0.659435, 0.0713927, 0.339058, 0.0384502 ,
   0.145974, 0.049395, 0.0403857, 0.302676, 0.04925 ,
   0.428821, 0.0640999, 0.368384, 0.0519584, 0.0224072 ,
   0.425792, 0.0341016, 0.0290549, 0.0390711, 0.0227266 ,
   0.453561, 0.623889, 0.0443806, 0.0432896, 0.0173471 ,
   0.754851, 0.236843, 0.570091, 0.48421, 0.0622915 ,
   0.804932, 0.782549, 0.752207, 0.390957, 0.102058    
};

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

void quant_pitch(float *gains, int len, ec_enc *enc)
{
   int i, id;
   float g2[len];
#if 0
   for (i=0;i<len;i++)
      g2[i] = gains[i]*gains[i];
   id = vq_index(g2, cdbk_pitch, len, 32);
   ec_enc_uint(enc, id, 32);
   for (i=0;i<len;i++)
      gains[i] = sqrt(cdbk_pitch[id*len+i]);
#else
   //for (i=0;i<len;i++) printf ("%f ", gains[i]);printf ("\n");
   for (i=0;i<len;i++)
      g2[i] = 1-sqrt(1-gains[i]*gains[i]);
   id = vq_index(g2, pgain_table, len, 128);
   ec_enc_uint(enc, id, 128);
   //for (i=0;i<len;i++) printf ("%f ", pgain_table[id*len+i]);printf ("\n");   
   for (i=0;i<len;i++)
      gains[i] = (sqrt(1-(1-pgain_table[id*len+i])*(1-pgain_table[id*len+i])));
   //for (i=0;i<len;i++) printf ("%f ", g2[i]);printf ("\n");
   //for (i=0;i<len;i++) printf ("%f ", gains[i]);printf ("\n");
   //printf ("\n");
#endif
}

void unquant_pitch(float *gains, int len, ec_dec *dec)
{
   int i, id;
#if 0
   id = ec_dec_uint(dec, 32);
   for (i=0;i<len;i++)
      gains[i] = sqrt(cdbk_pitch[id*len+i]);
#else
   id = ec_dec_uint(dec, 128);
   for (i=0;i<len;i++)
      gains[i] = (sqrt(1-(1-pgain_table[id*len+i])*(1-pgain_table[id*len+i])));
#endif
}
