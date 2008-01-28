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


#include "celt.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FRAME_SIZE 256
#define CHANNELS 1

int main(int argc, char *argv[])
{
   int i;
   char *inFile, *outFile;
   FILE *fin, *fout;
   short in[FRAME_SIZE*CHANNELS];
   short out[FRAME_SIZE*CHANNELS];
   CELTEncoder *enc;
   CELTDecoder *dec;
   int len;
   char data[1024];

   double rmsd = 0;
   int count = 0;
   
   inFile = argv[1];
   fin = fopen(inFile, "rb");
   outFile = argv[2];
   fout = fopen(outFile, "wb+");
   
   /* Use mode4 for stereo and don't forget to change the value of CHANNEL above */
   enc = celt_encoder_new(celt_mono);
   dec = celt_decoder_new(celt_mono);
   
   while (!feof(fin))
   {
      fread(in, sizeof(short), FRAME_SIZE*CHANNELS, fin);
      len = celt_encode(enc, in, data, 32);
      //printf ("\n");
      //printf ("%d\n", len);
      /* This is to simulate packet loss */
      if (rand()%100==-1)
         celt_decode(dec, NULL, len, out);
      else
         celt_decode(dec, data, len, out);
      //printf ("\n");
      for (i=0;i<FRAME_SIZE*CHANNELS;i++)
         rmsd += (in[i]-out[i])*1.0*(in[i]-out[i]);
      count++;
      fwrite(out, sizeof(short), FRAME_SIZE*CHANNELS, fout);
   }
   celt_encoder_destroy(enc);
   celt_decoder_destroy(dec);
   fclose(fin);
   fclose(fout);
   if (rmsd > 0)
   {
      rmsd = sqrt(rmsd/(1.0*FRAME_SIZE*CHANNELS*count));
      fprintf (stderr, "Error: encoder doesn't match decoder\n");
      fprintf (stderr, "RMS mismatch is %f\n", rmsd);
      return 1;
   } else {
      fprintf (stderr, "Encoder matches decoder!!\n");
   }
   return 0;
}

