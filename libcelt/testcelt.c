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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "celt.h"
#include "arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(int argc, char *argv[])
{
   int i;
   char *inFile, *outFile;
   FILE *fin, *fout;
   CELTMode *mode=NULL;
   CELTEncoder *enc;
   CELTDecoder *dec;
   int len;
   celt_int32_t frame_size, channels;
   int bytes_per_packet;
   unsigned char data[1024];
   int rate, overlap;
   double rmsd = 0;
   int count = 0;
   int skip;
   if (argc != 8)
   {
      fprintf (stderr, "Usage: testcelt <rate> <channels> <frame size> <overlap> <bytes per packet> <input> <output>\n");
      return 1;
   }
   
   rate = atoi(argv[1]);
   channels = atoi(argv[2]);
   frame_size = atoi(argv[3]);
   overlap = atoi(argv[4]);
   skip = overlap;
   mode = celt_mode_create(rate, channels, frame_size, overlap, NULL);
   
   if (mode == NULL)
   {
      fprintf(stderr, "failed to create a mode\n");
      return 1;
   }
   
   bytes_per_packet = atoi(argv[5]);
   if (bytes_per_packet < 0 || bytes_per_packet > 120)
   {
      fprintf (stderr, "bytes per packet must be between 10 and 120\n");
      return 1;
   }
   inFile = argv[6];
   fin = fopen(inFile, "rb");
   if (!fin)
   {
      fprintf (stderr, "Could not open input file %s\n", argv[6]);
      return 1;
   }
   outFile = argv[7];
   fout = fopen(outFile, "wb+");
   if (!fout)
   {
      fprintf (stderr, "Could not open output file %s\n", argv[7]);
      return 1;
   }
   
   /* Use mode4 for stereo and don't forget to change the value of CHANNEL above */
   enc = celt_encoder_new(mode);
   dec = celt_decoder_new(mode);
   
   celt_mode_info(mode, CELT_GET_FRAME_SIZE, &frame_size);
   celt_mode_info(mode, CELT_GET_NB_CHANNELS, &channels);
   while (!feof(fin))
   {
      VARDECL(celt_int16_t *in);
      VARDECL(celt_int16_t *out);
      ALLOC(in, frame_size*channels, celt_int16_t);
      ALLOC(out, frame_size*channels, celt_int16_t);
      fread(in, sizeof(short), frame_size*channels, fin);
      if (feof(fin))
         break;
      len = celt_encode(enc, in, data, bytes_per_packet);
      if (len <= 0)
      {
         fprintf (stderr, "celt_encode() returned %d\n", len);
         return 1;
      }
      /* This is to simulate packet loss */
#if 1
      if (rand()%100==-1)
         celt_decode(dec, NULL, len, out);
      else
         celt_decode(dec, data, len, out);
#else
      for (i=0;i<frame_size*channels;i++)
         out[i] = in[i];
#endif
      for (i=0;i<frame_size*channels;i++)
         rmsd += (in[i]-out[i])*1.0*(in[i]-out[i]);
      count++;
      fwrite(out, sizeof(short), (frame_size-skip)*channels, fout);
      skip = 0;
   }
   celt_encoder_destroy(enc);
   celt_decoder_destroy(dec);
   fclose(fin);
   fclose(fout);
   if (rmsd > 0)
   {
      rmsd = sqrt(rmsd/(1.0*frame_size*channels*count));
      fprintf (stderr, "Error: encoder doesn't match decoder\n");
      fprintf (stderr, "RMS mismatch is %f\n", rmsd);
      return 1;
   } else {
      fprintf (stderr, "Encoder matches decoder!!\n");
   }
   celt_mode_destroy(mode);
   return 0;
}

