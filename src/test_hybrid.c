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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "hybrid.h"


#define MAX_PACKET 1024

int main(int argc, char *argv[])
{
   int err;
   char *inFile, *outFile;
   FILE *fin, *fout;
   HybridEncoder *enc;
   HybridDecoder *dec;
   int len;
   int frame_size, channels;
   int bytes_per_packet;
   unsigned char data[MAX_PACKET];
   int rate;
   int count = 0;
   int skip;
   short *in, *out;
   int mode=MODE_HYBRID;
   if (argc != 9 && argc != 8 && argc != 7)
   {
      fprintf (stderr, "Usage: test_hybrid <rate> <channels> <frame size> "
               " <bytes per packet>  "
               "<input> <output>\n");
      return 1;
   }

   rate = atoi(argv[1]);
   channels = atoi(argv[2]);
   frame_size = atoi(argv[3]);

   bytes_per_packet = atoi(argv[4]);
   if (bytes_per_packet < 0 || bytes_per_packet > MAX_PACKET)
   {
      fprintf (stderr, "bytes per packet must be between 0 and %d\n",
                        MAX_PACKET);
      return 1;
   }

   inFile = argv[argc-2];
   fin = fopen(inFile, "rb");
   if (!fin)
   {
      fprintf (stderr, "Could not open input file %s\n", argv[argc-2]);
      return 1;
   }
   outFile = argv[argc-1];
   fout = fopen(outFile, "wb+");
   if (!fout)
   {
      fprintf (stderr, "Could not open output file %s\n", argv[argc-1]);
      return 1;
   }

   enc = hybrid_encoder_create();
   dec = hybrid_decoder_create();

   mode = MODE_HYBRID;
   hybrid_encoder_ctl(enc, HYBRID_SET_BANDWIDTH(BANDWIDTH_FULLBAND));
   hybrid_encoder_ctl(enc, HYBRID_SET_MODE(mode));

   hybrid_decoder_ctl(dec, HYBRID_SET_BANDWIDTH(BANDWIDTH_FULLBAND));
   hybrid_decoder_ctl(dec, HYBRID_SET_MODE(mode));

   in = (short*)malloc(frame_size*channels*sizeof(short));
   out = (short*)malloc(frame_size*channels*sizeof(short));
   while (!feof(fin))
   {
      err = fread(in, sizeof(short), frame_size*channels, fin);
      if (feof(fin))
         break;
      len = hybrid_encode(enc, in, frame_size, data, bytes_per_packet);
      if (len <= 0)
      {
         fprintf (stderr, "hybrid_encode() returned %d\n", len);
         return 1;
      }
      hybrid_decode(dec, data, len, out, frame_size);
      count++;
      fwrite(out+skip, sizeof(short), (frame_size-skip)*channels, fout);
      skip = 0;
   }

   hybrid_encoder_destroy(enc);
   hybrid_decoder_destroy(dec);
   fclose(fin);
   fclose(fout);
   free(in);
   free(out);
   return 0;
}
