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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "opus.h"


#define MAX_PACKET 1275

int main(int argc, char *argv[])
{
   int err;
   char *inFile, *outFile;
   FILE *fin, *fout;
   OpusDecoder *dec;
   int len;
   int frame_size, channels;
   int bytes_per_packet;
   unsigned char data[MAX_PACKET];
   int rate;
   int loss = 0;
   int count = 0;
   int stop=0;
   int vbr=0;
   int tot_read=0;
   short *in, *out;
   int mode=MODE_HYBRID;
   double bits=0;
   if (argc != 5 && argc != 6)
   {
      fprintf (stderr, "Usage: test_opus <rate (kHz)> <channels> "
               "[<packet loss rate>] "
               "<input> <output>\n");
      return 1;
   }

   rate = atoi(argv[1]);
   channels = atoi(argv[2]);

   if (argc >= 7)
       loss = atoi(argv[3]);

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

   dec = opus_decoder_create(rate, channels);

   out = (short*)malloc(960*channels*sizeof(short));
   while (!stop)
   {
      len = ((fgetc(fin)<<8)&0xFF00) | (fgetc(fin)&0xFF);
      if (feof(fin) || len>MAX_PACKET)
          break;
      bits += len*8;
      err = fread(data, 1, len, fin);
      frame_size = opus_decode(dec, rand()%100<loss ? NULL : data, len, out, 960);
      count+=frame_size;
      fwrite(out, sizeof(short), frame_size*channels, fout);
   }
   fprintf (stderr, "average bit-rate: %f kb/s\n", bits*rate/((double)count));
   opus_decoder_destroy(dec);
   fclose(fin);
   fclose(fout);
   free(in);
   free(out);
   return 0;
}
