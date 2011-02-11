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


#define MAX_PACKET 1024

int main(int argc, char *argv[])
{
   int err;
   char *inFile, *outFile;
   FILE *fin, *fout;
   OpusEncoder *enc;
   int len;
   int frame_size, channels;
   int bytes_per_packet;
   unsigned char data[MAX_PACKET];
   int rate;
   int count = 0;
   int stop=0;
   int vbr=0;
   int tot_read=0, tot_written=0;
   short *in, *out;
   int mode=MODE_HYBRID;
   double bits=0;
   if (argc != 9 && argc != 8 && argc != 7)
   {
      fprintf (stderr, "Usage: test_opus <rate (kHz)> <channels> <frame size> "
               " <bytes per packet>  [<VBR rate (kb/s)>] [<packet loss rate>] "
               "<input> <output>\n");
      return 1;
   }

   rate = atoi(argv[1]);
   channels = atoi(argv[2]);
   frame_size = atoi(argv[3]);

   bytes_per_packet = atoi(argv[4]);

   if (argc >= 8)
       vbr = atoi(argv[5]);

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

   enc = opus_encoder_create(rate, channels);

   mode = MODE_HYBRID;
   opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(BANDWIDTH_FULLBAND));
   opus_encoder_ctl(enc, OPUS_SET_MODE(mode));

   in = (short*)malloc(frame_size*channels*sizeof(short));
   while (!stop)
   {
      int write_samples;
      err = fread(in, sizeof(short), frame_size*channels, fin);
      tot_read += err;
      if (err < frame_size*channels)
      {
          int i;
          for (i=err;i<frame_size*channels;i++)
              in[i] = 0;
          stop = 1;
      }
      len = opus_encode(enc, in, frame_size, data, bytes_per_packet);
      if (len <= 0)
      {
         fprintf (stderr, "opus_encode() returned %d\n", len);
         return 1;
      }
      bits += len*8;
      count++;
      fputc((len>>8)&0xFF, fout);
      fputc((len)&0xFF, fout);
      fwrite(data, 1, len, fout);
   }
   fprintf (stderr, "average bit-rate: %f kb/s\n", bits*rate/(frame_size*(double)count));
   opus_encoder_destroy(enc);
   fclose(fin);
   fclose(fout);
   free(in);
   free(out);
   return 0;
}
