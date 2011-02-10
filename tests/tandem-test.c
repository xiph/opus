/* (C) 2009 Gregory Maxwell

   This test runs pink noise through the encoder and decoder many times 
   while feeding the output back into the input. It checks that after
   a number of cycles the energy has not increased or decreased by too
   large an amount.

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

#include "celt.h"
#include "arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <math.h>
#include <string.h>

#ifndef _MSC_VER
#include <unistd.h>
#endif

int async_tandem(int rate, int frame_size, int channels, int bitrate_min,
                 int bitrate_max)
{
    int error;
    unsigned char data[648];
    CELTMode *mode = NULL;
    CELTEncoder *enc;
    short carry[2];
    short pcm[960 * 2];
    CELTDecoder *dec;
    int bmin, bmax;
    float ms;
    int ret, i, j, bytes_per_frame;
    int increment = 1;

    bmin = floor((bitrate_min / (rate / (float) frame_size)) / 8.0);
    bmax = ceil((bitrate_max / (rate / (float) frame_size)) / 8.0);
    if (bmin < 8)
        bmin = 8;
    if (bmax > 640)
        bmax = 640;
    if (bmin >= bmax)
        bmax = bmin + 8;

    increment += (bmax - bmin) / 128; 

    printf ("Testing asynchronous tandeming (%dHz, %dch, %d samples, %d - %d bytes).\n",
         rate, channels, frame_size, bmin, bmax);

    mode = celt_mode_create(rate, frame_size, &error);
    if (mode == NULL || error) {
        fprintf(stderr, "Error: failed to create a mode: %s\n", celt_strerror(error));
        exit(1);
    }

    dec = celt_decoder_create_custom(mode, channels, &error);
    if (error){
      fprintf(stderr, "Error: celt_decoder_create returned %s\n", celt_strerror(error));
      exit(1);
    }
    enc = celt_encoder_create_custom(mode, channels, &error);
    if (error){
      fprintf(stderr, "Error: celt_encoder_create returned %s\n", celt_strerror(error));
      exit(1);
    }

    for (j = 0; j < frame_size * channels; j++)
        pcm[j] = 0;

    for (bytes_per_frame = bmin; bytes_per_frame <= bmax;
         bytes_per_frame += increment) {
        /*Prime the encoder and decoder */
        for (i = 0; i < (1024 + (frame_size >> 1)) / frame_size + 2; i++) {

            for (j = 0; j < channels; j++)
                pcm[j] = pcm[frame_size * channels - (channels - j + 1)];
            for (j = channels; j < frame_size * channels - 1; j++)
                pcm[j] = ((rand() % 4096) - 2048) + .9 * pcm[j - channels];

            ret = celt_encode(enc, pcm, frame_size, data, bytes_per_frame);
            if (ret != bytes_per_frame) {
                fprintf(stderr, "Error: celt_encode returned %s\n", celt_strerror(ret));
                exit(1);
            }

            ret = celt_decode(dec, data, ret, pcm, frame_size);
            if (ret != CELT_OK) {
                fprintf(stderr, "Error: celt_decode returned %s\n", celt_strerror(ret));
            }
        }

        for (j = 0; j < channels; j++)
            pcm[j] = pcm[frame_size * channels - (channels - j)];
        for (j = channels; j < frame_size * channels - 1; j++)
            pcm[j] = ((rand() % 4096) - 2048) + .9 * pcm[j - channels];

        for (i = 0; i < 8; i++) {
            for (j = 0; j < channels; j++)
                carry[j] = pcm[frame_size * channels - (channels - j)];
            memmove(pcm + channels, pcm, sizeof(short) * frame_size * channels);
            for (j = 0; j < channels; j++)
                pcm[j] = carry[j];

            ret = celt_encode(enc, pcm, frame_size, data, bytes_per_frame);
            if (ret != bytes_per_frame) {
                fprintf(stderr, "Error: at %d bytes_per_frame celt_encode returned %s\n",
                        bytes_per_frame, celt_strerror(ret));
                exit(1);
            }

            ret = celt_decode(dec, data, ret, pcm, frame_size);
            if (ret != CELT_OK) {
                fprintf(stderr, "Error: at %d bytes_per_frame celt_decode returned %s\n",
                        bytes_per_frame, celt_strerror(ret));
                exit(1);
            }
        }
        ms = 0;
        for (j = 0; j < frame_size * channels; j++)
            ms += pcm[j] * pcm[j];
        ms = sqrt(ms / (frame_size * channels));
        if (ms > 7000 || ms < 1000) {
            fprintf(stderr, "Error: Signal out of expected range. %d %d %d %d %f\n",
                    rate, channels, frame_size, bytes_per_frame, ms);
            exit(1);
        }
    }

    celt_encoder_destroy(enc);
    celt_decoder_destroy(dec);
    celt_mode_destroy(mode);

    return 0;
}

int main(int argc, char *argv[])
{
#ifdef CUSTOM_MODES
    int sizes[8]={960,512,480,256,240,128,120,64};
#else
    int sizes[4]={960,480,240,120};
#endif
    unsigned int seed;
    int ch, n;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [<seed>]\n", argv[0]);
        return 1;
    }

    if (argc > 1)
        seed = atoi(argv[1]);
    else
        seed = (time(NULL) ^ ((getpid() % 65536) << 16));

    srand(seed);
    printf("CELT codec tests. Random seed: %u (%.4X)\n", seed, rand() % 65536);

#ifdef CUSTOM_MODES
    for (n = 0; n < 8; n++) {
        for (ch = 1; ch <= 2; ch++) {
            async_tandem(48000, sizes[n], ch, 12000 * ch, 128000 * ch);
            async_tandem(44100, sizes[n], ch, 12000 * ch, 128000 * ch);
            if(n>0)async_tandem(32000, sizes[n], ch, 12000 * ch, 128000 * ch);
            if(n>2)async_tandem(16000, sizes[n], ch, 12000 * ch, 64000 * ch);
        }
    }
#else
    for (n = 0; n < 4; n++) {
        for (ch = 1; ch <= 2; ch++) {
            async_tandem(48000, sizes[n], ch, 12000 * ch, 128000 * ch);
        }
    }
#endif
    return 0;
}
