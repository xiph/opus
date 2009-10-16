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
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <string.h>


int async_tandem(int rate, int frame_size, int channels, int bitrate_min,
                 int bitrate_max)
{
    unsigned char data[250];
    CELTMode *mode = NULL;
    CELTEncoder *enc;
    short carry[2];
    short pcm[512 * 2];
    CELTDecoder *dec;
    int bmin, bmax;
    float ms;
    int ret, i, j, bytes_per_frame;
    int increment = 1;

    bmin = floor((bitrate_min / (rate / (float) frame_size)) / 8.0);
    bmax = ceil((bitrate_max / (rate / (float) frame_size)) / 8.0);
    if (bmin < 12)
        bmin = 12;
    if (bmax > 250)
        bmax = 250;
    if (bmin >= bmax)
        bmax = bmin + 8;

    /*increment += (bmax - bmin) / 64; */

    printf ("Testing asynchronous tandeming (%dHz, %dch, %d samples, %d - %d bytes).\n",
         rate, channels, frame_size, bmin, bmax);

    mode = celt_mode_create(rate, frame_size, NULL);
    if (mode == NULL) {
        fprintf(stderr, "Error: failed to create a mode\n");
        exit(1);
    }

    dec = celt_decoder_create(mode, channels, NULL);
    enc = celt_encoder_create(mode, channels, NULL);

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

            ret = celt_encode(enc, pcm, NULL, data, bytes_per_frame);
            if (ret != bytes_per_frame) {
                fprintf(stderr, "Error: during init celt_encode returned %d\n", ret);
                exit(1);
            }

            ret = celt_decode(dec, data, ret, pcm);
            if (ret != CELT_OK) {
                fprintf(stderr, "Error: during init celt_decode returned %d\n", ret);
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

            ret = celt_encode(enc, pcm, NULL, data, bytes_per_frame);
            if (ret != bytes_per_frame) {
                fprintf(stderr, "Error: at %d bytes_per_frame celt_encode returned %d\n",
                        bytes_per_frame, ret);
                exit(1);
            }

            ret = celt_decode(dec, data, ret, pcm);
            if (ret != CELT_OK) {
                fprintf(stderr, "Error: at %d bytes_per_frame celt_decode returned %d\n",
                        bytes_per_frame, ret);
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

    for (n = 6; n < 10; n++) {
        for (ch = 1; ch <= 2; ch++) {
            async_tandem(44100, 1 << n, ch, 47000 * ch, 77000 * ch);
            async_tandem(48000, 1 << n, ch, 47000 * ch, 77000 * ch);
            async_tandem(32000, 1 << n, ch, 31000 * ch, 65000 * ch);
        }
    }

    for (ch = 1; ch <= 2; ch++)
        async_tandem(32000, 320, ch, 31000 * ch, 65000 * ch);

    for (ch = 1; ch <= 2; ch++)
        async_tandem(48000, 480, ch, 31000 * ch, 77000 * ch);

    return 0;
}
