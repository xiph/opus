/* Copyright (c) 2018 Mozilla */
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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "arch.h"
#include "lpcnet.h"
#include "os_support.h"
#include "cpu_support.h"
#include "osce_features.h"


void usage(void) {
    fprintf(stderr, "usage: bwe_demo <input.pcm> <output.pcm>\n");
    exit(1);
}

int main(int argc, char **argv) {
    int arch;
    FILE *fin, *fout;
    arch = opus_select_arch();
    if (argc != 3) usage();
    fin = fopen(argv[1], "rb");
    if (fin == NULL) {
        fprintf(stderr, "Can't open %s\n", argv[2]);
        exit(1);
    }

    fout = fopen(argv[2], "wb");
    if (fout == NULL) {
        fprintf(stderr, "Can't open %s\n", argv[3]);
        exit(1);
    }

    printf("Feature calculation with signal (100 * (n % 90)) - 8900\n");
    int n = 0, i;
    opus_int16 frame[160];
    int frame_counter = 0;
    float features[32 + 2 * 41];

    for (frame_counter = 0; frame_counter < 10; frame_counter ++)
    {
        for (i = 0; i < 160; i ++ )
        {
            frame[i] = 100 * n++ - 8900;
            n = n % 90;
        }

        osce_bwe_calculate_features(features, frame, 160);

        printf("frame[%d]\n", frame_counter);
        printf("lmspec: ");
        for (i = 0; i < 32; i ++)
        {
            printf(" %f ", features[i]);
        }
        printf("\nphasediff: ");
        for (;i < 32 + 2 * 41; i ++)
        {
            printf(" %f ", features[i]);
        }
        printf("\n\n");

        fwrite(frame, sizeof(frame[0]), 160, fout);

    }



    fclose(fin);
    fclose(fout);

    return 0;
}
