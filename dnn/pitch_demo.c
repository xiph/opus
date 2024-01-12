/* Copyright (c) 2018 Mozilla
                 2024 Amazon */
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
#include <stdlib.h>
#include "arch.h"
#include "lpcnet.h"
#include "cpu_support.h"

void usage(void) {
    fprintf(stderr, "usage: lpcnet_demo <input.pcm> <output.txt>\n");
    fprintf(stderr, "       where input.pcm is a 16 kHz raw (no wav heaver) 16-bit PCM file\n");
    exit(1);
}

int main(int argc, char **argv) {
    int arch;
    FILE *fin, *fout;
    LPCNetEncState *net;
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
    net = lpcnet_encoder_create();

    while (1) {
        float features[NB_TOTAL_FEATURES];
        opus_int16 pcm[LPCNET_FRAME_SIZE];
        size_t ret;
        ret = fread(pcm, sizeof(pcm[0]), LPCNET_FRAME_SIZE, fin);
        if (feof(fin) || ret != LPCNET_FRAME_SIZE) break;
        lpcnet_compute_single_frame_features(net, pcm, features, arch);
        fprintf(fout, "%f\n", 62.5 * pow(2.f, (features[18]+1.5)));
    }
    lpcnet_encoder_destroy(net);
    fclose(fin);
    fclose(fout);
    return 0;
}
