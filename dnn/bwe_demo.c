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
#include "osce_structs.h"
#include "osce.h"
#include "silk/structs.h"


void usage(void) {
    fprintf(stderr, "usage: bwe_demo <input.pcm> <output.pcm>\n");
    exit(1);
}

#define BWE_FRAME_SIZE 320

int main(int argc, char **argv) {
    int arch;
    FILE *fin, *fout;
    silk_OSCE_BWE_struct *hOSCEBWE;
    OSCEModel *osce;


    arch = opus_select_arch();
    hOSCEBWE = (silk_OSCE_BWE_struct *)calloc(1, sizeof(*hOSCEBWE));
    osce = (OSCEModel *)calloc(1, sizeof(*osce));
    osce_load_models(osce, NULL, arch);
    osce_bwe_reset(hOSCEBWE);

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

    int16_t x_in[BWE_FRAME_SIZE];
    int16_t x_out[3 * BWE_FRAME_SIZE];


    while(fread(x_in, sizeof(x_in[0]), BWE_FRAME_SIZE, fin) == BWE_FRAME_SIZE) {
        osce_bwe(
            osce,
            hOSCEBWE,
            x_out,
            x_in,
            BWE_FRAME_SIZE,
            arch
        );

        fwrite(x_out, sizeof(x_out[0]), 3 * BWE_FRAME_SIZE, fout);
    }

    free(hOSCEBWE);
    free(osce);

    fclose(fin);
    fclose(fout);

    return 0;
}
