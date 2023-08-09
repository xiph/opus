//Modified version of lpcnet_demo to extract xcorr features (and pitch for reference)
/*
Do the following modifications to the files"
    1. Makefile.am: 
        (Add lpcnet_xcorr_extractor to noinst_PROGRAMS, and then link its source file)
        noinst_PROGRAMS = lpcnet_demo dump_data lpcnet_xcorr_extractor
        lpcnet_xcorr_extractor_SOURCES = src/lpcnet_xcorr_extractor.c
        lpcnet_xcorr_extractor_LDADD = liblpcnet.la
    2. include/lpcnet.h:
        LPCNET_EXPORT int lpcnet_compute_single_frame_features_dump(LPCNetEncState *st, const short *pcm, float features[NB_TOTAL_FEATURES], FILE *fout);
    3. src/lpcnet_enc.c:
    int lpcnet_compute_single_frame_features_dump(LPCNetEncState *st, const short *pcm, float features[NB_TOTAL_FEATURES], FILE *fout) {
    int i;
    float x[FRAME_SIZE];
    for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
    preemphasis(x, &st->mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(st, x);
    process_single_frame(st, NULL);
    RNN_COPY(features, &st->features[0][0], NB_TOTAL_FEATURES);
    fwrite(st->xc[2+2*st->pcount], sizeof(float),PITCH_MAX_PERIOD, fout);
    return 0;
    }
    4. Build LPCNet as usual
    5. Copy lpcnet_xcorr.sh, change the directories in the script and run it to generate the LPCNet extracted xcorr/f0 dumps
*/
// For input file, it creates and dumps the pitch and xcorr values to a new file with same name as input

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <libgen.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "arch.h"
#include "lpcnet_private.h"
#include "freq.h"

void usage(void) {
    fprintf(stderr, "usage: lpcnet_xcorr_extractor <input.pcm> <output.pcm>\n");
    exit(1);
}

int main(int argc, char **argv) {
    FILE *fin, *fout;
    FILE *fpitch;
    if(argc == 4){
        fpitch = fopen(argv[3], "w");
    if (fpitch == NULL) {
        fprintf(stderr, "Can't open %s\n", argv[3]);
        exit(1);
    }
    }
    fin = fopen(argv[1], "r");
    if (fin == NULL) {
        fprintf(stderr, "Can't open %s\n", argv[1]);
        exit(1);
    }
    fout = fopen(argv[2], "w");
    if (fout == NULL) {
        fprintf(stderr, "Can't open %s\n", argv[1]);
        exit(1);
    }
    rewind(fin);
    rewind(fout);
    
    struct LPCNetEncState *net;
    net = lpcnet_encoder_create();

    struct LPCNetEncState *net_pitch;
    net_pitch = lpcnet_encoder_create();
    // Only write the pitch and residual cross correlation to a file
    while (1) {
        float period;
        float features[NB_TOTAL_FEATURES];
        short pcm[LPCNET_FRAME_SIZE];
        size_t ret;
        ret = fread(pcm, sizeof(pcm[0]), LPCNET_FRAME_SIZE, fin);
        if (feof(fin) || ret != LPCNET_FRAME_SIZE) break;
        lpcnet_compute_single_frame_features_dump(net, pcm, fout);
        if(argc == 4){
        float features_pitch[NB_TOTAL_FEATURES];
        lpcnet_compute_single_frame_features(net_pitch, pcm, features_pitch);
        // Write Pitch period
        period = (.1 + features_pitch[18]*50+100);
        fwrite(&period, sizeof(float), 1, fpitch);
        }
    }
    lpcnet_encoder_destroy(net);
    
    fclose(fin);
    fclose(fout);
    if(argc == 4){
        fclose(fpitch);
        }
    return 0;
}
