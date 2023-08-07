#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "arch.h"
#include "neural_pitch.h"
#include "freq.h"
#include "os_support.h"

void usage(void) {
    fprintf(stderr, "usage: nnpitch_demo <input.f32> <output.f32>\n");
}

int main(int argc, char **argv) {
    FILE *fin, *fout;
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
    
    // State struct for the neural pitch model
    neural_pitch_model net;
    npm_init(&net);

    while (1) {
            float input[NEURAL_PITCH_FRAME_SIZE] = {0.0};
            float output[PITCH_NET_OUTPUT] = {0.0};
            opus_int16 pcm[NEURAL_PITCH_FRAME_SIZE];
            size_t ret;
            ret = fread(pcm, sizeof(pcm[0]), NEURAL_PITCH_FRAME_SIZE, fin);
            for(int i =0;i<NEURAL_PITCH_FRAME_SIZE;i++){
                input[i] = pcm[i]/(32767.0);
            }
            if (feof(fin) || ret != NEURAL_PITCH_FRAME_SIZE) break;

            pitch_model(&net,output,input);
            short cent_pred[1];
            cent_pred[0] = 20*argmax(output);
            fwrite(cent_pred, sizeof(cent_pred[0]), 1, fout);
}
fclose(fin);
fclose(fout);
return 0;
}
