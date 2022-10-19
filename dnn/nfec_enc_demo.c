#include <stdlib.h>
#include <stdio.h>

#include "nfec_enc.h"

void usage()
{
    printf("nfec_enc_demo <features> <latents path> <states path>\n");
    exit(1);
}

int main(int argc, char **argv)
{
    struct NFECEncState enc_state;
    float feature_buffer[32];
    float dframe[2 * 20];
    float latents[80];
    float initial_state[24];
    int index = 0;
    FILE *fid, *latents_fid, *states_fid;

    memset(&enc_state, 0, sizeof(enc_state));

    if (argc < 4)
    {
        usage();
    }

    fid = fopen(argv[1], "rb");
    if (fid == NULL)
    {
        fprintf(stderr, "could not open feature file %s\n", argv[1]);
        usage();
    }

    latents_fid = fopen(argv[2], "wb");
    if (latents_fid == NULL)
    {
        fprintf(stderr, "could not open latents file %s\n", argv[2]);
        usage();
    }

    states_fid = fopen(argv[3], "wb");
    if (fid == NULL)
    {
        fprintf(stderr, "could not open states file %s\n", argv[3]);
        usage();
    }


    while (fread(feature_buffer, sizeof(float), 32, fid) == 32)
    {
        memcpy(&dframe[16 * index++], feature_buffer, 16*sizeof(float));

        if (index == 2)
        {
            nfec_encode_dframe(&enc_state, latents, initial_state, dframe);
            index = 0;
            fwrite(latents, sizeof(float), NFEC_LATENT_DIM, latents_fid);
            fwrite(initial_state, sizeof(float), GDENSE2_OUT_SIZE, states_fid);
        }
    }

    fclose(fid);
    fclose(states_fid);
    fclose(latents_fid);
}

/* gcc -DDISABLE_DOT_PROD -DDISABLE_NEON nfec_enc_demo.c nfec_enc.c nnet.c nfec_enc_data.c kiss99.c -o nfec_enc_demo */