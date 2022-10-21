#include <stdlib.h>
#include <stdio.h>

#include "dred_rdovae_enc.h"
#include "dred_rdovae_constants.h"

void usage()
{
    printf("dred_rdovae_enc_demo <features> <latents path> <states path>\n");
    exit(1);
}

int main(int argc, char **argv)
{
    RDOVAEEnc enc_state;
    float feature_buffer[36];
    float dframe[2 * DRED_NUM_FEATURES];
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
    if (states_fid == NULL)
    {
        fprintf(stderr, "could not open states file %s\n", argv[3]);
        usage();
    }


    while (fread(feature_buffer, sizeof(float), 36, fid) == 36)
    {
        memcpy(&dframe[DRED_NUM_FEATURES * index++], feature_buffer, DRED_NUM_FEATURES*sizeof(float));

        if (index == 2)
        {
            dred_rdovae_encode_dframe(&enc_state, latents, initial_state, dframe);
            index = 0;
            fwrite(latents, sizeof(float), DRED_LATENT_DIM, latents_fid);
            fwrite(initial_state, sizeof(float), GDENSE2_OUT_SIZE, states_fid);
        }
    }

    fclose(fid);
    fclose(states_fid);
    fclose(latents_fid);

    return 0;
}

/* gcc -DDISABLE_DOT_PROD -DDISABLE_NEON dred_rdovae_enc_demo.c dred_rdovae_enc.c nnet.c dred_rdovae_enc_data.c dred_rdovae_stats_data.c kiss99.c -g -o dred_rdovae_enc_demo */