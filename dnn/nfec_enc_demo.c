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
    float feature_buffer[36];
    float dframe[2 * NFEC_NUM_FEATURES];
    float latents[80];
    float initial_state[24];
    int quantized_latents[NFEC_LATENT_DIM];
    int index = 0;
    FILE *fid, *latents_fid, *quantized_latents_fid, *states_fid;

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

    char filename[256];
    strcpy(filename, argv[2]);
    strcat(filename, ".quantized.f32");
    quantized_latents_fid = fopen(filename, "wb");
    if (latents_fid == NULL)
    {
        fprintf(stderr, "could not open latents file %s\n", filename);
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
        memcpy(&dframe[NFEC_NUM_FEATURES * index++], feature_buffer, NFEC_NUM_FEATURES*sizeof(float));

        if (index == 2)
        {
            nfec_encode_dframe(&enc_state, latents, initial_state, dframe);
            nfec_quantize_latent_vector(quantized_latents, latents, 0);
            index = 0;
            fwrite(latents, sizeof(float), NFEC_LATENT_DIM, latents_fid);
            fwrite(quantized_latents, sizeof(int), NFEC_LATENT_DIM, quantized_latents_fid);
            fwrite(initial_state, sizeof(float), GDENSE2_OUT_SIZE, states_fid);
        }
    }

    fclose(fid);
    fclose(states_fid);
    fclose(latents_fid);
    fclose(quantized_latents_fid);

    return 0;
}

/* gcc -DDISABLE_DOT_PROD -DDISABLE_NEON nfec_enc_demo.c nfec_enc.c nnet.c nfec_enc_data.c nfec_stats_data.c kiss99.c -g -o nfec_enc_demo */