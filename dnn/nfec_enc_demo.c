#include <stdlib.h>
#include <stdio.h>

#include "nfec_enc.h"

void usage()
{
    printf("nfec_enc_demo <features>");
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
    FILE *fid;

    if (argc < 2)
    {
        usage();
    }

    fid = fopen(argv[1], "rb");
    if (fid == NULL)
    {
        fprintf(stderr, "could not open feature file %s\n", argv[1]);
        usage();
    }

    while (fread(feature_buffer, sizeof(float), 32, fid) == 32)
    {
        memcpy(dframe[16 * index++], feature_buffer, 16*sizeof(float));

        if (index == 2)
        {
            nfec_encode_dframe(&enc_state, latents, initial_state, dframe);
            index = 0;
        }
    }
}

/* gcc -DDISABLE_DOT_PROD nfec_enc_demo.c nfec_enc.c nnet.c nfec_enc_data.c -o nfec_enc_demo */