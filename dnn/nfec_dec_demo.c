#include <stdlib.h>
#include <stdio.h>

#include "nfec_dec.h"
#include "nfec_enc.h"


void usage()
{
    printf("nfec_dec_demo <input> <output>\n");
    exit(1);
}

int main(int argc, char **argv)
{
    NFECDecState dec_state;
    float feature_buffer[36];
    float qframe[4 * NFEC_DEC_NUM_FEATURES];
    float latents[80];
    float initial_state[24];
    int quantized_latents[80];
    int index = 0;
    FILE *in_fid, *out_fid;
    int qlevel = 0;

    memset(&dec_state, 0, sizeof(dec_state));

    if (argc < 3) usage();

    in_fid = fopen(argv[1], "rb");
    if (in_fid == NULL)
    {
        perror("Could not open input file");
        usage();
    }

    out_fid = fopen(argv[2], "wb");
    if (out_fid == NULL)
    {
        perror("Could not open output file");
        usage();
    }

    /* read initial state from input stream */
    if (fread(initial_state, sizeof(float), 24, in_fid) != 24)
    {
        perror("error while reading initial state");
        return 1;
    }

    /* initialize GRU states */
    nfec_dec_init_states(&dec_state, initial_state);

    /* start decoding */
    while (fread(latents, sizeof(float), 80, in_fid) == 80)
    {
        nfec_decode_qframe(&dec_state, qframe, latents);
        fwrite(qframe, sizeof(float), 4*20, out_fid);
    }

    fclose(in_fid);
    fclose(out_fid);


    return 0;
}

/* gcc -DDISABLE_DOT_PROD -DDISABLE_NEON nfec_dec_demo.c nfec_dec.c nnet.c nfec_dec_data.c nfec_stats_data.c kiss99.c -g -o nfec_dec_demo */