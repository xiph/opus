#include "nfec_dec.h"

//#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#endif

void nfec_dec_init_states(
    NFECDecState *h,            /* io: state buffer handle */
    const float *initial_state  /* i: initial state */
    )
{
    /* initialize GRU states from initial state */
    compute_dense(&state1, h->dense2_state, initial_state);
    compute_dense(&state2, h->dense4_state, initial_state);
    compute_dense(&state3, h->dense6_state, initial_state);
}

void nfec_dec_unquantize_latent_vector(
    float *z,       /* o: unquantized latent vector */
    const int *zq,  /* i: quantized latent vector */
    int quant_level /* i: quantization level */
    )
{
    int i;
    /* inverse scaling and type conversion */
    for (i = 0; i < NFEC_STATS_NUM_LATENTS; i ++)
    {
        z[i] = (float) zq[i] / nfec_stats_quant_scales[quant_level * NFEC_STATS_NUM_LATENTS + i];
    }
}

void nfec_decode_qframe(
    NFECDecState *dec_state,    /* io: state buffer handle */
    float *qframe,              /* o: quadruple feature frame (four concatenated frames) */
    const float *input          /* i: latent vector */
    )
{
    float buffer[DEC_DENSE1_OUT_SIZE + DEC_DENSE2_OUT_SIZE + DEC_DENSE3_OUT_SIZE + DEC_DENSE4_OUT_SIZE + DEC_DENSE5_OUT_SIZE + DEC_DENSE6_OUT_SIZE + DEC_DENSE7_OUT_SIZE + DEC_DENSE8_OUT_SIZE];
    int output_index = 0;
    int input_index = 0;
#ifdef DEBUG
    static FILE *fids[8] = {NULL};
    int i;
    char filename[256];

    for (i=0; i < 8; i ++)
    {
        if (fids[i] == NULL)
        {
            sprintf(filename, "y%d.f32", i + 1);
            fids[i] = fopen(filename, "wb");
        }
    }
#endif

    /* run encoder stack and concatenate output in buffer*/
    compute_dense(&dec_dense1, &buffer[output_index], input);
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), DEC_DENSE1_OUT_SIZE, fids[0]);
#endif
    input_index = output_index;
    output_index += DEC_DENSE1_OUT_SIZE;

    compute_gru2(&dec_dense2, dec_state->dense2_state, &buffer[input_index]);
    memcpy(&buffer[output_index], dec_state->dense2_state, DEC_DENSE2_OUT_SIZE * sizeof(float));
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), DEC_DENSE2_OUT_SIZE, fids[1]);
#endif
    input_index = output_index;
    output_index += DEC_DENSE2_OUT_SIZE;

    compute_dense(&dec_dense3, &buffer[output_index], &buffer[input_index]);
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), DEC_DENSE3_OUT_SIZE, fids[2]);
#endif
    input_index = output_index;
    output_index += DEC_DENSE3_OUT_SIZE;

    compute_gru2(&dec_dense4, dec_state->dense4_state, &buffer[input_index]);
    memcpy(&buffer[output_index], dec_state->dense4_state, DEC_DENSE4_OUT_SIZE * sizeof(float));
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), DEC_DENSE4_OUT_SIZE, fids[3]);
#endif
    input_index = output_index;
    output_index += DEC_DENSE4_OUT_SIZE;

    compute_dense(&dec_dense5, &buffer[output_index], &buffer[input_index]);
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), DEC_DENSE5_OUT_SIZE, fids[4]);
#endif
    input_index = output_index;
    output_index += DEC_DENSE5_OUT_SIZE;

    compute_gru2(&dec_dense6, dec_state->dense6_state, &buffer[input_index]);
    memcpy(&buffer[output_index], dec_state->dense6_state, DEC_DENSE6_OUT_SIZE * sizeof(float));
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), DEC_DENSE6_OUT_SIZE, fids[5]);
#endif
    input_index = output_index;
    output_index += DEC_DENSE6_OUT_SIZE;

    compute_dense(&dec_dense7, &buffer[output_index], &buffer[input_index]);
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), DEC_DENSE7_OUT_SIZE, fids[6]);
#endif
    input_index = output_index;
    output_index += DEC_DENSE7_OUT_SIZE;

    compute_dense(&dec_dense8, &buffer[output_index], &buffer[input_index]);
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), DEC_DENSE8_OUT_SIZE, fids[7]);
#endif
    output_index += DEC_DENSE8_OUT_SIZE;

    compute_dense(&dec_final, qframe, buffer);
}