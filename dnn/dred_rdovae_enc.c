#include <math.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include "dred_rdovae_enc.h"


//#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#endif

void dred_rdovae_encode_dframe(
    RDOVAEEnc *enc_state,           /* io: encoder state */
    float *latents,                 /* o: latent vector */
    float *initial_state,           /* o: initial state */
    const float *input              /* i: double feature frame (concatenated) */
    )
{
    float buffer[ENC_DENSE1_OUT_SIZE + ENC_DENSE2_OUT_SIZE + ENC_DENSE3_OUT_SIZE + ENC_DENSE4_OUT_SIZE + ENC_DENSE5_OUT_SIZE + ENC_DENSE6_OUT_SIZE + ENC_DENSE7_OUT_SIZE + ENC_DENSE8_OUT_SIZE + GDENSE1_OUT_SIZE];
    int output_index = 0;
    int input_index = 0;
#ifdef DEBUG
    static FILE *fids[8] = {NULL};
    static FILE *fpre = NULL;
    int i;
    char filename[256];

    for (i=0; i < 8; i ++)
    {
        if (fids[i] == NULL)
        {
            sprintf(filename, "x%d.f32", i + 1);
            fids[i] = fopen(filename, "wb");
        }
    }
    if (fpre == NULL)
    {
        fpre = fopen("x_pre.f32", "wb");
    }
#endif


    /* run encoder stack and concatenate output in buffer*/
    _lpcnet_compute_dense(&enc_dense1, &buffer[output_index], input);
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), ENC_DENSE1_OUT_SIZE, fids[0]);
#endif
    input_index = output_index;
    output_index += ENC_DENSE1_OUT_SIZE;

    compute_gru2(&enc_dense2, enc_state->dense2_state, &buffer[input_index]);
    memcpy(&buffer[output_index], enc_state->dense2_state, ENC_DENSE2_OUT_SIZE * sizeof(float));
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), ENC_DENSE2_OUT_SIZE, fids[1]);
#endif
    input_index = output_index;
    output_index += ENC_DENSE2_OUT_SIZE;

    _lpcnet_compute_dense(&enc_dense3, &buffer[output_index], &buffer[input_index]);
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), ENC_DENSE3_OUT_SIZE, fids[2]);
#endif
    input_index = output_index;
    output_index += ENC_DENSE3_OUT_SIZE;

    compute_gru2(&enc_dense4, enc_state->dense4_state, &buffer[input_index]);
    memcpy(&buffer[output_index], enc_state->dense4_state, ENC_DENSE4_OUT_SIZE * sizeof(float));
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), ENC_DENSE4_OUT_SIZE, fids[3]);
#endif
    input_index = output_index;
    output_index += ENC_DENSE4_OUT_SIZE;

    _lpcnet_compute_dense(&enc_dense5, &buffer[output_index], &buffer[input_index]);
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), ENC_DENSE5_OUT_SIZE, fids[4]);
#endif
    input_index = output_index;
    output_index += ENC_DENSE5_OUT_SIZE;

    compute_gru2(&enc_dense6, enc_state->dense6_state, &buffer[input_index]);
    memcpy(&buffer[output_index], enc_state->dense6_state, ENC_DENSE6_OUT_SIZE * sizeof(float));
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), ENC_DENSE6_OUT_SIZE, fids[5]);
#endif
    input_index = output_index;
    output_index += ENC_DENSE6_OUT_SIZE;

    _lpcnet_compute_dense(&enc_dense7, &buffer[output_index], &buffer[input_index]);
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), ENC_DENSE7_OUT_SIZE, fids[6]);
#endif
    input_index = output_index;
    output_index += ENC_DENSE7_OUT_SIZE;

    _lpcnet_compute_dense(&enc_dense8, &buffer[output_index], &buffer[input_index]);
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), ENC_DENSE8_OUT_SIZE, fids[7]);
#endif
    output_index += ENC_DENSE8_OUT_SIZE;

    /* compute latents from concatenated input buffer */
#ifdef DEBUG
    fwrite(buffer, sizeof(buffer[0]), bits_dense.nb_inputs, fpre);
#endif
    compute_conv1d(&bits_dense, latents, enc_state->bits_dense_state, buffer);


    /* next, calculate initial state */
    _lpcnet_compute_dense(&gdense1, &buffer[output_index], buffer);
    input_index = output_index;
    _lpcnet_compute_dense(&gdense2, initial_state, &buffer[input_index]);

}
