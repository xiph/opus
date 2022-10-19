#include "nfec_enc.h"
#include "nnet.h"
#include "nfec_enc_data.h"

void nfec_encode_dframe(struct NFECEncState *enc_state, float *latents, float *initial_state, const float *input)
{
    float buffer[ENC_DENSE1_OUT_SIZE + ENC_DENSE2_OUT_SIZE + ENC_DENSE3_OUT_SIZE + ENC_DENSE4_OUT_SIZE + ENC_DENSE5_OUT_SIZE + ENC_DENSE6_OUT_SIZE + ENC_DENSE7_OUT_SIZE + ENC_DENSE8_OUT_SIZE + GDENSE1_OUT_SIZE];
    int output_index = 0;
    int input_index = 0;

    /* run encoder stack and concatenate output in buffer*/
    compute_dense(&enc_dense1, &buffer[output_index], input);
    input_index = output_index;
    output_index += ENC_DENSE1_OUT_SIZE;

    compute_gru3(&enc_dense2, enc_state->dense2_state, &buffer[input_index]);
    memcpy(&buffer[output_index], enc_state->dense2_state, ENC_DENSE2_OUT_SIZE * sizeof(float));
    input_index = output_index;
    output_index += ENC_DENSE2_OUT_SIZE;

    compute_dense(&enc_dense3, &buffer[output_index], &buffer[input_index]);
    input_index = output_index;
    output_index += ENC_DENSE3_OUT_SIZE;

    compute_gru3(&enc_dense4, enc_state->dense4_state, &buffer[input_index]);
    memcpy(&buffer[output_index], enc_state->dense4_state, ENC_DENSE4_OUT_SIZE * sizeof(float));
    input_index = output_index;
    output_index += ENC_DENSE4_OUT_SIZE;

    compute_dense(&enc_dense5, &buffer[output_index], &buffer[input_index]);
    input_index = output_index;
    output_index += ENC_DENSE5_OUT_SIZE;

    compute_gru3(&enc_dense6, enc_state->dense6_state, &buffer[input_index]);
    memcpy(&buffer[output_index], enc_state->dense6_state, ENC_DENSE6_OUT_SIZE * sizeof(float));
    input_index = output_index;
    output_index += ENC_DENSE6_OUT_SIZE;

    compute_dense(&enc_dense7, &buffer[output_index], &buffer[input_index]);
    input_index = output_index;
    output_index += ENC_DENSE7_OUT_SIZE;

    compute_dense(&enc_dense8, &buffer[output_index], &buffer[input_index]);
    output_index += ENC_DENSE8_OUT_SIZE;

    /* compute latents from concatenated input buffer */
    compute_conv1d(&bits_dense, latents, enc_state->bits_dense_state, buffer);

    /* next, calculate initial state */
    compute_dense(&gdense1, &buffer[output_index], buffer);
    input_index = output_index;
    compute_dense(&gdense2, initial_state, &buffer[input_index]);

}