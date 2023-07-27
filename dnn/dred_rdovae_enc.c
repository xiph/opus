/* Copyright (c) 2022 Amazon
   Written by Jan Buethe */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <math.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include "dred_rdovae_enc.h"
#include "os_support.h"

void dred_rdovae_encode_dframe(
    RDOVAEEncState *enc_state,           /* io: encoder state */
    const RDOVAEEnc *model,
    float *latents,                 /* o: latent vector */
    float *initial_state,           /* o: initial state */
    const float *input              /* i: double feature frame (concatenated) */
    )
{
    float buffer[ENC_DENSE1_OUT_SIZE + ENC_DENSE2_OUT_SIZE + ENC_DENSE3_OUT_SIZE + ENC_DENSE4_OUT_SIZE + ENC_DENSE5_OUT_SIZE + ENC_DENSE6_OUT_SIZE + ENC_DENSE7_OUT_SIZE + ENC_DENSE8_OUT_SIZE + GDENSE1_OUT_SIZE];
    int output_index = 0;
    int input_index = 0;

    /* run encoder stack and concatenate output in buffer*/
    compute_generic_dense(&model->enc_dense1, &buffer[output_index], input, ACTIVATION_TANH);
    input_index = output_index;
    output_index += ENC_DENSE1_OUT_SIZE;

    compute_generic_gru(&model->enc_dense2_input, &model->enc_dense2_recurrent, enc_state->dense2_state, &buffer[input_index]);
    OPUS_COPY(&buffer[output_index], enc_state->dense2_state, ENC_DENSE2_OUT_SIZE);
    input_index = output_index;
    output_index += ENC_DENSE2_OUT_SIZE;

    compute_generic_dense(&model->enc_dense3, &buffer[output_index], &buffer[input_index], ACTIVATION_TANH);
    input_index = output_index;
    output_index += ENC_DENSE3_OUT_SIZE;

    compute_generic_gru(&model->enc_dense4_input, &model->enc_dense4_recurrent, enc_state->dense4_state, &buffer[input_index]);
    OPUS_COPY(&buffer[output_index], enc_state->dense4_state, ENC_DENSE4_OUT_SIZE);
    input_index = output_index;
    output_index += ENC_DENSE4_OUT_SIZE;

    compute_generic_dense(&model->enc_dense5, &buffer[output_index], &buffer[input_index], ACTIVATION_TANH);
    input_index = output_index;
    output_index += ENC_DENSE5_OUT_SIZE;

    compute_generic_gru(&model->enc_dense6_input, &model->enc_dense6_recurrent, enc_state->dense6_state, &buffer[input_index]);
    OPUS_COPY(&buffer[output_index], enc_state->dense6_state, ENC_DENSE6_OUT_SIZE);
    input_index = output_index;
    output_index += ENC_DENSE6_OUT_SIZE;

    compute_generic_dense(&model->enc_dense7, &buffer[output_index], &buffer[input_index], ACTIVATION_TANH);
    input_index = output_index;
    output_index += ENC_DENSE7_OUT_SIZE;

    compute_generic_dense(&model->enc_dense8, &buffer[output_index], &buffer[input_index], ACTIVATION_TANH);
    output_index += ENC_DENSE8_OUT_SIZE;

    /* compute latents from concatenated input buffer */
    compute_generic_conv1d(&model->bits_dense, latents, enc_state->bits_dense_state, buffer, BITS_DENSE_IN_SIZE, ACTIVATION_LINEAR);


    /* next, calculate initial state */
    compute_generic_dense(&model->gdense1, &buffer[output_index], buffer, ACTIVATION_TANH);
    input_index = output_index;
    compute_generic_dense(&model->gdense2, initial_state, &buffer[input_index], ACTIVATION_TANH);

}
