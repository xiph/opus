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
