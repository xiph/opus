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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include "dred_rdovae_dec.h"
#include "dred_rdovae_constants.h"
//#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#endif

void dred_rdovae_dec_init_states(
    RDOVAEDec *h,            /* io: state buffer handle */
    const float *initial_state  /* i: initial state */
    )
{
    /* initialize GRU states from initial state */
    _lpcnet_compute_dense(&state1, h->dense2_state, initial_state);
    _lpcnet_compute_dense(&state2, h->dense4_state, initial_state);
    _lpcnet_compute_dense(&state3, h->dense6_state, initial_state);
}


void dred_rdovae_decode_qframe(
    RDOVAEDec *dec_state,       /* io: state buffer handle */
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
    _lpcnet_compute_dense(&dec_dense1, &buffer[output_index], input);
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

    _lpcnet_compute_dense(&dec_dense3, &buffer[output_index], &buffer[input_index]);
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

    _lpcnet_compute_dense(&dec_dense5, &buffer[output_index], &buffer[input_index]);
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

    _lpcnet_compute_dense(&dec_dense7, &buffer[output_index], &buffer[input_index]);
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), DEC_DENSE7_OUT_SIZE, fids[6]);
#endif
    input_index = output_index;
    output_index += DEC_DENSE7_OUT_SIZE;

    _lpcnet_compute_dense(&dec_dense8, &buffer[output_index], &buffer[input_index]);
#ifdef DEBUG
    fwrite(&buffer[output_index], sizeof(buffer[0]), DEC_DENSE8_OUT_SIZE, fids[7]);
#endif
    output_index += DEC_DENSE8_OUT_SIZE;

    _lpcnet_compute_dense(&dec_final, qframe, buffer);

    /* restore correct order of frames */
    memmove(buffer, qframe, 4 * DRED_NUM_FEATURES * sizeof(*qframe));
    memmove(qframe + 0 * DRED_NUM_FEATURES, buffer + 3 * DRED_NUM_FEATURES, DRED_NUM_FEATURES * sizeof(*qframe));
    memmove(qframe + 1 * DRED_NUM_FEATURES, buffer + 2 * DRED_NUM_FEATURES, DRED_NUM_FEATURES * sizeof(*qframe));
    memmove(qframe + 2 * DRED_NUM_FEATURES, buffer + 1 * DRED_NUM_FEATURES, DRED_NUM_FEATURES * sizeof(*qframe));
    memmove(qframe + 3 * DRED_NUM_FEATURES, buffer + 0 * DRED_NUM_FEATURES, DRED_NUM_FEATURES * sizeof(*qframe));
}