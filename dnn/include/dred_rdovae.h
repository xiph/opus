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

#ifndef DRED_RDOVAE_H
#define DRED_RDOVAE_H

#include <stdlib.h>

#include "opus_types.h"

typedef struct RDOVAEDecStruct RDOVAEDec;
typedef struct RDOVAEEncStruct RDOVAEEnc;

void DRED_rdovae_decode_all(float *features, const float *state, const float *latents, int nb_latents);


size_t DRED_rdovae_get_enc_size(void);

size_t DRED_rdovae_get_dec_size(void);

RDOVAEDec * DRED_rdovae_create_decoder(void);
RDOVAEEnc * DRED_rdovae_create_encoder(void);
void DRED_rdovae_destroy_decoder(RDOVAEDec* h);
void DRED_rdovae_destroy_encoder(RDOVAEEnc* h);


void DRED_rdovae_init_encoder(RDOVAEEnc *enc_state);

void DRED_rdovae_encode_dframe(RDOVAEEnc *enc_state, float *latents, float *initial_state, const float *input);

void DRED_rdovae_dec_init_states(RDOVAEDec *h, const float * initial_state);

void DRED_rdovae_decode_qframe(RDOVAEDec *h, float *qframe, const float * z);

const opus_uint16 * DRED_rdovae_get_p0_pointer(void);
const opus_uint16 * DRED_rdovae_get_dead_zone_pointer(void);
const opus_uint16 * DRED_rdovae_get_r_pointer(void);
const opus_uint16 * DRED_rdovae_get_quant_scales_pointer(void);

#endif
