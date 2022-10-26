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


#include "dred_rdovae.h"
#include "dred_rdovae_enc.h"
#include "dred_rdovae_dec.h"
#include "dred_rdovae_stats_data.h"

size_t DRED_rdovae_get_enc_size()
{
    return sizeof(RDOVAEEnc);
}

size_t DRED_rdovae_get_dec_size()
{
    return sizeof(RDOVAEDec);
}

void DRED_rdovae_init_encoder(RDOVAEEnc *enc_state)
{
    memset(enc_state, 0, sizeof(*enc_state));

}

void DRED_rdovae_init_decoder(RDOVAEDec *dec_state)
{
    memset(dec_state, 0, sizeof(*dec_state));
}


RDOVAEEnc * DRED_rdovae_create_encoder()
{
    RDOVAEEnc *enc;
    enc = (RDOVAEEnc*) calloc(sizeof(*enc), 1);
    DRED_rdovae_init_encoder(enc);
    return enc;
}

RDOVAEDec * DRED_rdovae_create_decoder()
{
    RDOVAEDec *dec;
    dec = (RDOVAEDec*) calloc(sizeof(*dec), 1);
    DRED_rdovae_init_decoder(dec);
    return dec;
}

void DRED_rdovae_destroy_decoder(RDOVAEDec* dec)
{
    free(dec);
}

void DRED_rdovae_destroy_encoder(RDOVAEEnc* enc)
{
    free(enc);
}

void DRED_rdovae_encode_dframe(RDOVAEEnc *enc_state, float *latents, float *initial_state, const float *input)
{
    dred_rdovae_encode_dframe(enc_state, latents, initial_state, input);
}

void DRED_rdovae_dec_init_states(RDOVAEDec *h, const float * initial_state)
{
    dred_rdovae_dec_init_states(h, initial_state);
}

void DRED_rdovae_decode_qframe(RDOVAEDec *h, float *qframe, const float *z)
{
    dred_rdovae_decode_qframe(h, qframe, z);
}


const opus_int16 * DRED_rdovae_get_p0_pointer(void)
{
    return &dred_p0_q15[0];
}

const opus_int16 * DRED_rdovae_get_dead_zone_pointer(void)
{
    return &dred_dead_zone_q10[0];
}

const opus_int16 * DRED_rdovae_get_r_pointer(void)
{
    return &dred_r_q15[0];
}

const opus_int16 * DRED_rdovae_get_quant_scales_pointer(void)
{
    return &dred_quant_scales_q8[0];
}