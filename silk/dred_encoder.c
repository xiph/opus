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

#include <string.h>

#include <stdio.h>
#include <math.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "dred_encoder.h"
#include "dred_coding.h"
#include "celt/entenc.h"

void init_dred_encoder(DREDEnc* enc)
{
    memset(enc, 0, sizeof(*enc));
    enc->lpcnet_enc_state = lpcnet_encoder_create();
    enc->rdovae_enc = DRED_rdovae_create_encoder();
}

void dred_encode_silk_frame(DREDEnc *enc, const opus_int16 *silk_frame)
{
    const opus_uint16 *dead_zone       = DRED_rdovae_get_dead_zone_pointer();
    const opus_uint16 *p0              = DRED_rdovae_get_p0_pointer();
    const opus_uint16 *quant_scales    = DRED_rdovae_get_quant_scales_pointer();
    const opus_uint16 *r               = DRED_rdovae_get_r_pointer();
    
    float input_buffer[2*DRED_NUM_FEATURES] = {0};

    int bytes;
    int q_level;
    int i;
    int offset;

    /* delay signal by 79 samples */
    memmove(enc->input_buffer, enc->input_buffer + DRED_DFRAME_SIZE, DRED_SILK_ENCODER_DELAY * sizeof(*enc->input_buffer));
    memcpy(enc->input_buffer + DRED_SILK_ENCODER_DELAY, silk_frame, DRED_DFRAME_SIZE * sizeof(*silk_frame));

    /* shift latents buffer */
    memmove(enc->latents_buffer + DRED_LATENT_DIM, enc->latents_buffer, (DRED_MAX_FRAMES - 1) * DRED_LATENT_DIM * sizeof(*enc->latents_buffer));

    /* calculate LPCNet features */
    lpcnet_compute_single_frame_features(enc->lpcnet_enc_state, enc->input_buffer, enc->feature_buffer);
    lpcnet_compute_single_frame_features(enc->lpcnet_enc_state, enc->input_buffer + DRED_FRAME_SIZE, enc->feature_buffer + 36);

    /* prepare input buffer (discard LPC coefficients) */
    memcpy(input_buffer, enc->feature_buffer, DRED_NUM_FEATURES * sizeof(input_buffer[0]));
    memcpy(input_buffer + DRED_NUM_FEATURES, enc->feature_buffer + 36, DRED_NUM_FEATURES * sizeof(input_buffer[0]));

    /* run RDOVAE encoder */
    DRED_rdovae_encode_dframe(enc->rdovae_enc, enc->latents_buffer, enc->state_buffer, input_buffer);

    /* entropy coding of state and latents */
    ec_enc_init(&enc->ec_encoder, enc->ec_buffer, DRED_MAX_DATA_SIZE);
    dred_encode_state(&enc->ec_encoder, enc->state_buffer);   

    for (i = 0; i < DRED_NUM_REDUNDANCY_FRAMES; i += 2)
    {
        q_level = (int) round(DRED_ENC_Q0 + 1.f * (DRED_ENC_Q1 - DRED_ENC_Q0) * i / (DRED_NUM_REDUNDANCY_FRAMES - 2));
        offset = q_level * DRED_LATENT_DIM;

        dred_encode_latents(
            &enc->ec_encoder,
            enc->latents_buffer + i * DRED_LATENT_DIM,
            quant_scales + offset,
            dead_zone + offset,
            r + offset,
            p0 + offset
        );
    }

    bytes = (ec_tell(&enc->ec_encoder)+7)/8;
    ec_enc_shrink(&enc->ec_encoder, bytes);
    ec_enc_done(&enc->ec_encoder);

#if 1
    printf("packet size: %d\n", bytes*8);

    static FILE *fbs = NULL;
    if (fbs == NULL)
    {
        fbs = fopen("dred_bitstream.bin", "wb");
    }
    fwrite(&bytes, sizeof(bytes), 1, fbs);
    fwrite(ec_get_buffer(&enc->ec_encoder), 1, bytes, fbs);
#endif


#if 0
    /* trial decoding */
    float state[24];
    float features[4 * 20];
    float latents[80];
    float zeros[36 - 20] = {0};
    static FILE *fid;
    RDOVAEDec *rdovae_dec = DRED_rdovae_create_decoder();

    if (fid == NULL)
    {
        fid = fopen("features_last.f32", "wb");
    }

    /* decode state */
    ec_enc ec_dec;
    ec_dec_init(&ec_dec, ec_get_buffer(&enc->ec_encoder), bytes);
    dred_decode_state(&ec_dec, state);

    q_level = DRED_ENC_Q0;
    offset = q_level * DRED_LATENT_DIM;
    dred_decode_latents(
        &ec_dec,
        latents,
        quant_scales + offset,
        r + offset,
        p0 + offset
        );

    DRED_rdovae_dec_init_states(rdovae_dec, state);

    DRED_rdovae_decode_qframe(rdovae_dec, features, latents);

    DRED_rdovae_destroy_decoder(rdovae_dec);

    fwrite(features + 40, sizeof(float), 20, fid);
    fwrite(zeros, sizeof(float), 16, fid);
    fwrite(features + 60, sizeof(float), 20, fid);
    fwrite(zeros, sizeof(float), 16, fid);

#endif


}