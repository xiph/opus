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

#include <string.h>

#if 1
#include <stdio.h>
#include <math.h>
#endif

#include "dred_encoder.h"
#include "dred_coding.h"
#include "celt/entenc.h"

#include "dred_decoder.h"

void init_dred_encoder(DREDEnc* enc)
{
    memset(enc, 0, sizeof(*enc));
    lpcnet_encoder_init(&enc->lpcnet_enc_state);
    DRED_rdovae_init_encoder(&enc->rdovae_enc);
}

void dred_process_silk_frame(DREDEnc *enc, const opus_int16 *silk_frame)
{
    float feature_buffer[2 * 36];

    float input_buffer[2*DRED_NUM_FEATURES] = {0};
    /* delay signal by 79 samples */
    memmove(enc->input_buffer, enc->input_buffer + DRED_DFRAME_SIZE, DRED_SILK_ENCODER_DELAY * sizeof(*enc->input_buffer));
    memcpy(enc->input_buffer + DRED_SILK_ENCODER_DELAY, silk_frame, DRED_DFRAME_SIZE * sizeof(*silk_frame));

    /* shift latents buffer */
    memmove(enc->latents_buffer + DRED_LATENT_DIM, enc->latents_buffer, (DRED_MAX_FRAMES - 1) * DRED_LATENT_DIM * sizeof(*enc->latents_buffer));

    /* calculate LPCNet features */
    lpcnet_compute_single_frame_features(&enc->lpcnet_enc_state, enc->input_buffer, feature_buffer);
    lpcnet_compute_single_frame_features(&enc->lpcnet_enc_state, enc->input_buffer + DRED_FRAME_SIZE, feature_buffer + 36);

    /* prepare input buffer (discard LPC coefficients) */
    memcpy(input_buffer, feature_buffer, DRED_NUM_FEATURES * sizeof(input_buffer[0]));
    memcpy(input_buffer + DRED_NUM_FEATURES, feature_buffer + 36, DRED_NUM_FEATURES * sizeof(input_buffer[0]));

    /* run RDOVAE encoder */
    DRED_rdovae_encode_dframe(&enc->rdovae_enc, enc->latents_buffer, enc->state_buffer, input_buffer);
    enc->latents_buffer_fill = IMIN(enc->latents_buffer_fill+1, DRED_NUM_REDUNDANCY_FRAMES);
}

int dred_encode_silk_frame(DREDEnc *enc, unsigned char *buf, int max_chunks, int max_bytes) {
    const opus_uint16 *dead_zone       = DRED_rdovae_get_dead_zone_pointer();
    const opus_uint16 *p0              = DRED_rdovae_get_p0_pointer();
    const opus_uint16 *quant_scales    = DRED_rdovae_get_quant_scales_pointer();
    const opus_uint16 *r               = DRED_rdovae_get_r_pointer();
    ec_enc ec_encoder;

    int q_level;
    int i;
    int offset;
    int ec_buffer_fill;

    /* entropy coding of state and latents */
    ec_enc_init(&ec_encoder, buf, max_bytes);
    dred_encode_state(&ec_encoder, enc->state_buffer);

    for (i = 0; i < IMIN(2*max_chunks, enc->latents_buffer_fill-1); i += 2)
    {
        ec_enc ec_bak;
        ec_bak = ec_encoder;

        q_level = (int) floor(0.5f + DRED_ENC_Q0 + 1.f * (DRED_ENC_Q1 - DRED_ENC_Q0) * i / (DRED_NUM_REDUNDANCY_FRAMES - 2));
        offset = q_level * DRED_LATENT_DIM;

        dred_encode_latents(
            &ec_encoder,
            enc->latents_buffer + i * DRED_LATENT_DIM,
            quant_scales + offset,
            dead_zone + offset,
            r + offset,
            p0 + offset
        );
        if (ec_tell(&ec_encoder) > 8*max_bytes) {
          ec_encoder = ec_bak;
          break;
        }
    }

    ec_buffer_fill = (ec_tell(&ec_encoder)+7)/8;
    ec_enc_shrink(&ec_encoder, ec_buffer_fill);
    ec_enc_done(&ec_encoder);
    return ec_buffer_fill;
}
