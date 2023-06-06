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
#include "float_cast.h"
#include "os_support.h"

int dred_encoder_load_model(DREDEnc* enc, const unsigned char *data, int len)
{
    WeightArray *list;
    int ret;
    parse_weights(&list, data, len);
    ret = init_rdovaeenc(&enc->model, list);
    free(list);
    return (ret == 0) ? OPUS_OK : OPUS_BAD_ARG;
}

void dred_encoder_reset(DREDEnc* enc)
{
    RNN_CLEAR((char*)&enc->DREDENC_RESET_START,
              sizeof(DREDEnc)-
              ((char*)&enc->DREDENC_RESET_START - (char*)enc));
    enc->input_buffer_fill = DRED_SILK_ENCODER_DELAY;
    lpcnet_encoder_init(&enc->lpcnet_enc_state);
    DRED_rdovae_init_encoder(&enc->rdovae_enc);
}

void dred_encoder_init(DREDEnc* enc, opus_int32 Fs, int channels)
{
    enc->Fs = Fs;
    enc->channels = channels;
#ifndef USE_WEIGHTS_FILE
    init_rdovaeenc(&enc->model, rdovae_enc_arrays);
#endif
    dred_encoder_reset(enc);
}

static void dred_process_frame(DREDEnc *enc)
{
    float feature_buffer[2 * 36];
    float input_buffer[2*DRED_NUM_FEATURES] = {0};

    /* shift latents buffer */
    OPUS_MOVE(enc->latents_buffer + DRED_LATENT_DIM, enc->latents_buffer, (DRED_MAX_FRAMES - 1) * DRED_LATENT_DIM);

    /* calculate LPCNet features */
    lpcnet_compute_single_frame_features_float(&enc->lpcnet_enc_state, enc->input_buffer, feature_buffer);
    lpcnet_compute_single_frame_features_float(&enc->lpcnet_enc_state, enc->input_buffer + DRED_FRAME_SIZE, feature_buffer + 36);

    /* prepare input buffer (discard LPC coefficients) */
    OPUS_COPY(input_buffer, feature_buffer, DRED_NUM_FEATURES);
    OPUS_COPY(input_buffer + DRED_NUM_FEATURES, feature_buffer + 36, DRED_NUM_FEATURES);

    /* run RDOVAE encoder */
    DRED_rdovae_encode_dframe(&enc->rdovae_enc, &enc->model, enc->latents_buffer, enc->state_buffer, input_buffer);
    enc->latents_buffer_fill = IMIN(enc->latents_buffer_fill+1, DRED_NUM_REDUNDANCY_FRAMES);
}

void filter_df2t(const float *in, float *out, int len, float b0, const float *b, const float *a, int order, float *mem)
{
    int i;
    for (i=0;i<len;i++) {
        int j;
        float xi, yi, nyi;
        xi = in[i];
        yi = xi*b0 + mem[0];
        nyi = -yi;
        for (j=0;j<order;j++)
        {
           mem[j] = mem[j+1] + b[j]*xi + a[j]*nyi;
        }
        out[i] = yi;
        /*fprintf(stdout, "%f\n", out[i]);*/
    }
}

#define MAX_DOWNMIX_BUFFER (960*2)
static void dred_convert_to_16k(DREDEnc *enc, const float *in, int in_len, float *out, int out_len)
{
    float downmix[MAX_DOWNMIX_BUFFER];
    int i;
    int up;
    celt_assert(enc->channels*in_len <= MAX_DOWNMIX_BUFFER);
    celt_assert(in_len * (opus_int32)16000 == out_len * enc->Fs);
    switch(enc->Fs) {
        case 8000:
            up = 2;
            break;
        case 12000:
            up = 4;
            break;
        case 16000:
            up = 1;
            break;
        case 24000:
            up = 2;
            break;
        case 48000:
            up = 1;
            break;
        default:
            celt_assert(0);
    }
    OPUS_CLEAR(downmix, up*in_len);
    if (enc->channels == 1) {
        for (i=0;i<in_len;i++) downmix[up*i] = FLOAT2INT16(up*in[i]);
    } else {
        for (i=0;i<in_len;i++) downmix[up*i] = FLOAT2INT16(.5*up*(in[2*i]+in[2*i+1]));
    }
    if (enc->Fs == 16000) {
        OPUS_COPY(out, downmix, out_len);
    } else if (enc->Fs == 48000 || enc->Fs == 24000) {
        /* ellip(7, .2, 70, 7750/24000) */

        static const float filter_b[8] = { 0.005873358047f,  0.012980854831f, 0.014531340042f,  0.014531340042f, 0.012980854831f,  0.005873358047f, 0.004523418224f, 0.f};
        static const float filter_a[8] = {-3.878718597768f, 7.748834257468f, -9.653651699533f, 8.007342726666f, -4.379450178552f, 1.463182111810f, -0.231720677804f, 0.f};
        float b0 = 0.004523418224f;
        filter_df2t(downmix, downmix, up*in_len, b0, filter_b, filter_a, RESAMPLING_ORDER, enc->resample_mem);
        for (i=0;i<out_len;i++) out[i] = downmix[3*i];
    } else if (enc->Fs == 12000) {
        /* ellip(7, .2, 70, 7750/24000) */
        static const float filter_b[8] = {-0.001017101081f,  0.003673127243f,   0.001009165267f,  0.001009165267f,  0.003673127243f, -0.001017101081f,  0.002033596776f, 0.f};
        static const float filter_a[8] = {-4.930414411612f, 11.291643096504f, -15.322037343815f, 13.216403930898f, -7.220409219553f,  2.310550142771f, -0.334338618782f, 0.f};
        float b0 = 0.002033596776f;
        filter_df2t(downmix, downmix, up*in_len, b0, filter_b, filter_a, RESAMPLING_ORDER, enc->resample_mem);
        for (i=0;i<out_len;i++) out[i] = downmix[3*i];
    } else if (enc->Fs == 8000) {
        /* ellip(7, .2, 70, 3900/8000) */
        static const float filter_b[8] = { 0.081670120929f, 0.180401598565f,  0.259391051971f, 0.259391051971f,  0.180401598565f, 0.081670120929f,  0.020109185709f, 0.f};
        static const float filter_a[8] = {-1.393651933659f, 2.609789872676f, -2.403541968806f, 2.056814957331f, -1.148908574570f, 0.473001413788f, -0.110359852412f, 0.f};
        float b0 = 0.020109185709f;
        filter_df2t(downmix, out, out_len, b0, filter_b, filter_a, RESAMPLING_ORDER, enc->resample_mem);
    } else {
        celt_assert(0);
    }
}

void dred_compute_latents(DREDEnc *enc, const float *pcm, int frame_size)
{
    int frame_size16k = frame_size * 16000 / enc->Fs;
    while (frame_size16k > 0) {
        int process_size16k;
        int process_size;
        process_size16k = IMIN(2*DRED_FRAME_SIZE, frame_size16k);
        process_size = process_size16k * enc->Fs / 16000;
        dred_convert_to_16k(enc, pcm, process_size, &enc->input_buffer[enc->input_buffer_fill], process_size16k);
        enc->input_buffer_fill += process_size16k;
        if (enc->input_buffer_fill >= 2*DRED_FRAME_SIZE)
        {
          dred_process_frame(enc);
          enc->input_buffer_fill -= 2*DRED_FRAME_SIZE;
          OPUS_MOVE(&enc->input_buffer[0], &enc->input_buffer[2*DRED_FRAME_SIZE], enc->input_buffer_fill);
        }

        pcm += process_size;
        frame_size16k -= process_size16k;
    }
}

int dred_encode_silk_frame(const DREDEnc *enc, unsigned char *buf, int max_chunks, int max_bytes) {
    const opus_uint16 *dead_zone       = DRED_rdovae_get_dead_zone_pointer();
    const opus_uint16 *p0              = DRED_rdovae_get_p0_pointer();
    const opus_uint16 *quant_scales    = DRED_rdovae_get_quant_scales_pointer();
    const opus_uint16 *r               = DRED_rdovae_get_r_pointer();
    ec_enc ec_encoder;

    int q_level;
    int i;
    int offset;
    int ec_buffer_fill;
    int dred_offset;
    int q0;
    int dQ;

    /* entropy coding of state and latents */
    ec_enc_init(&ec_encoder, buf, max_bytes);
    dred_offset = 8; /* 20 ms */
    q0 = DRED_ENC_Q0;
    dQ = 3;
    ec_enc_uint(&ec_encoder, dred_offset, 32);
    ec_enc_uint(&ec_encoder, q0, 16);
    ec_enc_uint(&ec_encoder, dQ, 8);
    dred_encode_state(&ec_encoder, enc->state_buffer);

    for (i = 0; i < IMIN(2*max_chunks, enc->latents_buffer_fill-1); i += 2)
    {
        ec_enc ec_bak;
        ec_bak = ec_encoder;

        q_level = compute_quantizer(q0, dQ, i/2);
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
