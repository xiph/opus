#include <string.h>

#include "dred_encoder.h"


void init_dred_encoder(DREDEnc* enc)
{
    memset(enc, 0, sizeof(*enc));
    enc->lpcnet_enc_state = lpcnet_encoder_create();
    enc->rdovae_enc = DRED_rdovae_create_encoder();
}

void dred_encode_silk_frame(DREDEnc *enc, const opus_int16 *silk_frame)
{
    /* delay signal by 79 samples */
    memmove(enc->input_buffer, enc->input_buffer + DRED_SILK_ENCODER_DELAY, DRED_SILK_ENCODER_DELAY * sizeof(*enc->input_buffer));
    memcpy(enc->input_buffer + DRED_SILK_ENCODER_DELAY, silk_frame, DRED_DFRAME_SIZE * sizeof(*silk_frame));

    /* shift latents buffer */
    memmove(enc->latents_buffer + DRED_LATENT_DIM, enc->latents_buffer, DRED_LATENT_DIM * sizeof(*enc->latents_buffer));

    /* calculate LPCNet features */
    lpcnet_compute_single_frame_features(enc->lpcnet_enc_state, enc->input_buffer, enc->feature_buffer);
    lpcnet_compute_single_frame_features(enc->lpcnet_enc_state, enc->input_buffer + DRED_FRAME_SIZE, enc->feature_buffer + DRED_NUM_FEATURES);

    /* run RDOVAE encoder */
    DRED_rdovae_encode_dframe(enc->rdovae_enc, enc->latents_buffer, enc->state_buffer, enc->feature_buffer);

    /* entropy coding of state and latents */
}