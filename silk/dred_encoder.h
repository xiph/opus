#ifndef _DRED_ENCODER_H
#define _DRED_ENCODER_H

#include "lpcnet.h"
#include "dred_rdovae.h"
#include "entcode.h"

/* these are inpart duplicates to the values defined in dred_rdovae_constants.h */
#define DRED_NUM_FEATURES 20
#define DRED_LATENT_DIM 80
#define DRED_STATE_DIM 24
#define DRED_NUM_QUANTIZATION_LEVELS 40
#define DRED_MAX_FRAMES 100
#define DRED_SILK_ENCODER_DELAY 79
#define DRED_FRAME_SIZE 160
#define DRED_DFRAME_SIZE (2 * (DRED_FRAME_SIZE))
#define DRED_MAX_DATA_SIZE 1000
#define DRED_ENC_Q0 9
#define DRED_ENC_Q1 15
#define DRED_NUM_REDUNDANCY_FRAMES 50

typedef struct {
    opus_int16 input_buffer[DRED_DFRAME_SIZE + DRED_SILK_ENCODER_DELAY];
    float feature_buffer[2 * 36];
    float latents_buffer[DRED_MAX_FRAMES * DRED_LATENT_DIM];
    float state_buffer[24];
    unsigned char ec_buffer[DRED_MAX_DATA_SIZE];
    ec_enc ec_encoder;
    LPCNetEncState *lpcnet_enc_state;
    RDOVAEEnc *rdovae_enc;
} DREDEnc;


void init_dred_encoder(DREDEnc* enc);

void dred_encode_silk_frame(DREDEnc *enc, const opus_int16 *silk_frame);

#endif