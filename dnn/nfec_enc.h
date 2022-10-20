#ifndef _NFEC_ENC_H
#define _NFEC_ENC_H

#include "nfec_enc_data.h"

struct NFECEncState{
    float dense2_state[3 * ENC_DENSE2_STATE_SIZE];
    float dense4_state[3 * ENC_DENSE4_STATE_SIZE];
    float dense6_state[3 * ENC_DENSE6_STATE_SIZE];
    float bits_dense_state[BITS_DENSE_STATE_SIZE];
};

void nfec_encode_dframe(struct NFECEncState *enc_state, float *latents, float *initial_state, const float *input);
void nfec_quantize_latent_vector(int *z_q, const float *z, int quant_level);

#endif