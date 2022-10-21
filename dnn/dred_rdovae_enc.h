#ifndef _DRED_RDOVAE_ENC_H
#define _DRED_RDOVAE_ENC_H

#include "dred_rdovae.h"

#include "dred_rdovae_enc_data.h"

struct RDOVAEEncStruct {
    float dense2_state[3 * ENC_DENSE2_STATE_SIZE];
    float dense4_state[3 * ENC_DENSE4_STATE_SIZE];
    float dense6_state[3 * ENC_DENSE6_STATE_SIZE];
    float bits_dense_state[BITS_DENSE_STATE_SIZE];
};

void dred_rdovae_encode_dframe(RDOVAEEnc *enc_state, float *latents, float *initial_state, const float *input);


#endif