#ifndef _NFEC_DEC_H
#define _NFEC_DEC_H

#include "nfec_dec_data.h"
#include "nfec_stats_data.h"

typedef struct {
    float dense2_state[DEC_DENSE2_STATE_SIZE];
    float dense4_state[DEC_DENSE2_STATE_SIZE];
    float dense6_state[DEC_DENSE2_STATE_SIZE];
} NFECDecState;

void nfec_dec_init_states(NFECDecState *h, const float * initial_state);
void nfec_dec_unquantize_latent_vector(float *z, const int *zq, int quant_level);
void nfec_decode_qframe(NFECDecState *h, float *qframe, const float * z);

#endif