#ifndef _DRED_RDOVAE_DEC_H
#define _DRED_RDOVAE_DEC_H

#include "dred_rdovae.h"
#include "dred_rdovae_dec_data.h"
#include "dred_rdovae_stats_data.h"

struct RDOVAEDecStruct {
    float dense2_state[DEC_DENSE2_STATE_SIZE];
    float dense4_state[DEC_DENSE2_STATE_SIZE];
    float dense6_state[DEC_DENSE2_STATE_SIZE];
};

void dred_rdovae_dec_init_states(RDOVAEDec *h, const float * initial_state);
void dred_rdovae_decode_qframe(RDOVAEDec *h, float *qframe, const float * z);

#endif