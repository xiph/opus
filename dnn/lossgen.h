#ifndef LOSSGEN_H
#define LOSSGEN_H


#include "lossgen_data.h"

#define PITCH_MIN_PERIOD 32
#define PITCH_MAX_PERIOD 256

#define NB_XCORR_FEATURES (PITCH_MAX_PERIOD-PITCH_MIN_PERIOD)


typedef struct {
  LossGen model;
  float gru1_state[LOSSGEN_GRU1_STATE_SIZE];
  float gru2_state[LOSSGEN_GRU2_STATE_SIZE];
  int last_loss;
} LossGenState;


void lossgen_init(LossGenState *st);
int lossgen_load_model(LossGenState *st, const unsigned char *data, int len);

int sample_loss(
    LossGenState *st,
    float percent_loss);

#endif
