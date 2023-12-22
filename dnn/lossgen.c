#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include "lossgen.h"
#include "os_support.h"
#include "nnet.h"
#include "lpcnet_private.h"

int sample_loss(
    LossGenState *st,
    float percent_loss,
    int arch
    )
{
  float input[2];
  float tmp[LOSSGEN_DENSE_IN_OUT_SIZE];
  float out;
  int loss;
  LossGen *model = &st->model;
  input[0] = st->last_loss;
  input[1] = percent_loss;
  compute_generic_dense(&model->lossgen_dense_in, tmp, input, ACTIVATION_TANH, arch);
  compute_generic_gru(&model->lossgen_gru1_input, &model->lossgen_gru1_recurrent, st->gru1_state, tmp, arch);
  compute_generic_gru(&model->lossgen_gru2_input, &model->lossgen_gru2_recurrent, st->gru2_state, st->gru1_state, arch);
  compute_generic_dense(&model->lossgen_dense_out, &out, st->gru2_state, ACTIVATION_SIGMOID, arch);
  loss = (float)rand()/RAND_MAX < out;
  st->last_loss = loss;
  return loss;
}


void lossgen_init(LossGenState *st)
{
  int ret;
  OPUS_CLEAR(st, 1);
#ifndef USE_WEIGHTS_FILE
  ret = init_lossgen(&st->model, lossgen_arrays);
#else
  ret = 0;
#endif
  celt_assert(ret == 0);
}

int lossgen_load_model(LossGenState *st, const unsigned char *data, int len) {
  WeightArray *list;
  int ret;
  parse_weights(&list, data, len);
  ret = init_lossgen(&st->model, list);
  opus_free(list);
  if (ret == 0) return 0;
  else return -1;
}

#if 0
#include <stdio.h>
int main(int argc, char **argv) {
  int i, N;
  float p;
  LossGenState st;
  lossgen_init(&st);
  p = atof(argv[1]);
  N = atoi(argv[2]);
  for (i=0;i<N;i++) {
    printf("%d\n", sample_loss(&st, p, 0));
  }
}
#endif
