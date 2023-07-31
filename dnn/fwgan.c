/* Copyright (c) 2023 Amazon */
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

#include "fwgan.h"
#include "os_support.h"
#include "freq.h"
#include "fwgan_data.h"
#include "lpcnet.h"
#include "pitch.h"
#include "nnet.h"

#define NB_SUBFRAMES 4
#define SUBFRAME_SIZE 40
#define FWGAN_FRAME_SIZE (NB_SUBFRAMES*SUBFRAME_SIZE)
#define CONT_PCM_INPUTS 320
#define MAX_CONT_SIZE CONT_NET_0_OUT_SIZE
#define FWGAN_GAMMA 0.9f
#define FWGAN_DEEMPHASIS 0.85f

#define FEAT_IN_SIZE (BFCC_WITH_CORR_UPSAMPLER_FC_OUT_SIZE/4 + FWGAN_FRAME_SIZE/2)

#define FWGAN_FEATURES (NB_FEATURES-1)

static void pitch_embeddings(float *pembed, float *phase, float w0) {
  int i;
  /* FIXME: This could be speeded up by making phase a unit-norm complex value, rotating it
     by exp(-i*w0) each sample, and renormalizing once in while.  */
  for (i=0;i<SUBFRAME_SIZE;i++) {
    *phase += w0;
    pembed[i] = sin(*phase);
    pembed[SUBFRAME_SIZE+i] = cos(*phase);
  }
}

static void run_fwgan_upsampler(FWGANState *st, float *cond, const float *features)
{
  FWGAN *model;
  model = &st->model;
  celt_assert(FWGAN_FEATURES == model->bfcc_with_corr_upsampler_fc.nb_inputs);
  celt_assert(BFCC_WITH_CORR_UPSAMPLER_FC_OUT_SIZE == model->bfcc_with_corr_upsampler_fc.nb_outputs);
  compute_generic_dense(&model->bfcc_with_corr_upsampler_fc, cond, features, ACTIVATION_TANH);
}

void fwgan_cont(FWGANState *st, const float *pcm0, const float *features0)
{
  int i;
  float norm2, norm_1;
  float cont_inputs[CONT_PCM_INPUTS+1];
  float tmp1[MAX_CONT_SIZE];
  float tmp2[MAX_CONT_SIZE];
  FWGAN *model;
  model = &st->model;
  norm2 = celt_inner_prod(pcm0, pcm0, CONT_PCM_INPUTS, st->arch);
  norm_1 = 1.f/sqrt(1e-8f + norm2);
  for (i=0;i<CONT_PCM_INPUTS;i++) cont_inputs[i+1] = norm_1*pcm0[i];
  cont_inputs[0] = log(sqrt(norm2) + 1e-7f);

  compute_generic_dense(&model->cont_net_0, tmp1, cont_inputs, ACTIVATION_TANH);
  compute_generic_dense(&model->cont_net_2, tmp2, tmp1, ACTIVATION_TANH);
  compute_generic_dense(&model->cont_net_4, tmp1, tmp2, ACTIVATION_TANH);
  compute_generic_dense(&model->cont_net_6, tmp2, tmp1, ACTIVATION_TANH);
  compute_generic_dense(&model->cont_net_8, tmp1, tmp2, ACTIVATION_TANH);
  celt_assert(CONT_NET_10_OUT_SIZE == model->cont_net_10.nb_outputs);
  compute_generic_dense(&model->cont_net_10, cont_inputs, tmp1, ACTIVATION_TANH);

  celt_assert(RNN_GRU_STATE_SIZE == model->rnn_cont_fc_0.nb_outputs);
  compute_generic_dense(&model->rnn_cont_fc_0, st->rnn_state, cont_inputs, ACTIVATION_TANH);

  celt_assert(FWC1_STATE_SIZE == model->fwc1_cont_fc_0.nb_outputs);
  compute_generic_dense(&model->fwc1_cont_fc_0, st->fwc1_state, cont_inputs, ACTIVATION_TANH);
  celt_assert(FWC2_STATE_SIZE == model->fwc2_cont_fc_0.nb_outputs);
  compute_generic_dense(&model->fwc2_cont_fc_0, st->fwc2_state, cont_inputs, ACTIVATION_TANH);
  celt_assert(FWC3_STATE_SIZE == model->fwc3_cont_fc_0.nb_outputs);
  compute_generic_dense(&model->fwc3_cont_fc_0, st->fwc3_state, cont_inputs, ACTIVATION_TANH);
  celt_assert(FWC4_STATE_SIZE == model->fwc4_cont_fc_0.nb_outputs);
  compute_generic_dense(&model->fwc4_cont_fc_0, st->fwc4_state, cont_inputs, ACTIVATION_TANH);
  celt_assert(FWC5_STATE_SIZE == model->fwc5_cont_fc_0.nb_outputs);
  compute_generic_dense(&model->fwc5_cont_fc_0, st->fwc5_state, cont_inputs, ACTIVATION_TANH);
  celt_assert(FWC6_STATE_SIZE == model->fwc6_cont_fc_0.nb_outputs);
  compute_generic_dense(&model->fwc6_cont_fc_0, st->fwc6_state, cont_inputs, ACTIVATION_TANH);
  celt_assert(FWC7_STATE_SIZE == model->fwc7_cont_fc_0.nb_outputs);
  compute_generic_dense(&model->fwc7_cont_fc_0, st->fwc7_state, cont_inputs, ACTIVATION_TANH);

  /* FIXME: Do we need to handle initial features? How? */

  st->cont_initialized = 1;
}

static void apply_gain(float *pcm, float c0) {
  int i;
  float gain = pow(10.f, (0.5f*c0/sqrt(18.f)));
  for (i=0;i<FWGAN_FRAME_SIZE;i++) pcm[i] *= gain;
}

static void fwgan_lpc_syn(float *pcm, float *mem, const float *lpc) {
  int i;
  for (i=0;i<FWGAN_FRAME_SIZE;i++) {
    int j;
    for (j=0;j<LPC_ORDER;j++) pcm[i] -= mem[j]*lpc[j];
    OPUS_MOVE(&mem[1], &mem[0], LPC_ORDER-1);
    mem[0] = pcm[i];
  }
}

static void fwgan_deemphasis(float *pcm, float *deemph_mem) {
  int i;
  for (i=0;i<FWGAN_FRAME_SIZE;i++) {
    pcm[i] += FWGAN_DEEMPHASIS * *deemph_mem;
    *deemph_mem = pcm[i];
  }
}

static void run_fwgan_subframe(FWGANState *st, float *pcm, const float *cond, float w0)
{
  float tmp1[FWC1_FC_0_OUT_SIZE];
  float tmp2[IMAX(RNN_GRU_STATE_SIZE, FWC2_FC_0_OUT_SIZE)];
  float feat_in[FEAT_IN_SIZE];
  float rnn_in[FEAT_IN_CONV1_CONV_OUT_SIZE];
  float pembed[FWGAN_FRAME_SIZE/2];
  FWGAN *model;
  model = &st->model;

  pitch_embeddings(pembed, &st->embed_phase, w0);
  /* Interleave bfcc_cond and pembed for each subframe in feat_in. */
  OPUS_COPY(&feat_in[0], &cond[0], BFCC_WITH_CORR_UPSAMPLER_FC_OUT_SIZE/4);
  OPUS_COPY(&feat_in[BFCC_WITH_CORR_UPSAMPLER_FC_OUT_SIZE/4], &pembed[0], FWGAN_FRAME_SIZE/2);

  compute_generic_conv1d(&model->feat_in_conv1_conv, rnn_in, st->cont_conv1_mem, feat_in, FEAT_IN_CONV1_CONV_IN_SIZE, ACTIVATION_LINEAR);
  celt_assert(FEAT_IN_NL1_GATE_OUT_SIZE == model->feat_in_nl1_gate.nb_outputs);
  compute_gated_activation(&model->feat_in_nl1_gate, rnn_in, rnn_in, ACTIVATION_TANH);


  compute_generic_gru(&model->rnn_gru_input, &model->rnn_gru_recurrent, st->rnn_state, rnn_in);
  celt_assert(IMAX(RNN_GRU_STATE_SIZE, FWC2_FC_0_OUT_SIZE) >= model->rnn_nl_gate.nb_outputs);
  compute_gated_activation(&model->rnn_nl_gate, tmp2, st->rnn_state, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc1_fc_0, tmp1, st->fwc1_state, tmp2, RNN_GRU_STATE_SIZE, ACTIVATION_TANH);
  compute_gated_activation(&model->fwc1_fc_1_gate, tmp1, tmp1, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc2_fc_0, tmp2, st->fwc2_state, tmp1, FWC1_FC_0_OUT_SIZE, ACTIVATION_TANH);
  compute_gated_activation(&model->fwc2_fc_1_gate, tmp2, tmp2, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc3_fc_0, tmp1, st->fwc3_state, tmp2, FWC2_FC_0_OUT_SIZE, ACTIVATION_TANH);
  compute_gated_activation(&model->fwc3_fc_1_gate, tmp1, tmp1, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc4_fc_0, tmp2, st->fwc4_state, tmp1, FWC3_FC_0_OUT_SIZE, ACTIVATION_TANH);
  compute_gated_activation(&model->fwc4_fc_1_gate, tmp2, tmp2, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc5_fc_0, tmp1, st->fwc5_state, tmp2, FWC4_FC_0_OUT_SIZE, ACTIVATION_TANH);
  compute_gated_activation(&model->fwc5_fc_1_gate, tmp1, tmp1, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc6_fc_0, tmp2, st->fwc6_state, tmp1, FWC5_FC_0_OUT_SIZE, ACTIVATION_TANH);
  compute_gated_activation(&model->fwc6_fc_1_gate, tmp2, tmp2, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc7_fc_0, tmp1, st->fwc7_state, tmp2, FWC6_FC_0_OUT_SIZE, ACTIVATION_TANH);
  compute_gated_activation(&model->fwc7_fc_1_gate, pcm, tmp1, ACTIVATION_TANH);
}



void fwgan_init(FWGANState *st)
{
  int ret;
  OPUS_CLEAR(st, 1);
  ret = init_fwgan(&st->model, fwgan_arrays);
  celt_assert(ret == 0);
  /* FIXME: perform arch detection. */
}

void fwgan_synthesize(FWGANState *st, float *pcm, const float *features)
{
  int subframe;
  float lpc[LPC_ORDER];
  float cond[BFCC_WITH_CORR_UPSAMPLER_FC_OUT_SIZE];
  float w0;
  int period;
  float lpc_weight;
  int i;
  celt_assert(st->cont_initialized);
  period = (int)floor(.1 + 50*features[NB_BANDS]+100);
  w0 = 2*M_PI/period;
  lpc_from_cepstrum(lpc, features);
  lpc_weight = 1.f;
  for (i=0;i<LPC_ORDER;i++) {
    lpc_weight *= FWGAN_GAMMA;
    lpc[i] *= lpc_weight;
  }
  run_fwgan_upsampler(st, cond, features);
  for (subframe=0;subframe<NB_SUBFRAMES;subframe++) {
    float *sub_cond;
    sub_cond = &cond[subframe*FEAT_IN_SIZE/4];
    run_fwgan_subframe(st, &pcm[subframe*SUBFRAME_SIZE], sub_cond, w0);
  }
  apply_gain(pcm, features[0]);
  fwgan_lpc_syn(pcm, st->syn_mem, lpc);
  fwgan_deemphasis(pcm, &st->deemph_mem);
}
