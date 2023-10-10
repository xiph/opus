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

#include "fargan.h"
#include "os_support.h"
#include "freq.h"
#include "fargan_data.h"
#include "lpcnet.h"
#include "pitch.h"
#include "nnet.h"
#include "lpcnet_private.h"

#define FARGAN_FEATURES (NB_FEATURES)

static void compute_fargan_cond(FARGANState *st, float *cond, const float *features)
{
  FARGAN *model;
  float conv1_in[COND_NET_FCONV1_IN_SIZE];
  float conv2_in[COND_NET_FCONV2_IN_SIZE];
  model = &st->model;
  celt_assert(FARGAN_FEATURES == model->cond_net_fdense1.nb_inputs);
  celt_assert(COND_NET_FCONV1_IN_SIZE == model->cond_net_fdense1.nb_outputs);
  celt_assert(COND_NET_FCONV2_IN_SIZE == model->cond_net_fconv1.nb_outputs);

  compute_generic_dense(&model->cond_net_fdense1, conv1_in, features, ACTIVATION_TANH);
  compute_generic_conv1d(&model->cond_net_fconv1, conv2_in, st->cond_conv1_state, conv1_in, COND_NET_FCONV1_IN_SIZE, ACTIVATION_TANH);
  compute_generic_conv1d(&model->cond_net_fconv2, cond, st->cond_conv2_state, conv2_in, COND_NET_FCONV2_IN_SIZE, ACTIVATION_TANH);
}

static void fargan_preemphasis(float *pcm, float *preemph_mem) {
  int i;
  for (i=0;i<FARGAN_SUBFRAME_SIZE;i++) {
    float tmp = pcm[i];
    pcm[i] -= FARGAN_DEEMPHASIS * *preemph_mem;
    *preemph_mem = tmp;
  }
}

static void fargan_deemphasis(float *pcm, float *deemph_mem) {
  int i;
  for (i=0;i<FARGAN_SUBFRAME_SIZE;i++) {
    pcm[i] += FARGAN_DEEMPHASIS * *deemph_mem;
    *deemph_mem = pcm[i];
  }
}

static void run_fargan_subframe(FARGANState *st, float *pcm, const float *cond, int period)
{
  int i, pos;
  float tmp1[FWC1_FC_0_OUT_SIZE];
  float tmp2[IMAX(RNN_GRU_STATE_SIZE, FWC2_FC_0_OUT_SIZE)];
  float fwc0_in[FARGAN_COND_SIZE+2*FARGAN_SUBFRAME_SIZE];
  float rnn_in[FEAT_IN_CONV1_CONV_OUT_SIZE];
  float pembed[FARGAN_FRAME_SIZE/2];
  FARGAN *model;
  model = &st->model;

  /* Interleave bfcc_cond and pembed for each subframe in feat_in. */
  OPUS_COPY(&fwc0_in[0], &cond[0], FARGAN_COND_SIZE);
  pos = PITCH_MAX_PERIOD-period;
  for (i=0;i<FARGAN_SUBFRAME_SIZE;i++) {
    fwc0_in[FARGAN_COND_SIZE+i] = st->pitch_buf[pos++];
    if (pos == PITCH_MAX_PERIOD) pos -= period;
  }
  OPUS_COPY(&fwc0_in[FARGAN_COND_SIZE], st->pitch_buf[PITCH_MAX_PERIOD-FARGAN_SUBFRAME_SIZE], FARGAN_SUBFRAME_SIZE);

  compute_generic_conv1d(&model->sig_net_fwc0_conv, gru1_in, st->fwc0_mem, fwc0_in, FARGAN_COND_SIZE+2*FARGAN_SUBFRAME_SIZE, ACTIVATION_TANH);
  celt_assert(FEAT_IN_NL1_GATE_OUT_SIZE == model->feat_in_nl1_gate.nb_outputs);
  compute_glu(&model->sig_net_fwc0_glu_gate, rnn_in, rnn_in);

  compute_generic_gru(&model->sig_net_gru1_input, &model->sig_net_gru1_recurrent, st->gru1_state, rnn_in);
  celt_assert(IMAX(RNN_GRU_STATE_SIZE, FWC2_FC_0_OUT_SIZE) >= model->rnn_nl_gate.nb_outputs);
  compute_glu(&model->rnn_nl_gate, tmp2, st->gru1_state);

  compute_generic_conv1d(&model->fwc1_fc_0, tmp1, st->fwc1_state, tmp2, RNN_GRU_STATE_SIZE, ACTIVATION_LINEAR);
  compute_gated_activation(&model->fwc1_fc_1_gate, tmp1, tmp1, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc2_fc_0, tmp2, st->fwc2_state, tmp1, FWC1_FC_0_OUT_SIZE, ACTIVATION_LINEAR);
  compute_gated_activation(&model->fwc2_fc_1_gate, tmp2, tmp2, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc3_fc_0, tmp1, st->fwc3_state, tmp2, FWC2_FC_0_OUT_SIZE, ACTIVATION_LINEAR);
  compute_gated_activation(&model->fwc3_fc_1_gate, tmp1, tmp1, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc4_fc_0, tmp2, st->fwc4_state, tmp1, FWC3_FC_0_OUT_SIZE, ACTIVATION_LINEAR);
  compute_gated_activation(&model->fwc4_fc_1_gate, tmp2, tmp2, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc5_fc_0, tmp1, st->fwc5_state, tmp2, FWC4_FC_0_OUT_SIZE, ACTIVATION_LINEAR);
  compute_gated_activation(&model->fwc5_fc_1_gate, tmp1, tmp1, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc6_fc_0, tmp2, st->fwc6_state, tmp1, FWC5_FC_0_OUT_SIZE, ACTIVATION_LINEAR);
  compute_gated_activation(&model->fwc6_fc_1_gate, tmp2, tmp2, ACTIVATION_TANH);

  compute_generic_conv1d(&model->fwc7_fc_0, tmp1, st->fwc7_state, tmp2, FWC6_FC_0_OUT_SIZE, ACTIVATION_LINEAR);
  compute_gated_activation(&model->fwc7_fc_1_gate, pcm, tmp1, ACTIVATION_TANH);

  apply_gain(pcm, c0, &st->last_gain);
  fargan_preemphasis(pcm, &st->preemph_mem);
  fargan_lpc_syn(pcm, st->syn_mem, lpc, st->last_lpc);
  fargan_deemphasis(pcm, &st->deemph_mem);
}

void fargan_cont(FARGANState *st, const float *pcm0, const float *features0)
{
  int i;
  float norm2, norm_1;
  float wpcm0[CONT_PCM_INPUTS];
  float cont_inputs[CONT_PCM_INPUTS+1];
  float tmp1[MAX_CONT_SIZE];
  float tmp2[MAX_CONT_SIZE];
  float lpc[LPC_ORDER];
  float new_pcm[FARGAN_FRAME_SIZE];
  FARGAN *model;
  st->embed_phase[0] = 1;
  model = &st->model;
  compute_wlpc(lpc, features0);
  /* Deemphasis memory is just the last continuation sample. */
  st->deemph_mem = pcm0[CONT_PCM_INPUTS-1];

  /* Apply analysis filter, considering that the preemphasis and deemphasis filter
     cancel each other in this case since the LPC filter is constant across that boundary.
     */
  for (i=LPC_ORDER;i<CONT_PCM_INPUTS;i++) {
    int j;
    wpcm0[i] = pcm0[i];
    for (j=0;j<LPC_ORDER;j++) wpcm0[i] += lpc[j]*pcm0[i-j-1];
  }
  /* FIXME: Make this less stupid. */
  for (i=0;i<LPC_ORDER;i++) wpcm0[i] = wpcm0[LPC_ORDER];

  /* The memory of the pre-empahsis is the last sample of the weighted signal
     (ignoring preemphasis+deemphasis combination). */
  st->preemph_mem = wpcm0[CONT_PCM_INPUTS-1];
  /* The memory of the synthesis filter is the pre-emphasized continuation. */
  for (i=0;i<LPC_ORDER;i++) st->syn_mem[i] = pcm0[CONT_PCM_INPUTS-1-i] - FARGAN_DEEMPHASIS*pcm0[CONT_PCM_INPUTS-2-i];

  norm2 = celt_inner_prod(wpcm0, wpcm0, CONT_PCM_INPUTS, st->arch);
  norm_1 = 1.f/sqrt(1e-8f + norm2);
  for (i=0;i<CONT_PCM_INPUTS;i++) cont_inputs[i+1] = norm_1*wpcm0[i];
  cont_inputs[0] = log(sqrt(norm2) + 1e-7f);


  st->cont_initialized = 1;
  /* Process the first frame, discard the first subframe, and keep the rest for the first
     synthesis call. */
  fargan_synthesize_impl(st, new_pcm, lpc, features0);
  OPUS_COPY(st->pcm_buf, &new_pcm[SUBFRAME_SIZE], FARGAN_FRAME_SIZE-SUBFRAME_SIZE);
}


void fargan_init(FARGANState *st)
{
  int ret;
  OPUS_CLEAR(st, 1);
  ret = init_fargan(&st->model, fwgan_arrays);
  celt_assert(ret == 0);
  /* FIXME: perform arch detection. */
}

int fargan_load_model(FARGANState *st, const unsigned char *data, int len) {
  WeightArray *list;
  int ret;
  parse_weights(&list, data, len);
  ret = init_fargan(&st->model, list);
  free(list);
  if (ret == 0) return 0;
  else return -1;
}

static void fargan_synthesize_impl(FARGANState *st, float *pcm, const float *features)
{
  int subframe;
  float cond[COND_NET_FCONV2_OUT_SIZE];
  int period;
  celt_assert(st->cont_initialized);

  period = (int)floor(.5+256./pow(2.f,((1./60.)*((features[NB_BANDS]+1.5)*60))));
  compute_fargan_cond(st, cond, features);
  for (subframe=0;subframe<FARGAN_NB_SUBFRAMES;subframe++) {
    float *sub_cond;
    sub_cond = &cond[subframe*FARGAN_COND_SIZE];
    run_fargan_subframe(st, &pcm[subframe*FARGAN_SUBFRAME_SIZE], sub_cond, period);
  }
}

void fargan_synthesize(FARGANState *st, float *pcm, const float *features)
{
  fargan_synthesize_impl(st, pcm, features);
}

void fargan_synthesize_int(FARGANState *st, opus_int16 *pcm, const float *features)
{
  int i;
  float fpcm[FARGAN_FRAME_SIZE];
  fargan_synthesize(st, fpcm, features);
  for (i=0;i<LPCNET_FRAME_SIZE;i++) pcm[i] = (int)floor(.5 + MIN32(32767, MAX32(-32767, 32768.f*fpcm[i])));
}
