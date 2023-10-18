/* Copyright (c) 2021 Amazon */
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
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
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

#include "lpcnet_private.h"
#include "lpcnet.h"
#include "plc_data.h"
#include "os_support.h"
#include "common.h"

#ifndef M_PI
#define M_PI 3.141592653
#endif

/* Comment this out to have LPCNet update its state on every good packet (slow). */
#define PLC_SKIP_UPDATES

int lpcnet_plc_get_size() {
  return sizeof(LPCNetPLCState);
}

void lpcnet_plc_reset(LPCNetPLCState *st) {
  OPUS_CLEAR((char*)&st->LPCNET_PLC_RESET_START,
          sizeof(LPCNetPLCState)-
          ((char*)&st->LPCNET_PLC_RESET_START - (char*)st));
  lpcnet_encoder_init(&st->enc);
  OPUS_CLEAR(st->pcm, PLC_BUF_SIZE);
  st->blend = 0;
  st->loss_count = 0;
}

int lpcnet_plc_init(LPCNetPLCState *st, int options) {
  int ret;
  fargan_init(&st->fargan);
  lpcnet_encoder_init(&st->enc);
  if ((options&0x3) == LPCNET_PLC_CAUSAL) {
    st->enable_blending = 1;
  } else if ((options&0x3) == LPCNET_PLC_CODEC) {
    st->enable_blending = 0;
  } else {
    return -1;
  }
#ifndef USE_WEIGHTS_FILE
  ret = init_plc_model(&st->model, lpcnet_plc_arrays);
#else
  ret = 0;
#endif
  celt_assert(ret == 0);
  lpcnet_plc_reset(st);
  return ret;
}

int lpcnet_plc_load_model(LPCNetPLCState *st, const unsigned char *data, int len) {
  WeightArray *list;
  int ret;
  parse_weights(&list, data, len);
  ret = init_plc_model(&st->model, list);
  free(list);
  if (ret == 0) {
    return fargan_load_model(&st->fargan, data, len);
  }
  else return -1;
}

LPCNetPLCState *lpcnet_plc_create(int options) {
  LPCNetPLCState *st;
  st = calloc(sizeof(*st), 1);
  lpcnet_plc_init(st, options);
  return st;
}

void lpcnet_plc_destroy(LPCNetPLCState *st) {
  free(st);
}

void lpcnet_plc_fec_add(LPCNetPLCState *st, const float *features) {
  if (features == NULL) {
    st->fec_skip++;
    return;
  }
  if (st->fec_fill_pos == PLC_MAX_FEC) {
    if (st->fec_keep_pos == 0) {
      fprintf(stderr, "FEC buffer full\n");
      return;
    }
    OPUS_MOVE(&st->fec[0][0], &st->fec[st->fec_keep_pos][0], (st->fec_fill_pos-st->fec_keep_pos)*NB_FEATURES);
    st->fec_fill_pos = st->fec_fill_pos-st->fec_keep_pos;
    st->fec_read_pos -= st->fec_keep_pos;
    st->fec_keep_pos = 0;
  }
  OPUS_COPY(&st->fec[st->fec_fill_pos][0], features, NB_FEATURES);
  st->fec_fill_pos++;
}

void lpcnet_plc_fec_clear(LPCNetPLCState *st) {
  st->fec_keep_pos = st->fec_read_pos = st->fec_fill_pos = st-> fec_skip = 0;
}


static void compute_plc_pred(LPCNetPLCState *st, float *out, const float *in) {
  float zeros[3*PLC_MAX_RNN_NEURONS] = {0};
  float dense_out[PLC_DENSE1_OUT_SIZE];
  PLCNetState *net = &st->plc_net;
  _lpcnet_compute_dense(&st->model.plc_dense1, dense_out, in);
  compute_gruB(&st->model.plc_gru1, zeros, net->plc_gru1_state, dense_out);
  compute_gruB(&st->model.plc_gru2, zeros, net->plc_gru2_state, net->plc_gru1_state);
  _lpcnet_compute_dense(&st->model.plc_out, out, net->plc_gru2_state);
}

static int get_fec_or_pred(LPCNetPLCState *st, float *out) {
  if (st->fec_read_pos != st->fec_fill_pos && st->fec_skip==0) {
    float plc_features[2*NB_BANDS+NB_FEATURES+1] = {0};
    float discard[NB_FEATURES];
    OPUS_COPY(out, &st->fec[st->fec_read_pos][0], NB_FEATURES);
    st->fec_read_pos++;
    /* Make sure we can rewind a few frames back at resync time. */
    st->fec_keep_pos = IMAX(0, IMAX(st->fec_keep_pos, st->fec_read_pos-FEATURES_DELAY-1));
    /* Update PLC state using FEC, so without Burg features. */
    OPUS_COPY(&plc_features[2*NB_BANDS], out, NB_FEATURES);
    plc_features[2*NB_BANDS+NB_FEATURES] = -1;
    compute_plc_pred(st, discard, plc_features);
    return 1;
  } else {
    float zeros[2*NB_BANDS+NB_FEATURES+1] = {0};
    compute_plc_pred(st, out, zeros);
    if (st->fec_skip > 0) st->fec_skip--;
    return 0;
  }
}

/* In this causal version of the code, the DNN model implemented by compute_plc_pred()
   needs to generate two feature vectors to conceal the first lost packet.*/

int lpcnet_plc_update(LPCNetPLCState *st, opus_int16 *pcm) {
  int i;
  float x[FRAME_SIZE];
  float plc_features[2*NB_BANDS+NB_FEATURES+1];
  for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
  burg_cepstral_analysis(plc_features, x);
  if (st->blend) {
    if (FEATURES_DELAY > 0) st->plc_net = st->plc_copy[FEATURES_DELAY-1];
  }
  /* Update state. */
  /*fprintf(stderr, "update state\n");*/
  for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
  preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
  compute_frame_features(&st->enc, x);
  process_single_frame(&st->enc, NULL);
  if (!st->blend) {
    OPUS_COPY(&plc_features[2*NB_BANDS], st->enc.features, NB_FEATURES);
    plc_features[2*NB_BANDS+NB_FEATURES] = 1;
    compute_plc_pred(st, st->features, plc_features);
    /* Discard an FEC frame that we know we will no longer need. */
    OPUS_MOVE(&st->cont_features[0], &st->cont_features[NB_FEATURES], (CONT_VECTORS-1)*NB_FEATURES);
  }
  OPUS_COPY(&st->cont_features[(CONT_VECTORS-1)*NB_FEATURES], st->enc.features, NB_FEATURES);
  OPUS_MOVE(st->pcm, &st->pcm[FRAME_SIZE], FARGAN_CONT_SAMPLES-FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) st->pcm[FARGAN_CONT_SAMPLES-FRAME_SIZE+i] = (1.f/32768.f)*pcm[i];
  st->loss_count = 0;
  st->blend = 0;
  return 0;
}

static const float att_table[10] = {0, 0,  -.2, -.2,  -.4, -.4,  -.8, -.8, -1.6, -1.6};
int lpcnet_plc_conceal(LPCNetPLCState *st, opus_int16 *pcm) {
  int i;
  if (st->blend == 0) {
    get_fec_or_pred(st, st->features);
    OPUS_MOVE(&st->cont_features[0], &st->cont_features[NB_FEATURES], (CONT_VECTORS-1)*NB_FEATURES);
    OPUS_COPY(&st->cont_features[(CONT_VECTORS-1)*NB_FEATURES], st->features, NB_FEATURES);
    get_fec_or_pred(st, st->features);
    OPUS_MOVE(&st->cont_features[0], &st->cont_features[NB_FEATURES], (CONT_VECTORS-1)*NB_FEATURES);
    OPUS_COPY(&st->cont_features[(CONT_VECTORS-1)*NB_FEATURES], st->features, NB_FEATURES);
    fargan_cont(&st->fargan, st->pcm, st->cont_features);
  }
  OPUS_MOVE(&st->plc_copy[1], &st->plc_copy[0], FEATURES_DELAY);
  st->plc_copy[0] = st->plc_net;
  if (get_fec_or_pred(st, st->features)) st->loss_count = 0;
  else st->loss_count++;
  if (st->loss_count >= 10) st->features[0] = MAX16(-10, st->features[0]+att_table[9] - 2*(st->loss_count-9));
  else st->features[0] = MAX16(-10, st->features[0]+att_table[st->loss_count]);
  fargan_synthesize_int(&st->fargan, pcm, &st->features[0]);
  {
    float x[FRAME_SIZE];
    /* FIXME: Can we do better? */
    for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
    preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(&st->enc, x);
    process_single_frame(&st->enc, NULL);
  }
  OPUS_MOVE(&st->cont_features[0], &st->cont_features[NB_FEATURES], (CONT_VECTORS-1)*NB_FEATURES);
  OPUS_COPY(&st->cont_features[(CONT_VECTORS-1)*NB_FEATURES], st->enc.features, NB_FEATURES);
  OPUS_MOVE(st->pcm, &st->pcm[FRAME_SIZE], FARGAN_CONT_SAMPLES-FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) st->pcm[FARGAN_CONT_SAMPLES-FRAME_SIZE+i] = (1.f/32768.f)*pcm[i];
  st->blend = 1;
  return 0;
}
