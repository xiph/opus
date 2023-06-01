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

#ifndef M_PI
#define M_PI 3.141592653
#endif

/* Comment this out to have LPCNet update its state on every good packet (slow). */
#define PLC_SKIP_UPDATES

LPCNET_EXPORT int lpcnet_plc_get_size() {
  return sizeof(LPCNetPLCState);
}

LPCNET_EXPORT void lpcnet_plc_reset(LPCNetPLCState *st) {
  RNN_CLEAR((char*)&st->LPCNET_PLC_RESET_START,
          sizeof(LPCNetPLCState)-
          ((char*)&st->LPCNET_PLC_RESET_START - (char*)st));
  lpcnet_reset(&st->lpcnet);
  lpcnet_encoder_init(&st->enc);
  RNN_CLEAR(st->pcm, PLC_BUF_SIZE);
  st->pcm_fill = PLC_BUF_SIZE;
  st->skip_analysis = 0;
  st->blend = 0;
  st->loss_count = 0;
  st->dc_mem = 0;
  st->queued_update = 0;
}

LPCNET_EXPORT int lpcnet_plc_init(LPCNetPLCState *st, int options) {
  int ret;
  lpcnet_init(&st->lpcnet);
  lpcnet_encoder_init(&st->enc);
  if ((options&0x3) == LPCNET_PLC_CAUSAL) {
    st->enable_blending = 1;
    st->non_causal = 0;
  } else if ((options&0x3) == LPCNET_PLC_NONCAUSAL) {
    st->enable_blending = 1;
    st->non_causal = 1;
  } else if ((options&0x3) == LPCNET_PLC_CODEC) {
    st->enable_blending = 0;
    st->non_causal = 0;
  } else {
    return -1;
  }
  st->remove_dc = !!(options&LPCNET_PLC_DC_FILTER);
#ifndef USE_WEIGHTS_FILE
  ret = init_plc_model(&st->model, lpcnet_plc_arrays);
#else
  ret = 0;
#endif
  celt_assert(ret == 0);
  lpcnet_plc_reset(st);
  return ret;
}

LPCNET_EXPORT int lpcnet_plc_load_model(LPCNetPLCState *st, const unsigned char *data, int len) {
  WeightArray *list;
  int ret;
  parse_weights(&list, data, len);
  ret = init_plc_model(&st->model, list);
  free(list);
  if (ret == 0) {
    return lpcnet_load_model(&st->lpcnet, data, len);
  }
  else return -1;
}

LPCNET_EXPORT LPCNetPLCState *lpcnet_plc_create(int options) {
  LPCNetPLCState *st;
  st = calloc(sizeof(*st), 1);
  lpcnet_plc_init(st, options);
  return st;
}

LPCNET_EXPORT void lpcnet_plc_destroy(LPCNetPLCState *st) {
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
    RNN_MOVE(&st->fec[0][0], &st->fec[st->fec_keep_pos][0], (st->fec_fill_pos-st->fec_keep_pos)*NB_FEATURES);
    st->fec_fill_pos = st->fec_fill_pos-st->fec_keep_pos;
    st->fec_read_pos -= st->fec_keep_pos;
    st->fec_keep_pos = 0;
  }
  RNN_COPY(&st->fec[st->fec_fill_pos][0], features, NB_FEATURES);
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
  /* Artificially boost the correlation to make harmonics cleaner. */
  out[19] = MIN16(.5f, out[19]+.1f);
}

static int get_fec_or_pred(LPCNetPLCState *st, float *out) {
  if (st->fec_read_pos != st->fec_fill_pos && st->fec_skip==0) {
    float plc_features[2*NB_BANDS+NB_FEATURES+1] = {0};
    float discard[NB_FEATURES];
    RNN_COPY(out, &st->fec[st->fec_read_pos][0], NB_FEATURES);
    st->fec_read_pos++;
    /* Make sure we can rewind a few frames back at resync time. */
    st->fec_keep_pos = IMAX(0, IMAX(st->fec_keep_pos, st->fec_read_pos-FEATURES_DELAY-1));
    /* Update PLC state using FEC, so without Burg features. */
    RNN_COPY(&plc_features[2*NB_BANDS], out, NB_FEATURES);
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

static void fec_rewind(LPCNetPLCState *st, int offset) {
  st->fec_read_pos -= offset;
  if (st->fec_read_pos < st->fec_keep_pos) {
    st->fec_read_pos = st->fec_keep_pos;
  }
}

void clear_state(LPCNetPLCState *st) {
  RNN_CLEAR(st->lpcnet.last_sig, LPC_ORDER);
  st->lpcnet.last_exc = lin2ulaw(0.f);
  st->lpcnet.deemph_mem = 0;
  RNN_CLEAR(st->lpcnet.nnet.gru_a_state, GRU_A_STATE_SIZE);
  RNN_CLEAR(st->lpcnet.nnet.gru_b_state, GRU_B_STATE_SIZE);
}

#define DC_CONST 0.003

/* In this causal version of the code, the DNN model implemented by compute_plc_pred()
   needs to generate two feature vectors to conceal the first lost packet.*/

static int lpcnet_plc_update_causal(LPCNetPLCState *st, short *pcm) {
  int i;
  float x[FRAME_SIZE];
  short output[FRAME_SIZE];
  float plc_features[2*NB_BANDS+NB_FEATURES+1];
  short lp[FRAME_SIZE]={0};
  int delta = 0;
  if (st->remove_dc) {
    st->dc_mem += st->syn_dc;
    delta = st->syn_dc;
    st->syn_dc = 0;
    for (i=0;i<FRAME_SIZE;i++) {
      lp[i] = (int)floor(.5 + st->dc_mem);
      st->dc_mem += DC_CONST*(pcm[i] - st->dc_mem);
      pcm[i] -= lp[i];
    }
  }
  for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
  burg_cepstral_analysis(plc_features, x);
  st->enc.pcount = 0;
  if (st->skip_analysis) {
    /*fprintf(stderr, "skip update\n");*/
    if (st->blend) {
      short tmp[FRAME_SIZE-TRAINING_OFFSET];
      float zeros[2*NB_BANDS+NB_FEATURES+1] = {0};
      RNN_COPY(zeros, plc_features, 2*NB_BANDS);
      zeros[2*NB_BANDS+NB_FEATURES] = 1;
      if (st->enable_blending) {
        LPCNetState copy;
        st->plc_net = st->plc_copy[FEATURES_DELAY];
        compute_plc_pred(st, st->features, zeros);
        for (i=0;i<FEATURES_DELAY;i++) {
          /* FIXME: backtrack state, replace features. */
          run_frame_network_deferred(&st->lpcnet, st->features);
        }
        copy = st->lpcnet;
        lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], tmp, FRAME_SIZE-TRAINING_OFFSET, 0);
        for (i=0;i<FRAME_SIZE-TRAINING_OFFSET;i++) {
          float w;
          w = .5 - .5*cos(M_PI*i/(FRAME_SIZE-TRAINING_OFFSET));
          pcm[i] = (int)floor(.5 + w*pcm[i] + (1-w)*(tmp[i]-delta));
        }
        st->lpcnet = copy;
        lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], pcm, FRAME_SIZE-TRAINING_OFFSET, FRAME_SIZE-TRAINING_OFFSET);
      } else {
        if (FEATURES_DELAY > 0) st->plc_net = st->plc_copy[FEATURES_DELAY-1];
        fec_rewind(st, FEATURES_DELAY);
#ifdef PLC_SKIP_UPDATES
        lpcnet_reset_signal(&st->lpcnet);
#else
        RNN_COPY(tmp, pcm, FRAME_SIZE-TRAINING_OFFSET);
        lpcnet_synthesize_tail_impl(&st->lpcnet, tmp, FRAME_SIZE-TRAINING_OFFSET, FRAME_SIZE-TRAINING_OFFSET);
#endif
      }
      RNN_COPY(st->pcm, &pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET);
      st->pcm_fill = TRAINING_OFFSET;
    } else {
      RNN_COPY(&st->pcm[st->pcm_fill], pcm, FRAME_SIZE);
      st->pcm_fill += FRAME_SIZE;
    }
  }
  /* Update state. */
  /*fprintf(stderr, "update state\n");*/
  for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
  preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
  compute_frame_features(&st->enc, x);
  process_single_frame(&st->enc, NULL);
  if (!st->blend) {
    RNN_COPY(&plc_features[2*NB_BANDS], st->enc.features[0], NB_FEATURES);
    plc_features[2*NB_BANDS+NB_FEATURES] = 1;
    compute_plc_pred(st, st->features, plc_features);
    /* Discard an FEC frame that we know we will no longer need. */
    if (st->fec_skip) st->fec_skip--;
    else if (st->fec_read_pos < st->fec_fill_pos) st->fec_read_pos++;
    st->fec_keep_pos = IMAX(0, IMAX(st->fec_keep_pos, st->fec_read_pos-FEATURES_DELAY-1));
  }
  if (st->skip_analysis) {
    if (st->enable_blending) {
      /* FIXME: backtrack state, replace features. */
      run_frame_network_deferred(&st->lpcnet, st->enc.features[0]);
    }
    st->skip_analysis--;
  } else {
    for (i=0;i<FRAME_SIZE;i++) st->pcm[PLC_BUF_SIZE+i] = pcm[i];
    RNN_COPY(output, &st->pcm[0], FRAME_SIZE);
#ifdef PLC_SKIP_UPDATES
    {
      run_frame_network_deferred(&st->lpcnet, st->enc.features[0]);
    }
#else
    lpcnet_synthesize_impl(&st->lpcnet, st->enc.features[0], output, FRAME_SIZE, FRAME_SIZE);
#endif
    RNN_MOVE(st->pcm, &st->pcm[FRAME_SIZE], PLC_BUF_SIZE);
  }
  st->loss_count = 0;
  if (st->remove_dc) {
    for (i=0;i<FRAME_SIZE;i++) {
      pcm[i] += lp[i];
    }
  }
  st->blend = 0;
  return 0;
}

static const float att_table[10] = {0, 0,  -.2, -.2,  -.4, -.4,  -.8, -.8, -1.6, -1.6};
static int lpcnet_plc_conceal_causal(LPCNetPLCState *st, short *pcm) {
  int i;
  short output[FRAME_SIZE];
  run_frame_network_flush(&st->lpcnet);
  st->enc.pcount = 0;
  /* If we concealed the previous frame, finish synthesizing the rest of the samples. */
  /* FIXME: Copy/predict features. */
  while (st->pcm_fill > 0) {
    /*fprintf(stderr, "update state for PLC %d\n", st->pcm_fill);*/
    int update_count;
    update_count = IMIN(st->pcm_fill, FRAME_SIZE);
    RNN_COPY(output, &st->pcm[0], update_count);
    RNN_MOVE(&st->plc_copy[1], &st->plc_copy[0], FEATURES_DELAY);
    st->plc_copy[0] = st->plc_net;
    get_fec_or_pred(st, st->features);
    lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], output, update_count, update_count);
    RNN_MOVE(st->pcm, &st->pcm[FRAME_SIZE], PLC_BUF_SIZE);
    st->pcm_fill -= update_count;
    st->skip_analysis++;
  }
  RNN_MOVE(&st->plc_copy[1], &st->plc_copy[0], FEATURES_DELAY);
  st->plc_copy[0] = st->plc_net;
  lpcnet_synthesize_tail_impl(&st->lpcnet, pcm, FRAME_SIZE-TRAINING_OFFSET, 0);
  if (get_fec_or_pred(st, st->features)) st->loss_count = 0;
  else st->loss_count++;
  if (st->loss_count >= 10) st->features[0] = MAX16(-10, st->features[0]+att_table[9] - 2*(st->loss_count-9));
  else st->features[0] = MAX16(-10, st->features[0]+att_table[st->loss_count]);
  lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], &pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET, 0);
  {
    float x[FRAME_SIZE];
    /* FIXME: Can we do better? */
    for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
    preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(&st->enc, x);
    process_single_frame(&st->enc, NULL);
  }
  st->blend = 1;
  if (st->remove_dc) {
    for (i=0;i<FRAME_SIZE;i++) {
      st->syn_dc += DC_CONST*(pcm[i] - st->syn_dc);
      pcm[i] += (int)floor(.5 + st->dc_mem);
    }
  }
  return 0;
}

/* In this non-causal version of the code, the DNN model implemented by compute_plc_pred()
   is always called once per frame. We process audio up to the current position minus TRAINING_OFFSET. */

void process_queued_update(LPCNetPLCState *st) {
  if (st->queued_update) {
    lpcnet_synthesize_impl(&st->lpcnet, st->features, st->queued_samples, FRAME_SIZE, FRAME_SIZE);
    st->queued_update=0;
  }
}

static int lpcnet_plc_update_non_causal(LPCNetPLCState *st, short *pcm) {
  int i;
  float x[FRAME_SIZE];
  short pcm_save[FRAME_SIZE];
  float plc_features[2*NB_BANDS+NB_FEATURES+1];
  short lp[FRAME_SIZE]={0};
  double mem_bak=0;
  int delta = st->syn_dc;
  if (FEATURES_DELAY != 0) {
    fprintf(stderr, "Non-causal PLC cannot work with non-zero FEATURES_DELAY\n");
    fprintf(stderr, "Recompile with a no-lookahead model (see README.md)\n");
    exit(1);
  }
  process_queued_update(st);
  if (st->remove_dc) {
    st->dc_mem += st->syn_dc;
    st->syn_dc = 0;
    mem_bak = st->dc_mem;
    for (i=0;i<FRAME_SIZE;i++) {
      lp[i] = (int)floor(.5 + st->dc_mem);
      st->dc_mem += DC_CONST*(pcm[i] - st->dc_mem);
      pcm[i] -= lp[i];
    }
  }
  RNN_COPY(pcm_save, pcm, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
  burg_cepstral_analysis(plc_features, x);
  st->enc.pcount = 0;
  if (st->loss_count > 0) {
    LPCNetState copy;
    /* Handle blending. */
    float zeros[2*NB_BANDS+NB_FEATURES+1] = {0};
    RNN_COPY(zeros, plc_features, 2*NB_BANDS);
    zeros[2*NB_BANDS+NB_FEATURES] = 1;
    compute_plc_pred(st, st->features, zeros);
    copy = st->lpcnet;
    lpcnet_synthesize_impl(&st->lpcnet, st->features, &st->pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET, 0);
    /* Undo initial DC offset removal so that we can take into account the last 5ms of synthesis. */
    if (st->remove_dc) {
      for (i=0;i<FRAME_SIZE;i++) pcm[i] += lp[i];
      st->dc_mem = mem_bak;
      for (i=0;i<TRAINING_OFFSET;i++) st->syn_dc += DC_CONST*(st->pcm[FRAME_SIZE-TRAINING_OFFSET+i] - st->syn_dc);
      st->dc_mem += st->syn_dc;
      delta += st->syn_dc;
      st->syn_dc = 0;
      for (i=0;i<FRAME_SIZE;i++) {
        lp[i] = (int)floor(.5 + st->dc_mem);
        st->dc_mem += DC_CONST*(pcm[i] - st->dc_mem);
        pcm[i] -= lp[i];
      }
      RNN_COPY(pcm_save, pcm, FRAME_SIZE);
    }
    {
      short rev[FRAME_SIZE];
      for (i=0;i<FRAME_SIZE;i++) rev[i] = pcm[FRAME_SIZE-i-1];
      clear_state(st);
      lpcnet_synthesize_impl(&st->lpcnet, st->features, rev, FRAME_SIZE, FRAME_SIZE);
      lpcnet_synthesize_tail_impl(&st->lpcnet, rev, TRAINING_OFFSET, 0);
      for (i=0;i<TRAINING_OFFSET;i++) {
        float w;
        w = .5 - .5*cos(M_PI*i/(TRAINING_OFFSET));
        st->pcm[FRAME_SIZE-1-i] = (int)floor(.5 + w*st->pcm[FRAME_SIZE-1-i] + (1-w)*(rev[i]+delta));
      }
      
    }
    st->lpcnet = copy;
#if 1
    st->queued_update = 1;
    RNN_COPY(&st->queued_samples[0], &st->pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET);
    RNN_COPY(&st->queued_samples[TRAINING_OFFSET], pcm, FRAME_SIZE-TRAINING_OFFSET);
#else
    lpcnet_synthesize_impl(&st->lpcnet, st->features, &st->pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET, TRAINING_OFFSET);
    lpcnet_synthesize_tail_impl(&st->lpcnet, pcm, FRAME_SIZE-TRAINING_OFFSET, FRAME_SIZE-TRAINING_OFFSET);
#endif
    for (i=0;i<FRAME_SIZE;i++) x[i] = st->pcm[i];
    preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(&st->enc, x);
    process_single_frame(&st->enc, NULL);
    
  }
  for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
  preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
  compute_frame_features(&st->enc, x);
  process_single_frame(&st->enc, NULL);
  if (st->loss_count == 0) {
    RNN_COPY(&plc_features[2*NB_BANDS], st->enc.features[0], NB_FEATURES);
    plc_features[2*NB_BANDS+NB_FEATURES] = 1;
    compute_plc_pred(st, st->features, plc_features);
    lpcnet_synthesize_impl(&st->lpcnet, st->enc.features[0], &st->pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET, TRAINING_OFFSET);
    lpcnet_synthesize_tail_impl(&st->lpcnet, pcm, FRAME_SIZE-TRAINING_OFFSET, FRAME_SIZE-TRAINING_OFFSET);
  }
  RNN_COPY(&pcm[FRAME_SIZE-TRAINING_OFFSET], pcm, TRAINING_OFFSET);
  RNN_COPY(pcm, &st->pcm[TRAINING_OFFSET], FRAME_SIZE-TRAINING_OFFSET);
  RNN_COPY(st->pcm, pcm_save, FRAME_SIZE);
  st->loss_count = 0;
  if (st->remove_dc) {
    for (i=0;i<TRAINING_OFFSET;i++) pcm[i] += st->dc_buf[i];
    for (;i<FRAME_SIZE;i++) pcm[i] += lp[i-TRAINING_OFFSET];
    for (i=0;i<TRAINING_OFFSET;i++) st->dc_buf[i] = lp[FRAME_SIZE-TRAINING_OFFSET+i];
  }
  return 0;
}

static int lpcnet_plc_conceal_non_causal(LPCNetPLCState *st, short *pcm) {
  int i;
  float x[FRAME_SIZE];
  float zeros[2*NB_BANDS+NB_FEATURES+1] = {0};
  process_queued_update(st);
  st->enc.pcount = 0;

  compute_plc_pred(st, st->features, zeros);
  if (st->loss_count >= 10) st->features[0] = MAX16(-10, st->features[0]+att_table[9] - 2*(st->loss_count-9));
  else st->features[0] = MAX16(-10, st->features[0]+att_table[st->loss_count]);

  if (st->loss_count == 0) {
    RNN_COPY(pcm, &st->pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET);
    lpcnet_synthesize_impl(&st->lpcnet, st->features, &st->pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET, TRAINING_OFFSET);
    lpcnet_synthesize_tail_impl(&st->lpcnet, &pcm[TRAINING_OFFSET], FRAME_SIZE-TRAINING_OFFSET, 0);
  } else {
    lpcnet_synthesize_impl(&st->lpcnet, st->features, pcm, TRAINING_OFFSET, 0);
    lpcnet_synthesize_tail_impl(&st->lpcnet, &pcm[TRAINING_OFFSET], FRAME_SIZE-TRAINING_OFFSET, 0);

    RNN_COPY(&st->pcm[FRAME_SIZE-TRAINING_OFFSET], pcm, TRAINING_OFFSET);
    for (i=0;i<FRAME_SIZE;i++) x[i] = st->pcm[i];
    preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(&st->enc, x);
    process_single_frame(&st->enc, NULL);
  }
  RNN_COPY(st->pcm, &pcm[TRAINING_OFFSET], FRAME_SIZE-TRAINING_OFFSET);

  if (st->remove_dc) {
    int dc = (int)floor(.5 + st->dc_mem);
    if (st->loss_count == 0) {
        for (i=TRAINING_OFFSET;i<FRAME_SIZE;i++) st->syn_dc += DC_CONST*(pcm[i] - st->syn_dc);
    } else {
        for (i=0;i<FRAME_SIZE;i++) st->syn_dc += DC_CONST*(pcm[i] - st->syn_dc);
    }
    for (i=0;i<TRAINING_OFFSET;i++) pcm[i] += st->dc_buf[i];
    for (;i<FRAME_SIZE;i++) pcm[i] += dc;
    for (i=0;i<TRAINING_OFFSET;i++) st->dc_buf[i] = dc;
  }
  st->loss_count++;
  return 0;
}


LPCNET_EXPORT int lpcnet_plc_update(LPCNetPLCState *st, short *pcm) {
  if (st->non_causal) return lpcnet_plc_update_non_causal(st, pcm);
  else return lpcnet_plc_update_causal(st, pcm);
}

LPCNET_EXPORT int lpcnet_plc_conceal(LPCNetPLCState *st, short *pcm) {
  if (st->non_causal) return lpcnet_plc_conceal_non_causal(st, pcm);
  else return lpcnet_plc_conceal_causal(st, pcm);
}
