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

LPCNET_EXPORT int lpcnet_plc_get_size() {
  return sizeof(LPCNetPLCState);
}

LPCNET_EXPORT void lpcnet_plc_init(LPCNetPLCState *st) {
  lpcnet_init(&st->lpcnet);
  lpcnet_encoder_init(&st->enc);
  RNN_CLEAR(st->pcm, PLC_BUF_SIZE);
  st->pcm_fill = PLC_BUF_SIZE;
  st->skip_analysis = 0;
  st->blend = 0;
}

LPCNET_EXPORT LPCNetPLCState *lpcnet_plc_create() {
  LPCNetPLCState *st;
  st = malloc(sizeof(*st));
  lpcnet_plc_init(st);
  return st;
}

LPCNET_EXPORT void lpcnet_plc_destroy(LPCNetPLCState *st) {
  free(st);
}

LPCNET_EXPORT int lpcnet_plc_update(LPCNetPLCState *st, short *pcm) {
  int i;
  float x[FRAME_SIZE];
  short output[FRAME_SIZE];
  st->enc.pcount = 0;
  if (st->skip_analysis) {
    /*fprintf(stderr, "skip update\n");*/
    if (st->blend) {
      short tmp[FRAME_SIZE-TRAINING_OFFSET];
      lpcnet_synthesize_tail_impl(&st->lpcnet, tmp, FRAME_SIZE-TRAINING_OFFSET, 0);
      for (i=0;i<FRAME_SIZE-TRAINING_OFFSET;i++) {
        float w;
        w = .5 - .5*cos(M_PI*i/(FRAME_SIZE-TRAINING_OFFSET));
        pcm[i] = (int)floor(.5 + w*pcm[i] + (1-w)*tmp[i]);
      }
      st->blend = 0;
      RNN_COPY(st->pcm, &pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET);
      st->pcm_fill = TRAINING_OFFSET;
    } else {
      RNN_COPY(&st->pcm[st->pcm_fill], pcm, FRAME_SIZE);
      st->pcm_fill += FRAME_SIZE;
    }
    /*fprintf(stderr, "fill at %d\n", st->pcm_fill);*/
  }
  /* Update state. */
  /*fprintf(stderr, "update state\n");*/
  for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
  preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
  compute_frame_features(&st->enc, x);
  process_single_frame(&st->enc, NULL);
  if (st->skip_analysis) {
    float lpc[LPC_ORDER];
    float gru_a_condition[3*GRU_A_STATE_SIZE];
    float gru_b_condition[3*GRU_B_STATE_SIZE];
    /* FIXME: backtrack state, replace features. */
    run_frame_network(&st->lpcnet, gru_a_condition, gru_b_condition, lpc, st->enc.features[0]);
    st->skip_analysis--;
  } else {
    for (i=0;i<FRAME_SIZE;i++) st->pcm[PLC_BUF_SIZE+i] = pcm[i];
    RNN_COPY(output, &st->pcm[0], FRAME_SIZE);
    lpcnet_synthesize_impl(&st->lpcnet, st->enc.features[0], output, FRAME_SIZE, FRAME_SIZE);

    RNN_MOVE(st->pcm, &st->pcm[FRAME_SIZE], PLC_BUF_SIZE);
  }
  RNN_COPY(st->features, st->enc.features[0], NB_TOTAL_FEATURES);
  return 0;
}

LPCNET_EXPORT int lpcnet_plc_conceal(LPCNetPLCState *st, short *pcm) {
  short output[FRAME_SIZE];
  st->enc.pcount = 0;
  /* If we concealed the previous frame, finish synthesizing the rest of the samples. */
  /* FIXME: Copy/predict features. */
  while (st->pcm_fill > 0) {
    /*fprintf(stderr, "update state for PLC %d\n", st->pcm_fill);*/
    int update_count;
    update_count = IMIN(st->pcm_fill, FRAME_SIZE);
    RNN_COPY(output, &st->pcm[0], update_count);

    lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], output, update_count, update_count);
    RNN_MOVE(st->pcm, &st->pcm[FRAME_SIZE], PLC_BUF_SIZE);
    st->pcm_fill -= update_count;
    st->skip_analysis++;
  }
  lpcnet_synthesize_tail_impl(&st->lpcnet, pcm, FRAME_SIZE-TRAINING_OFFSET, 0);
  lpcnet_synthesize_impl(&st->lpcnet, &st->features[0], &pcm[FRAME_SIZE-TRAINING_OFFSET], TRAINING_OFFSET, 0);
  {
    int i;
    float x[FRAME_SIZE];
    /* FIXME: Can we do better? */
    for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[i];
    preemphasis(x, &st->enc.mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(&st->enc, x);
    process_single_frame(&st->enc, NULL);
  }
  st->blend = 1;
  return 0;
}
