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

#ifndef FWGAN_H
#define FWGAN_H

#include "freq.h"
#include "fwgan_data.h"

#define FWGAN_CONT_SAMPLES 320

/* FIXME: Derive those from the model rather than hardcoding. */
#define FWC1_STATE_SIZE 512
#define FWC2_STATE_SIZE 512
#define FWC3_STATE_SIZE 256
#define FWC4_STATE_SIZE 256
#define FWC5_STATE_SIZE 128
#define FWC6_STATE_SIZE 128
#define FWC7_STATE_SIZE 80

typedef struct {
  FWGAN model;
  int arch;
  int cont_initialized;
  double embed_phase;
  float last_gain;
  float syn_mem[LPC_ORDER];
  float deemph_mem;
  float cont_conv1_mem[FEAT_IN_CONV1_CONV_STATE_SIZE];
  float cont[FEAT_IN_NL1_GATE_OUT_SIZE];
  float rnn_state[RNN_GRU_STATE_SIZE];
  float fwc1_state[FWC1_STATE_SIZE];
  float fwc2_state[FWC2_STATE_SIZE];
  float fwc3_state[FWC3_STATE_SIZE];
  float fwc4_state[FWC4_STATE_SIZE];
  float fwc5_state[FWC5_STATE_SIZE];
  float fwc6_state[FWC6_STATE_SIZE];
  float fwc7_state[FWC7_STATE_SIZE];
} FWGANState;

void fwgan_init(FWGANState *st);

void fwgan_cont(FWGANState *st, const float *pcm0, const float *features0);

void fwgan_synthesize(FWGANState *st, float *pcm, const float *features);


#endif /* FWGAN_H */
