/* Copyright (c) 2018 Mozilla */
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

#include <math.h>
#include <stdio.h>
#include "nnet_data.h"
#include "nnet.h"
#include "common.h"
#include "arch.h"
#include "lpcnet.h"

#define NB_FEATURES 38
#define NB_TOTAL_FEATURES 55

#define LPC_ORDER 16
#define PREEMPH 0.85f

#define PITCH_GAIN_FEATURE 37
#define PDF_FLOOR 0.002

#define FRAME_INPUT_SIZE (NB_FEATURES + EMBED_PITCH_OUT_SIZE)

#define SAMPLE_INPUT_SIZE (2*EMBED_SIG_OUT_SIZE + EMBED_EXC_OUT_SIZE + FEATURE_DENSE2_OUT_SIZE)

#define FEATURES_DELAY (FEATURE_CONV1_DELAY + FEATURE_CONV2_DELAY)
struct LPCNetState {
    NNetState nnet;
    int last_exc;
    float last_sig[LPC_ORDER];
    float old_input[FEATURES_DELAY][FEATURE_CONV2_OUT_SIZE];
    float old_lpc[FEATURES_DELAY][LPC_ORDER];
    int frame_count;
    float deemph_mem;
};


static float ulaw2lin(float u)
{
    float s;
    float scale_1 = 32768.f/255.f;
    u = u - 128;
    s = u >= 0 ? 1 : -1;
    u = fabs(u);
    return s*scale_1*(exp(u/128.*log(256))-1);
}

static int lin2ulaw(float x)
{
    float u;
    float scale = 255.f/32768.f;
    int s = x >= 0 ? 1 : -1;
    x = fabs(x);
    u = (s*(128*log(1+scale*x)/log(256)));
    u = 128 + u;
    if (u < 0) u = 0;
    if (u > 255) u = 255;
    return (int)floor(.5 + u);
}

#if 0
static void print_vector(float *x, int N)
{
    int i;
    for (i=0;i<N;i++) printf("%f ", x[i]);
    printf("\n");
}
#endif

void run_frame_network(LPCNetState *lpcnet, float *condition, float *gru_a_condition, const float *features, int pitch)
{
    int i;
    NNetState *net;
    float in[FRAME_INPUT_SIZE];
    float conv1_out[FEATURE_CONV1_OUT_SIZE];
    float conv2_out[FEATURE_CONV2_OUT_SIZE];
    float dense1_out[FEATURE_DENSE1_OUT_SIZE];
    net = &lpcnet->nnet;
    RNN_COPY(in, features, NB_FEATURES);
    compute_embedding(&embed_pitch, &in[NB_FEATURES], pitch);
    compute_conv1d(&feature_conv1, conv1_out, net->feature_conv1_state, in);
    if (lpcnet->frame_count < FEATURE_CONV1_DELAY) RNN_CLEAR(conv1_out, FEATURE_CONV1_OUT_SIZE);
    compute_conv1d(&feature_conv2, conv2_out, net->feature_conv2_state, conv1_out);
    celt_assert(FRAME_INPUT_SIZE == FEATURE_CONV2_OUT_SIZE);
    if (lpcnet->frame_count < FEATURES_DELAY) RNN_CLEAR(conv2_out, FEATURE_CONV2_OUT_SIZE);
    for (i=0;i<FEATURE_CONV2_OUT_SIZE;i++) conv2_out[i] += lpcnet->old_input[FEATURES_DELAY-1][i];
    memmove(lpcnet->old_input[1], lpcnet->old_input[0], (FEATURES_DELAY-1)*FRAME_INPUT_SIZE*sizeof(in[0]));
    memcpy(lpcnet->old_input[0], in, FRAME_INPUT_SIZE*sizeof(in[0]));
    compute_dense(&feature_dense1, dense1_out, conv2_out);
    compute_dense(&feature_dense2, condition, dense1_out);
    compute_dense(&gru_a_dense_feature, gru_a_condition, condition);
    if (lpcnet->frame_count < 1000) lpcnet->frame_count++;
}

void run_sample_network(NNetState *net, float *pdf, const float *condition, const float *gru_a_condition, int last_exc, int last_sig, int pred)
{
    float gru_a_input[3*GRU_A_STATE_SIZE];
    float in_b[GRU_A_STATE_SIZE+FEATURE_DENSE2_OUT_SIZE];
    RNN_COPY(gru_a_input, gru_a_condition, 3*GRU_A_STATE_SIZE);
    accum_embedding(&gru_a_embed_sig, gru_a_input, last_sig);
    accum_embedding(&gru_a_embed_pred, gru_a_input, pred);
    accum_embedding(&gru_a_embed_exc, gru_a_input, last_exc);
    /*compute_gru3(&gru_a, net->gru_a_state, gru_a_input);*/
    compute_sparse_gru(&sparse_gru_a, net->gru_a_state, gru_a_input);
    RNN_COPY(in_b, net->gru_a_state, GRU_A_STATE_SIZE);
    RNN_COPY(&in_b[GRU_A_STATE_SIZE], condition, FEATURE_DENSE2_OUT_SIZE);
    compute_gru2(&gru_b, net->gru_b_state, in_b);
    compute_mdense(&dual_fc, pdf, net->gru_b_state);
}

LPCNetState *lpcnet_create()
{
    LPCNetState *lpcnet;
    lpcnet = (LPCNetState *)calloc(sizeof(LPCNetState), 1);
    return lpcnet;
}

void lpcnet_destroy(LPCNetState *lpcnet)
{
    free(lpcnet);
}

void lpcnet_synthesize(LPCNetState *lpcnet, short *output, const float *features, const float *new_lpc, int N)
{
    int i;
    float condition[FEATURE_DENSE2_OUT_SIZE];
    float lpc[LPC_ORDER];
    float pdf[DUAL_FC_OUT_SIZE];
    float gru_a_condition[3*GRU_A_STATE_SIZE];
    int pitch;
    float pitch_gain;
    /* FIXME: Remove this -- it's just a temporary hack to match the Python code. */
    static int start = LPC_ORDER+1;
    /* FIXME: Do proper rounding once the Python code rounds properly. */
    pitch = (int)floor(50*features[36]+100);
    /* FIXME: get the pitch gain from 2 frames in the past. */
    pitch_gain = features[PITCH_GAIN_FEATURE];
    run_frame_network(lpcnet, condition, gru_a_condition, features, pitch);
    memcpy(lpc, lpcnet->old_lpc[FEATURES_DELAY-1], LPC_ORDER*sizeof(lpc[0]));
    memmove(lpcnet->old_lpc[1], lpcnet->old_lpc[0], (FEATURES_DELAY-1)*LPC_ORDER*sizeof(lpc[0]));
    memcpy(lpcnet->old_lpc[0], new_lpc, LPC_ORDER*sizeof(lpc[0]));
    if (lpcnet->frame_count <= FEATURES_DELAY)
    {
        RNN_CLEAR(output, N);
        return;
    }
    for (i=start;i<N;i++)
    {
        int j;
        float pcm;
        int exc;
        int last_sig_ulaw;
        int pred_ulaw;
        float pred = 0;
        for (j=0;j<LPC_ORDER;j++) pred -= lpcnet->last_sig[j]*lpc[j];
        last_sig_ulaw = lin2ulaw(lpcnet->last_sig[0]);
        pred_ulaw = lin2ulaw(pred);
        run_sample_network(&lpcnet->nnet, pdf, condition, gru_a_condition, lpcnet->last_exc, last_sig_ulaw, pred_ulaw);
        exc = sample_from_pdf(pdf, DUAL_FC_OUT_SIZE, MAX16(0, 1.5f*pitch_gain - .5f), PDF_FLOOR);
        pcm = pred + ulaw2lin(exc);
        RNN_MOVE(&lpcnet->last_sig[1], &lpcnet->last_sig[0], LPC_ORDER-1);
        lpcnet->last_sig[0] = pcm;
        lpcnet->last_exc = exc;
        pcm += PREEMPH*lpcnet->deemph_mem;
        lpcnet->deemph_mem = pcm;
        if (pcm<-32767) pcm = -32767;
        if (pcm>32767) pcm = 32767;
        output[i] = (int)floor(.5 + pcm);
    }
    start = 0;
}

#if 1
#define FRAME_SIZE 160
int main(int argc, char **argv) {
    FILE *fin, *fout;
    LPCNetState *net;
    net = lpcnet_create();
    if (argc != 3)
    {
        fprintf(stderr, "usage: test_lpcnet <features.f32> <output.pcm>\n");
        return 0;
    }
    fin = fopen(argv[1], "rb");
    fout = fopen(argv[2], "wb");
    while (1) {
        float in_features[NB_TOTAL_FEATURES];
        float features[NB_FEATURES];
        float lpc[LPC_ORDER];
        short pcm[FRAME_SIZE];
        fread(in_features, sizeof(features[0]), NB_TOTAL_FEATURES, fin);
        RNN_COPY(features, in_features, NB_FEATURES);
        RNN_CLEAR(&features[18], 18);
        RNN_COPY(lpc, &in_features[NB_TOTAL_FEATURES-LPC_ORDER], LPC_ORDER);
        if (feof(fin)) break;
        lpcnet_synthesize(net, pcm, features, lpc, FRAME_SIZE);
        fwrite(pcm, sizeof(pcm[0]), FRAME_SIZE, fout);
    }
    fclose(fin);
    fclose(fout);
    lpcnet_destroy(net);
    return 0;
}
#endif
