#ifndef LPCNET_PRIVATE_H
#define LPCNET_PRIVATE_H

#include <stdio.h>
#include "freq.h"
#include "lpcnet.h"
#include "nnet_data.h"
#include "plc_data.h"
#include "kiss99.h"
#include "neural_pitch.h"

#define PITCH_MIN_PERIOD 32
#define PITCH_MAX_PERIOD 256

#define PITCH_FRAME_SIZE 320
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

#define PLC_MAX_FEC 100
#define MAX_FEATURE_BUFFER_SIZE 4

struct LPCNetState {
    LPCNetModel model;
    int arch;
    float sampling_logit_table[256];
    kiss99_ctx rng;

#define LPCNET_RESET_START nnet
    NNetState nnet;
    int last_exc;
    float last_sig[LPC_ORDER];
    float feature_buffer[NB_FEATURES*MAX_FEATURE_BUFFER_SIZE];
    int feature_buffer_fill;
    float last_features[NB_FEATURES];
#if FEATURES_DELAY>0
    float old_lpc[FEATURES_DELAY][LPC_ORDER];
#endif
    float gru_a_condition[3*GRU_A_STATE_SIZE];
    float gru_b_condition[3*GRU_B_STATE_SIZE];
    int frame_count;
    float deemph_mem;
    float lpc[LPC_ORDER];
};

struct LPCNetEncState{
  int arch;
  float analysis_mem[OVERLAP_SIZE];
  float mem_preemph;
  float pitch_mem[LPC_ORDER];
  float pitch_filt;
  float xc[2][PITCH_MAX_PERIOD+1];
  float frame_weight[2];
  float exc_buf[PITCH_BUF_SIZE];
  float pitch_max_path[2][PITCH_MAX_PERIOD];
  float pitch_max_path_all;
  int best_i;
  float last_gain;
  int last_period;
  float lpc[LPC_ORDER];
  float vq_mem[NB_BANDS];
  float features[NB_TOTAL_FEATURES];
  float sig_mem[LPC_ORDER];
  int exc_mem;
  float burg_cepstrum[2*NB_BANDS];
};

#define PLC_BUF_SIZE (FEATURES_DELAY*FRAME_SIZE + FRAME_SIZE)
struct LPCNetPLCState {
  PLCModel model;
  LPCNetState lpcnet;
  LPCNetEncState enc;
  int arch;
  int enable_blending;

#define LPCNET_PLC_RESET_START fec
  float fec[PLC_MAX_FEC][NB_FEATURES];
  int fec_keep_pos;
  int fec_read_pos;
  int fec_fill_pos;
  int fec_skip;
  opus_int16 pcm[PLC_BUF_SIZE+FRAME_SIZE];
  int pcm_fill;
  int skip_analysis;
  int blend;
  float features[NB_TOTAL_FEATURES];
  int loss_count;
  PLCNetState plc_net;
  PLCNetState plc_copy[FEATURES_DELAY+1];
};

void preemphasis(float *y, float *mem, const float *x, float coef, int N);

void compute_frame_features(LPCNetEncState *st, const float *in);
void compute_frame_features_xcorronly(LPCNetEncState *st, const float *in);
int lpcnet_compute_single_frame_features_dump(LPCNetEncState *st, const short *pcm, FILE *fout);

void lpcnet_reset_signal(LPCNetState *lpcnet);
void run_frame_network(LPCNetState *lpcnet, float *gru_a_condition, float *gru_b_condition, float *lpc, const float *features);
void run_frame_network_deferred(LPCNetState *lpcnet, const float *features);
void run_frame_network_flush(LPCNetState *lpcnet);


void lpcnet_synthesize_tail_impl(LPCNetState *lpcnet, opus_int16 *output, int N, int preload);
void lpcnet_synthesize_impl(LPCNetState *lpcnet, const float *features, opus_int16 *output, int N, int preload);
void lpcnet_synthesize_blend_impl(LPCNetState *lpcnet, const opus_int16 *pcm_in, opus_int16 *output, int N);
void process_single_frame(LPCNetEncState *st, FILE *ffeat);
int lpcnet_compute_single_frame_features(LPCNetEncState *st, const opus_int16 *pcm, float features[NB_TOTAL_FEATURES]);

void process_single_frame(LPCNetEncState *st, FILE *ffeat);
void process_single_frame_neuralpitch(LPCNetEncState *st, FILE *ffeat, neural_pitch_model *npm, float *input);

void run_frame_network(LPCNetState *lpcnet, float *gru_a_condition, float *gru_b_condition, float *lpc, const float *features);

int parse_weights(WeightArray **list, const unsigned char *data, int len);
#endif
