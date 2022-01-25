#ifndef LPCNET_PRIVATE_H
#define LPCNET_PRIVATE_H

#include <stdio.h>
#include "common.h"
#include "freq.h"
#include "lpcnet.h"
#include "nnet_data.h"
#include "kiss99.h"

#define BITS_PER_CHAR 8

#define PITCH_MIN_PERIOD 32
#define PITCH_MAX_PERIOD 256

#define PITCH_FRAME_SIZE 320
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

#define MULTI 4
#define MULTI_MASK (MULTI-1)

#define FORBIDDEN_INTERP 7

#define FEATURES_DELAY (FEATURE_CONV1_DELAY + FEATURE_CONV2_DELAY)

struct LPCNetState {
    NNetState nnet;
    int last_exc;
    float last_sig[LPC_ORDER];
#if FEATURES_DELAY>0
    float old_lpc[FEATURES_DELAY][LPC_ORDER];
#endif
    float sampling_logit_table[256];
    float gru_a_condition[3*GRU_A_STATE_SIZE];
    float gru_b_condition[3*GRU_B_STATE_SIZE];
    int frame_count;
    float deemph_mem;
    float lpc[LPC_ORDER];
    kiss99_ctx rng;
};

struct LPCNetDecState {
    LPCNetState lpcnet_state;
    float vq_mem[NB_BANDS];
};

struct LPCNetEncState{
  float analysis_mem[OVERLAP_SIZE];
  float mem_preemph;
  int pcount;
  float pitch_mem[LPC_ORDER];
  float pitch_filt;
  float xc[10][PITCH_MAX_PERIOD+1];
  float frame_weight[10];
  float exc_buf[PITCH_BUF_SIZE];
  float pitch_max_path[2][PITCH_MAX_PERIOD];
  float pitch_max_path_all;
  int best_i;
  float last_gain;
  int last_period;
  float lpc[LPC_ORDER];
  float vq_mem[NB_BANDS];
  float features[4][NB_TOTAL_FEATURES];
  float sig_mem[LPC_ORDER];
  int exc_mem;
};

#define PLC_BUF_SIZE (FEATURES_DELAY*FRAME_SIZE + TRAINING_OFFSET)
struct LPCNetPLCState {
  LPCNetState lpcnet;
  LPCNetEncState enc;
  short pcm[PLC_BUF_SIZE+FRAME_SIZE];
  int pcm_fill;
  int skip_analysis;
  int blend;
  float features[NB_TOTAL_FEATURES];
};

extern float ceps_codebook1[];
extern float ceps_codebook2[];
extern float ceps_codebook3[];
extern float ceps_codebook_diff4[];

void preemphasis(float *y, float *mem, const float *x, float coef, int N);

void perform_double_interp(float features[4][NB_TOTAL_FEATURES], const float *mem, int best_id);

void process_superframe(LPCNetEncState *st, unsigned char *buf, FILE *ffeat, int encode, int quantize);

void compute_frame_features(LPCNetEncState *st, const float *in);

void decode_packet(float features[4][NB_TOTAL_FEATURES], float *vq_mem, const unsigned char buf[8]);

void run_frame_network(LPCNetState *lpcnet, float *gru_a_condition, float *gru_b_condition, float *lpc, const float *features);
void lpcnet_synthesize_tail_impl(LPCNetState *lpcnet, short *output, int N, int preload);
void lpcnet_synthesize_impl(LPCNetState *lpcnet, const float *features, short *output, int N, int preload);
void lpcnet_synthesize_blend_impl(LPCNetState *lpcnet, const short *pcm_in, short *output, int N);
void process_single_frame(LPCNetEncState *st, FILE *ffeat);
int lpcnet_compute_single_frame_features(LPCNetEncState *st, const short *pcm, float features[NB_TOTAL_FEATURES]);

void process_single_frame(LPCNetEncState *st, FILE *ffeat);

void run_frame_network(LPCNetState *lpcnet, float *gru_a_condition, float *gru_b_condition, float *lpc, const float *features);
#endif
