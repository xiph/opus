#ifndef LPCNET_PRIVATE_H
#define LPCNET_PRIVATE_H

#include <stdio.h>
#include "common.h"
#include "freq.h"
#include "lpcnet.h"
#include "nnet_data.h"
#include "plc_data.h"
#include "kiss99.h"

#define BITS_PER_CHAR 8

#define PITCH_MIN_PERIOD 32
#define PITCH_MAX_PERIOD 256

#define PITCH_FRAME_SIZE 320
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

#define MULTI 4
#define MULTI_MASK (MULTI-1)

#define FORBIDDEN_INTERP 7

#define PLC_MAX_FEC 100

struct LPCNetState {
    NNetState nnet;
    int last_exc;
    float last_sig[LPC_ORDER];
    float last_features[NB_FEATURES];
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
  float burg_cepstrum[2*NB_BANDS];
};

#define PLC_BUF_SIZE (FEATURES_DELAY*FRAME_SIZE + TRAINING_OFFSET)
struct LPCNetPLCState {
  LPCNetState lpcnet;
  LPCNetEncState enc;
  float fec[PLC_MAX_FEC][NB_FEATURES];
  int fec_keep_pos;
  int fec_read_pos;
  int fec_fill_pos;
  int fec_active;
  short pcm[PLC_BUF_SIZE+FRAME_SIZE];
  int pcm_fill;
  int skip_analysis;
  int blend;
  float features[NB_TOTAL_FEATURES];
  int loss_count;
  PLCNetState plc_net;
  PLCNetState plc_copy[FEATURES_DELAY+1];
  int enable_blending;
  int non_causal;
  double dc_mem;
  double syn_dc;
  int remove_dc;

  short dc_buf[TRAINING_OFFSET];
  int queued_update;
  short queued_samples[FRAME_SIZE];
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
