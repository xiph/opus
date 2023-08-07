#ifndef neural_pitch_h
#define neural_pitch_h

#include <stdlib.h>

#include "opus_types.h"
#include "neural_pitch_data.h"
#include "kiss_fft.h"

#define PITCH_FFT_SIZE (320)
#define NEURAL_PITCH_FRAME_SIZE (160)
#define PITCH_NET_OUTPUT (192)
#define IF_DIMENSION (90)

// Struct for IF computation state
typedef struct {
    kiss_fft_cpx fft_previous[PITCH_FFT_SIZE/2 + 1];
    float analysis_mem[NEURAL_PITCH_FRAME_SIZE];
} if_state;

// Struct to store Neural Pitch Model State
typedef struct {
    float gru_state[GRU_STATE_SIZE];
    nnpitch model;
    if_state ifstate;
} neural_pitch_model;

void pitch_model(
    neural_pitch_model *npm,
    float *output,
    const float *input             
    );

void IF_computation(
    if_state *state,
    float *output,
    const float *input             
    );

short argmax(float *input);

void npm_init(neural_pitch_model *st);
void if_state_init(if_state *st);
#endif
