#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string.h>
#include <stdio.h>
#include "neural_pitch.h"
#include "os_support.h"
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "freq.h"
#include "pitch.h"
#include "arch.h"
#include <assert.h>
#include "nnet.h"

void pitch_model(
    neural_pitch_model *npm,
    float *output,
    const float *input             
    )
{   
    // First compute IF Features from the input
    float if_features[IF_DIMENSION] = {0.0};
    IF_computation(&npm->ifstate,if_features,input);

    nnpitch *model = &npm->model;
    float buffer[INITIAL_OUT_SIZE] = {0.0};

    // Run forward pass
    compute_generic_dense(&model->initial, buffer, if_features, ACTIVATION_TANH);
    compute_generic_gru(&model->gru_input, &model->gru_recurrent, npm->gru_state, buffer);
    compute_generic_dense(&model->upsample, output, npm->gru_state, ACTIVATION_TANH);
}

void IF_computation(
    if_state *state,
    float *output,
    const float *input             
    )
    {
        // Consider overlap from previous and update
        float x[PITCH_FFT_SIZE];
        OPUS_COPY(x, state->analysis_mem, OVERLAP_SIZE);
        OPUS_COPY(&x[OVERLAP_SIZE], input, FRAME_SIZE);
        OPUS_COPY(state->analysis_mem, &input[FRAME_SIZE-OVERLAP_SIZE], OVERLAP_SIZE);
        kiss_fft_cpx current_fft[PITCH_FFT_SIZE/2 + 1];
        float eps = 1.0e-8;
        float a_t,a_tm1,b_t,b_tm1,m_t,m_tm1;
        forward_transform(current_fft,x);

        for(int i = 0;i < IF_DIMENSION/3;i++){
            // Only scale current FFT by N and not past, will store scaled into past
            a_t = current_fft[i].r*PITCH_FFT_SIZE;
            b_t = current_fft[i].i*PITCH_FFT_SIZE;
            // No scaling on past
            a_tm1 = state->fft_previous[i].r;
            b_tm1 = state->fft_previous[i].i;
            m_t = sqrt(a_t*a_t + b_t*b_t);
            m_tm1 = sqrt(a_tm1*a_tm1 + b_tm1*b_tm1);
            output[i] = log(m_t + eps);
            output[i + IF_DIMENSION/3] = (a_t*a_tm1 + b_t*b_tm1)/(m_t*m_tm1 + eps);
            output[i + 2*IF_DIMENSION/3] = (b_t*a_tm1 - a_t*b_tm1)/(m_t*m_tm1 + eps);

            // Update State with scaled present
            state->fft_previous[i].r = a_t;
            state->fft_previous[i].i = b_t;
        }
    }

void npm_init(neural_pitch_model *st)
    {   
        int ret;
        OPUS_CLEAR(st, 1);
        ret = init_nnpitch(&st->model, nnpitch_arrays);
        celt_assert(ret == 0);
  /* FIXME: perform arch detection. */
}

void if_state_init(if_state *st)
    {
        OPUS_CLEAR(st, 1);
    }

short argmax(float *input)
{
    float maxVal = input[0];
    short pos = 0;

    for(short i =0;i<PITCH_NET_OUTPUT;i++)
    {
        if(input[i] > maxVal){
            maxVal = input[i];
            pos=i;
        }
    }
    return pos;
}