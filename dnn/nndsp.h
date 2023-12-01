#ifndef NNDSP_H
#define NNDSP_H

#include "opus_types.h"
#include "nnet.h"
#include <string.h>


#define ADACONV_MAX_KERNEL_SIZE 15
#define ADACONV_MAX_INPUT_CHANNELS 2
#define ADACONV_MAX_OUTPUT_CHANNELS 2
#define ADACONV_MAX_FRAME_SIZE 80
#define ADACONV_MAX_OVERLAP_SIZE 40

#define ADACOMB_MAX_LAG 300
#define ADACOMB_MAX_KERNEL_SIZE 15
#define ADACOMB_MAX_FRAME_SIZE 80
#define ADACOMB_MAX_OVERLAP_SIZE 40

#define ADASHAPE_MAX_INPUT_DIM 512
#define ADASHAPE_MAX_FRAME_SIZE 160

/*#define DEBUG_NNDSP*/
#ifdef DEBUG_NNDSP
#include <stdio.h>
#endif

void print_float_vector(const char* name, const float *vec, int length);

typedef struct {
    float history[ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS];
    float last_kernel[ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS * ADACONV_MAX_OUTPUT_CHANNELS];
    float last_gain;
} AdaConvState;


typedef struct {
    float history[ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG];
    float last_kernel[ADACOMB_MAX_KERNEL_SIZE];
    float last_global_gain;
    int last_pitch_lag;
} AdaCombState;


typedef struct {
    float conv_alpha1_state[ADASHAPE_MAX_INPUT_DIM];
    float conv_alpha2_state[ADASHAPE_MAX_FRAME_SIZE];
} AdaShapeState;

void init_adaconv_state(AdaConvState *hAdaConv);

void init_adacomb_state(AdaCombState *hAdaComb);

void init_adashape_state(AdaShapeState *hAdaShape);

void adaconv_process_frame(
    AdaConvState* hAdaConv,
    float *x_out,
    const float *x_in,
    const float *features,
    const LinearLayer *kernel_layer,
    const LinearLayer *gain_layer,
    int feature_dim, /* not strictly necessary */
    int frame_size,
    int overlap_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int left_padding,
    float filter_gain_a,
    float filter_gain_b,
    float shape_gain,
    float *window,
    int arch
);

void adacomb_process_frame(
    AdaCombState* hAdaComb,
    float *x_out,
    const float *x_in,
    const float *features,
    const LinearLayer *kernel_layer,
    const LinearLayer *gain_layer,
    const LinearLayer *global_gain_layer,
    int pitch_lag,
    int feature_dim,
    int frame_size,
    int overlap_size,
    int kernel_size,
    int left_padding,
    float filter_gain_a,
    float filter_gain_b,
    float log_gain_limit,
    float *window,
    int arch
);

void adashape_process_frame(
    AdaShapeState *hAdaShape,
    float *x_out,
    const float *x_in,
    const float *features,
    const LinearLayer *alpha1,
    const LinearLayer *alpha2,
    int feature_dim,
    int frame_size,
    int avg_pool_k,
    int in_stride,
    int in_offset,
    int out_stride,
    int out_offset,
    int arch
);

#endif
