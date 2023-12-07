#ifndef OSCE_STRUCTS_H
#define OSCE_STRUCTS_H

#include "opus_types.h"
#include "osce_config.h"
#include "lace_data.h"
#include "nolace_data.h"
#include "nndsp.h"
#include "nnet.h"

/* feature calculation */

typedef struct {
    float               numbits_smooth;
    int                 pitch_hangover_count;
    int                 last_lag;
    int                 last_type;
    float               signal_history[OSCE_FEATURES_MAX_HISTORY];
} OSCEFeatureState;

/* LACE */

typedef struct {
    float feature_net_conv2_state[LACE_FEATURE_NET_CONV2_STATE_SIZE];
    float feature_net_gru_state[LACE_COND_DIM]; /* ToDo: fix! */
    AdaCombState cf1_state;
    AdaCombState cf2_state;
    AdaConvState af1_state;
    float preemph_mem;
    float deemph_mem;
} LACEState;

typedef struct
{
    LACELayers layers;
    LACEState state;
    float window[LACE_OVERLAP_SIZE];
} LACE;

/* NoLACE */

typedef struct {
    float feature_net_conv2_state[NOLACE_FEATURE_NET_CONV2_STATE_SIZE];
    float feature_net_gru_state[NOLACE_COND_DIM];
    float post_cf1_state[NOLACE_COND_DIM];
    float post_cf2_state[NOLACE_COND_DIM];
    float post_af1_state[NOLACE_COND_DIM];
    float post_af2_state[NOLACE_COND_DIM];
    float post_af3_state[NOLACE_COND_DIM];
    AdaCombState cf1_state;
    AdaCombState cf2_state;
    AdaConvState af1_state;
    AdaConvState af2_state;
    AdaConvState af3_state;
    AdaConvState af4_state;
    AdaShapeState tdshape1_state;
    AdaShapeState tdshape2_state;
    AdaShapeState tdshape3_state;
    float preemph_mem;
    float deemph_mem;
} NoLACEState;

typedef struct {
    NOLACELayers layers;
    NoLACEState state;
    float window[LACE_OVERLAP_SIZE];
} NoLACE;

/* OSCEModel */
typedef struct {
    LACE lace;
    NoLACE nolace;
} OSCEModel;

typedef union {
    LACEState lace;
    NoLACEState nolace;
} OSCEState;

#endif