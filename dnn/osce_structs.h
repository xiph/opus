#ifndef OSCE_STRUCTS_H
#define OSCE_STRUCTS_H

#include "opus_types.h"
#include "osce_config.h"
#include "lace_data.h"
#include "nolace_data.h"
#include "nndsp.h"
#include "nnet.h"

typedef struct {
    float               numbits_smooth;
    int                 pitch_hangover_count;
    int                 last_lag;
    int                 last_type;
    float               signal_history[OSCE_FEATURES_MAX_HISTORY];
} OSCEFeatureState;

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


typedef struct NOLACE NOLACE;
typedef struct NoLACEState NoLACEState;

typedef union {
    LACE lace;
} OSCEModel;


#endif