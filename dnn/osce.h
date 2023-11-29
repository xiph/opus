#ifndef OSCE_H
#define OSCE_H


#include "opus_types.h"
#include "lace_data.h"
#include "nolace_data.h"
#include "nndsp.h"
#include "nnet.h"

extern const WeightArray lacelayers_arrays[];
extern const WeightArray nolacelayers_arrays[];

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

typedef LACE OSCEModel;

typedef struct NOLACE NOLACE;
typedef struct NoLACEState NoLACEState;

void init_lace(LACE *hLACE);

void lace_process_20ms_frame(
    LACE* hLACE,
    float *x_out,
    const float *x_in,
    const float *features,
    const float *numbits,
    const int *periods
);

#define init_osce(x) init_lace(x)

#endif