#ifndef OSCE_H
#define OSCE_H


#include "opus_types.h"
//#include "osce_config.h"
#include "lace_data.h"
#include "nolace_data.h"
#include "nndsp.h"
#include "nnet.h"
#include "osce_structs.h"
#include "structs.h"

extern const WeightArray lacelayers_arrays[];
extern const WeightArray nolacelayers_arrays[];


void init_lace(LACE *hLACE);

void lace_process_20ms_frame(
    LACE* hLACE,
    float *x_out,
    const float *x_in,
    const float *features,
    const float *numbits,
    const int *periods,
    int arch
);

/* API */


void osce_enhance_frame(
    silk_decoder_state          *psDec,                         /* I/O  Decoder state                               */
    silk_decoder_control        *psDecCtrl,                     /* I    Decoder control                             */
    opus_int16                  xq[],                           /* I/O  Decoded speech                              */
    opus_int32                  num_bits,                       /* I    Size of SILK payload in bits                */
    int                         arch                            /* I    Run-time architecture                       */
);

#define init_osce(x) init_lace(x)



#endif
