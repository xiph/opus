#ifndef OSCE_FEATURES_H
#define OSCE_FEATURES_H


#include "structs.h"
#include "opus_types.h"

#define OSCE_NUMBITS_BUGFIX

void osce_calculate_features(
    silk_decoder_state          *psDec,                         /* I/O  Decoder state                               */
    silk_decoder_control        *psDecCtrl,                     /* I    Decoder control                             */
    float                       *features,                      /* O    input features                              */
    float                       *numbits,                       /* O    numbits and smoothed numbits                */
    int                         *periods,                       /* O    pitch lags on subframe basis                */
    const opus_int16            xq[],                           /* I    Decoded speech                              */
    opus_int32                  num_bits                        /* I    Size of SILK payload in bits                */
);


#endif