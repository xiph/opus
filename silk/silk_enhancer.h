#ifndef SILK_ENHANCER_H
#define SILK_ENHANCER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "main.h"

void silk_enhancer(
    silk_decoder_state          *psDec,                         /* I/O  Decoder state                               */
    silk_decoder_control        *psDecCtrl,                     /* I    Decoder control                             */
    opus_int16                  xq[],                           /* I/O  Decoded speech                              */
    opus_int32                  num_bits,                       /* I    Size of SILK payload in bits                */
    int                         arch                            /* I    Run-time architecture                       */
);

#ifdef __cplusplus
}
#endif
#endif
