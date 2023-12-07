#ifndef OSCE_H
#define OSCE_H


#include "opus_types.h"
/*#include "osce_config.h"*/
#ifndef DISABLE_LACE
#include "lace_data.h"
#endif
#ifndef DISABLE_NOLACE
#include "nolace_data.h"
#endif
#include "nndsp.h"
#include "nnet.h"
#include "osce_structs.h"
#include "structs.h"

#define OSCE_METHOD_NONE 0
#ifndef DISABLE_LACE
#define OSCE_METHOD_LACE 1
#endif
#ifndef DISABLE_NOLACE
#define OSCE_METHOD_NOLACE 2
#endif

#if !defined(DISABLE_NOLACE)
#define OSCE_DEFAULT_METHOD OSCE_METHOD_NOLACE
#elif !defined(DISABLE_LACE)
#define OSCE_DEFAULT_METHOD OSCE_METHOD_LACE
#else
#define OSCE_DEFAULT_METHOD OSCE_METHOD_NONE
#endif




/* API */


void osce_enhance_frame(
    silk_decoder_state          *psDec,                         /* I/O  Decoder state                               */
    silk_decoder_control        *psDecCtrl,                     /* I    Decoder control                             */
    opus_int16                  xq[],                           /* I/O  Decoded speech                              */
    opus_int32                  num_bits,                       /* I    Size of SILK payload in bits                */
    int                         arch                            /* I    Run-time architecture                       */
);


void osce_init(silk_OSCE_struct *hOSCE, int method, WeightArray **weights);
void osce_reset(silk_OSCE_struct *hOSCE, int method);


#endif
