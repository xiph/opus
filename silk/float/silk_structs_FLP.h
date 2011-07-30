/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, (subject to the limitations in the disclaimer below)
are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Skype Limited, nor the names of specific
contributors, may be used to endorse or promote products derived from
this software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED
BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifndef SILK_STRUCTS_FLP_H
#define SILK_STRUCTS_FLP_H

#include "silk_typedef.h"
#include "silk_main.h"
#include "silk_structs.h"

#ifdef __cplusplus
extern "C"
{
#endif

/********************************/
/* Noise shaping analysis state */
/********************************/
typedef struct {
    opus_int8    LastGainIndex;
    SKP_float   HarmBoost_smth;
    SKP_float   HarmShapeGain_smth;
    SKP_float   Tilt_smth;
} silk_shape_state_FLP;

/********************************/
/* Prefilter state              */
/********************************/
typedef struct {
    SKP_float   sLTP_shp[ LTP_BUF_LENGTH ];
    SKP_float   sAR_shp[ MAX_SHAPE_LPC_ORDER + 1 ];
    opus_int     sLTP_shp_buf_idx;
    SKP_float   sLF_AR_shp;
    SKP_float   sLF_MA_shp;
    SKP_float   sHarmHP;
    opus_int32   rand_seed;
    opus_int     lagPrev;
} silk_prefilter_state_FLP;

/********************************/
/* Encoder state FLP            */
/********************************/
typedef struct {
    silk_encoder_state          sCmn;                       /* Common struct, shared with fixed-point code */
    silk_shape_state_FLP        sShape;                     /* Noise shaping state */
    silk_prefilter_state_FLP    sPrefilt;                   /* Prefilter State */

    /* Buffer for find pitch and noise shape analysis */
    SKP_float                   x_buf[ 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX ];/* Buffer for find pitch and noise shape analysis */
    SKP_float                   LTPCorr;                    /* Normalized correlation from pitch lag estimator */

    /* Parameters for LTP scaling control */
    SKP_float                   prevLTPredCodGain;
    SKP_float                   HPLTPredCodGain;
} silk_encoder_state_FLP;

/************************/
/* Encoder control FLP  */
/************************/
typedef struct {
    /* Prediction and coding parameters */
    SKP_float                    Gains[ MAX_NB_SUBFR ];
    SKP_float                    PredCoef[ 2 ][ MAX_LPC_ORDER ];        /* holds interpolated and final coefficients */
    SKP_float                    LTPCoef[LTP_ORDER * MAX_NB_SUBFR];
    SKP_float                    LTP_scale;
    opus_int                     pitchL[ MAX_NB_SUBFR ];

    /* Noise shaping parameters */
    SKP_float                    AR1[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ];
    SKP_float                    AR2[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ];
    SKP_float                    LF_MA_shp[     MAX_NB_SUBFR ];
    SKP_float                    LF_AR_shp[     MAX_NB_SUBFR ];
    SKP_float                    GainsPre[      MAX_NB_SUBFR ];
    SKP_float                    HarmBoost[     MAX_NB_SUBFR ];
    SKP_float                    Tilt[          MAX_NB_SUBFR ];
    SKP_float                    HarmShapeGain[ MAX_NB_SUBFR ];
    SKP_float                    Lambda;
    SKP_float                    input_quality;
    SKP_float                    coding_quality;

    /* Measures */
    SKP_float                    sparseness;
    SKP_float                   predGain;
    SKP_float                    LTPredCodGain;
    SKP_float                    ResNrg[ MAX_NB_SUBFR ];                    /* Residual energy per subframe */
} silk_encoder_control_FLP;

/************************/
/* Encoder Super Struct */
/************************/
typedef struct {
    silk_encoder_state_FLP      state_Fxx[ ENCODER_NUM_CHANNELS ];
    stereo_enc_state            sStereo;
    opus_int32                   nBitsExceeded;
    opus_int                     nChannelsAPI;
    opus_int                     nChannelsInternal;
    opus_int                     timeSinceSwitchAllowed_ms;
    opus_int                     allowBandwidthSwitch;
} silk_encoder;

#ifdef __cplusplus
}
#endif

#endif
