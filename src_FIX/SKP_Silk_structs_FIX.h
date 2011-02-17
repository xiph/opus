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

#ifndef SKP_SILK_STRUCTS_FIX_H
#define SKP_SILK_STRUCTS_FIX_H

#include "SKP_Silk_typedef.h"
#include "SKP_Silk_main.h"
#include "SKP_Silk_structs.h"

#ifdef __cplusplus
extern "C"
{
#endif

/********************************/
/* Noise shaping analysis state */
/********************************/
typedef struct {
    SKP_int8    LastGainIndex;
    SKP_int32   HarmBoost_smth_Q16;
    SKP_int32   HarmShapeGain_smth_Q16;
    SKP_int32   Tilt_smth_Q16;
} SKP_Silk_shape_state_FIX;

/********************************/
/* Prefilter state              */
/********************************/
typedef struct {
    SKP_int16   sLTP_shp[ LTP_BUF_LENGTH ];
    SKP_int32   sAR_shp[ MAX_SHAPE_LPC_ORDER + 1 ];
    SKP_int     sLTP_shp_buf_idx;
    SKP_int32   sLF_AR_shp_Q12;
    SKP_int32   sLF_MA_shp_Q12;
    SKP_int     sHarmHP;
    SKP_int32   rand_seed;
    SKP_int     lagPrev;
} SKP_Silk_prefilter_state_FIX;

/********************************/
/* Encoder state FIX            */
/********************************/
typedef struct {
    SKP_Silk_encoder_state          sCmn;                           /* Common struct, shared with floating-point code */
    SKP_Silk_shape_state_FIX        sShape;                         /* Shape state                                                          */
    SKP_Silk_prefilter_state_FIX    sPrefilt;                       /* Prefilter State                                                      */

    /* Buffer for find pitch and noise shape analysis */
    SKP_DWORD_ALIGN SKP_int16 x_buf[ 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX ];
    SKP_int                         LTPCorr_Q15;                    /* Normalized correlation from pitch lag estimator                      */
    SKP_int32                       SNR_dB_Q7;                      /* Quality setting                                                      */
    SKP_int                         BufferedInChannel_ms;           /* Simulated number of ms buffer because of exceeded TargetRate_bps     */

    /* Parameters For LTP scaling Control */
    SKP_int                         prevLTPredCodGain_Q7;
    SKP_int                         HPLTPredCodGain_Q7;

    SKP_int32                       inBandFEC_SNR_comp_Q7;          /* Compensation to SNR_dB when using inband FEC Voiced                  */

} SKP_Silk_encoder_state_FIX;

/************************/
/* Encoder control FIX  */
/************************/
typedef struct {
    /* Prediction and coding parameters */
    SKP_int32                   Gains_Q16[ MAX_NB_SUBFR ];
    SKP_DWORD_ALIGN SKP_int16   PredCoef_Q12[ 2 ][ MAX_LPC_ORDER ];
    SKP_int16                   LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ];
    SKP_int                     LTP_scale_Q14;
    SKP_int                     pitchL[ MAX_NB_SUBFR ];

    /* Noise shaping parameters */
    /* Testing */
    SKP_DWORD_ALIGN SKP_int16 AR1_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ];
    SKP_DWORD_ALIGN SKP_int16 AR2_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ];
    SKP_int32   LF_shp_Q14[        MAX_NB_SUBFR ];          /* Packs two int16 coefficients per int32 value             */
    SKP_int     GainsPre_Q14[      MAX_NB_SUBFR ];
    SKP_int     HarmBoost_Q14[     MAX_NB_SUBFR ];
    SKP_int     Tilt_Q14[          MAX_NB_SUBFR ];
    SKP_int     HarmShapeGain_Q14[ MAX_NB_SUBFR ];
    SKP_int     Lambda_Q10;
    SKP_int     input_quality_Q14;
    SKP_int     coding_quality_Q14;
    SKP_int     current_SNR_dB_Q7;

    /* measures */
    SKP_int     sparseness_Q8;
    SKP_int32   predGain_Q16;
    SKP_int     LTPredCodGain_Q7;
    SKP_int32   ResNrg[ MAX_NB_SUBFR ];             /* Residual energy per subframe                             */
    SKP_int     ResNrgQ[ MAX_NB_SUBFR ];            /* Q domain for the residual energy > 0                     */
    
} SKP_Silk_encoder_control_FIX;


#ifdef __cplusplus
}
#endif

#endif
