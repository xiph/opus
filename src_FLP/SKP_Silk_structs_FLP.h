/***********************************************************************
Copyright (c) 2006-2010, Skype Limited. All rights reserved. 
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

#ifndef SKP_SILK_STRUCTS_FLP_H
#define SKP_SILK_STRUCTS_FLP_H

#include "SKP_Silk_typedef.h"
#include "SKP_Silk_main.h"

#ifdef __cplusplus
extern "C"
{
#endif

/********************************/
/* Noise shaping analysis state */
/********************************/
typedef struct {
    SKP_int     LastGainIndex;
    SKP_float   HarmBoost_smth;
    SKP_float   HarmShapeGain_smth;
    SKP_float   Tilt_smth;
} SKP_Silk_shape_state_FLP;

/********************************/
/* Prefilter state              */
/********************************/
typedef struct {
    SKP_float   sLTP_shp[ LTP_BUF_LENGTH ];
    SKP_float   sAR_shp[ MAX_SHAPE_LPC_ORDER + 1 ];
    SKP_int     sLTP_shp_buf_idx;
    SKP_float   sLF_AR_shp;
    SKP_float   sLF_MA_shp;
    SKP_float   sHarmHP;
    SKP_int32   rand_seed;
    SKP_int     lagPrev;
} SKP_Silk_prefilter_state_FLP;

/*****************************/
/* Prediction analysis state */
/*****************************/
typedef struct {
    SKP_int     pitch_LPC_win_length;
    SKP_int     min_pitch_lag;                      /* Lowest possible pitch lag (samples)  */
    SKP_int     max_pitch_lag;                      /* Highest possible pitch lag (samples) */
    SKP_float   prev_NLSFq[ MAX_LPC_ORDER ];        /* Previously quantized NLSF vector     */
} SKP_Silk_predict_state_FLP;

/*******************************************/
/* Structure containing NLSF MSVQ codebook */
/*******************************************/
/* structure for one stage of MSVQ */
typedef struct {
    const SKP_int32     nVectors;
    const SKP_float     *CB;
    const SKP_float     *Rates;
} SKP_Silk_NLSF_CBS_FLP;

typedef struct {
    const SKP_int32                         nStages;

    /* fields for (de)quantizing */
    const SKP_Silk_NLSF_CBS_FLP *CBStages;
    const SKP_float                         *NDeltaMin;

    /* fields for arithmetic (de)coding */
    const SKP_uint16                        *CDF;
    const SKP_uint16 * const                *StartPtr;
    const SKP_int                           *MiddleIx;
} SKP_Silk_NLSF_CB_FLP;

/********************************/
/* Encoder state FLP            */
/********************************/
typedef struct {
    SKP_Silk_encoder_state              sCmn;                       /* Common struct, shared with fixed-point code */

    SKP_float                           variable_HP_smth1;          /* State of first smoother */
    SKP_float                           variable_HP_smth2;          /* State of second smoother */

    SKP_Silk_shape_state_FLP            sShape;                     /* Noise shaping state */
    SKP_Silk_prefilter_state_FLP        sPrefilt;                   /* Prefilter State */
    SKP_Silk_predict_state_FLP          sPred;                      /* Prediction State */
    SKP_Silk_nsq_state                  sNSQ;                       /* Noise Shape Quantizer State */
    SKP_Silk_nsq_state                  sNSQ_LBRR;                  /* Noise Shape Quantizer State ( for low bitrate redundancy )*/

    /* Buffer for find pitch and noise shape analysis */
    SKP_float                           x_buf[ 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX ];/* Buffer for find pitch and noise shape analysis */
    SKP_float                           LTPCorr;                    /* Normalized correlation from pitch lag estimator */
    SKP_float                           mu_LTP;                     /* Rate-distortion tradeoff in LTP quantization */
    SKP_float                           SNR_dB;                     /* Quality setting */
    SKP_float                           avgGain;                    /* average gain during active speech */
    SKP_float                           BufferedInChannel_ms;       /* Simulated number of ms buffer in channel because of exceeded TargetRate_bps */
    SKP_float                           speech_activity;            /* Speech activity */

    /* Parameters for LTP scaling control */
    SKP_float                           prevLTPredCodGain;
    SKP_float                           HPLTPredCodGain;

    SKP_float                           inBandFEC_SNR_comp;         /* Compensation to SNR_DB when using inband FEC Voiced */

    const SKP_Silk_NLSF_CB_FLP  *psNLSF_CB_FLP[ 2 ];        /* Pointers to voiced/unvoiced NLSF codebooks */
} SKP_Silk_encoder_state_FLP;


/************************/
/* Encoder control FLP  */
/************************/
typedef struct {
    SKP_Silk_encoder_control    sCmn;                               /* Common struct, shared with fixed-point code */

    /* Prediction and coding parameters */
	SKP_float					Gains[MAX_NB_SUBFR];
	SKP_float					PredCoef[ 2 ][ MAX_LPC_ORDER ];		/* holds interpolated and final coefficients */
	SKP_float					LTPCoef[LTP_ORDER * MAX_NB_SUBFR];
	SKP_float					LTP_scale;

    /* Noise shaping parameters */
	SKP_float					AR1[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ];
	SKP_float					AR2[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ];
	SKP_float					LF_MA_shp[     MAX_NB_SUBFR ];
	SKP_float					LF_AR_shp[     MAX_NB_SUBFR ];
	SKP_float					GainsPre[      MAX_NB_SUBFR ];
	SKP_float					HarmBoost[     MAX_NB_SUBFR ];
	SKP_float					Tilt[          MAX_NB_SUBFR ];
	SKP_float					HarmShapeGain[ MAX_NB_SUBFR ];
	SKP_float					Lambda;
	SKP_float					input_quality;
	SKP_float					coding_quality;
	SKP_float					pitch_freq_low_Hz;
	SKP_float					current_SNR_dB;

	/* Measures */
	SKP_float					sparseness;
    SKP_float                   predGain;
	SKP_float					LTPredCodGain;
	SKP_float					input_quality_bands[ VAD_N_BANDS ];
	SKP_float					input_tilt;
	SKP_float					ResNrg[ MAX_NB_SUBFR ];					/* Residual energy per subframe */
} SKP_Silk_encoder_control_FLP;

#ifdef __cplusplus
}
#endif

#endif
