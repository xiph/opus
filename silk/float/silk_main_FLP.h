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

#ifndef SILK_MAIN_FLP_H
#define SILK_MAIN_FLP_H

#include "silk_SigProc_FLP.h"
#include "silk_SigProc_FIX.h"
#include "silk_structs_FLP.h"
#include "silk_main.h"
#include "silk_define.h"
#include "silk_debug.h"
#include "entenc.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define silk_encoder_state_Fxx      silk_encoder_state_FLP
#define silk_encode_frame_Fxx       silk_encode_frame_FLP

/*********************/
/* Encoder Functions */
/*********************/

/* High-pass filter with cutoff frequency adaptation based on pitch lag statistics */
void silk_HP_variable_cutoff(
    silk_encoder_state_Fxx          state_Fxx[],        /* I/O  Encoder states                          */
    const opus_int                   nChannels           /* I    Number of channels                      */
);

/* Encoder main function */
opus_int silk_encode_frame_FLP( 
    silk_encoder_state_FLP          *psEnc,             /* I/O  Encoder state FLP                       */
    opus_int32                       *pnBytesOut,        /*   O  Number of payload bytes;                */
    ec_enc                          *psRangeEnc         /* I/O  compressor data structure               */
);

/* Low Bitrate Redundancy (LBRR) encoding. Reuse all parameters but encode with lower bitrate           */
void silk_LBRR_encode_FLP(
    silk_encoder_state_FLP          *psEnc,             /* I/O  Encoder state FLP                       */
    silk_encoder_control_FLP        *psEncCtrl,         /* I/O  Encoder control FLP                     */
    const SKP_float                 xfw[]               /* I    Input signal                            */
);

/* Initializes the Silk encoder state */
opus_int silk_init_encoder(
    silk_encoder_state_FLP          *psEnc              /* I/O  Encoder state FLP                       */
);

/* Control the Silk encoder */
opus_int silk_control_encoder( 
    silk_encoder_state_FLP          *psEnc,             /* I/O  Pointer to Silk encoder state FLP       */
    silk_EncControlStruct           *encControl,        /* I:   Control structure                       */
    const opus_int32                 TargetRate_bps,     /* I    Target max bitrate (bps)                */
    const opus_int                   allow_bw_switch,    /* I    Flag to allow switching audio bandwidth */
    const opus_int                   channelNb           /* I    Channel number                          */
);

/****************/
/* Prefiltering */
/****************/
void silk_prefilter_FLP(
    silk_encoder_state_FLP              *psEnc,         /* I/O  Encoder state FLP                       */
    const silk_encoder_control_FLP      *psEncCtrl,     /* I    Encoder control FLP                     */
          SKP_float                     xw[],           /* O    Weighted signal                         */
    const SKP_float                     x[]             /* I    Speech signal                           */
);

/**************************/
/* Noise shaping analysis */
/**************************/
/* Compute noise shaping coefficients and initial gain values */
void silk_noise_shape_analysis_FLP(
    silk_encoder_state_FLP          *psEnc,             /* I/O  Encoder state FLP                       */
    silk_encoder_control_FLP        *psEncCtrl,         /* I/O  Encoder control FLP                     */
    const SKP_float                 *pitch_res,         /* I    LPC residual from pitch analysis        */
    const SKP_float                 *x                  /* I    Input signal [frame_length + la_shape]  */
);

/* Autocorrelations for a warped frequency axis */
void silk_warped_autocorrelation_FLP( 
          SKP_float                 *corr,              /* O    Result [order + 1]                      */
    const SKP_float                 *input,             /* I    Input data to correlate                 */
    const SKP_float                 warping,            /* I    Warping coefficient                     */
    const opus_int                   length,             /* I    Length of input                         */
    const opus_int                   order               /* I    Correlation order (even)                */
);

/* Calculation of LTP state scaling */
void silk_LTP_scale_ctrl_FLP(
    silk_encoder_state_FLP          *psEnc,             /* I/O  Encoder state FLP                       */
    silk_encoder_control_FLP        *psEncCtrl          /* I/O  Encoder control FLP                     */
);

/**********************************************/
/* Prediction Analysis                        */
/**********************************************/
/* Find pitch lags */
void silk_find_pitch_lags_FLP(
    silk_encoder_state_FLP          *psEnc,             /* I/O  Encoder state FLP                       */
    silk_encoder_control_FLP        *psEncCtrl,         /* I/O  Encoder control FLP                     */
          SKP_float                 res[],              /* O    Residual                                */
    const SKP_float                 x[]                 /* I    Speech signal                           */
);

/* Find LPC and LTP coefficients */
void silk_find_pred_coefs_FLP(
    silk_encoder_state_FLP          *psEnc,             /* I/O  Encoder state FLP                       */
    silk_encoder_control_FLP        *psEncCtrl,         /* I/O  Encoder control FLP                     */
    const SKP_float                 res_pitch[],        /* I    Residual from pitch analysis            */
    const SKP_float                 x[]                 /* I    Speech signal                           */
);

/* LPC analysis */
void silk_find_LPC_FLP(
          opus_int16                 NLSF_Q15[],             /* O    NLSFs                                   */
          opus_int8                  *interpIndex,           /* O    NLSF interp. index for NLSF interp.     */
    const opus_int16                 prev_NLSFq_Q15[],       /* I    Previous NLSFs, for NLSF interpolation  */
    const opus_int                   useInterpNLSFs,         /* I    Flag                                    */
    const opus_int                   firstFrameAfterReset,   /* I    Flag                                    */
    const opus_int                   LPC_order,              /* I    LPC order                               */
    const SKP_float                 x[],                    /* I    Input signal                            */
    const opus_int                   subfr_length,           /* I    Subframe length incl preceeding samples */
    const opus_int                   nb_subfr                /* I:   Number of subframes                     */
);

/* LTP analysis */
void silk_find_LTP_FLP(
          SKP_float b[ MAX_NB_SUBFR * LTP_ORDER ],          /* O    LTP coefs                               */
          SKP_float WLTP[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ], /* O    Weight for LTP quantization       */
          SKP_float *LTPredCodGain,                         /* O    LTP coding gain                         */
    const SKP_float r_lpc[],                                /* I    LPC residual                            */
    const opus_int   lag[  MAX_NB_SUBFR ],                   /* I    LTP lags                                */
    const SKP_float Wght[ MAX_NB_SUBFR ],                   /* I    Weights                                 */
    const opus_int   subfr_length,                           /* I    Subframe length                         */
    const opus_int   nb_subfr,                               /* I    number of subframes                     */
    const opus_int   mem_offset                              /* I    Number of samples in LTP memory         */
);

void silk_LTP_analysis_filter_FLP(
          SKP_float         *LTP_res,                   /* O    LTP res MAX_NB_SUBFR*(pre_lgth+subfr_lngth) */
    const SKP_float         *x,                         /* I    Input signal, with preceeding samples       */
    const SKP_float         B[ LTP_ORDER * MAX_NB_SUBFR ],  /* I    LTP coefficients for each subframe      */
    const opus_int           pitchL[   MAX_NB_SUBFR ],   /* I    Pitch lags                                  */
    const SKP_float         invGains[ MAX_NB_SUBFR ],   /* I    Inverse quantization gains                  */
    const opus_int           subfr_length,               /* I    Length of each subframe                     */
    const opus_int           nb_subfr,                   /* I    number of subframes                         */
    const opus_int           pre_length                  /* I    Preceeding samples for each subframe        */
);

/* Calculates residual energies of input subframes where all subframes have LPC_order   */
/* of preceeding samples                                                                */
void silk_residual_energy_FLP(  
          SKP_float             nrgs[ MAX_NB_SUBFR ],   /* O    Residual energy per subframe            */
    const SKP_float             x[],                    /* I    Input signal                            */
          SKP_float             a[ 2 ][ MAX_LPC_ORDER ],/* I    AR coefs for each frame half            */
    const SKP_float             gains[],                /* I    Quantization gains                      */
    const opus_int               subfr_length,           /* I    Subframe length                         */
    const opus_int               nb_subfr,               /* I    number of subframes                     */
    const opus_int               LPC_order               /* I    LPC order                               */
);

/* 16th order LPC analysis filter */
void silk_LPC_analysis_filter_FLP(
          SKP_float                 r_LPC[],            /* O    LPC residual signal                     */
    const SKP_float                 PredCoef[],         /* I    LPC coefficients                        */
    const SKP_float                 s[],                /* I    Input signal                            */
    const opus_int                   length,             /* I    Length of input signal                  */
    const opus_int                   Order               /* I    LPC order                               */
);

/* LTP tap quantizer */
void silk_quant_LTP_gains_FLP(
          SKP_float B[ MAX_NB_SUBFR * LTP_ORDER ],              /* I/O  (Un-)quantized LTP gains                */
          opus_int8  cbk_index[ MAX_NB_SUBFR ],                  /* O    Codebook index                          */
          opus_int8  *periodicity_index,                         /* O    Periodicity index                       */
    const SKP_float W[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ],  /* I    Error weights                           */
    const opus_int   mu_Q10,                                     /* I    Mu value (R/D tradeoff)                 */
    const opus_int   lowComplexity,                              /* I    Flag for low complexity                 */
    const opus_int   nb_subfr                                    /* I    number of subframes                     */
);

/******************/
/* NLSF Quantizer */
/******************/
/* Limit, stabilize, and quantize NLSFs */
void silk_process_NLSFs_FLP(
    silk_encoder_state              *psEncC,                            /* I/O  Encoder state                               */
    SKP_float                       PredCoef[ 2 ][ MAX_LPC_ORDER ],     /* O    Prediction coefficients                     */
    opus_int16                       NLSF_Q15[      MAX_LPC_ORDER ],     /* I/O  Normalized LSFs (quant out) (0 - (2^15-1))  */
    const opus_int16                 prev_NLSF_Q15[ MAX_LPC_ORDER ]      /* I    Previous Normalized LSFs (0 - (2^15-1))     */
);

/* Residual energy: nrg = wxx - 2 * wXx * c + c' * wXX * c */
SKP_float silk_residual_energy_covar_FLP(           /* O    Weighted residual energy                */
    const SKP_float                 *c,                 /* I    Filter coefficients                     */
          SKP_float                 *wXX,               /* I/O  Weighted correlation matrix, reg. out   */
    const SKP_float                 *wXx,               /* I    Weighted correlation vector             */
    const SKP_float                 wxx,                /* I    Weighted correlation value              */
    const opus_int                   D                   /* I    Dimension                               */
);

/* Entropy constrained MATRIX-weighted VQ, for a single input data vector */
void silk_VQ_WMat_EC_FLP(
          opus_int                   *ind,               /* O    Index of best codebook vector           */
          SKP_float                 *rate_dist,         /* O    Best weighted quant. error + mu * rate  */
    const SKP_float                 *in,                /* I    Input vector to be quantized            */
    const SKP_float                 *W,                 /* I    Weighting matrix                        */
    const opus_int16                 *cb,                /* I    Codebook                                */
    const opus_int16                 *cl_Q6,             /* I    Code length for each codebook vector    */
    const SKP_float                 mu,                 /* I    Tradeoff between WSSE and rate          */
    const opus_int                   L                   /* I    Number of vectors in codebook           */
);

/* Processing of gains */
void silk_process_gains_FLP(
    silk_encoder_state_FLP          *psEnc,             /* I/O  Encoder state FLP                       */
    silk_encoder_control_FLP        *psEncCtrl          /* I/O  Encoder control FLP                     */
);

/******************/
/* Linear Algebra */
/******************/
/* Calculates correlation matrix X'*X */
void silk_corrMatrix_FLP(
    const SKP_float                 *x,                 /* I    x vector [ L+order-1 ] used to create X */
    const opus_int                   L,                  /* I    Length of vectors                       */
    const opus_int                   Order,              /* I    Max lag for correlation                 */
          SKP_float                 *XX                 /* O    X'*X correlation matrix [order x order] */
);

/* Calculates correlation vector X'*t */
void silk_corrVector_FLP(
    const SKP_float                 *x,                 /* I    x vector [L+order-1] used to create X   */
    const SKP_float                 *t,                 /* I    Target vector [L]                       */
    const opus_int                   L,                  /* I    Length of vecors                        */
    const opus_int                   Order,              /* I    Max lag for correlation                 */
          SKP_float                 *Xt                 /* O    X'*t correlation vector [order]         */
);

/* Add noise to matrix diagonal */
void silk_regularize_correlations_FLP(
          SKP_float                 *XX,                /* I/O  Correlation matrices                    */
          SKP_float                 *xx,                /* I/O  Correlation values                      */
    const SKP_float                 noise,              /* I    Noise energy to add                     */
    const opus_int                   D                   /* I    Dimension of XX                         */
);

/* Function to solve linear equation Ax = b, where A is an MxM symmetric matrix */
void silk_solve_LDL_FLP(
          SKP_float                 *A,                 /* I/O  Symmetric square matrix, out: reg.      */
    const opus_int                   M,                  /* I    Size of matrix                          */
    const SKP_float                 *b,                 /* I    Pointer to b vector                     */
          SKP_float                 *x                  /* O    Pointer to x solution vector            */
);

/* Apply sine window to signal vector.                                                                  */
/* Window types:                                                                                        */
/*  1 -> sine window from 0 to pi/2                                                                     */
/*  2 -> sine window from pi/2 to pi                                                                    */
void silk_apply_sine_window_FLP(
          SKP_float                 px_win[],           /* O    Pointer to windowed signal              */
    const SKP_float                 px[],               /* I    Pointer to input signal                 */
    const opus_int                   win_type,           /* I    Selects a window type                   */
    const opus_int                   length              /* I    Window length, multiple of 4            */
);

/* Wrapper functions. Call flp / fix code */

/* Convert AR filter coefficients to NLSF parameters */
void silk_A2NLSF_FLP( 
          opus_int16                 *NLSF_Q15,          /* O    NLSF vector      [ LPC_order ]          */
    const SKP_float                 *pAR,               /* I    LPC coefficients [ LPC_order ]          */
    const opus_int                   LPC_order           /* I    LPC order                               */
);

/* Convert NLSF parameters to AR prediction filter coefficients */
void silk_NLSF2A_FLP( 
          SKP_float                 *pAR,               /* O    LPC coefficients [ LPC_order ]          */
    const opus_int16                 *NLSF_Q15,          /* I    NLSF vector      [ LPC_order ]          */
    const opus_int                   LPC_order           /* I    LPC order                               */
);

/****************************************/
/* Floating-point Silk NSQ wrapper      */
/****************************************/
void silk_NSQ_wrapper_FLP(
    silk_encoder_state_FLP          *psEnc,         /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,     /* I/O  Encoder control FLP                         */
    SideInfoIndices                 *psIndices,     /* I/O  Quantization indices                        */
    silk_nsq_state                  *psNSQ,         /* I/O  Noise Shaping Quantzation state             */
          opus_int8                  pulses[],       /* O    Quantized pulse signal                      */
    const SKP_float                 x[]             /* I    Prefiltered input signal                    */
);

#ifdef __cplusplus
}
#endif

#endif
