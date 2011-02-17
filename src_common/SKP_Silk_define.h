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

#ifndef SKP_SILK_DEFINE_H
#define SKP_SILK_DEFINE_H

#include "SKP_Silk_errors.h"
#include "SKP_Silk_typedef.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#define MAX_FRAMES_PER_PACKET                   3

#ifndef FIXED_POINT
#    define FIXED_POINT                         0
#endif

/* Limits on bitrate */
#define MIN_TARGET_RATE_BPS                     5000
#define MAX_TARGET_RATE_BPS                     80000

/* Compensation in bitrate calculations for 10 ms modes */
#define REDUCE_BITRATE_10_MS_BPS                2200

/* Transition bitrates between modes */
#define WB2MB_BITRATE_BPS                       12000
#define MB2WB_BITRATE_BPS                       16000
#define MB2NB_BITRATE_BPS                        9000
#define NB2MB_BITRATE_BPS                       12000
#define TARGET_RATE_TAB_SZ                      8

/* DTX settings                                 */
#define NO_SPEECH_FRAMES_BEFORE_DTX             5       /* eq 100 ms */
#define MAX_CONSECUTIVE_DTX                     20      /* eq 400 ms */

/* LBRR thresholds */
#define LBRR_NB_MIN_RATE_BPS                    9000
#define LBRR_MB_MIN_RATE_BPS                    12000
#define LBRR_WB_MIN_RATE_BPS                    15000

/* Activate bandwidth transition filtering for mode switching */
#define SWITCH_TRANSITION_FILTERING             1

/* Decoder Parameters */
#define DEC_HP_ORDER                            2

/* Maximum sampling frequency, should be 16 for embedded */
#define MAX_FS_KHZ                              16 
#define MAX_API_FS_KHZ                          48

/* Signal types used by silk */
#define TYPE_NO_VOICE_ACTIVITY                  0
#define TYPE_UNVOICED                           1
#define TYPE_VOICED                             2

/* Number of subframes */
#define MAX_NB_SUBFR                            4

/* Number of samples per frame */ 
#define LTP_MEM_LENGTH_MS                       20
#define SUB_FRAME_LENGTH_MS                     5
#define MAX_SUB_FRAME_LENGTH                    ( SUB_FRAME_LENGTH_MS * MAX_FS_KHZ )
#define MAX_FRAME_LENGTH_MS                     ( SUB_FRAME_LENGTH_MS * MAX_NB_SUBFR )
#define MAX_FRAME_LENGTH                        ( MAX_FRAME_LENGTH_MS * MAX_FS_KHZ )

/* Milliseconds of lookahead for pitch analysis */
#define LA_PITCH_MS                             2
#define LA_PITCH_MAX                            ( LA_PITCH_MS * MAX_FS_KHZ )

/* Order of LPC used in find pitch */
#define MAX_FIND_PITCH_LPC_ORDER                16

/* Length of LPC window used in find pitch */
#define FIND_PITCH_LPC_WIN_MS                   ( 20 + (LA_PITCH_MS << 1) )
#define FIND_PITCH_LPC_WIN_MS_2_SF              ( 10 + (LA_PITCH_MS << 1) )
#define FIND_PITCH_LPC_WIN_MAX                  ( FIND_PITCH_LPC_WIN_MS * MAX_FS_KHZ )

/* Milliseconds of lookahead for noise shape analysis */
#define LA_SHAPE_MS                             5
#define LA_SHAPE_MAX                            ( LA_SHAPE_MS * MAX_FS_KHZ )

/* Maximum length of LPC window used in noise shape analysis */
#define SHAPE_LPC_WIN_MAX                       ( 15 * MAX_FS_KHZ )

/* dB level of lowest gain quantization level */
#define MIN_QGAIN_DB                            2
/* dB level of highest gain quantization level */
#define MAX_QGAIN_DB                            88
/* Number of gain quantization levels */
#define N_LEVELS_QGAIN                          64
/* Max increase in gain quantization index */
#define MAX_DELTA_GAIN_QUANT                    36
/* Max decrease in gain quantization index */
#define MIN_DELTA_GAIN_QUANT                    -4

/* Quantization offsets (multiples of 4) */
#define OFFSET_VL_Q10                           32
#define OFFSET_VH_Q10                           100
#define OFFSET_UVL_Q10                          100
#define OFFSET_UVH_Q10                          240

/* Maximum numbers of iterations used to stabilize a LPC vector */
#define MAX_LPC_STABILIZE_ITERATIONS            20

#define MAX_LPC_ORDER                           16
#define MIN_LPC_ORDER                           10

/* Find Pred Coef defines */
#define LTP_ORDER                               5

/* LTP quantization settings */
#define NB_LTP_CBKS                             3

/* Flag to use harmonic noise shaping */
#define USE_HARM_SHAPING                        1

/* Max LPC order of noise shaping filters */
#define MAX_SHAPE_LPC_ORDER                     16

#define HARM_SHAPE_FIR_TAPS                     3

/* Maximum number of delayed decision states */
#define MAX_DEL_DEC_STATES                      4

#define LTP_BUF_LENGTH                          512
#define LTP_MASK                                (LTP_BUF_LENGTH - 1)

#define DECISION_DELAY                          32
#define DECISION_DELAY_MASK                     (DECISION_DELAY - 1)

/* number of subframes for excitation entropy coding */
#define SHELL_CODEC_FRAME_LENGTH                16
#define LOG2_SHELL_CODEC_FRAME_LENGTH           4
#define MAX_NB_SHELL_BLOCKS                     (MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH)

/* number of rate levels, for entropy coding of excitation */
#define N_RATE_LEVELS                           10

/* maximum sum of pulses per shell coding frame */
#define MAX_PULSES                              16

#define MAX_MATRIX_SIZE                         MAX_LPC_ORDER /* Max of LPC Order and LTP order */

#if( MAX_LPC_ORDER > DECISION_DELAY )
# define NSQ_LPC_BUF_LENGTH                     MAX_LPC_ORDER
#else
# define NSQ_LPC_BUF_LENGTH                     DECISION_DELAY
#endif

/***********************/
/* High pass filtering */
/***********************/
#define HIGH_PASS_INPUT                         1

/***************************/
/* Voice activity detector */
/***************************/
#define VAD_N_BANDS                             4

#define VAD_INTERNAL_SUBFRAMES_LOG2             2
#define VAD_INTERNAL_SUBFRAMES                  (1 << VAD_INTERNAL_SUBFRAMES_LOG2)
    
#define VAD_NOISE_LEVEL_SMOOTH_COEF_Q16         1024    /* Must be <  4096                                  */
#define VAD_NOISE_LEVELS_BIAS                   50 

/* Sigmoid settings */
#define VAD_NEGATIVE_OFFSET_Q5                  128     /* sigmoid is 0 at -128                             */
#define VAD_SNR_FACTOR_Q16                      45000 

/* smoothing for SNR measurement */
#define VAD_SNR_SMOOTH_COEF_Q18                 4096

/******************/
/* NLSF quantizer */
/******************/
#define NLSF_MSVQ_MAX_CB_STAGES                      10 /* Update manually when changing codebooks      */
#define NLSF_MSVQ_MAX_VECTORS_IN_STAGE               64 /* Update manually when changing codebooks      */
#define NLSF_MSVQ_MAX_VECTORS_IN_STAGE_TWO_TO_END    16 /* Update manually when changing codebooks      */

#define NLSF_MSVQ_FLUCTUATION_REDUCTION         1
#define MAX_NLSF_MSVQ_SURVIVORS                 16

#define NLSF_Q_DOMAIN_STAGE_0                   8
#define NLSF_Q_DOMAIN_STAGE_2_TO_LAST           8

/* Based on above defines, calculate how much memory is necessary to allocate */
#if( NLSF_MSVQ_MAX_VECTORS_IN_STAGE > ( MAX_NLSF_MSVQ_SURVIVORS_LC_MODE * NLSF_MSVQ_MAX_VECTORS_IN_STAGE_TWO_TO_END ) )
#   define NLSF_MSVQ_TREE_SEARCH_MAX_VECTORS_EVALUATED_LC_MODE  NLSF_MSVQ_MAX_VECTORS_IN_STAGE
#else
#   define NLSF_MSVQ_TREE_SEARCH_MAX_VECTORS_EVALUATED_LC_MODE  MAX_NLSF_MSVQ_SURVIVORS_LC_MODE * NLSF_MSVQ_MAX_VECTORS_IN_STAGE_TWO_TO_END
#endif

#if( NLSF_MSVQ_MAX_VECTORS_IN_STAGE > ( MAX_NLSF_MSVQ_SURVIVORS * NLSF_MSVQ_MAX_VECTORS_IN_STAGE_TWO_TO_END ) )
#   define NLSF_MSVQ_TREE_SEARCH_MAX_VECTORS_EVALUATED  NLSF_MSVQ_MAX_VECTORS_IN_STAGE
#else
#   define NLSF_MSVQ_TREE_SEARCH_MAX_VECTORS_EVALUATED  MAX_NLSF_MSVQ_SURVIVORS * NLSF_MSVQ_MAX_VECTORS_IN_STAGE_TWO_TO_END
#endif

#define NLSF_MSVQ_SURV_MAX_REL_RD               0.1f    /* Must be < 0.5                                    */

/* Transition filtering for mode switching */
#if SWITCH_TRANSITION_FILTERING
#  define TRANSITION_TIME_MS                    5120 // 5120 = 64 * FRAME_LENGTH_MS * ( TRANSITION_INT_NUM - 1 ) = 64*(20*4)
#  define TRANSITION_NB                         3 /* Hardcoded in tables */
#  define TRANSITION_NA                         2 /* Hardcoded in tables */
#  define TRANSITION_INT_NUM                    5 /* Hardcoded in tables */
#  define TRANSITION_FRAMES                     ( TRANSITION_TIME_MS / MAX_FRAME_LENGTH_MS ) // todo: needs to be made flexible for 10 ms frames
#  define TRANSITION_INT_STEPS                  ( TRANSITION_FRAMES  / ( TRANSITION_INT_NUM - 1 ) )
#else
#  define TRANSITION_FRAMES                     0
#endif

/* BWE factors to apply after packet loss */
#define BWE_AFTER_LOSS_Q16                      63570

/* Defines for CN generation */
#define CNG_BUF_MASK_MAX                        255             /* 2^floor(log2(MAX_FRAME_LENGTH))-1    */
#define CNG_GAIN_SMTH_Q16                       4634            /* 0.25^(1/4)                           */
#define CNG_NLSF_SMTH_Q16                       16348           /* 0.25                                 */

#ifdef __cplusplus
}
#endif

#endif
