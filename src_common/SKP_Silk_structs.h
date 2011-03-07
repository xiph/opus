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

#ifndef SKP_SILK_STRUCTS_H
#define SKP_SILK_STRUCTS_H


#include "SKP_Silk_typedef.h"
#include "SKP_Silk_SigProc_FIX.h"
#include "SKP_Silk_define.h"
#include "entenc.h"
#include "entdec.h"

#ifdef __cplusplus
extern "C"
{
#endif

/************************************/
/* Noise shaping quantization state */
/************************************/
typedef struct {
    SKP_int16   xq[           2 * MAX_FRAME_LENGTH ]; /* Buffer for quantized output signal */
    SKP_int32   sLTP_shp_Q10[ 2 * MAX_FRAME_LENGTH ];
    SKP_int32   sLPC_Q14[ MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH ];
    SKP_int32   sAR2_Q14[ MAX_SHAPE_LPC_ORDER ];
    SKP_int32   sLF_AR_shp_Q12;
    SKP_int     lagPrev;
    SKP_int     sLTP_buf_idx;
    SKP_int     sLTP_shp_buf_idx;
    SKP_int32   rand_seed;
    SKP_int32   prev_inv_gain_Q16;
    SKP_int     rewhite_flag;
} SKP_Silk_nsq_state;

/********************************/
/* VAD state                    */
/********************************/
typedef struct {
    SKP_int32   AnaState[ 2 ];                  /* Analysis filterbank state: 0-8 kHz                       */
    SKP_int32   AnaState1[ 2 ];                 /* Analysis filterbank state: 0-4 kHz                       */
    SKP_int32   AnaState2[ 2 ];                 /* Analysis filterbank state: 0-2 kHz                       */
    SKP_int32   XnrgSubfr[ VAD_N_BANDS ];       /* Subframe energies                                        */
    SKP_int32   NrgRatioSmth_Q8[ VAD_N_BANDS ]; /* Smoothed energy level in each band                       */
    SKP_int16   HPstate;                        /* State of differentiator in the lowest band               */
    SKP_int32   NL[ VAD_N_BANDS ];              /* Noise energy level in each band                          */
    SKP_int32   inv_NL[ VAD_N_BANDS ];          /* Inverse noise energy level in each band                  */
    SKP_int32   NoiseLevelBias[ VAD_N_BANDS ];  /* Noise level estimator bias/offset                        */
    SKP_int32   counter;                        /* Frame counter used in the initial phase                  */
} SKP_Silk_VAD_state;

/* Variable cut-off low-pass filter state */
typedef struct {
    SKP_int32                   In_LP_State[ 2 ];           /* Low pass filter state */
    SKP_int32                   transition_frame_no;        /* Counter which is mapped to a cut-off frequency */
    SKP_int                     mode;                       /* Operating mode, <0: switch down, >0: switch up; 0: do nothing */
} SKP_Silk_LP_state;

/* Structure containing NLSF codebook */
typedef struct {
    const SKP_int16             nVectors;
    const SKP_int16             order;
    const SKP_int16             quantStepSize_Q16;
    const SKP_int16             invQuantStepSize_Q6;
    const SKP_uint8             *CB1_NLSF_Q8;
    const SKP_uint8             *CB1_iCDF;
    const SKP_uint8             *pred_Q8;
    const SKP_uint8             *ec_sel;
    const SKP_uint8             *ec_iCDF;
    const SKP_uint8             *ec_Rates_Q5;
    const SKP_int16             *deltaMin_Q15;
} SKP_Silk_NLSF_CB_struct;

typedef struct {
    SKP_int8        GainsIndices[ MAX_NB_SUBFR ];
    SKP_int8        LTPIndex[ MAX_NB_SUBFR ];
    SKP_int8        NLSFIndices[ MAX_LPC_ORDER + 1 ];
    SKP_int16       lagIndex;
    SKP_int8        contourIndex;
    SKP_int8        signalType;
    SKP_int8        quantOffsetType;
    SKP_int8        NLSFInterpCoef_Q2;
    SKP_int8        PERIndex;
    SKP_int8        LTP_scaleIndex;
    SKP_int8        Seed;
} SideInfoIndices;

/********************************/
/* Encoder state                */
/********************************/
typedef struct {
    SKP_int32                       In_HP_State[ 2 ];               /* High pass filter state                                               */
    SKP_int32                       variable_HP_smth1_Q15;          /* State of first smoother                                              */
    SKP_int32                       variable_HP_smth2_Q15;          /* State of second smoother                                             */
    SKP_Silk_LP_state               sLP;                            /* Low pass filter state                                                */
    SKP_Silk_VAD_state              sVAD;                           /* Voice activity detector state                                        */
    SKP_Silk_nsq_state              sNSQ;                           /* Noise Shape Quantizer State                                          */
    SKP_int16                       prev_NLSFq_Q15[ MAX_LPC_ORDER ];/* Previously quantized NLSF vector                                     */
    SKP_int                         speech_activity_Q8;             /* Speech activity                                                      */
    SKP_int8                        LBRRprevLastGainIndex;
    SKP_int8                        prevSignalType;
    SKP_int                         prevLag;
    SKP_int                         pitch_LPC_win_length;
    SKP_int                         max_pitch_lag;                  /* Highest possible pitch lag (samples)                                 */
    SKP_int32                       API_fs_Hz;                      /* API sampling frequency (Hz)                                          */
    SKP_int32                       prev_API_fs_Hz;                 /* Previous API sampling frequency (Hz)                                 */
    SKP_int                         maxInternal_fs_kHz;             /* Maximum internal sampling frequency (kHz)                            */
    SKP_int                         minInternal_fs_kHz;             /* Minimum internal sampling frequency (kHz)                            */
    SKP_int                         fs_kHz;                         /* Internal sampling frequency (kHz)                                    */
    SKP_int                         nb_subfr;                       /* Number of 5 ms subframes in a frame                                  */
    SKP_int                         frame_length;                   /* Frame length (samples)                                               */
    SKP_int                         subfr_length;                   /* Subframe length (samples)                                            */
    SKP_int                         ltp_mem_length;                 /* Length of LTP memory                                                 */
    SKP_int                         la_pitch;                       /* Look-ahead for pitch analysis (samples)                              */
    SKP_int                         la_shape;                       /* Look-ahead for noise shape analysis (samples)                        */
    SKP_int                         shapeWinLength;                 /* Window length for noise shape analysis (samples)                     */
    SKP_int32                       TargetRate_bps;                 /* Target bitrate (bps)                                                 */
    SKP_int                         PacketSize_ms;                  /* Number of milliseconds to put in each packet                         */
    SKP_int                         PacketLoss_perc;                /* Packet loss rate measured by farend                                  */
    SKP_int32                       frameCounter;
    SKP_int                         Complexity;                     /* Complexity setting: 0-> low; 1-> medium; 2->high                     */
    SKP_int                         nStatesDelayedDecision;         /* Number of states in delayed decision quantization                    */
    SKP_int                         useInterpolatedNLSFs;           /* Flag for using NLSF interpolation                                    */
    SKP_int                         shapingLPCOrder;                /* Filter order for noise shaping filters                               */
    SKP_int                         predictLPCOrder;                /* Filter order for prediction filters                                  */
    SKP_int                         pitchEstimationComplexity;      /* Complexity level for pitch estimator                                 */
    SKP_int                         pitchEstimationLPCOrder;        /* Whitening filter order for pitch estimator                           */
    SKP_int32                       pitchEstimationThreshold_Q16;   /* Threshold for pitch estimator                                        */
    SKP_int                         LTPQuantLowComplexity;          /* Flag for low complexity LTP quantization                             */
    SKP_int                         mu_LTP_Q9;                      /* Rate-distortion tradeoff in LTP quantization                         */
    SKP_int                         NLSF_MSVQ_Survivors;            /* Number of survivors in NLSF MSVQ                                     */
    SKP_int                         first_frame_after_reset;        /* Flag for deactivating NLSF interp. and fluc. reduction after resets  */
    SKP_int                         controlled_since_last_payload;  /* Flag for ensuring codec_control only runs once per packet            */
	SKP_int                         warping_Q16;                    /* Warping parameter for warped noise shaping                           */
    SKP_int                         useCBR;                         /* Flag to enable constant bitrate                                      */
    SKP_int                         prev_nBits;                     /* Use to track bits used by each frame in packet                       */
    SKP_int                         prefillFlag;                    /* Flag to indicate that only buffers are prefilled, no coding          */
    const SKP_uint8                 *pitch_lag_low_bits_iCDF;       /* Pointer to iCDF table for low bits of pitch lag index                */
    const SKP_uint8                 *pitch_contour_iCDF;            /* Pointer to iCDF table for pitch contour index                        */
    const SKP_Silk_NLSF_CB_struct   *psNLSF_CB;                     /* Pointer to NLSF codebook                                             */
    SKP_int                         input_quality_bands_Q15[ VAD_N_BANDS ];
    SKP_int                         input_tilt_Q15;

    SKP_int8                        VAD_flags[ MAX_FRAMES_PER_PACKET ];
    SKP_int8                        LBRR_flag;
    SKP_int                         LBRR_flags[ MAX_FRAMES_PER_PACKET ];

    SideInfoIndices                 indices;
	SKP_int8                        pulses[ MAX_FRAME_LENGTH ];

    /* Input/output buffering */
    SKP_int16                       inputBuf[ MAX_FRAME_LENGTH ];   /* buffer containing input signal                                       */
    SKP_int                         inputBufIx;
    SKP_int                         nFramesPerPacket;
    SKP_int                         nFramesAnalyzed;                /* Number of frames analyzed in current packet                          */

    /* Parameters For LTP scaling Control */
    SKP_int                         frames_since_onset;

    /* Specifically for entropy coding */
    SKP_int                         ec_prevSignalType;
    SKP_int16                       ec_prevLagIndex;

    SKP_Silk_resampler_state_struct resampler_state;

    /* DTX */
    SKP_int                         useDTX;                         /* Flag to enable DTX                                                   */
    SKP_int                         inDTX;                          /* Flag to signal DTX period                                            */
    SKP_int                         noSpeechCounter;                /* Counts concecutive nonactive frames, used by DTX                     */

    /* Inband Low Bitrate Redundancy (LBRR) data */ 
    SKP_int                         useInBandFEC;                   /* Saves the API setting for query                                      */
    SKP_int                         LBRR_enabled;                   /* Depends on useInBandFRC, bitrate and packet loss rate                */
    SKP_int                         LBRR_GainIncreases;             /* Number of shifts to Gains to get LBRR rate Voiced frames             */
    SideInfoIndices                 indices_LBRR[ MAX_FRAMES_PER_PACKET ];
	SKP_int8                        pulses_LBRR[ MAX_FRAMES_PER_PACKET ][ MAX_FRAME_LENGTH ];
} SKP_Silk_encoder_state;


/* Struct for Packet Loss Concealment */
typedef struct {
    SKP_int32   pitchL_Q8;                      /* Pitch lag to use for voiced concealment                  */
    SKP_int16   LTPCoef_Q14[ LTP_ORDER ];       /* LTP coeficients to use for voiced concealment            */
    SKP_int16   prevLPC_Q12[ MAX_LPC_ORDER ];
    SKP_int     last_frame_lost;                /* Was previous frame lost                                  */
    SKP_int32   rand_seed;                      /* Seed for unvoiced signal generation                      */
    SKP_int16   randScale_Q14;                  /* Scaling of unvoiced random signal                        */
    SKP_int32   conc_energy;
    SKP_int     conc_energy_shift;
    SKP_int16   prevLTP_scale_Q14;
    SKP_int32   prevGain_Q16[ MAX_NB_SUBFR ];
    SKP_int     fs_kHz;
} SKP_Silk_PLC_struct;

/* Struct for CNG */
typedef struct {
    SKP_int32   CNG_exc_buf_Q10[ MAX_FRAME_LENGTH ];
    SKP_int16   CNG_smth_NLSF_Q15[ MAX_LPC_ORDER ];
    SKP_int32   CNG_synth_state[ MAX_LPC_ORDER ];
    SKP_int32   CNG_smth_Gain_Q16;
    SKP_int32   rand_seed;
    SKP_int     fs_kHz;
} SKP_Silk_CNG_struct;

/********************************/
/* Decoder state                */
/********************************/
typedef struct {
    SKP_int32       prev_inv_gain_Q16;
    SKP_int32       sLTP_Q16[ 2 * MAX_FRAME_LENGTH ];
    SKP_int32       sLPC_Q14[ MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER ];
    SKP_int32       exc_Q10[ MAX_FRAME_LENGTH ];
    SKP_int16       outBuf[ 2 * MAX_FRAME_LENGTH ];             /* Buffer for output signal                                             */
    SKP_int         lagPrev;                                    /* Previous Lag                                                         */
    SKP_int8        LastGainIndex;                              /* Previous gain index                                                  */
    SKP_int32       HPState[ DEC_HP_ORDER ];                    /* HP filter state                                                      */
    const SKP_int32 *HP_A;                                      /* HP filter AR coefficients                                            */
    const SKP_int32 *HP_B;                                      /* HP filter MA coefficients                                            */
    SKP_int         fs_kHz;                                     /* Sampling frequency in kHz                                            */
    SKP_int32       prev_API_sampleRate;                        /* Previous API sample frequency (Hz)                                   */
    SKP_int         nb_subfr;                                   /* Number of 5 ms subframes in a frame                                  */
    SKP_int         frame_length;                               /* Frame length (samples)                                               */
    SKP_int         subfr_length;                               /* Subframe length (samples)                                            */
    SKP_int         ltp_mem_length;                             /* Length of LTP memory                                                 */
    SKP_int         LPC_order;                                  /* LPC order                                                            */
    SKP_int16       prevNLSF_Q15[ MAX_LPC_ORDER ];              /* Used to interpolate LSFs                                             */
    SKP_int         first_frame_after_reset;                    /* Flag for deactivating NLSF interp. and fluc. reduction after resets  */
    const SKP_uint8 *pitch_lag_low_bits_iCDF;                   /* Pointer to iCDF table for low bits of pitch lag index                */
    const SKP_uint8 *pitch_contour_iCDF;                        /* Pointer to iCDF table for pitch contour index                        */

    /* For buffering payload in case of more frames per packet */
    SKP_int         nFramesDecoded;
    SKP_int         nFramesPerPacket;

    /* Specifically for entropy coding */
    SKP_int         ec_prevSignalType;
    SKP_int16       ec_prevLagIndex;

    SKP_int         VAD_flags[ MAX_FRAMES_PER_PACKET ];
    SKP_int         LBRR_flag;
    SKP_int         LBRR_flags[ MAX_FRAMES_PER_PACKET ];

    SKP_Silk_resampler_state_struct resampler_state;

    const SKP_Silk_NLSF_CB_struct   *psNLSF_CB;                 /* Pointer to NLSF codebook                                             */

    /* Quantization indices */
    SideInfoIndices indices;
    
    /* CNG state */
    SKP_Silk_CNG_struct sCNG;

    /* Stuff used for PLC */
    SKP_int         lossCnt;
    SKP_int         prevSignalType;

    SKP_Silk_PLC_struct sPLC;

} SKP_Silk_decoder_state;

/************************/
/* Decoder control      */
/************************/
typedef struct {
    /* prediction and coding parameters */
    SKP_int             pitchL[ MAX_NB_SUBFR ];
    SKP_int32           Gains_Q16[ MAX_NB_SUBFR ];
    /* holds interpolated and final coefficients, 4-byte aligned */
    SKP_DWORD_ALIGN SKP_int16 PredCoef_Q12[ 2 ][ MAX_LPC_ORDER ];
    SKP_int16           LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ];
    SKP_int             LTP_scale_Q14;
} SKP_Silk_decoder_control;

#ifdef __cplusplus
}
#endif

#endif
