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

#ifndef SKP_SILK_DEFINE_FLP_H
#define SKP_SILK_DEFINE_FLP_H

#ifdef __cplusplus
extern "C"
{
#endif

/*******************/
/* Pitch estimator */
/*******************/

/* Level of noise floor for whitening filter LPC analysis in pitch analysis */
#define FIND_PITCH_WHITE_NOISE_FRACTION                 1e-3f

/* Bandwidth expansion for whitening filter in pitch analysis */
#define FIND_PITCH_BANDWITH_EXPANSION                   0.99f

/* Threshold used by pitch estimator for early escape */
#define FIND_PITCH_CORRELATION_THRESHOLD_HC_MODE        0.7f
#define FIND_PITCH_CORRELATION_THRESHOLD_MC_MODE        0.75f
#define FIND_PITCH_CORRELATION_THRESHOLD_LC_MODE        0.8f

/***********************/
/* Long-Term predictor */
/***********************/

/* Regualarization factor for correlation matrix. Equivalent to adding noise at -50 dB */
#define FIND_LTP_COND_FAC                               1e-5f
#define FIND_LPC_COND_FAC                               6e-5f

/* Find prediction coefficients defines */
#define LTP_DAMPING                                     0.001f
#define LTP_SMOOTHING                                   0.1f

/* LTP quantization settings */
#define MU_LTP_QUANT_NB                                 0.03f
#define MU_LTP_QUANT_MB                                 0.025f
#define MU_LTP_QUANT_WB                                 0.02f
#define MU_LTP_QUANT_SWB                                0.016f

/***********************/
/* High pass filtering */
/***********************/

/* Smoothing parameters for low end of pitch frequency range estimation */
#define VARIABLE_HP_SMTH_COEF1                          0.1f
#define VARIABLE_HP_SMTH_COEF2                          0.015f

/* Min and max values for low end of pitch frequency range estimation */
#define VARIABLE_HP_MIN_FREQ                            80.0f
#define VARIABLE_HP_MAX_FREQ                            150.0f

/* Max absolute difference between log2 of pitch frequency and smoother state, to enter the smoother */
#define VARIABLE_HP_MAX_DELTA_FREQ                      0.4f

/***********/
/* Various */
/***********/

/* Required speech activity for counting frame as active */
#define WB_DETECT_ACTIVE_SPEECH_LEVEL_THRES             0.7f        

#define SPEECH_ACTIVITY_DTX_THRES                       0.1f

/* Speech Activity LBRR enable threshold (needs tuning) */
#define LBRR_SPEECH_ACTIVITY_THRES                      0.5f        

#define Q14_CONVERSION_FAC                              6.1035e-005f // 1 / 2^14


#ifdef __cplusplus
}
#endif

#endif
