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

#ifndef SKP_SILK_PERCEPTUAL_PARAMETERS_H
#define SKP_SILK_PERCEPTUAL_PARAMETERS_H

#ifdef __cplusplus
extern "C"
{
#endif

/* reduction in coding SNR during low speech activity */
#define BG_SNR_DECR_dB                              4.0f

/* factor for reducing quantization noise during voiced speech */
#define HARM_SNR_INCR_dB                            2.0f

/* factor for reducing quantization noise for unvoiced sparse signals */
#define SPARSE_SNR_INCR_dB                          2.0f

/* threshold for sparseness measure above which to use lower quantization offset during unvoiced */
#define SPARSENESS_THRESHOLD_QNT_OFFSET             0.75f


/* noise shaping filter chirp factor */
#define BANDWIDTH_EXPANSION                         0.95f

/* difference between chirp factors for analysis and synthesis noise shaping filters at low bitrates */
#define LOW_RATE_BANDWIDTH_EXPANSION_DELTA          0.01f

/* factor to reduce all bandwdith expansion coefficients for super wideband, relative to wideband */
#define SWB_BANDWIDTH_EXPANSION_REDUCTION           1.0f

/* gain reduction for fricatives */
#define DE_ESSER_COEF_SWB_dB                        2.0f
#define DE_ESSER_COEF_WB_dB                         1.0f

/* extra harmonic boosting (signal shaping) at low bitrates */
#define LOW_RATE_HARMONIC_BOOST                     0.1f

/* extra harmonic boosting (signal shaping) for noisy input signals */
#define LOW_INPUT_QUALITY_HARMONIC_BOOST            0.1f

/* harmonic noise shaping */
#define HARMONIC_SHAPING                            0.3f

/* extra harmonic noise shaping for high bitrates or noisy input */
#define HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING   0.2f


/* parameter for shaping noise towards higher frequencies */
#define HP_NOISE_COEF                               0.3f

/* parameter for shaping noise extra towards higher frequencies during voiced speech */
#define HARM_HP_NOISE_COEF                          0.4f

/* parameter for applying a high-pass tilt to the input signal */
#define INPUT_TILT                                  0.04f

/* parameter for extra high-pass tilt to the input signal at high rates */
#define HIGH_RATE_INPUT_TILT                        0.04f

/* parameter for reducing noise at the very low frequencies */
#define LOW_FREQ_SHAPING                            3.0f

/* less reduction of noise at the very low frequencies for signals with low SNR at low frequencies */
#define LOW_QUALITY_LOW_FREQ_SHAPING_DECR           0.5f

/* fraction added to first autocorrelation value */
#define SHAPE_WHITE_NOISE_FRACTION                  3.8147e-5f

/* fraction of first autocorrelation value added to residual energy value; limits prediction gain */
#define SHAPE_MIN_ENERGY_RATIO                      1.526e-5f       // 1.526e-5 = 1/65536

/* noise floor to put a low limit on the quantization step size */
#define NOISE_FLOOR_dB                              4.0f

/* noise floor relative to active speech gain level */
#define RELATIVE_MIN_GAIN_dB                        -50.0f

/* subframe smoothing coefficient for determining active speech gain level (lower -> more smoothing) */
#define GAIN_SMOOTHING_COEF                         1e-3f

/* subframe smoothing coefficient for HarmBoost, HarmShapeGain, Tilt (lower -> more smoothing) */
#define SUBFR_SMTH_COEF                             0.4f

#ifdef __cplusplus
}
#endif

#endif
