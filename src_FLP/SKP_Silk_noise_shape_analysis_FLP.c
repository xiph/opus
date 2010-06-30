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

#include "SKP_Silk_main_FLP.h"
#include "SKP_Silk_perceptual_parameters.h"

/* Compute gain to make warped filter coefficients have a zero mean log frequency response on a     */
/* non-warped frequency scale. (So that it can be implemented with a minimum-phase monic filter.)   */
SKP_INLINE SKP_float warped_gain( 
    const SKP_float     *coefs, 
    SKP_float           lambda, 
    SKP_int             order 
) {
    SKP_int   i;
    SKP_float gain;

    lambda = -lambda;
    gain = coefs[ order - 1 ];
    for( i = order - 2; i >= 0; i-- ) {
        gain = lambda * gain + coefs[ i ];
    }
    return (SKP_float)( 1.0f / ( 1.0f - lambda * gain ) );
}

/* Convert warped filter coefficients to monic pseudo-warped coefficients */
SKP_INLINE void warped_true2monic_coefs( 
    SKP_float           *coefs, 
    SKP_float           lambda, 
    SKP_int             order 
) {
    SKP_int   i;
    SKP_float gain;

    lambda = -lambda;
    for( i = order - 1; i > 0; i-- ) {
        coefs[ i - 1 ] += lambda * coefs[ i ];
    }
    gain = ( 1.0f - lambda * lambda ) / ( 1.0f - lambda * coefs[ 0 ] );
    for( i = 0; i < order; i++ ) {
        coefs[ i ] *= gain;
    }
}

/* Limit max amplitude of monic warped coefficients by using bandwidth expansion on the true coefficients */
SKP_INLINE void limit_warped_coefs( 
    SKP_float           *coefs_syn,
    SKP_float           *coefs_ana,
    SKP_float           lambda,
    SKP_float           limit,
    SKP_int             order
) {
    SKP_int   i, iter, ind;
    SKP_float tmp, maxabs, chirp;

    for( iter = 0; iter < 10; iter++ ) {
        /* Find maximum absolute value */
        ind = 1;
        maxabs = SKP_abs( coefs_syn[ ind ] );
        for( i = 2; i < order - 1; i++ ) {
            tmp = SKP_abs( coefs_syn[ i ] );
            if( tmp > maxabs ) {
                maxabs = tmp;
                ind = i;
            }
        }
        if( maxabs <= limit ) {
            return;
        }

        /* Convert to true warped coefficients */
        for( i = 1; i < order; i++ ) {
            coefs_syn[ i - 1 ] += lambda * coefs_syn[ i ];
            coefs_ana[ i - 1 ] += lambda * coefs_ana[ i ];
        }

        /* Apply bandwidth expansion */
        chirp = 0.99f - ( 0.8f + 0.1f * iter ) * ( maxabs - limit ) / ( maxabs * ( ind + 1 ) );
        SKP_Silk_bwexpander_FLP( coefs_syn, order, chirp );
        SKP_Silk_bwexpander_FLP( coefs_ana, order, chirp );

        /* Convert back to monic warped coefficients */
        for( i = order - 1; i > 0; i-- ) {
            coefs_syn[ i - 1 ] -= lambda * coefs_syn[ i ];
            coefs_ana[ i - 1 ] -= lambda * coefs_ana[ i ];
        }
    }
}

/* Compute noise shaping coefficients and initial gain values */
void SKP_Silk_noise_shape_analysis_FLP(
    SKP_Silk_encoder_state_FLP      *psEnc,             /* I/O  Encoder state FLP                       */
    SKP_Silk_encoder_control_FLP    *psEncCtrl,         /* I/O  Encoder control FLP                     */
    const SKP_float                 *pitch_res,         /* I    LPC residual from pitch analysis        */
    const SKP_float                 *x                  /* I    Input signal [frame_length + la_shape]  */
)
{
    SKP_Silk_shape_state_FLP *psShapeSt = &psEnc->sShape;
    SKP_int     k, nSamples;
    SKP_float   SNR_adj_dB, HarmBoost, HarmShapeGain, Tilt;
    SKP_float   nrg, pre_nrg, log_energy, log_energy_prev, energy_variation;
    SKP_float   delta, BWExp1, BWExp2, gain_mult, gain_add, strength, b, warping;
    SKP_float   x_windowed[ SHAPE_LPC_WIN_MAX ];
    SKP_float   auto_corr[ MAX_SHAPE_LPC_ORDER + 1 ];
    const SKP_float *x_ptr, *pitch_res_ptr;

    /* Point to start of first LPC analysis block */
    x_ptr = x + psEnc->sCmn.la_shape - SHAPE_LPC_WIN_MS * psEnc->sCmn.fs_kHz + psEnc->sCmn.subfr_length;

    /****************/
    /* CONTROL SNR  */
    /****************/
    /* Reduce SNR_dB values if recent bitstream has exceeded TargetRate */
    psEncCtrl->current_SNR_dB = psEnc->SNR_dB - 0.05f * psEnc->BufferedInChannel_ms;

    /* Reduce SNR_dB if inband FEC used */
    if( psEnc->speech_activity > LBRR_SPEECH_ACTIVITY_THRES ) {
        psEncCtrl->current_SNR_dB -= psEnc->inBandFEC_SNR_comp;
    }

    /****************/
    /* GAIN CONTROL */
    /****************/
    /* Input quality is the average of the quality in the lowest two VAD bands */
    psEncCtrl->input_quality = 0.5f * ( psEncCtrl->input_quality_bands[ 0 ] + psEncCtrl->input_quality_bands[ 1 ] );

    /* Coding quality level, between 0.0 and 1.0 */
    psEncCtrl->coding_quality = SKP_sigmoid( 0.25f * ( psEncCtrl->current_SNR_dB - 17.0f ) );

    /* Reduce coding SNR during low speech activity */
    b = 1.0f - psEnc->speech_activity;
    SNR_adj_dB = psEncCtrl->current_SNR_dB - 
        BG_SNR_DECR_dB * psEncCtrl->coding_quality * ( 0.5f + 0.5f * psEncCtrl->input_quality ) * b * b;

    if( psEncCtrl->sCmn.sigtype == SIG_TYPE_VOICED ) {
        /* Reduce gains for periodic signals */
        SNR_adj_dB += HARM_SNR_INCR_dB * psEnc->LTPCorr;
    } else { 
        /* For unvoiced signals and low-quality input, adjust the quality slower than SNR_dB setting */
        SNR_adj_dB += ( -0.4f * psEncCtrl->current_SNR_dB + 6.0f ) * ( 1.0f - psEncCtrl->input_quality );
    }

    /*************************/
    /* SPARSENESS PROCESSING */
    /*************************/
    /* Set quantizer offset */
    if( psEncCtrl->sCmn.sigtype == SIG_TYPE_VOICED ) {
        /* Initally set to 0; may be overruled in process_gains(..) */
        psEncCtrl->sCmn.QuantOffsetType = 0;
        psEncCtrl->sparseness = 0.0f;
    } else {
        /* Sparseness measure, based on relative fluctuations of energy per 2 milliseconds */
        nSamples = 2 * psEnc->sCmn.fs_kHz;
        energy_variation = 0.0f;
        log_energy_prev  = 0.0f;
        pitch_res_ptr = pitch_res;
        for( k = 0; k < SKP_SMULBB( SUB_FRAME_LENGTH_MS, psEnc->sCmn.nb_subfr ) / 2; k++ ) {
            nrg = ( SKP_float )nSamples + ( SKP_float )SKP_Silk_energy_FLP( pitch_res_ptr, nSamples );
            log_energy = SKP_Silk_log2( nrg );
            if( k > 0 ) {
                energy_variation += SKP_abs_float( log_energy - log_energy_prev );
            }
            log_energy_prev = log_energy;
            pitch_res_ptr += nSamples;
        }
        psEncCtrl->sparseness = SKP_sigmoid( 0.4f * ( energy_variation - 5.0f ) );

        /* Set quantization offset depending on sparseness measure */
        if( psEncCtrl->sparseness > SPARSENESS_THRESHOLD_QNT_OFFSET ) {
            psEncCtrl->sCmn.QuantOffsetType = 0;
        } else {
            psEncCtrl->sCmn.QuantOffsetType = 1;
        }
        
        /* Increase coding SNR for sparse signals */
        SNR_adj_dB += SPARSE_SNR_INCR_dB * ( psEncCtrl->sparseness - 0.5f );
    }

    /*******************************/
    /* Control bandwidth expansion */
    /*******************************/
    delta  = LOW_RATE_BANDWIDTH_EXPANSION_DELTA * ( 1.0f - 0.75f * psEncCtrl->coding_quality );
    BWExp1 = BANDWIDTH_EXPANSION - delta;
    BWExp2 = BANDWIDTH_EXPANSION + delta;
    if( psEnc->sCmn.fs_kHz == 24 ) {
        /* Less bandwidth expansion for super wideband */
        BWExp1 = 1.0f - ( 1.0f - BWExp1 ) * SWB_BANDWIDTH_EXPANSION_REDUCTION;
        BWExp2 = 1.0f - ( 1.0f - BWExp2 ) * SWB_BANDWIDTH_EXPANSION_REDUCTION;
    }
    /* BWExp1 will be applied after BWExp2, so make it relative */
    BWExp1 /= BWExp2;

    /* Warping coefficient */
    psEncCtrl->sCmn.warping_Q16 = psEnc->sCmn.fs_kHz * WARPING_MULTIPLIER_Q16;
    psEncCtrl->sCmn.warping_Q16 = SKP_min( psEncCtrl->sCmn.warping_Q16, 32767 );
    warping = (SKP_float)psEncCtrl->sCmn.warping_Q16 / 65536.0f;

    /********************************************/
    /* Compute noise shaping AR coefs and gains */
    /********************************************/
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        /* Apply window */
        SKP_Silk_apply_sine_window_FLP( x_windowed, x_ptr, 0, SHAPE_LPC_WIN_MS * psEnc->sCmn.fs_kHz );

        /* Update pointer: next LPC analysis block */
        x_ptr += psEnc->sCmn.subfr_length;

        /* Calculate warped auto correlation */
        SKP_Silk_warped_autocorrelation_FLP( auto_corr, x_windowed, warping, 
            SHAPE_LPC_WIN_MS * psEnc->sCmn.fs_kHz, psEnc->sCmn.shapingLPCOrder );

        /* Add white noise, as a fraction of energy */
        auto_corr[ 0 ] += auto_corr[ 0 ] * SHAPE_WHITE_NOISE_FRACTION; 

        /* Convert correlations to prediction coefficients, and compute residual energy */
        nrg = SKP_Silk_levinsondurbin_FLP( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], auto_corr, psEnc->sCmn.shapingLPCOrder );

        /* Convert residual energy to non-warped scale */
        gain_mult = warped_gain( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], warping, psEnc->sCmn.shapingLPCOrder );
        nrg *= gain_mult * gain_mult; 

        /* Bandwidth expansion for synthesis filter shaping */
        SKP_Silk_bwexpander_FLP( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], psEnc->sCmn.shapingLPCOrder, BWExp2 );

        /* Compute noise shaping filter coefficients */
        SKP_memcpy(
            &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ], 
            &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], 
            psEnc->sCmn.shapingLPCOrder * sizeof( SKP_float ) );

        /* Bandwidth expansion for analysis filter shaping */
        SKP_Silk_bwexpander_FLP( &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ], psEnc->sCmn.shapingLPCOrder, BWExp1 );

        /* Increase residual energy */
        nrg += SHAPE_MIN_ENERGY_RATIO * auto_corr[ 0 ];
        psEncCtrl->Gains[ k ] = ( SKP_float )sqrt( nrg );
        
        /* Ratio of prediction gains, in energy domain */
        SKP_Silk_LPC_inverse_pred_gain_FLP( &pre_nrg, &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], psEnc->sCmn.shapingLPCOrder );
        SKP_Silk_LPC_inverse_pred_gain_FLP( &nrg,     &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ], psEnc->sCmn.shapingLPCOrder );
        psEncCtrl->GainsPre[ k ] = 1.0f - 0.7f * ( 1.0f - pre_nrg / nrg );

        /* Convert to monic warped prediction coefficients */
        warped_true2monic_coefs( &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ], warping, psEnc->sCmn.shapingLPCOrder );
        warped_true2monic_coefs( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], warping, psEnc->sCmn.shapingLPCOrder );

        /* Limit absolute values */
        limit_warped_coefs( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ], 
            warping, 3.999f, psEnc->sCmn.shapingLPCOrder );
    }

    /*****************/
    /* Gain tweaking */
    /*****************/
    /* Increase gains during low speech activity and put lower limit on gains */
    gain_mult = ( SKP_float )pow( 2.0f, -0.16f * SNR_adj_dB );
    gain_add  = ( SKP_float )pow( 2.0f,  0.16f * NOISE_FLOOR_dB ) + 
                ( SKP_float )pow( 2.0f,  0.16f * RELATIVE_MIN_GAIN_dB ) * psEnc->avgGain;
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        psEncCtrl->Gains[ k ] *= gain_mult;
        psEncCtrl->Gains[ k ] += gain_add;
        psEnc->avgGain += psEnc->speech_activity * GAIN_SMOOTHING_COEF * ( psEncCtrl->Gains[ k ] - psEnc->avgGain );
    }

    /************************************************/
    /* Decrease level during fricatives (de-essing) */
    /************************************************/
    gain_mult = 1.0f + INPUT_TILT + psEncCtrl->coding_quality * HIGH_RATE_INPUT_TILT;
    if( psEncCtrl->input_tilt <= 0.0f && psEncCtrl->sCmn.sigtype == SIG_TYPE_UNVOICED ) {
        SKP_float essStrength = -psEncCtrl->input_tilt * psEnc->speech_activity * ( 1.0f - psEncCtrl->sparseness );
        if( psEnc->sCmn.fs_kHz == 24 ) {
            gain_mult *= ( SKP_float )pow( 2.0f, -0.16f * DE_ESSER_COEF_SWB_dB * essStrength );
        } else if( psEnc->sCmn.fs_kHz == 16 ) {
            gain_mult *= (SKP_float)pow( 2.0f, -0.16f * DE_ESSER_COEF_WB_dB * essStrength );
        } else {
            SKP_assert( psEnc->sCmn.fs_kHz == 12 || psEnc->sCmn.fs_kHz == 8 );
        }
    }

    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        psEncCtrl->GainsPre[ k ] *= gain_mult;
    }

    /************************************************/
    /* Control low-frequency shaping and noise tilt */
    /************************************************/
    /* Less low frequency shaping for noisy inputs */
    strength = LOW_FREQ_SHAPING * ( 1.0f + LOW_QUALITY_LOW_FREQ_SHAPING_DECR * ( psEncCtrl->input_quality_bands[ 0 ] - 1.0f ) );
    if( psEncCtrl->sCmn.sigtype == SIG_TYPE_VOICED ) {
        /* Reduce low frequencies quantization noise for periodic signals, depending on pitch lag */
        /*f = 400; freqz([1, -0.98 + 2e-4 * f], [1, -0.97 + 7e-4 * f], 2^12, Fs); axis([0, 1000, -10, 1])*/
        for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
            b = 0.2f / psEnc->sCmn.fs_kHz + 3.0f / psEncCtrl->sCmn.pitchL[ k ];
            psEncCtrl->LF_MA_shp[ k ] = -1.0f + b;
            psEncCtrl->LF_AR_shp[ k ] =  1.0f - b - b * strength;
        }
        Tilt = - HP_NOISE_COEF - 
            (1 - HP_NOISE_COEF) * HARM_HP_NOISE_COEF * psEnc->speech_activity;
    } else {
        b = 1.3f / psEnc->sCmn.fs_kHz;
        psEncCtrl->LF_MA_shp[ 0 ] = -1.0f + b;
        psEncCtrl->LF_AR_shp[ 0 ] =  1.0f - b - b * strength * 0.6f;
        for( k = 1; k < psEnc->sCmn.nb_subfr; k++ ) {
            psEncCtrl->LF_MA_shp[ k ] = psEncCtrl->LF_MA_shp[ k - 1 ];
            psEncCtrl->LF_AR_shp[ k ] = psEncCtrl->LF_AR_shp[ k - 1 ];
        }
        Tilt = -HP_NOISE_COEF;
    }

    /****************************/
    /* HARMONIC SHAPING CONTROL */
    /****************************/
    /* Control boosting of harmonic frequencies */
    HarmBoost = LOW_RATE_HARMONIC_BOOST * ( 1.0f - psEncCtrl->coding_quality ) * psEnc->LTPCorr;

    /* More harmonic boost for noisy input signals */
    HarmBoost += LOW_INPUT_QUALITY_HARMONIC_BOOST * ( 1.0f - psEncCtrl->input_quality );

    if( USE_HARM_SHAPING && psEncCtrl->sCmn.sigtype == SIG_TYPE_VOICED ) {
        /* Harmonic noise shaping */
        HarmShapeGain = HARMONIC_SHAPING;

        /* More harmonic noise shaping for high bitrates or noisy input */
        HarmShapeGain += HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING * 
            ( 1.0f - ( 1.0f - psEncCtrl->coding_quality ) * psEncCtrl->input_quality );

        /* Less harmonic noise shaping for less periodic signals */
        HarmShapeGain *= ( SKP_float )sqrt( psEnc->LTPCorr );
    } else {
        HarmShapeGain = 0.0f;
    }

    /*************************/
    /* Smooth over subframes */
    /*************************/
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        psShapeSt->HarmBoost_smth     += SUBFR_SMTH_COEF * ( HarmBoost - psShapeSt->HarmBoost_smth );
        psEncCtrl->HarmBoost[ k ]      = psShapeSt->HarmBoost_smth;
        psShapeSt->HarmShapeGain_smth += SUBFR_SMTH_COEF * ( HarmShapeGain - psShapeSt->HarmShapeGain_smth );
        psEncCtrl->HarmShapeGain[ k ]  = psShapeSt->HarmShapeGain_smth;
        psShapeSt->Tilt_smth          += SUBFR_SMTH_COEF * ( Tilt - psShapeSt->Tilt_smth );
        psEncCtrl->Tilt[ k ]           = psShapeSt->Tilt_smth;
    }
}
