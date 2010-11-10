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
#include "SKP_Silk_tuning_parameters.h"

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

/* Convert warped filter coefficients to monic pseudo-warped coefficients and limit maximum     */
/* amplitude of monic warped coefficients by using bandwidth expansion on the true coefficients */
SKP_INLINE void warped_true2monic_coefs( 
    SKP_float           *coefs_syn,
    SKP_float           *coefs_ana,
    SKP_float           lambda,
    SKP_float           limit,
    SKP_int             order
) {
    SKP_int   i, iter, ind = 0;
    SKP_float tmp, maxabs, chirp, gain_syn, gain_ana;

    /* Convert to monic coefficients */
    for( i = order - 1; i > 0; i-- ) {
        coefs_syn[ i - 1 ] -= lambda * coefs_syn[ i ];
        coefs_ana[ i - 1 ] -= lambda * coefs_ana[ i ];
    } 
    gain_syn = ( 1.0f - lambda * lambda ) / ( 1.0f + lambda * coefs_syn[ 0 ] );
    gain_ana = ( 1.0f - lambda * lambda ) / ( 1.0f + lambda * coefs_ana[ 0 ] );
    for( i = 0; i < order; i++ ) {
        coefs_syn[ i ] *= gain_syn;
        coefs_ana[ i ] *= gain_ana;
    }

    /* Limit */
    for( iter = 0; iter < 10; iter++ ) {
        /* Find maximum absolute value */
        maxabs = -1.0f;
        for( i = 0; i < order; i++ ) {
            tmp = SKP_max( SKP_abs_float( coefs_syn[ i ] ), SKP_abs_float( coefs_ana[ i ] ) );
            if( tmp > maxabs ) {
                maxabs = tmp;
                ind = i;
            }
        }
        if( maxabs <= limit ) {
            /* Coefficients are within range - done */
            return;
        }

        /* Convert back to true warped coefficients */
        for( i = 1; i < order; i++ ) {
            coefs_syn[ i - 1 ] += lambda * coefs_syn[ i ];
            coefs_ana[ i - 1 ] += lambda * coefs_ana[ i ];
        }
        gain_syn = 1.0f / gain_syn;
        gain_ana = 1.0f / gain_ana;
        for( i = 0; i < order; i++ ) {
            coefs_syn[ i ] *= gain_syn;
            coefs_ana[ i ] *= gain_ana;
        }

        /* Apply bandwidth expansion */
        chirp = 0.99f - ( 0.8f + 0.1f * iter ) * ( maxabs - limit ) / ( maxabs * ( ind + 1 ) );
        SKP_Silk_bwexpander_FLP( coefs_syn, order, chirp );
        SKP_Silk_bwexpander_FLP( coefs_ana, order, chirp );

        /* Convert to monic warped coefficients */
        for( i = order - 1; i > 0; i-- ) {
            coefs_syn[ i - 1 ] -= lambda * coefs_syn[ i ];
            coefs_ana[ i - 1 ] -= lambda * coefs_ana[ i ];
        }
        gain_syn = ( 1.0f - lambda * lambda ) / ( 1.0f + lambda * coefs_syn[ 0 ] );
        gain_ana = ( 1.0f - lambda * lambda ) / ( 1.0f + lambda * coefs_ana[ 0 ] );
        for( i = 0; i < order; i++ ) {
            coefs_syn[ i ] *= gain_syn;
            coefs_ana[ i ] *= gain_ana;
        }
    }
	SKP_assert( 0 );
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
    x_ptr = x - psEnc->sCmn.la_shape;

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
    psEncCtrl->coding_quality = SKP_sigmoid( 0.25f * ( psEncCtrl->current_SNR_dB - 18.0f ) );

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
    /* More BWE for signals with high prediction gain */
    strength = FIND_PITCH_WHITE_NOISE_FRACTION * psEncCtrl->predGain;           /* between 0.0 and 1.0 */
    BWExp1 = BWExp2 = BANDWIDTH_EXPANSION / ( 1.0f + strength * strength );
    delta  = LOW_RATE_BANDWIDTH_EXPANSION_DELTA * ( 1.0f - 0.75f * psEncCtrl->coding_quality );
    BWExp1 -= delta;
    BWExp2 += delta;
    /* BWExp1 will be applied after BWExp2, so make it relative */
    BWExp1 /= BWExp2;

    if( psEnc->sCmn.warping_Q16 > 0 ) {
        /* Slightly more warping in analysis will move quantization noise up in frequency, where it's better masked */
        warping = (SKP_float)psEnc->sCmn.warping_Q16 / 65536.0f + 0.01f * psEncCtrl->coding_quality;
    } else {
        warping = 0.0f;
    }

    /********************************************/
    /* Compute noise shaping AR coefs and gains */
    /********************************************/
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        /* Apply window: sine slope followed by flat part followed by cosine slope */
        SKP_int shift, slope_part, flat_part;
        flat_part = psEnc->sCmn.fs_kHz * 3;
        slope_part = ( psEnc->sCmn.shapeWinLength - flat_part ) / 2;

        SKP_Silk_apply_sine_window_FLP( x_windowed, x_ptr, 1, slope_part );
        shift = slope_part;
        SKP_memcpy( x_windowed + shift, x_ptr + shift, flat_part * sizeof(SKP_float) );
        shift += flat_part;
        SKP_Silk_apply_sine_window_FLP( x_windowed + shift, x_ptr + shift, 2, slope_part );

        /* Update pointer: next LPC analysis block */
        x_ptr += psEnc->sCmn.subfr_length;

        if( psEnc->sCmn.warping_Q16 > 0 ) {
            /* Calculate warped auto correlation */
            SKP_Silk_warped_autocorrelation_FLP( auto_corr, x_windowed, warping, 
                psEnc->sCmn.shapeWinLength, psEnc->sCmn.shapingLPCOrder );
        } else {
            /* Calculate regular auto correlation */
            SKP_Silk_autocorrelation_FLP( auto_corr, x_windowed, psEnc->sCmn.shapeWinLength, psEnc->sCmn.shapingLPCOrder + 1 );
        }

        /* Add white noise, as a fraction of energy */
        auto_corr[ 0 ] += auto_corr[ 0 ] * SHAPE_WHITE_NOISE_FRACTION; 

        /* Convert correlations to prediction coefficients, and compute residual energy */
        nrg = SKP_Silk_levinsondurbin_FLP( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], auto_corr, psEnc->sCmn.shapingLPCOrder );
        psEncCtrl->Gains[ k ] = ( SKP_float )sqrt( nrg );

        if( psEnc->sCmn.warping_Q16 > 0 ) {
            /* Adjust gain for warping */
            psEncCtrl->Gains[ k ] *= warped_gain( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], warping, psEnc->sCmn.shapingLPCOrder );
        }

        /* Bandwidth expansion for synthesis filter shaping */
        SKP_Silk_bwexpander_FLP( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], psEnc->sCmn.shapingLPCOrder, BWExp2 );

        /* Compute noise shaping filter coefficients */
        SKP_memcpy(
            &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ], 
            &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], 
            psEnc->sCmn.shapingLPCOrder * sizeof( SKP_float ) );

        /* Bandwidth expansion for analysis filter shaping */
        SKP_Silk_bwexpander_FLP( &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ], psEnc->sCmn.shapingLPCOrder, BWExp1 );

        /* Ratio of prediction gains, in energy domain */
        SKP_Silk_LPC_inverse_pred_gain_FLP( &pre_nrg, &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], psEnc->sCmn.shapingLPCOrder );
        SKP_Silk_LPC_inverse_pred_gain_FLP( &nrg,     &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ], psEnc->sCmn.shapingLPCOrder );
        psEncCtrl->GainsPre[ k ] = 1.0f - 0.7f * ( 1.0f - pre_nrg / nrg );

        /* Convert to monic warped prediction coefficients and limit absolute values */
        warped_true2monic_coefs( &psEncCtrl->AR2[ k * MAX_SHAPE_LPC_ORDER ], &psEncCtrl->AR1[ k * MAX_SHAPE_LPC_ORDER ], 
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

    gain_mult = 1.0f + INPUT_TILT + psEncCtrl->coding_quality * HIGH_RATE_INPUT_TILT;

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
            psEncCtrl->LF_MA_shp[ k ] = psEncCtrl->LF_MA_shp[ 0 ];
            psEncCtrl->LF_AR_shp[ k ] = psEncCtrl->LF_AR_shp[ 0 ];
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
