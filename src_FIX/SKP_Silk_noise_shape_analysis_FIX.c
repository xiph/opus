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

#include "SKP_Silk_main_FIX.h"
#include "SKP_Silk_perceptual_parameters_FIX.h"

/* Compute gain to make warped filter coefficients have a zero mean log frequency response on a     */
/* non-warped frequency scale. (So that it can be implemented with a minimum-phase monic filter.)   */
SKP_INLINE SKP_int32 warped_gain( // gain in Q16
    const SKP_int32     *coefs_Q24, 
    SKP_int             lambda_Q16, 
    SKP_int             order 
) {
    SKP_int   i;
    SKP_int32 gain_Q24;

    lambda_Q16 = -lambda_Q16;
    gain_Q24 = coefs_Q24[ order - 1 ];
    for( i = order - 2; i >= 0; i-- ) {
        gain_Q24 = SKP_SMLAWB( coefs_Q24[ i ], gain_Q24, lambda_Q16 );
    }
    gain_Q24  = SKP_SMLAWB( 1 << 24, gain_Q24, -lambda_Q16 );
    return SKP_INVERSE32_varQ( gain_Q24, 40 );
}

/* Convert warped filter coefficients to monic pseudo-warped coefficients */
SKP_INLINE void warped_true2monic_coefs( 
    SKP_int32           *coefs_Q24,
    SKP_int             lambda_Q16,
    SKP_int             order 
) {
    SKP_int   i;
    SKP_int32 gain_Q16, nom_Q16, den_Q24;

    lambda_Q16 = -lambda_Q16;
    for( i = order - 1; i > 0; i-- ) {
        coefs_Q24[ i - 1 ] = SKP_SMLAWB( coefs_Q24[ i - 1 ], coefs_Q24[ i ], lambda_Q16 );
    }
    nom_Q16  = SKP_SMLAWB( 1 << 16, lambda_Q16,     -lambda_Q16 );
    den_Q24  = SKP_SMLAWB( 1 << 24, coefs_Q24[ 0 ], -lambda_Q16 );
    gain_Q16 = SKP_DIV32_varQ( nom_Q16, den_Q24, 24 );
    for( i = 0; i < order; i++ ) {
        coefs_Q24[ i ] = SKP_SMULWW( gain_Q16, coefs_Q24[ i ] );
    }
}

/* Limit max amplitude of monic warped coefficients by using bandwidth expansion on the true coefficients */
SKP_INLINE void limit_warped_coefs( 
    SKP_int32           *coefs_syn_Q24,
    SKP_int32           *coefs_ana_Q24,
    SKP_int             lambda_Q16,
    SKP_int32           limit_Q24,
    SKP_int             order
) {
    SKP_int   i, iter, ind;
    SKP_int32 tmp, maxabs_Q24, chirp_Q16;

    for( iter = 0; iter < 10; iter++ ) {
        /* Find maximum absolute value */
        ind = 1;
        maxabs_Q24 = SKP_abs( coefs_syn_Q24[ ind ] );
        for( i = 2; i < order - 1; i++ ) {
            tmp = SKP_abs( coefs_syn_Q24[ i ] );
            if( tmp > maxabs_Q24 ) {
                maxabs_Q24 = tmp;
                ind = i;
            }
        }
        if( maxabs_Q24 <= limit_Q24 ) {
            return;
        }

        /* Convert to true warped coefficients */
        for( i = 1; i < order; i++ ) {
            coefs_syn_Q24[ i - 1 ] = SKP_SMLAWB( coefs_syn_Q24[ i - 1 ], coefs_syn_Q24[ i ], lambda_Q16 );
            coefs_ana_Q24[ i - 1 ] = SKP_SMLAWB( coefs_ana_Q24[ i - 1 ], coefs_ana_Q24[ i ], lambda_Q16 );
        }

        /* Apply bandwidth expansion */
        chirp_Q16 = SKP_FIX_CONST( 0.99, 16 ) - SKP_DIV32_varQ(
            SKP_SMULWB( maxabs_Q24 - limit_Q24, SKP_SMLABB( SKP_FIX_CONST( 0.8, 10 ), SKP_FIX_CONST( 0.1, 10 ), iter ) ), 
            SKP_MUL( maxabs_Q24, ind + 1 ), 22 );
        SKP_Silk_bwexpander_32( coefs_syn_Q24, order, chirp_Q16 );
        SKP_Silk_bwexpander_32( coefs_ana_Q24, order, chirp_Q16 );

        /* Convert back to monic warped coefficients */
        lambda_Q16 = -lambda_Q16;
        for( i = order - 1; i > 0; i-- ) {
            coefs_syn_Q24[ i - 1 ] = SKP_SMLAWB( coefs_syn_Q24[ i - 1 ], coefs_syn_Q24[ i ], lambda_Q16 );
            coefs_ana_Q24[ i - 1 ] = SKP_SMLAWB( coefs_ana_Q24[ i - 1 ], coefs_ana_Q24[ i ], lambda_Q16 );
        }
        lambda_Q16 = -lambda_Q16;
    }
}

/**************************************************************/
/* Compute noise shaping coefficients and initial gain values */
/**************************************************************/
void SKP_Silk_noise_shape_analysis_FIX(
    SKP_Silk_encoder_state_FIX      *psEnc,         /* I/O  Encoder state FIX                           */
    SKP_Silk_encoder_control_FIX    *psEncCtrl,     /* I/O  Encoder control FIX                         */
    const SKP_int16                 *pitch_res,     /* I    LPC residual from pitch analysis            */
    const SKP_int16                 *x              /* I    Input signal [ frame_length + la_shape ]    */
)
{
    SKP_Silk_shape_state_FIX *psShapeSt = &psEnc->sShape;
    SKP_int     k, i, nSamples, Qnrg, b_Q14, scale = 0, sz;
    SKP_int32   SNR_adj_dB_Q7, HarmBoost_Q16, HarmShapeGain_Q16, Tilt_Q16, tmp32;
    SKP_int32   nrg, pre_nrg_Q30, log_energy_Q7, log_energy_prev_Q7, energy_variation_Q7;
    SKP_int32   delta_Q16, BWExp1_Q16, BWExp2_Q16, gain_mult_Q16, gain_add_Q16, strength_Q16, b_Q8;
    SKP_int32   auto_corr[     MAX_SHAPE_LPC_ORDER + 1 ];
    SKP_int32   refl_coef_Q16[ MAX_SHAPE_LPC_ORDER ];
    SKP_int32   AR1_Q24[       MAX_SHAPE_LPC_ORDER ];
    SKP_int32   AR2_Q24[       MAX_SHAPE_LPC_ORDER ];
    SKP_int16   x_windowed[    SHAPE_LPC_WIN_MAX ];
    const SKP_int16 *x_ptr, *pitch_res_ptr;

    SKP_int32   sqrt_nrg[ MAX_NB_SUBFR ], Qnrg_vec[ MAX_NB_SUBFR ];

    /* Point to start of first LPC analysis block */
    x_ptr = x + psEnc->sCmn.la_shape - SKP_SMULBB( SHAPE_LPC_WIN_MS, psEnc->sCmn.fs_kHz ) + psEnc->sCmn.subfr_length;

    /****************/
    /* CONTROL SNR  */
    /****************/
#if DEACTIVATE_SNR_FEEDBACK
    psEncCtrl->current_SNR_dB_Q7 = psEnc->SNR_dB_Q7;
#else
    /* Reduce SNR_dB values if recent bitstream has exceeded TargetRate */
    psEncCtrl->current_SNR_dB_Q7 = psEnc->SNR_dB_Q7 - SKP_SMULWB( SKP_LSHIFT( ( SKP_int32 )psEnc->BufferedInChannel_ms, 7 ), 3277 );
#endif

    /* Reduce SNR_dB if inband FEC used */
    if( psEnc->speech_activity_Q8 > LBRR_SPEECH_ACTIVITY_THRES_Q8 ) {
        psEncCtrl->current_SNR_dB_Q7 -= SKP_RSHIFT( psEnc->inBandFEC_SNR_comp_Q8, 1 );
    }

    /****************/
    /* GAIN CONTROL */
    /****************/
    /* Input quality is the average of the quality in the lowest two VAD bands */
    psEncCtrl->input_quality_Q14 = ( SKP_int )SKP_RSHIFT( ( SKP_int32 )psEncCtrl->input_quality_bands_Q15[ 0 ] 
        + psEncCtrl->input_quality_bands_Q15[ 1 ], 2 );

    /* Coding quality level, between 0.0_Q0 and 1.0_Q0, but in Q14 */
    psEncCtrl->coding_quality_Q14 = SKP_RSHIFT( SKP_Silk_sigm_Q15( SKP_RSHIFT_ROUND( psEncCtrl->current_SNR_dB_Q7 - ( 17 << 7 ), 4 ) ), 1 );

    /* Reduce coding SNR during low speech activity */
    b_Q8 = ( 1 << 8 ) - psEnc->speech_activity_Q8;
    b_Q8 = SKP_SMULWB( SKP_LSHIFT( b_Q8, 8 ), b_Q8 );
    SNR_adj_dB_Q7 = SKP_SMLAWB( psEncCtrl->current_SNR_dB_Q7,
        SKP_SMULBB( -BG_SNR_DECR_dB_Q7 >> ( 4 + 1 ), b_Q8 ),                                            // Q11
        SKP_SMULWB( ( 1 << 14 ) + psEncCtrl->input_quality_Q14, psEncCtrl->coding_quality_Q14 ) );      // Q12

    if( psEncCtrl->sCmn.sigtype == SIG_TYPE_VOICED ) {
        /* Reduce gains for periodic signals */
        SNR_adj_dB_Q7 = SKP_SMLAWB( SNR_adj_dB_Q7, HARM_SNR_INCR_dB_Q7 << 1, psEnc->LTPCorr_Q15 );
    } else { 
        /* For unvoiced signals and low-quality input, adjust the quality slower than SNR_dB setting */
        SNR_adj_dB_Q7 = SKP_SMLAWB( SNR_adj_dB_Q7, 
            SKP_SMLAWB( 6 << ( 7 + 2 ), -104856, psEncCtrl->current_SNR_dB_Q7 ),    //-104856_Q18 = -0.4_Q0, Q9
            ( 1 << 14 ) - psEncCtrl->input_quality_Q14 );                           // Q14
    }

    /*************************/
    /* SPARSENESS PROCESSING */
    /*************************/
    /* Set quantizer offset */
    if( psEncCtrl->sCmn.sigtype == SIG_TYPE_VOICED ) {
        /* Initally set to 0; may be overruled in process_gains(..) */
        psEncCtrl->sCmn.QuantOffsetType = 0;
        psEncCtrl->sparseness_Q8 = 0;
    } else {
        /* Sparseness measure, based on relative fluctuations of energy per 2 milliseconds */
        nSamples = SKP_LSHIFT( psEnc->sCmn.fs_kHz, 1 );
        energy_variation_Q7 = 0;
        log_energy_prev_Q7  = 0;
        pitch_res_ptr = pitch_res;
        for( k = 0; k < SKP_SMULBB( SUB_FRAME_LENGTH_MS, psEnc->sCmn.nb_subfr ) / 2; k++ ) {
            SKP_Silk_sum_sqr_shift( &nrg, &scale, pitch_res_ptr, nSamples );
            nrg += SKP_RSHIFT( nSamples, scale );           // Q(-scale)
            
            log_energy_Q7 = SKP_Silk_lin2log( nrg );
            if( k > 0 ) {
                energy_variation_Q7 += SKP_abs( log_energy_Q7 - log_energy_prev_Q7 );
            }
            log_energy_prev_Q7 = log_energy_Q7;
            pitch_res_ptr += nSamples;
        }

        psEncCtrl->sparseness_Q8 = SKP_RSHIFT( SKP_Silk_sigm_Q15( SKP_SMULWB( energy_variation_Q7 - ( 5 << 7 ), 6554 ) ), 7 ); // 6554_Q16 = 0.1_Q0

        /* Set quantization offset depending on sparseness measure */
        if( psEncCtrl->sparseness_Q8 > SPARSENESS_THRESHOLD_QNT_OFFSET_Q8 ) {
            psEncCtrl->sCmn.QuantOffsetType = 0;
        } else {
            psEncCtrl->sCmn.QuantOffsetType = 1;
        }
        
        /* Increase coding SNR for sparse signals */
        SNR_adj_dB_Q7 = SKP_SMLAWB( SNR_adj_dB_Q7, SPARSE_SNR_INCR_dB_Q7 << 8, psEncCtrl->sparseness_Q8 - ( 1 << 7 ) );
    }

    /*******************************/
    /* Control bandwidth expansion */
    /*******************************/
    delta_Q16  = SKP_SMULWB( ( 1 << 16 ) - SKP_SMULBB( 3, psEncCtrl->coding_quality_Q14 ), LOW_RATE_BANDWIDTH_EXPANSION_DELTA_Q16 );
    BWExp1_Q16 = BANDWIDTH_EXPANSION_Q16 - delta_Q16;
    BWExp2_Q16 = BANDWIDTH_EXPANSION_Q16 + delta_Q16;
    if( psEnc->sCmn.fs_kHz == 24 ) {
        /* Less bandwidth expansion for super wideband */
        BWExp1_Q16 = ( 1 << 16 ) - SKP_SMULWB( SWB_BANDWIDTH_EXPANSION_REDUCTION_Q16, ( 1 << 16 ) - BWExp1_Q16 );
        BWExp2_Q16 = ( 1 << 16 ) - SKP_SMULWB( SWB_BANDWIDTH_EXPANSION_REDUCTION_Q16, ( 1 << 16 ) - BWExp2_Q16 );
    }
    /* BWExp1 will be applied after BWExp2, so make it relative */
    BWExp1_Q16 = SKP_DIV32_16( SKP_LSHIFT( BWExp1_Q16, 14 ), SKP_RSHIFT( BWExp2_Q16, 2 ) );

    /* Warping coefficient */
    psEncCtrl->sCmn.warping_Q16 = psEnc->sCmn.fs_kHz * WARPING_MULTIPLIER_Q16;
    psEncCtrl->sCmn.warping_Q16 = SKP_min( psEncCtrl->sCmn.warping_Q16, 32767 );

    /********************************************/
    /* Compute noise shaping AR coefs and gains */
    /********************************************/
    sz = ( SKP_int )SKP_SMULBB( SHAPE_LPC_WIN_MS, psEnc->sCmn.fs_kHz );
    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        /* Apply window */
        SKP_Silk_apply_sine_window( x_windowed, x_ptr, 0, SHAPE_LPC_WIN_MS * psEnc->sCmn.fs_kHz );

        /* Update pointer: next LPC analysis block */
        x_ptr += psEnc->sCmn.subfr_length;

        /* Calculate warped auto correlation */
        SKP_Silk_warped_autocorrelation_FIX( auto_corr, &scale, x_windowed, psEncCtrl->sCmn.warping_Q16, sz, psEnc->sCmn.shapingLPCOrder ); 

        /* Add white noise, as a fraction of energy */
        auto_corr[0] = SKP_ADD32( auto_corr[0], SKP_max_32( SKP_SMULWB( SKP_RSHIFT( auto_corr[ 0 ], 4 ), SHAPE_WHITE_NOISE_FRACTION_Q20 ), 1 ) ); 

        /* Calculate the reflection coefficients using schur */
        nrg = SKP_Silk_schur64( refl_coef_Q16, auto_corr, psEnc->sCmn.shapingLPCOrder );
        SKP_assert( nrg >= 0 );

        /* Convert reflection coefficients to prediction coefficients */
        SKP_Silk_k2a_Q16( AR2_Q24, refl_coef_Q16, psEnc->sCmn.shapingLPCOrder );

        /* Convert residual energy to non-warped scale */
        gain_mult_Q16 = warped_gain( AR2_Q24, psEncCtrl->sCmn.warping_Q16, psEnc->sCmn.shapingLPCOrder );
        gain_mult_Q16 = SKP_SMULWW( gain_mult_Q16, gain_mult_Q16 );
        SKP_assert( gain_mult_Q16 >= 0 ); // If breaking, add dynamic scaling
        nrg = SKP_SMULWW( nrg, gain_mult_Q16 );
        SKP_assert( nrg >= 0 ); // If breaking, add dynamic scaling

        /* Bandwidth expansion for synthesis filter shaping */
        SKP_Silk_bwexpander_32( AR2_Q24, psEnc->sCmn.shapingLPCOrder, BWExp2_Q16 );

        /* Compute noise shaping filter coefficients */
        SKP_memcpy( AR1_Q24, AR2_Q24, psEnc->sCmn.shapingLPCOrder * sizeof( SKP_int32 ) );

        /* Bandwidth expansion for analysis filter shaping */
        SKP_assert( BWExp1_Q16 <= ( 1 << 16 ) ); // If ever breaking, use LPC_stabilize() in these cases to stay within range
        SKP_Silk_bwexpander_32( AR1_Q24, psEnc->sCmn.shapingLPCOrder, BWExp1_Q16 );

        /* Increase residual energy */
        nrg = SKP_SMLAWB( nrg, SKP_RSHIFT( auto_corr[ 0 ], 8 ), SHAPE_MIN_ENERGY_RATIO_Q24 );

        Qnrg = -scale;          // range: -12...30
        SKP_assert( Qnrg >= -12 );
        SKP_assert( Qnrg <=  30 );

        /* Make sure that Qnrg is an even number */
        if( Qnrg & 1 ) {
            Qnrg -= 1;
            nrg >>= 1;
        }

        tmp32 = SKP_Silk_SQRT_APPROX( nrg );
        Qnrg >>= 1;             // range: -6...15

        sqrt_nrg[ k ] = tmp32;
        Qnrg_vec[ k ] = Qnrg;

        psEncCtrl->Gains_Q16[ k ] = SKP_LSHIFT_SAT32( tmp32, 16 - Qnrg );

        /* Ratio of prediction gains, in energy domain */
        SKP_Silk_LPC_inverse_pred_gain_Q24( &pre_nrg_Q30, AR2_Q24, psEnc->sCmn.shapingLPCOrder );
        SKP_Silk_LPC_inverse_pred_gain_Q24( &nrg,         AR1_Q24, psEnc->sCmn.shapingLPCOrder );

        //psEncCtrl->GainsPre[ k ] = 1.0f - 0.7f * ( 1.0f - pre_nrg / nrg ) = 0.3f + 0.7f * pre_nrg / nrg;
        pre_nrg_Q30 = SKP_LSHIFT32( SKP_SMULWB( pre_nrg_Q30, SKP_FIX_CONST( 0.7, 15 ) ), 1 );
        psEncCtrl->GainsPre_Q14[ k ] = ( SKP_int ) SKP_FIX_CONST( 0.3, 14 ) + SKP_DIV32_varQ( pre_nrg_Q30, nrg, 14 );

        /* Convert to monic warped prediction coefficients */
        warped_true2monic_coefs( AR1_Q24, psEncCtrl->sCmn.warping_Q16, psEnc->sCmn.shapingLPCOrder );
        warped_true2monic_coefs( AR2_Q24, psEncCtrl->sCmn.warping_Q16, psEnc->sCmn.shapingLPCOrder );

        /* Limit absolute values */
        limit_warped_coefs( AR2_Q24, AR1_Q24, psEncCtrl->sCmn.warping_Q16, SKP_FIX_CONST( 3.999, 24 ), psEnc->sCmn.shapingLPCOrder );

        /* Convert from Q24 to Q13 and store in int16 */
        for( i = 0; i < psEnc->sCmn.shapingLPCOrder; i++ ) {
            psEncCtrl->AR1_Q13[ k * MAX_SHAPE_LPC_ORDER + i ] = (SKP_int16)SKP_SAT16( SKP_RSHIFT_ROUND( AR1_Q24[ i ], 11 ) );
            psEncCtrl->AR2_Q13[ k * MAX_SHAPE_LPC_ORDER + i ] = (SKP_int16)SKP_SAT16( SKP_RSHIFT_ROUND( AR2_Q24[ i ], 11 ) );
        }
    }

    /*****************/
    /* Gain tweaking */
    /*****************/
    /* Increase gains during low speech activity and put lower limit on gains */
    gain_mult_Q16 = SKP_Silk_log2lin( -SKP_SMLAWB( -16 << 7, SNR_adj_dB_Q7,           10486 ) ); // 10486_Q16 = 0.16_Q0
    gain_add_Q16  = SKP_Silk_log2lin(  SKP_SMLAWB(  16 << 7, NOISE_FLOOR_dB_Q7,       10486 ) ); // 10486_Q16 = 0.16_Q0
    tmp32         = SKP_Silk_log2lin(  SKP_SMLAWB(  16 << 7, RELATIVE_MIN_GAIN_dB_Q7, 10486 ) ); // 10486_Q16 = 0.16_Q0
    tmp32 = SKP_SMULWW( psEnc->avgGain_Q16, tmp32 );
    gain_add_Q16 = SKP_ADD_SAT32( gain_add_Q16, tmp32 );
    SKP_assert( gain_mult_Q16 >= 0 );

    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        psEncCtrl->Gains_Q16[ k ] = SKP_SMULWW( psEncCtrl->Gains_Q16[ k ], gain_mult_Q16 );
        SKP_assert( psEncCtrl->Gains_Q16[ k ] >= 0 );
    }

    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        psEncCtrl->Gains_Q16[ k ] = SKP_ADD_POS_SAT32( psEncCtrl->Gains_Q16[ k ], gain_add_Q16 );
        psEnc->avgGain_Q16 = SKP_ADD_SAT32( 
            psEnc->avgGain_Q16, 
            SKP_SMULWB(
                psEncCtrl->Gains_Q16[ k ] - psEnc->avgGain_Q16, 
                SKP_RSHIFT_ROUND( SKP_SMULBB( psEnc->speech_activity_Q8, GAIN_SMOOTHING_COEF_Q10 ), 2 ) 
            ) );
    }

    /************************************************/
    /* Decrease level during fricatives (de-essing) */
    /************************************************/
    gain_mult_Q16 = ( 1 << 16 ) + SKP_RSHIFT_ROUND( SKP_MLA( INPUT_TILT_Q26, psEncCtrl->coding_quality_Q14, HIGH_RATE_INPUT_TILT_Q12 ), 10 );

    if( psEncCtrl->input_tilt_Q15 <= 0 && psEncCtrl->sCmn.sigtype == SIG_TYPE_UNVOICED ) {
        if( psEnc->sCmn.fs_kHz == 24 ) {
            SKP_int32 essStrength_Q15 = SKP_SMULWW( -psEncCtrl->input_tilt_Q15, 
                SKP_SMULBB( psEnc->speech_activity_Q8, ( 1 << 8 ) - psEncCtrl->sparseness_Q8 ) );
            tmp32 = SKP_Silk_log2lin( ( 16 << 7 ) - SKP_SMULWB( essStrength_Q15, 
                SKP_SMULWB( DE_ESSER_COEF_SWB_dB_Q7, 20972 ) ) ); // 20972_Q17 = 0.16_Q0
            gain_mult_Q16 = SKP_SMULWW( gain_mult_Q16, tmp32 );
        } else if( psEnc->sCmn.fs_kHz == 16 ) {
            SKP_int32 essStrength_Q15 = SKP_SMULWW(-psEncCtrl->input_tilt_Q15, 
                SKP_SMULBB( psEnc->speech_activity_Q8, ( 1 << 8 ) - psEncCtrl->sparseness_Q8 ));
            tmp32 = SKP_Silk_log2lin( ( 16 << 7 ) - SKP_SMULWB( essStrength_Q15, 
                SKP_SMULWB( DE_ESSER_COEF_WB_dB_Q7, 20972 ) ) ); // 20972_Q17 = 0.16_Q0
            gain_mult_Q16 = SKP_SMULWW( gain_mult_Q16, tmp32 );
        } else {
            SKP_assert( psEnc->sCmn.fs_kHz == 12 || psEnc->sCmn.fs_kHz == 8 );
        }
    }

    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        psEncCtrl->GainsPre_Q14[ k ] = SKP_SMULWB( gain_mult_Q16, psEncCtrl->GainsPre_Q14[ k ] );
    }

    /************************************************/
    /* Control low-frequency shaping and noise tilt */
    /************************************************/
    /* Less low frequency shaping for noisy inputs */
    strength_Q16 = SKP_MUL( LOW_FREQ_SHAPING_Q0, ( 1 << 16 ) + SKP_SMULBB( LOW_QUALITY_LOW_FREQ_SHAPING_DECR_Q1, psEncCtrl->input_quality_bands_Q15[ 0 ] - ( 1 << 15 ) ) );
    if( psEncCtrl->sCmn.sigtype == SIG_TYPE_VOICED ) {
        /* Reduce low frequencies quantization noise for periodic signals, depending on pitch lag */
        /*f = 400; freqz([1, -0.98 + 2e-4 * f], [1, -0.97 + 7e-4 * f], 2^12, Fs); axis([0, 1000, -10, 1])*/
        SKP_int fs_kHz_inv = SKP_DIV32_16( 3277, psEnc->sCmn.fs_kHz );      // 0.2_Q0 = 3277_Q14
        for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
            b_Q14 = fs_kHz_inv + SKP_DIV32_16( ( 3 << 14 ), psEncCtrl->sCmn.pitchL[ k ] ); 
            /* Pack two coefficients in one int32 */
            psEncCtrl->LF_shp_Q14[ k ]  = SKP_LSHIFT( ( 1 << 14 ) - b_Q14 - SKP_SMULWB( strength_Q16, b_Q14 ), 16 );
            psEncCtrl->LF_shp_Q14[ k ] |= (SKP_uint16)( b_Q14 - ( 1 << 14 ) );
        }
        SKP_assert( HARM_HP_NOISE_COEF_Q24 < ( 1 << 23 ) ); // Guarantees that second argument to SMULWB() is within range of an SKP_int16
        Tilt_Q16 = - HP_NOISE_COEF_Q16 - 
            SKP_SMULWB( ( 1 << 16 ) - HP_NOISE_COEF_Q16, SKP_SMULWB( HARM_HP_NOISE_COEF_Q24, psEnc->speech_activity_Q8 ) );
    } else {
        b_Q14 = SKP_DIV32_16( 21299, psEnc->sCmn.fs_kHz ); // 1.3_Q0 = 21299_Q14
        /* Pack two coefficients in one int32 */
        psEncCtrl->LF_shp_Q14[ 0 ]  = SKP_LSHIFT( ( 1 << 14 ) - b_Q14 - SKP_SMULWB( strength_Q16, SKP_SMULWB( 39322, b_Q14 ) ), 16 ); // 0.6_Q0 = 39322_Q16
        psEncCtrl->LF_shp_Q14[ 0 ] |= (SKP_uint16)( b_Q14 - ( 1 << 14 ) );
        for( k = 1; k < psEnc->sCmn.nb_subfr; k++ ) {
            psEncCtrl->LF_shp_Q14[ k ] = psEncCtrl->LF_shp_Q14[ k - 1 ];
        }
        Tilt_Q16 = -HP_NOISE_COEF_Q16;
    }

    /****************************/
    /* HARMONIC SHAPING CONTROL */
    /****************************/
    /* Control boosting of harmonic frequencies */
    HarmBoost_Q16 = SKP_SMULWB( SKP_SMULWB( ( 1 << 17 ) - SKP_LSHIFT( psEncCtrl->coding_quality_Q14, 3 ), 
        psEnc->LTPCorr_Q15 ), LOW_RATE_HARMONIC_BOOST_Q16 );

    /* More harmonic boost for noisy input signals */
    HarmBoost_Q16 = SKP_SMLAWB( HarmBoost_Q16, 
        ( 1 << 16 ) - SKP_LSHIFT( psEncCtrl->input_quality_Q14, 2 ), LOW_INPUT_QUALITY_HARMONIC_BOOST_Q16 );

    if( USE_HARM_SHAPING && psEncCtrl->sCmn.sigtype == SIG_TYPE_VOICED ) {
        /* More harmonic noise shaping for high bitrates or noisy input */
        HarmShapeGain_Q16 = SKP_SMLAWB( HARMONIC_SHAPING_Q16, 
                ( 1 << 16 ) - SKP_SMULWB( ( 1 << 18 ) - SKP_LSHIFT( psEncCtrl->coding_quality_Q14, 4 ),
                psEncCtrl->input_quality_Q14 ), HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING_Q16 );

        /* Less harmonic noise shaping for less periodic signals */
        HarmShapeGain_Q16 = SKP_SMULWB( SKP_LSHIFT( HarmShapeGain_Q16, 1 ), 
            SKP_Silk_SQRT_APPROX( SKP_LSHIFT( psEnc->LTPCorr_Q15, 15 ) ) );
    } else {
        HarmShapeGain_Q16 = 0;
    }

    /*************************/
    /* Smooth over subframes */
    /*************************/
    for( k = 0; k < MAX_NB_SUBFR; k++ ) {
        psShapeSt->HarmBoost_smth_Q16 =
            SKP_SMLAWB( psShapeSt->HarmBoost_smth_Q16,     HarmBoost_Q16     - psShapeSt->HarmBoost_smth_Q16,     SUBFR_SMTH_COEF_Q16 );
        psShapeSt->HarmShapeGain_smth_Q16 =
            SKP_SMLAWB( psShapeSt->HarmShapeGain_smth_Q16, HarmShapeGain_Q16 - psShapeSt->HarmShapeGain_smth_Q16, SUBFR_SMTH_COEF_Q16 );
        psShapeSt->Tilt_smth_Q16 =
            SKP_SMLAWB( psShapeSt->Tilt_smth_Q16,          Tilt_Q16          - psShapeSt->Tilt_smth_Q16,          SUBFR_SMTH_COEF_Q16 );

        psEncCtrl->HarmBoost_Q14[ k ]     = ( SKP_int )SKP_RSHIFT_ROUND( psShapeSt->HarmBoost_smth_Q16,     2 );
        psEncCtrl->HarmShapeGain_Q14[ k ] = ( SKP_int )SKP_RSHIFT_ROUND( psShapeSt->HarmShapeGain_smth_Q16, 2 );
        psEncCtrl->Tilt_Q14[ k ]          = ( SKP_int )SKP_RSHIFT_ROUND( psShapeSt->Tilt_smth_Q16,          2 );
    }
}
