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

#include "SKP_Silk_main.h"

/* Limit, stabilize, convert and quantize NLSFs.    */ 
void SKP_Silk_process_NLSFs(
    SKP_Silk_encoder_state          *psEncC,                                /* I/O  Encoder state                               */
    SKP_int16                       PredCoef_Q12[ 2 ][ MAX_LPC_ORDER ],     /* O    Prediction coefficients                     */
    SKP_int                         pNLSF_Q15[         MAX_LPC_ORDER ],     /* I/O  Normalized LSFs (quant out) (0 - (2^15-1))  */
    const SKP_int                   prev_NLSFq_Q15[    MAX_LPC_ORDER ]      /* I    Previous Normalized LSFs (0 - (2^15-1))     */
)
{
    SKP_int     i, doInterpolate;
    SKP_int     pNLSFW_Q5[ MAX_LPC_ORDER ];
    SKP_int     NLSF_mu_Q15, NLSF_mu_fluc_red_Q16;
    SKP_int32   i_sqr_Q15;
    SKP_int     pNLSF0_temp_Q15[ MAX_LPC_ORDER ];
    SKP_int     pNLSFW0_temp_Q5[ MAX_LPC_ORDER ];
    const SKP_Silk_NLSF_CB_struct *psNLSF_CB;

    SKP_assert( psEncC->speech_activity_Q8 >=   0 );
    SKP_assert( psEncC->speech_activity_Q8 <= SKP_FIX_CONST( 1.0, 8 ) );

    /***********************/
    /* Calculate mu values */
    /***********************/
    if( psEncC->indices.signalType == TYPE_VOICED ) {
        /* NLSF_mu           = 0.002f - 0.001f * psEnc->speech_activity; */
        /* NLSF_mu_fluc_red  = 0.1f   - 0.05f  * psEnc->speech_activity; */
        NLSF_mu_Q15          = SKP_SMLAWB( SKP_FIX_CONST( 0.002, 15 ), SKP_FIX_CONST( -0.001, 23 ), psEncC->speech_activity_Q8 );
        NLSF_mu_fluc_red_Q16 = SKP_SMLAWB( SKP_FIX_CONST( 0.1,   16 ), SKP_FIX_CONST( -0.05,  24 ), psEncC->speech_activity_Q8 );
    } else { 
        /* NLSF_mu           = 0.005f - 0.004f * psEnc->speech_activity; */
        /* NLSF_mu_fluc_red  = 0.2f   - 0.1f   * psEnc->speech_activity - 0.1f * psEncCtrl->sparseness; */
        NLSF_mu_Q15          = SKP_SMLAWB( SKP_FIX_CONST( 0.005, 15 ), SKP_FIX_CONST( -0.004, 23 ), psEncC->speech_activity_Q8 );
        NLSF_mu_fluc_red_Q16 = SKP_SMLAWB( SKP_FIX_CONST( 0.15,  16 ), SKP_FIX_CONST( -0.1,   24 ), psEncC->speech_activity_Q8 ); 
    }
    SKP_assert( NLSF_mu_Q15          >= 0 );
    SKP_assert( NLSF_mu_Q15          <= SKP_FIX_CONST( 0.005, 15 ) );
    SKP_assert( NLSF_mu_fluc_red_Q16 >= 0 );
    SKP_assert( NLSF_mu_fluc_red_Q16 <= SKP_FIX_CONST( 0.15, 16 ) );

    NLSF_mu_Q15 = SKP_max( NLSF_mu_Q15, 1 );

    /* Calculate NLSF weights */
    SKP_Silk_NLSF_VQ_weights_laroia( pNLSFW_Q5, pNLSF_Q15, psEncC->predictLPCOrder );

    /* Update NLSF weights for interpolated NLSFs */
    doInterpolate = ( psEncC->useInterpolatedNLSFs == 1 ) && ( psEncC->indices.NLSFInterpCoef_Q2 < 4 );
    if( doInterpolate ) {
        /* Calculate the interpolated NLSF vector for the first half */
        SKP_Silk_interpolate( pNLSF0_temp_Q15, prev_NLSFq_Q15, pNLSF_Q15, 
            psEncC->indices.NLSFInterpCoef_Q2, psEncC->predictLPCOrder );

        /* Calculate first half NLSF weights for the interpolated NLSFs */
        SKP_Silk_NLSF_VQ_weights_laroia( pNLSFW0_temp_Q5, pNLSF0_temp_Q15, psEncC->predictLPCOrder );

        /* Update NLSF weights with contribution from first half */
        i_sqr_Q15 = SKP_LSHIFT( SKP_SMULBB( psEncC->indices.NLSFInterpCoef_Q2, psEncC->indices.NLSFInterpCoef_Q2 ), 11 );
        for( i = 0; i < psEncC->predictLPCOrder; i++ ) {
            pNLSFW_Q5[ i ] = SKP_SMLAWB( SKP_RSHIFT( pNLSFW_Q5[ i ], 1 ), pNLSFW0_temp_Q5[ i ], i_sqr_Q15 );
            SKP_assert( pNLSFW_Q5[ i ] <= SKP_int16_MAX );
            SKP_assert( pNLSFW_Q5[ i ] >= 1 );
        }
    }

    /* Set pointer to the NLSF codebook for the current signal type and LPC order */
    psNLSF_CB = psEncC->psNLSF_CB[ 1 - ( psEncC->indices.signalType >> 1 ) ];

    /* Quantize NLSF parameters given the trained NLSF codebooks */
    TIC(MSVQ_encode_FIX)
    SKP_Silk_NLSF_MSVQ_encode( psEncC->indices.NLSFIndices, pNLSF_Q15, psNLSF_CB, prev_NLSFq_Q15, pNLSFW_Q5, NLSF_mu_Q15, 
        NLSF_mu_fluc_red_Q16, psEncC->NLSF_MSVQ_Survivors, psEncC->predictLPCOrder, psEncC->first_frame_after_reset );
    TOC(MSVQ_encode_FIX)

    /* Convert quantized NLSFs back to LPC coefficients */
    SKP_Silk_NLSF2A_stable( PredCoef_Q12[ 1 ], pNLSF_Q15, psEncC->predictLPCOrder );

    if( doInterpolate ) {
        /* Calculate the interpolated, quantized LSF vector for the first half */
        SKP_Silk_interpolate( pNLSF0_temp_Q15, prev_NLSFq_Q15, pNLSF_Q15, 
            psEncC->indices.NLSFInterpCoef_Q2, psEncC->predictLPCOrder );

        /* Convert back to LPC coefficients */
        SKP_Silk_NLSF2A_stable( PredCoef_Q12[ 0 ], pNLSF0_temp_Q15, psEncC->predictLPCOrder );

    } else {
        /* Copy LPC coefficients for first half from second half */
        SKP_memcpy( PredCoef_Q12[ 0 ], PredCoef_Q12[ 1 ], psEncC->predictLPCOrder * sizeof( SKP_int16 ) );
    }
}
