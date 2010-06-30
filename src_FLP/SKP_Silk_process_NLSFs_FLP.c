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

#include <stdlib.h>
#include "SKP_Silk_main_FLP.h"

/* Limit, stabilize, convert and quantize NLSFs */
void SKP_Silk_process_NLSFs_FLP(
    SKP_Silk_encoder_state_FLP      *psEnc,             /* I/O  Encoder state FLP                       */
    SKP_Silk_encoder_control_FLP    *psEncCtrl,         /* I/O  Encoder control FLP                     */
    SKP_float                       *pNLSF              /* I/O  NLSFs (quantized output)                */
)
{
    SKP_int     doInterpolate;
    SKP_float   pNLSFW[ MAX_LPC_ORDER ];
    SKP_float   NLSF_mu, NLSF_mu_fluc_red, i_sqr, NLSF_interpolation_factor = 0.0f;
    const SKP_Silk_NLSF_CB_FLP *psNLSF_CB_FLP;

    /* Used only for NLSF interpolation */
    SKP_float   pNLSF0_temp[  MAX_LPC_ORDER ];
    SKP_float   pNLSFW0_temp[ MAX_LPC_ORDER ];
    SKP_int     i;

    SKP_assert( psEncCtrl->sCmn.sigtype == SIG_TYPE_VOICED || psEncCtrl->sCmn.sigtype == SIG_TYPE_UNVOICED );

    /***********************/
    /* Calculate mu values */
    /***********************/
    if( psEncCtrl->sCmn.sigtype == SIG_TYPE_VOICED ) {
        NLSF_mu          = 0.002f - 0.001f * psEnc->speech_activity;
        NLSF_mu_fluc_red = 0.1f   - 0.05f  * psEnc->speech_activity;
    } else { 
        NLSF_mu          = 0.005f - 0.004f * psEnc->speech_activity;
        NLSF_mu_fluc_red = 0.2f   - 0.1f   * ( psEnc->speech_activity + psEncCtrl->sparseness );
    }

    /* Calculate NLSF weights */
    SKP_Silk_NLSF_VQ_weights_laroia_FLP( pNLSFW, pNLSF, psEnc->sCmn.predictLPCOrder );

    /* Update NLSF weights for interpolated NLSFs */
    doInterpolate = ( psEnc->sCmn.useInterpolatedNLSFs == 1 ) && ( psEncCtrl->sCmn.NLSFInterpCoef_Q2 < ( 1 << 2 ) );
    if( doInterpolate ) {

        /* Calculate the interpolated NLSF vector for the first half */
        NLSF_interpolation_factor = 0.25f * psEncCtrl->sCmn.NLSFInterpCoef_Q2;
        SKP_Silk_interpolate_wrapper_FLP( pNLSF0_temp, psEnc->sPred.prev_NLSFq, pNLSF, 
            NLSF_interpolation_factor, psEnc->sCmn.predictLPCOrder );

        /* Calculate first half NLSF weights for the interpolated NLSFs */
        SKP_Silk_NLSF_VQ_weights_laroia_FLP( pNLSFW0_temp, pNLSF0_temp, psEnc->sCmn.predictLPCOrder );

        /* Update NLSF weights with contribution from first half */
        i_sqr = NLSF_interpolation_factor * NLSF_interpolation_factor;
        for( i = 0; i < psEnc->sCmn.predictLPCOrder; i++ ) {
            pNLSFW[ i ] = 0.5f * ( pNLSFW[ i ] + i_sqr * pNLSFW0_temp[ i ] );
        }
    }

    /* Set pointer to the NLSF codebook for the current signal type and LPC order */
    psNLSF_CB_FLP = psEnc->psNLSF_CB_FLP[ psEncCtrl->sCmn.sigtype ];

    /* Quantize NLSF parameters given the trained NLSF codebooks */
    SKP_Silk_NLSF_MSVQ_encode_FLP( psEncCtrl->sCmn.NLSFIndices, pNLSF, psNLSF_CB_FLP, psEnc->sPred.prev_NLSFq, pNLSFW, NLSF_mu, 
        NLSF_mu_fluc_red, psEnc->sCmn.NLSF_MSVQ_Survivors, psEnc->sCmn.predictLPCOrder, psEnc->sCmn.first_frame_after_reset );

    /* Convert quantized NLSFs back to LPC coefficients */
    SKP_Silk_NLSF2A_stable_FLP( psEncCtrl->PredCoef[ 1 ], pNLSF, psEnc->sCmn.predictLPCOrder );

    if( doInterpolate ) {
        /* Calculate the interpolated, quantized NLSF vector for the first half */
        SKP_Silk_interpolate_wrapper_FLP( pNLSF0_temp, psEnc->sPred.prev_NLSFq, pNLSF, 
            NLSF_interpolation_factor, psEnc->sCmn.predictLPCOrder );

        /* Convert back to LPC coefficients */
        SKP_Silk_NLSF2A_stable_FLP( psEncCtrl->PredCoef[ 0 ], pNLSF0_temp, psEnc->sCmn.predictLPCOrder );

    } else {
        /* Copy LPC coefficients for first half from second half */
        SKP_memcpy( psEncCtrl->PredCoef[ 0 ], psEncCtrl->PredCoef[ 1 ], psEnc->sCmn.predictLPCOrder * sizeof( SKP_float ) );
    }
}

