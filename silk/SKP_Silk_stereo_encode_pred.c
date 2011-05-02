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

/* Quantize mid/side predictors and entropy code the quantization indices */
void SKP_Silk_stereo_encode_pred(
    ec_enc              *psRangeEnc,                    /* I/O  Compressor data structure                   */
    SKP_int32           pred_Q13[]                      /* I/O  Predictors (out: quantized)                 */
)
{
    SKP_int   i, j, n, ibest[ 2 ] = { 0 }, jbest[ 2 ] = { 0 }, kbest[ 2 ];
    SKP_int32 low_Q13, step_Q13, lvl_Q13, err_min_Q13, err_Q13, quant_pred_Q13 = 0;

    /* Quantize */
    for( n = 0; n < 2; n++ ) {
        /* Brute-force search over quantization levels */
        err_min_Q13 = SKP_int32_MAX;
        for( i = 0; i < STEREO_QUANT_TAB_SIZE - 1; i++ ) {
            low_Q13 = SKP_Silk_stereo_pred_quant_Q13[ i ];
            step_Q13 = SKP_SMULWB( SKP_Silk_stereo_pred_quant_Q13[ i + 1 ] - low_Q13, 
                SKP_FIX_CONST( 0.5 / STEREO_QUANT_SUB_STEPS, 16 ) );
            for( j = 0; j < STEREO_QUANT_SUB_STEPS; j++ ) {
                lvl_Q13 = SKP_SMLABB( low_Q13, step_Q13, 2 * j + 1 );
                err_Q13 = SKP_abs( pred_Q13[ n ] - lvl_Q13 );
                if( err_Q13 < err_min_Q13 ) {
                    err_min_Q13 = err_Q13;
                    quant_pred_Q13 = lvl_Q13;
                    ibest[ n ] = i;
                    jbest[ n ] = j;
                } else {
                    /* Error increasing, so we're past the optimum */
                    goto done;
                }
            }
        }
        done:
        kbest[ n ]  = SKP_DIV32_16( ibest[ n ], 3 );
        ibest[ n ] -= kbest[ n ] * 3;
        pred_Q13[ n ] = quant_pred_Q13; 
    }

    /* Subtract second from first predictor (helps when actually applying these) */
    pred_Q13[ 0 ] -= pred_Q13[ 1 ];
    
    /* Entropy coding */
    i = 5 * kbest[ 0 ] + kbest[ 1 ];
    SKP_assert( i < 25 );
    ec_enc_icdf( psRangeEnc, i, SKP_Silk_stereo_pred_joint_iCDF, 8 );
    for( n = 0; n < 2; n++ ) {
        SKP_assert( ibest[ n ] < 3 );
        SKP_assert( jbest[ n ] < STEREO_QUANT_SUB_STEPS );
        ec_enc_icdf( psRangeEnc, ibest[ n ], SKP_Silk_uniform3_iCDF, 8 );
        ec_enc_icdf( psRangeEnc, jbest[ n ], SKP_Silk_uniform5_iCDF, 8 );
    }
}
