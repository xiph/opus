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

/* Convert Left/Right stereo signal to adaptive Mid/Side representation */
void SKP_Silk_stereo_LR_to_MS( 
    stereo_state        *state,                         /* I/O  State                                       */
    SKP_int16           x1[],                           /* I/O  Left input signal, becomes mid signal       */
    SKP_int16           x2[],                           /* I/O  Right input signal, becomes side signal     */
    SKP_int             *predictorIx,                   /* O    Index for predictor filter                  */
    SKP_int             fs_kHz,                         /* I    Samples rate (kHz)                          */
    SKP_int             frame_length                    /* I    Number of samples                           */
)
{
    SKP_int   n, scale, scale1, scale2;
    SKP_int32 sum, diff, nrg1, nrg2, corr, predictor_Q16, pred_Q16, delta_Q16;

    /* Convert to basic mid/side signals */
    for( n = 0; n < frame_length; n++ ) {
        sum  = x1[ n ] + (SKP_int32)x2[ n ];
        diff = x1[ n ] - (SKP_int32)x2[ n ];
        x1[ n ] = (SKP_int16)SKP_RSHIFT32( sum + 1,  1 );
        x2[ n ] = (SKP_int16)SKP_RSHIFT32( diff, 1 );
    }

    /* Find  predictor */
    SKP_Silk_sum_sqr_shift( &nrg1, &scale1, x1, frame_length );
    SKP_Silk_sum_sqr_shift( &nrg2, &scale2, x2, frame_length );
    if( scale1 > scale2 ) {
        scale = scale1;
    } else {
        scale = scale2;
        nrg1 = SKP_RSHIFT32( nrg1, scale2 - scale1 );
    }
    corr = SKP_Silk_inner_prod_aligned_scale( x1, x2, scale, frame_length );
    predictor_Q16 = SKP_DIV32_varQ( corr, nrg1 + 1, 16 );

    /* Hysteresis */
    if( predictor_Q16 > state->predictor_prev_Q16 ) {
        predictor_Q16 -= SKP_FIX_CONST( STEREO_QUANT_HYSTERESIS / STEREO_QUANT_STEPS, 16 );
    } else {
        predictor_Q16 += SKP_FIX_CONST( STEREO_QUANT_HYSTERESIS / STEREO_QUANT_STEPS, 16 );
    }

    /* Quantize */
    *predictorIx = SKP_RSHIFT_ROUND( SKP_MUL( predictor_Q16 + 65536, STEREO_QUANT_STEPS - 1 ), 17 );
    *predictorIx = SKP_LIMIT( *predictorIx, 0, STEREO_QUANT_STEPS - 1 );

    predictor_Q16 = SKP_SMLABB( -65536, *predictorIx, ( 1 << 17 ) / ( STEREO_QUANT_STEPS - 1 ) );

    /* Subtract prediction from side channel */
    if( predictor_Q16 != state->predictor_prev_Q16 ) {
        /* Interpolate predictor */
        pred_Q16 = -state->predictor_prev_Q16;
        delta_Q16 = -SKP_DIV32_16( predictor_Q16 - state->predictor_prev_Q16, STEREO_INTERPOL_LENGTH_MS * fs_kHz );
        for( n = 0; n < STEREO_INTERPOL_LENGTH_MS * fs_kHz; n++ ) {
            pred_Q16 += delta_Q16;
            x2[ n ] = (SKP_int16)SKP_SAT16( SKP_SMLAWB( x2[ n ], pred_Q16, x1[ n ] ) );
        }
    } else {
        n = 0;
    }
    pred_Q16 = -predictor_Q16;
    for( ; n < frame_length; n++ ) {
        x2[ n ] = (SKP_int16)SKP_SAT16( SKP_SMLAWB( x2[ n ], pred_Q16, x1[ n ] ) );
    }

    state->predictor_prev_Q16 = predictor_Q16;
}
