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
    ec_enc              *psRangeEnc,                    /* I/O  Compressor data structure                   */
    stereo_state        *state,                         /* I/O  State                                       */
    SKP_int16           x1[],                           /* I/O  Left input signal, becomes mid signal       */
    SKP_int16           x2[],                           /* I/O  Right input signal, becomes side signal     */
    SKP_int             fs_kHz,                         /* I    Samples rate (kHz)                          */
    SKP_int             frame_length                    /* I    Number of samples                           */
)
{
    SKP_int   n, denom_Q16, delta0_Q13, delta1_Q13;
    SKP_int32 sum, diff, pred_Q13[ 2 ], pred0_Q13, pred1_Q13;
    SKP_int16 mid[ MAX_FRAME_LENGTH + 2 ], side[ MAX_FRAME_LENGTH + 2 ];
    SKP_int16 LP_mid[  MAX_FRAME_LENGTH ], HP_mid[  MAX_FRAME_LENGTH ];
    SKP_int16 LP_side[ MAX_FRAME_LENGTH ], HP_side[ MAX_FRAME_LENGTH ];

    /* Convert to basic mid/side signals */
    for( n = 0; n < frame_length; n++ ) {
        sum  = x1[ n ] + (SKP_int32)x2[ n ];
        diff = x1[ n ] - (SKP_int32)x2[ n ];
        mid[  n + 2 ] = (SKP_int16)SKP_RSHIFT_ROUND( sum,  1 );
        side[ n + 2 ] = (SKP_int16)SKP_SAT16( SKP_RSHIFT_ROUND( diff, 1 ) );
    }

    /* Buffering */
    SKP_memcpy( mid,  state->sMid,  2 * sizeof( SKP_int16 ) );
    SKP_memcpy( side, state->sSide, 2 * sizeof( SKP_int16 ) );
    SKP_memcpy( state->sMid,  &mid[  frame_length ], 2 * sizeof( SKP_int16 ) );
    SKP_memcpy( state->sSide, &side[ frame_length ], 2 * sizeof( SKP_int16 ) );

    /* LP and HP filter mid signal */
    for( n = 0; n < frame_length; n++ ) {
        sum = SKP_RSHIFT_ROUND( SKP_ADD_LSHIFT( mid[ n ] + mid[ n + 2 ], mid[ n + 1 ], 1 ), 2 );
        LP_mid[ n ] = sum;
        HP_mid[ n ] = mid[ n + 1 ] - sum;
    }

    /* LP and HP filter side signal */
    for( n = 0; n < frame_length; n++ ) {
        sum = SKP_RSHIFT_ROUND( SKP_ADD_LSHIFT( side[ n ] + side[ n + 2 ], side[ n + 1 ], 1 ), 2 );
        LP_side[ n ] = sum;
        HP_side[ n ] = side[ n + 1 ] - sum;
    }

    /* Find predictors */
    pred_Q13[ 0 ] = SKP_Silk_stereo_find_predictor( LP_mid, LP_side, frame_length );
    pred_Q13[ 1 ] = SKP_Silk_stereo_find_predictor( HP_mid, HP_side, frame_length );

    /* Quantize and encode predictors */
    SKP_Silk_stereo_encode_pred( psRangeEnc, pred_Q13 );

    /* Interpolate predictors and subtract prediction from side channel */
    pred0_Q13  = -state->pred_prev_Q13[ 0 ];
    pred1_Q13  = -state->pred_prev_Q13[ 1 ];
    denom_Q16  = SKP_DIV32_16( 1 << 16, STEREO_INTERP_LEN_MS * fs_kHz );
    delta0_Q13 = -SKP_RSHIFT_ROUND( SKP_SMULBB( pred_Q13[ 0 ] - state->pred_prev_Q13[ 0 ], denom_Q16 ), 16 );
    delta1_Q13 = -SKP_RSHIFT_ROUND( SKP_SMULBB( pred_Q13[ 1 ] - state->pred_prev_Q13[ 1 ], denom_Q16 ), 16 );
    for( n = 0; n < STEREO_INTERP_LEN_MS * fs_kHz; n++ ) {
        pred0_Q13 += delta0_Q13;
        pred1_Q13 += delta1_Q13;
        sum = SKP_LSHIFT( SKP_ADD_LSHIFT( mid[ n ] + mid[ n + 2 ], mid[ n + 1 ], 1 ), 9 );      /* Q11 */ 
        sum = SKP_SMLAWB( SKP_LSHIFT( ( SKP_int32 )side[ n + 1 ], 8 ), sum, pred0_Q13 );        /* Q8  */
        sum = SKP_SMLAWB( sum, SKP_LSHIFT( ( SKP_int32 )mid[ n + 1 ], 11 ), pred1_Q13 );        /* Q8  */
        x2[ n ] = (SKP_int16)SKP_SAT16( SKP_RSHIFT_ROUND( sum, 8 ) );
    }
    pred0_Q13 = -pred_Q13[ 0 ];
    pred1_Q13 = -pred_Q13[ 1 ];
    for( n = STEREO_INTERP_LEN_MS * fs_kHz; n < frame_length; n++ ) {
        sum = SKP_LSHIFT( SKP_ADD_LSHIFT( mid[ n ] + mid[ n + 2 ], mid[ n + 1 ], 1 ), 9 );      /* Q11 */ 
        sum = SKP_SMLAWB( SKP_LSHIFT( ( SKP_int32 )side[ n + 1 ], 8 ), sum, pred0_Q13 );        /* Q8  */
        sum = SKP_SMLAWB( sum, SKP_LSHIFT( ( SKP_int32 )mid[ n + 1 ], 11 ), pred1_Q13 );        /* Q8  */
        x2[ n ] = (SKP_int16)SKP_SAT16( SKP_RSHIFT_ROUND( sum, 8 ) );
    }
    state->pred_prev_Q13[ 0 ] = pred_Q13[ 0 ];
    state->pred_prev_Q13[ 1 ] = pred_Q13[ 1 ];

    SKP_memcpy( x1, mid + 1, frame_length * sizeof( SKP_int16 ) );
}
