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

#include "silk_main.h"

/* Convert adaptive Mid/Side representation to Left/Right stereo signal */
void silk_stereo_MS_to_LR( 
    stereo_dec_state    *state,                         /* I/O  State                                       */
    SKP_int16           x1[],                           /* I/O  Left input signal, becomes mid signal       */
    SKP_int16           x2[],                           /* I/O  Right input signal, becomes side signal     */
    const SKP_int32     pred_Q13[],                     /* I    Predictors                                  */
    SKP_int             fs_kHz,                         /* I    Samples rate (kHz)                          */
    SKP_int             frame_length                    /* I    Number of samples                           */
)
{
    SKP_int   n, denom_Q16, delta0_Q13, delta1_Q13;
    SKP_int32 sum, diff, pred0_Q13, pred1_Q13;

    /* Buffering */
    SKP_memcpy( x1, state->sMid,  2 * sizeof( SKP_int16 ) );
    SKP_memcpy( x2, state->sSide, 2 * sizeof( SKP_int16 ) );
    SKP_memcpy( state->sMid,  &x1[ frame_length ], 2 * sizeof( SKP_int16 ) );
    SKP_memcpy( state->sSide, &x2[ frame_length ], 2 * sizeof( SKP_int16 ) );

    /* Interpolate predictors and add prediction to side channel */
    pred0_Q13  = state->pred_prev_Q13[ 0 ];
    pred1_Q13  = state->pred_prev_Q13[ 1 ];
    denom_Q16  = SKP_DIV32_16( 1 << 16, STEREO_INTERP_LEN_MS * fs_kHz );
    delta0_Q13 = SKP_RSHIFT_ROUND( SKP_SMULBB( pred_Q13[ 0 ] - state->pred_prev_Q13[ 0 ], denom_Q16 ), 16 );
    delta1_Q13 = SKP_RSHIFT_ROUND( SKP_SMULBB( pred_Q13[ 1 ] - state->pred_prev_Q13[ 1 ], denom_Q16 ), 16 );
    for( n = 0; n < STEREO_INTERP_LEN_MS * fs_kHz; n++ ) {
        pred0_Q13 += delta0_Q13;
        pred1_Q13 += delta1_Q13;
        sum = SKP_LSHIFT( SKP_ADD_LSHIFT( x1[ n ] + x1[ n + 2 ], x1[ n + 1 ], 1 ), 9 );         /* Q11 */ 
        sum = SKP_SMLAWB( SKP_LSHIFT( ( SKP_int32 )x2[ n + 1 ], 8 ), sum, pred0_Q13 );          /* Q8  */
        sum = SKP_SMLAWB( sum, SKP_LSHIFT( ( SKP_int32 )x1[ n + 1 ], 11 ), pred1_Q13 );         /* Q8  */
        x2[ n + 1 ] = (SKP_int16)SKP_SAT16( SKP_RSHIFT_ROUND( sum, 8 ) );
    }
    pred0_Q13 = pred_Q13[ 0 ];
    pred1_Q13 = pred_Q13[ 1 ];
    for( n = STEREO_INTERP_LEN_MS * fs_kHz; n < frame_length; n++ ) {
        sum = SKP_LSHIFT( SKP_ADD_LSHIFT( x1[ n ] + x1[ n + 2 ], x1[ n + 1 ], 1 ), 9 );         /* Q11 */ 
        sum = SKP_SMLAWB( SKP_LSHIFT( ( SKP_int32 )x2[ n + 1 ], 8 ), sum, pred0_Q13 );          /* Q8  */
        sum = SKP_SMLAWB( sum, SKP_LSHIFT( ( SKP_int32 )x1[ n + 1 ], 11 ), pred1_Q13 );         /* Q8  */
        x2[ n + 1 ] = (SKP_int16)SKP_SAT16( SKP_RSHIFT_ROUND( sum, 8 ) );
    }
    state->pred_prev_Q13[ 0 ] = pred_Q13[ 0 ];
    state->pred_prev_Q13[ 1 ] = pred_Q13[ 1 ];

    /* Convert to left/right signals */
    for( n = 0; n < frame_length; n++ ) {
        sum  = x1[ n + 1 ] + (SKP_int32)x2[ n + 1 ];
        diff = x1[ n + 1 ] - (SKP_int32)x2[ n + 1 ];
        x1[ n + 1 ] = (SKP_int16)SKP_SAT16( sum );
        x2[ n + 1 ] = (SKP_int16)SKP_SAT16( diff );
    }
}
