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

/* Convert Left/Right stereo signal to adaptive Mid/Side representation */
void silk_stereo_LR_to_MS( 
    stereo_enc_state    *state,                         /* I/O  State                                       */
    opus_int16           x1[],                           /* I/O  Left input signal, becomes mid signal       */
    opus_int16           x2[],                           /* I/O  Right input signal, becomes side signal     */
    opus_int8            ix[ 2 ][ 4 ],                   /* O    Quantization indices                        */
    opus_int32           mid_side_rates_bps[],           /* O    Bitrates for mid and side signals           */
    opus_int32           total_rate_bps,                 /* I    Total bitrate                               */
    opus_int             prev_speech_act_Q8,             /* I    Speech activity level in previous frame     */
    opus_int             fs_kHz,                         /* I    Sample rate (kHz)                           */
    opus_int             frame_length                    /* I    Number of samples                           */
)
{
    opus_int   n, is10msFrame, denom_Q16, delta0_Q13, delta1_Q13;
    opus_int32 sum, diff, smooth_coef_Q16, pred_Q13[ 2 ], pred0_Q13, pred1_Q13;
    opus_int32 LP_ratio_Q14, HP_ratio_Q14, frac_Q16, frac_3_Q16, min_mid_rate_bps, width_Q14, w_Q24, deltaw_Q24;
    opus_int16 side[ MAX_FRAME_LENGTH + 2 ];
    opus_int16 LP_mid[  MAX_FRAME_LENGTH ], HP_mid[  MAX_FRAME_LENGTH ];
    opus_int16 LP_side[ MAX_FRAME_LENGTH ], HP_side[ MAX_FRAME_LENGTH ];
    opus_int16 *mid = &x1[ -2 ];

    /* Convert to basic mid/side signals */
    for( n = 0; n < frame_length + 2; n++ ) {
        sum  = x1[ n - 2 ] + (opus_int32)x2[ n - 2 ];
        diff = x1[ n - 2 ] - (opus_int32)x2[ n - 2 ];
        mid[  n ] = (opus_int16)SKP_RSHIFT_ROUND( sum, 1 );
        side[ n ] = (opus_int16)SKP_SAT16( SKP_RSHIFT_ROUND( diff, 1 ) );
    }

    /* Buffering */
    SKP_memcpy( mid,  state->sMid,  2 * sizeof( opus_int16 ) );
    SKP_memcpy( side, state->sSide, 2 * sizeof( opus_int16 ) );
    SKP_memcpy( state->sMid,  &mid[  frame_length ], 2 * sizeof( opus_int16 ) );
    SKP_memcpy( state->sSide, &side[ frame_length ], 2 * sizeof( opus_int16 ) );

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

    /* Find energies and predictors */
    is10msFrame = frame_length == 10 * fs_kHz;
    smooth_coef_Q16 = is10msFrame ? 
        SILK_FIX_CONST( STEREO_RATIO_SMOOTH_COEF / 2, 16 ) : 
        SILK_FIX_CONST( STEREO_RATIO_SMOOTH_COEF,     16 );
    smooth_coef_Q16 = SKP_SMULWB( SKP_SMULBB( prev_speech_act_Q8 , prev_speech_act_Q8 ), smooth_coef_Q16 );

    pred_Q13[ 0 ] = silk_stereo_find_predictor( &LP_ratio_Q14, LP_mid, LP_side, &state->mid_side_amp_Q0[ 0 ], frame_length, smooth_coef_Q16 );
    pred_Q13[ 1 ] = silk_stereo_find_predictor( &HP_ratio_Q14, HP_mid, HP_side, &state->mid_side_amp_Q0[ 2 ], frame_length, smooth_coef_Q16 );
    /* Ratio of the norms of residual and mid signals */
    frac_Q16 = SKP_SMLABB( HP_ratio_Q14, LP_ratio_Q14, 3 );
    frac_Q16 = SKP_min( frac_Q16, SILK_FIX_CONST( 1, 16 ) );

    /* Determine bitrate distribution between mid and side, and possibly reduce stereo width */
    total_rate_bps -= is10msFrame ? 1200 : 600;      /* Subtract approximate bitrate for coding stereo parameters */
    min_mid_rate_bps = SKP_SMLABB( 2000, fs_kHz, 900 );
    SKP_assert( min_mid_rate_bps < 32767 );
    /* Default bitrate distribution: 8 parts for Mid and (5+3*frac) parts for Side. so: mid_rate = ( 8 / ( 13 + 3 * frac ) ) * total_ rate */
    frac_3_Q16 = SKP_MUL( 3, frac_Q16 );
    mid_side_rates_bps[ 0 ] = silk_DIV32_varQ( total_rate_bps, SILK_FIX_CONST( 8 + 5, 16 ) + frac_3_Q16, 16+3 );
    /* If Mid bitrate below minimum, reduce stereo width */
    if( mid_side_rates_bps[ 0 ] < min_mid_rate_bps ) {
        mid_side_rates_bps[ 0 ] = min_mid_rate_bps;
        mid_side_rates_bps[ 1 ] = total_rate_bps - mid_side_rates_bps[ 0 ];
        /* width = 4 * ( 2 * side_rate - min_rate ) / ( ( 1 + 3 * frac ) * min_rate ) */
        width_Q14 = silk_DIV32_varQ( SKP_LSHIFT( mid_side_rates_bps[ 1 ], 1 ) - min_mid_rate_bps, 
            SKP_SMULWB( SILK_FIX_CONST( 1, 16 ) + frac_3_Q16, min_mid_rate_bps ), 14+2 );
        width_Q14 = SKP_LIMIT( width_Q14, 0, SILK_FIX_CONST( 1, 14 ) );
    } else {
        mid_side_rates_bps[ 1 ] = total_rate_bps - mid_side_rates_bps[ 0 ];
        width_Q14 = SILK_FIX_CONST( 1, 14 );
    }

    /* Smoother */
    state->smth_width_Q14 = (opus_int16)SKP_SMLAWB( state->smth_width_Q14, width_Q14 - state->smth_width_Q14, smooth_coef_Q16 );

    /* Reduce predictors */
    pred_Q13[ 0 ] = SKP_RSHIFT( SKP_SMULBB( state->smth_width_Q14, pred_Q13[ 0 ] ), 14 );
    pred_Q13[ 1 ] = SKP_RSHIFT( SKP_SMULBB( state->smth_width_Q14, pred_Q13[ 1 ] ), 14 );

    ix[ 0 ][ 3 ] = 0;
    if( state->width_prev_Q14 == 0 && 
        ( 8 * total_rate_bps < 13 * min_mid_rate_bps || SKP_SMULWB( frac_Q16, state->smth_width_Q14 ) < SILK_FIX_CONST( 0.05, 14 ) ) ) 
    {
        width_Q14 = 0;
        /* Only encode mid channel */
        mid_side_rates_bps[ 0 ] = total_rate_bps;
        mid_side_rates_bps[ 1 ] = 0;
        ix[ 0 ][ 3 ] = 1;
    } else if( state->width_prev_Q14 != 0 && 
        ( 8 * total_rate_bps < 11 * min_mid_rate_bps || SKP_SMULWB( frac_Q16, state->smth_width_Q14 ) < SILK_FIX_CONST( 0.02, 14 ) ) ) 
    {
        width_Q14 = 0;
    } else if( state->smth_width_Q14 > SILK_FIX_CONST( 0.95, 14 ) ) {
        width_Q14 = SILK_FIX_CONST( 1, 14 );
    } else {
        width_Q14 = state->smth_width_Q14;
    }

#if 0
    DEBUG_STORE_DATA( midside.dat, &mid_side_rates_bps[ 0 ], 8 );
    DEBUG_STORE_DATA( norms0.pcm, &state->mid_side_amp_Q0[0], 8 );
    DEBUG_STORE_DATA( norms1.pcm, &state->mid_side_amp_Q0[2], 8 );
    DEBUG_STORE_DATA( width.pcm, &width_Q14, 4 );
#endif

    /* Quantize predictors */
    silk_stereo_quant_pred( state, pred_Q13, ix );

    /* Interpolate predictors and subtract prediction from side channel */
    pred0_Q13  = -state->pred_prev_Q13[ 0 ];
    pred1_Q13  = -state->pred_prev_Q13[ 1 ];
    w_Q24      =  SKP_LSHIFT( state->width_prev_Q14, 10 );
    denom_Q16  = SKP_DIV32_16( 1 << 16, STEREO_INTERP_LEN_MS * fs_kHz );
    delta0_Q13 = -SKP_RSHIFT_ROUND( SKP_SMULBB( pred_Q13[ 0 ] - state->pred_prev_Q13[ 0 ], denom_Q16 ), 16 );
    delta1_Q13 = -SKP_RSHIFT_ROUND( SKP_SMULBB( pred_Q13[ 1 ] - state->pred_prev_Q13[ 1 ], denom_Q16 ), 16 );
    deltaw_Q24 =  SKP_LSHIFT( SKP_SMULWB( width_Q14 - state->width_prev_Q14, denom_Q16 ), 10 );
    for( n = 0; n < STEREO_INTERP_LEN_MS * fs_kHz; n++ ) {
        pred0_Q13 += delta0_Q13;
        pred1_Q13 += delta1_Q13;
        w_Q24   += deltaw_Q24;
        sum = SKP_LSHIFT( SKP_ADD_LSHIFT( mid[ n ] + mid[ n + 2 ], mid[ n + 1 ], 1 ), 9 );      /* Q11 */ 
        sum = SKP_SMLAWB( SKP_SMULWB( w_Q24, side[ n + 1 ] ), sum, pred0_Q13 );                 /* Q8  */
        sum = SKP_SMLAWB( sum, SKP_LSHIFT( ( opus_int32 )mid[ n + 1 ], 11 ), pred1_Q13 );        /* Q8  */
        x2[ n - 1 ] = (opus_int16)SKP_SAT16( SKP_RSHIFT_ROUND( sum, 8 ) );
    }
    pred0_Q13 = -pred_Q13[ 0 ];
    pred1_Q13 = -pred_Q13[ 1 ];
    w_Q24     =  SKP_LSHIFT( width_Q14, 10 );
    for( n = STEREO_INTERP_LEN_MS * fs_kHz; n < frame_length; n++ ) {
        sum = SKP_LSHIFT( SKP_ADD_LSHIFT( mid[ n ] + mid[ n + 2 ], mid[ n + 1 ], 1 ), 9 );      /* Q11 */ 
        sum = SKP_SMLAWB( SKP_SMULWB( w_Q24, side[ n + 1 ] ), sum, pred0_Q13 );                 /* Q8  */
        sum = SKP_SMLAWB( sum, SKP_LSHIFT( ( opus_int32 )mid[ n + 1 ], 11 ), pred1_Q13 );        /* Q8  */
        x2[ n - 1 ] = (opus_int16)SKP_SAT16( SKP_RSHIFT_ROUND( sum, 8 ) );
    }
    state->pred_prev_Q13[ 0 ] = (opus_int16)pred_Q13[ 0 ];
    state->pred_prev_Q13[ 1 ] = (opus_int16)pred_Q13[ 1 ];
    state->width_prev_Q14     = (opus_int16)width_Q14;
}
