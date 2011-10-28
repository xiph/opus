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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "main.h"

/**********************************************************/
/* Core decoder. Performs inverse NSQ operation LTP + LPC */
/**********************************************************/
void silk_decode_core(
    silk_decoder_state          *psDec,                         /* I/O  Decoder state                               */
    silk_decoder_control        *psDecCtrl,                     /* I    Decoder control                             */
    opus_int16                  xq[],                           /* O    Decoded speech                              */
    const opus_int              pulses[ MAX_FRAME_LENGTH ]      /* I    Pulse signal                                */
)
{
    opus_int   i, j, k, lag = 0, start_idx, sLTP_buf_idx, NLSF_interpolation_flag, signalType;
    opus_int16 *A_Q12, *B_Q14, *pxq, A_Q12_tmp[ MAX_LPC_ORDER ];
    opus_int16 sLTP[ MAX_FRAME_LENGTH ];
    opus_int32 sLTP_Q16[ 2 * MAX_FRAME_LENGTH ];
    opus_int32 LTP_pred_Q14, LPC_pred_Q10, Gain_Q10, inv_gain_Q16, inv_gain_Q32, gain_adj_Q16, rand_seed, offset_Q10;
    opus_int32 *pred_lag_ptr, *pexc_Q10, *pres_Q10;
    opus_int32 res_Q10[ MAX_SUB_FRAME_LENGTH ];
    opus_int32 sLPC_Q14[ MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER ];

    silk_assert( psDec->prev_inv_gain_Q16 != 0 );

    offset_Q10 = silk_Quantization_Offsets_Q10[ psDec->indices.signalType >> 1 ][ psDec->indices.quantOffsetType ];

    if( psDec->indices.NLSFInterpCoef_Q2 < 1 << 2 ) {
        NLSF_interpolation_flag = 1;
    } else {
        NLSF_interpolation_flag = 0;
    }

    /* Decode excitation */
    rand_seed = psDec->indices.Seed;
    for( i = 0; i < psDec->frame_length; i++ ) {
        rand_seed = silk_RAND( rand_seed );
        psDec->exc_Q10[ i ] = silk_LSHIFT( ( opus_int32 )pulses[ i ], 10 );
        if( psDec->exc_Q10[ i ] > 0 ) {
            psDec->exc_Q10[ i ] -= QUANT_LEVEL_ADJUST_Q10;
        } else
        if( psDec->exc_Q10[ i ] < 0 ) {
            psDec->exc_Q10[ i ] += QUANT_LEVEL_ADJUST_Q10;
        }
        psDec->exc_Q10[ i ] += offset_Q10;
        psDec->exc_Q10[ i ] ^= silk_RSHIFT( rand_seed, 31 );

        rand_seed = silk_ADD32_ovflw(rand_seed, pulses[ i ]);
    }

    /* Copy LPC state */
    silk_memcpy( sLPC_Q14, psDec->sLPC_Q14_buf, MAX_LPC_ORDER * sizeof( opus_int32 ) );

    pexc_Q10 = psDec->exc_Q10;
    pxq      = xq;
    sLTP_buf_idx = psDec->ltp_mem_length;
    /* Loop over subframes */
    for( k = 0; k < psDec->nb_subfr; k++ ) {
        pres_Q10 = res_Q10;
        A_Q12 = psDecCtrl->PredCoef_Q12[ k >> 1 ];

        /* Preload LPC coeficients to array on stack. Gives small performance gain */
        silk_memcpy( A_Q12_tmp, A_Q12, psDec->LPC_order * sizeof( opus_int16 ) );
        B_Q14        = &psDecCtrl->LTPCoef_Q14[ k * LTP_ORDER ];
        signalType   = psDec->indices.signalType;

        Gain_Q10     = silk_RSHIFT( psDecCtrl->Gains_Q16[ k ], 6 );
        inv_gain_Q16 = silk_INVERSE32_varQ( psDecCtrl->Gains_Q16[ k ], 32 );
        inv_gain_Q16 = silk_min( inv_gain_Q16, silk_int16_MAX );

        /* Calculate Gain adjustment factor */
        gain_adj_Q16 = 1 << 16;
        if( inv_gain_Q16 != psDec->prev_inv_gain_Q16 ) {
            gain_adj_Q16 =  silk_DIV32_varQ( inv_gain_Q16, psDec->prev_inv_gain_Q16, 16 );

            /* Scale short term state */
            for( i = 0; i < MAX_LPC_ORDER; i++ ) {
                sLPC_Q14[ i ] = silk_SMULWW( gain_adj_Q16, sLPC_Q14[ i ] );
            }
        }

        /* Save inv_gain */
        silk_assert( inv_gain_Q16 != 0 );
        psDec->prev_inv_gain_Q16 = inv_gain_Q16;

        /* Avoid abrupt transition from voiced PLC to unvoiced normal decoding */
        if( psDec->lossCnt && psDec->prevSignalType == TYPE_VOICED &&
            psDec->indices.signalType != TYPE_VOICED && k < MAX_NB_SUBFR/2 ) {

            silk_memset( B_Q14, 0, LTP_ORDER * sizeof( opus_int16 ) );
            B_Q14[ LTP_ORDER/2 ] = SILK_FIX_CONST( 0.25, 14 );

            signalType = TYPE_VOICED;
            psDecCtrl->pitchL[ k ] = psDec->lagPrev;
        }

        if( signalType == TYPE_VOICED ) {
            /* Voiced */
            lag = psDecCtrl->pitchL[ k ];

            /* Re-whitening */
            if( k == 0 || ( k == 2 && NLSF_interpolation_flag ) ) {
                /* Rewhiten with new A coefs */
                start_idx = psDec->ltp_mem_length - lag - psDec->LPC_order - LTP_ORDER / 2;
                silk_assert( start_idx > 0 );

                if( k == 2 ) {
                    silk_memcpy( &psDec->outBuf[ psDec->ltp_mem_length ], xq, 2 * psDec->subfr_length * sizeof( opus_int16 ) );
                }

                silk_LPC_analysis_filter( &sLTP[ start_idx ], &psDec->outBuf[ start_idx + k * psDec->subfr_length ],
                    A_Q12, psDec->ltp_mem_length - start_idx, psDec->LPC_order );

                /* After rewhitening the LTP state is unscaled */
                inv_gain_Q32 = silk_LSHIFT( inv_gain_Q16, 16 );
                if( k == 0 ) {
                    /* Do LTP downscaling to reduce inter-packet dependency */
                    inv_gain_Q32 = silk_LSHIFT( silk_SMULWB( inv_gain_Q32, psDecCtrl->LTP_scale_Q14 ), 2 );
                }
                for( i = 0; i < lag + LTP_ORDER/2; i++ ) {
                    sLTP_Q16[ sLTP_buf_idx - i - 1 ] = silk_SMULWB( inv_gain_Q32, sLTP[ psDec->ltp_mem_length - i - 1 ] );
                }
            } else {
                /* Update LTP state when Gain changes */
                if( gain_adj_Q16 != 1 << 16 ) {
                    for( i = 0; i < lag + LTP_ORDER/2; i++ ) {
                        sLTP_Q16[ sLTP_buf_idx - i - 1 ] = silk_SMULWW( gain_adj_Q16, sLTP_Q16[ sLTP_buf_idx - i - 1 ] );
                    }
                }
            }
        }

        /* Long-term prediction */
        if( signalType == TYPE_VOICED ) {
            /* Setup pointer */
            pred_lag_ptr = &sLTP_Q16[ sLTP_buf_idx - lag + LTP_ORDER / 2 ];
            for( i = 0; i < psDec->subfr_length; i++ ) {
                /* Unrolled loop */
                LTP_pred_Q14 = silk_SMULWB(               pred_lag_ptr[  0 ], B_Q14[ 0 ] );
                LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[ -1 ], B_Q14[ 1 ] );
                LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[ -2 ], B_Q14[ 2 ] );
                LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[ -3 ], B_Q14[ 3 ] );
                LTP_pred_Q14 = silk_SMLAWB( LTP_pred_Q14, pred_lag_ptr[ -4 ], B_Q14[ 4 ] );
                pred_lag_ptr++;

                /* Generate LPC excitation */
                pres_Q10[ i ] = silk_ADD32( pexc_Q10[ i ], silk_RSHIFT_ROUND( LTP_pred_Q14, 4 ) );

                /* Update states */
                sLTP_Q16[ sLTP_buf_idx ] = silk_LSHIFT( pres_Q10[ i ], 6 );
                sLTP_buf_idx++;
            }
        } else {
            pres_Q10 = pexc_Q10;
        }

        for( i = 0; i < psDec->subfr_length; i++ ) {
            /* Partially unrolled */
            LPC_pred_Q10 = silk_SMULWB(               sLPC_Q14[ MAX_LPC_ORDER + i -  1 ], A_Q12_tmp[ 0 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  2 ], A_Q12_tmp[ 1 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  3 ], A_Q12_tmp[ 2 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  4 ], A_Q12_tmp[ 3 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  5 ], A_Q12_tmp[ 4 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  6 ], A_Q12_tmp[ 5 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  7 ], A_Q12_tmp[ 6 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  8 ], A_Q12_tmp[ 7 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i -  9 ], A_Q12_tmp[ 8 ] );
            LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i - 10 ], A_Q12_tmp[ 9 ] );
            for( j = 10; j < psDec->LPC_order; j++ ) {
                LPC_pred_Q10 = silk_SMLAWB( LPC_pred_Q10, sLPC_Q14[ MAX_LPC_ORDER + i - j - 1 ], A_Q12_tmp[ j ] );
            }

            /* Add prediction to LPC excitation */
            sLPC_Q14[ MAX_LPC_ORDER + i ] = silk_LSHIFT( silk_ADD32( pres_Q10[ i ], LPC_pred_Q10 ), 4 );

            /* Scale with Gain */
            pxq[ i ] = ( opus_int16 )silk_SAT16( silk_RSHIFT_ROUND( silk_SMULWW( sLPC_Q14[ MAX_LPC_ORDER + i ], Gain_Q10 ), 8 ) );
        }

        /* Update LPC filter state */
        silk_memcpy( sLPC_Q14, &sLPC_Q14[ psDec->subfr_length ], MAX_LPC_ORDER * sizeof( opus_int32 ) );
        pexc_Q10 += psDec->subfr_length;
        pxq      += psDec->subfr_length;
    }

    /* Save LPC state */
    silk_memcpy( psDec->sLPC_Q14_buf, sLPC_Q14, MAX_LPC_ORDER * sizeof( opus_int32 ) );
}
