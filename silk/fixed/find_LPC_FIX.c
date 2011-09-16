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

#include "main_FIX.h"
#include "tuning_parameters.h"

/* Finds LPC vector from correlations, and converts to NLSF */
void silk_find_LPC_FIX(
    opus_int16                       NLSF_Q15[],             /* O    NLSFs                                                           */
    opus_int8                        *interpIndex,           /* O    NLSF interpolation index, only used for NLSF interpolation      */
    const opus_int16                 prev_NLSFq_Q15[],       /* I    previous NLSFs, only used for NLSF interpolation                */
    const opus_int                   useInterpNLSFs,         /* I    Flag                                                            */
    const opus_int                   firstFrameAfterReset,   /* I    Flag                                                            */
    const opus_int                   LPC_order,              /* I    LPC order                                                       */
    const opus_int16                 x[],                    /* I    Input signal                                                    */
    const opus_int                   subfr_length,           /* I    Input signal subframe length including preceeding samples       */
    const opus_int                   nb_subfr                /* I:   Number of subframes                                             */
)
{
    opus_int     k;
    opus_int32   a_Q16[ MAX_LPC_ORDER ];
    opus_int     isInterpLower, shift;
    opus_int32   res_nrg0, res_nrg1;
    opus_int     rshift0, rshift1;

    /* Used only for LSF interpolation */
    opus_int32   a_tmp_Q16[ MAX_LPC_ORDER ], res_nrg_interp, res_nrg, res_tmp_nrg, res_nrg_2nd;
    opus_int     res_nrg_interp_Q, res_nrg_Q, res_tmp_nrg_Q, res_nrg_2nd_Q;
    opus_int16   a_tmp_Q12[ MAX_LPC_ORDER ];
    opus_int16   NLSF0_Q15[ MAX_LPC_ORDER ];
    opus_int16   LPC_res[ ( MAX_FRAME_LENGTH + MAX_NB_SUBFR * MAX_LPC_ORDER ) / 2 ];

    /* Default: no interpolation */
    *interpIndex = 4;

    /* Burg AR analysis for the full frame */
    silk_burg_modified( &res_nrg, &res_nrg_Q, a_Q16, x, subfr_length, nb_subfr, SILK_FIX_CONST( FIND_LPC_COND_FAC, 32 ), LPC_order );

    if( firstFrameAfterReset ) {
        silk_bwexpander_32( a_Q16, LPC_order, SILK_FIX_CONST( FIND_LPC_CHIRP_FIRST_FRAME, 16 ) );
    } else {
        silk_bwexpander_32( a_Q16, LPC_order, SILK_FIX_CONST( FIND_LPC_CHIRP, 16 ) );
    }

    if( useInterpNLSFs && !firstFrameAfterReset && nb_subfr == MAX_NB_SUBFR ) {

        /* Optimal solution for last 10 ms */
        silk_burg_modified( &res_tmp_nrg, &res_tmp_nrg_Q, a_tmp_Q16, x + ( MAX_NB_SUBFR >> 1 ) * subfr_length,
            subfr_length, ( MAX_NB_SUBFR >> 1 ), SILK_FIX_CONST( FIND_LPC_COND_FAC, 32 ), LPC_order );

        silk_bwexpander_32( a_tmp_Q16, LPC_order, SILK_FIX_CONST( FIND_LPC_CHIRP, 16 ) );

        /* subtract residual energy here, as that's easier than adding it to the    */
        /* residual energy of the first 10 ms in each iteration of the search below */
        shift = res_tmp_nrg_Q - res_nrg_Q;
        if( shift >= 0 ) {
            if( shift < 32 ) {
                res_nrg = res_nrg - silk_RSHIFT( res_tmp_nrg, shift );
            }
        } else {
            silk_assert( shift > -32 );
            res_nrg   = silk_RSHIFT( res_nrg, -shift ) - res_tmp_nrg;
            res_nrg_Q = res_tmp_nrg_Q;
        }

        /* Convert to NLSFs */
        silk_A2NLSF( NLSF_Q15, a_tmp_Q16, LPC_order );

        /* Search over interpolation indices to find the one with lowest residual energy */
        res_nrg_2nd = silk_int32_MAX;
        for( k = 3; k >= 0; k-- ) {
            /* Interpolate NLSFs for first half */
            silk_interpolate( NLSF0_Q15, prev_NLSFq_Q15, NLSF_Q15, k, LPC_order );

            /* Convert to LPC for residual energy evaluation */
            silk_NLSF2A( a_tmp_Q12, NLSF0_Q15, LPC_order );

            /* Calculate residual energy with NLSF interpolation */
            silk_LPC_analysis_filter( LPC_res, x, a_tmp_Q12, 2 * subfr_length, LPC_order );

            silk_sum_sqr_shift( &res_nrg0, &rshift0, LPC_res + LPC_order,                subfr_length - LPC_order );
            silk_sum_sqr_shift( &res_nrg1, &rshift1, LPC_res + LPC_order + subfr_length, subfr_length - LPC_order );

            /* Add subframe energies from first half frame */
            shift = rshift0 - rshift1;
            if( shift >= 0 ) {
                res_nrg1         = silk_RSHIFT( res_nrg1, shift );
                res_nrg_interp_Q = -rshift0;
            } else {
                res_nrg0         = silk_RSHIFT( res_nrg0, -shift );
                res_nrg_interp_Q = -rshift1;
            }
            res_nrg_interp = silk_ADD32( res_nrg0, res_nrg1 );

            /* Compare with first half energy without NLSF interpolation, or best interpolated value so far */
            shift = res_nrg_interp_Q - res_nrg_Q;
            if( shift >= 0 ) {
                if( silk_RSHIFT( res_nrg_interp, shift ) < res_nrg ) {
                    isInterpLower = silk_TRUE;
                } else {
                    isInterpLower = silk_FALSE;
                }
            } else {
                if( -shift < 32 ) {
                    if( res_nrg_interp < silk_RSHIFT( res_nrg, -shift ) ) {
                        isInterpLower = silk_TRUE;
                    } else {
                        isInterpLower = silk_FALSE;
                    }
                } else {
                    isInterpLower = silk_FALSE;
                }
            }

            /* Determine whether current interpolated NLSFs are best so far */
            if( isInterpLower == silk_TRUE ) {
                /* Interpolation has lower residual energy */
                res_nrg   = res_nrg_interp;
                res_nrg_Q = res_nrg_interp_Q;
                *interpIndex = (opus_int8)k;
            }
            res_nrg_2nd   = res_nrg_interp;
            res_nrg_2nd_Q = res_nrg_interp_Q;
        }
    }

    if( *interpIndex == 4 ) {
        /* NLSF interpolation is currently inactive, calculate NLSFs from full frame AR coefficients */
        silk_A2NLSF( NLSF_Q15, a_Q16, LPC_order );
    }

    silk_assert( *interpIndex == 4 || ( useInterpNLSFs && !firstFrameAfterReset && nb_subfr == MAX_NB_SUBFR ) );
}
