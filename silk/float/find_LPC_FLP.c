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

#include "main_FLP.h"
#include "tuning_parameters.h"

void silk_find_LPC_FLP(
    opus_int16                      NLSF_Q15[],                         /* O    NLSFs                                       */
    opus_int8                       *interpIndex,                       /* O    NLSF interp. index for NLSF interp.         */
    const opus_int16                prev_NLSFq_Q15[],                   /* I    Previous NLSFs, for NLSF interpolation      */
    const opus_int                  useInterpNLSFs,                     /* I    Flag                                        */
    const opus_int                  firstFrameAfterReset,               /* I    Flag                                        */
    const opus_int                  LPC_order,                          /* I    LPC order                                   */
    const silk_float                x[],                                /* I    Input signal                                */
    const opus_int                  subfr_length,                       /* I    Subframe length incl preceeding samples     */
    const opus_int                  nb_subfr                            /* I    Number of subframes                         */
)
{
    opus_int     k;
    silk_float   a[ MAX_LPC_ORDER ];

    /* Used only for NLSF interpolation */
    double      res_nrg, res_nrg_2nd, res_nrg_interp;
    opus_int16   NLSF0_Q15[ MAX_LPC_ORDER ];
    silk_float   a_tmp[ MAX_LPC_ORDER ];
    silk_float   LPC_res[ ( MAX_FRAME_LENGTH + MAX_NB_SUBFR * MAX_LPC_ORDER ) / 2 ];

    /* Default: No interpolation */
    *interpIndex = 4;

    /* Burg AR analysis for the full frame */
    res_nrg = silk_burg_modified_FLP( a, x, subfr_length, nb_subfr, FIND_LPC_COND_FAC, LPC_order );

    if( firstFrameAfterReset ) {
        silk_bwexpander_FLP( a, LPC_order, FIND_LPC_CHIRP_FIRST_FRAME );
    } else {
        silk_bwexpander_FLP( a_tmp, LPC_order, FIND_LPC_CHIRP );
    }

    if( useInterpNLSFs && !firstFrameAfterReset && nb_subfr == MAX_NB_SUBFR ) {
        /* Optimal solution for last 10 ms; subtract residual energy here, as that's easier than        */
        /* adding it to the residual energy of the first 10 ms in each iteration of the search below    */
        res_nrg -= silk_burg_modified_FLP( a_tmp, x + ( MAX_NB_SUBFR / 2 ) * subfr_length,
            subfr_length, MAX_NB_SUBFR / 2, FIND_LPC_COND_FAC, LPC_order );

        silk_bwexpander_FLP( a_tmp, LPC_order, FIND_LPC_CHIRP );

        /* Convert to NLSFs */
        silk_A2NLSF_FLP( NLSF_Q15, a_tmp, LPC_order );

        /* Search over interpolation indices to find the one with lowest residual energy */
        res_nrg_2nd = silk_float_MAX;
        for( k = 3; k >= 0; k-- ) {
            /* Interpolate NLSFs for first half */
            silk_interpolate( NLSF0_Q15, prev_NLSFq_Q15, NLSF_Q15, k, LPC_order );

            /* Convert to LPC for residual energy evaluation */
            silk_NLSF2A_FLP( a_tmp, NLSF0_Q15, LPC_order );

            /* Calculate residual energy with LSF interpolation */
            silk_LPC_analysis_filter_FLP( LPC_res, a_tmp, x, 2 * subfr_length, LPC_order );
            res_nrg_interp =
                silk_energy_FLP( LPC_res + LPC_order,                subfr_length - LPC_order ) +
                silk_energy_FLP( LPC_res + LPC_order + subfr_length, subfr_length - LPC_order );

            /* Determine whether current interpolated NLSFs are best so far */
            if( res_nrg_interp < res_nrg ) {
                /* Interpolation has lower residual energy */
                res_nrg = res_nrg_interp;
                *interpIndex = (opus_int8)k;
            } else if( res_nrg_interp > res_nrg_2nd ) {
                /* No reason to continue iterating - residual energies will continue to climb */
                break;
            }
            res_nrg_2nd = res_nrg_interp;
        }
    }

    if( *interpIndex == 4 ) {
        /* NLSF interpolation is currently inactive, calculate NLSFs from full frame AR coefficients */
        silk_A2NLSF_FLP( NLSF_Q15, a, LPC_order );
    }

    silk_assert( *interpIndex == 4 || ( useInterpNLSFs && !firstFrameAfterReset && nb_subfr == MAX_NB_SUBFR ) );
}
