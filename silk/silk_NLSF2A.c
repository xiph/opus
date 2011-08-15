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

/* conversion between prediction filter coefficients and LSFs   */
/* order should be even                                         */
/* a piecewise linear approximation maps LSF <-> cos(LSF)       */
/* therefore the result is not accurate LSFs, but the two       */
/* functions are accurate inverses of each other                */

#include "silk_SigProc_FIX.h"
#include "silk_tables.h"

#define QA      16

/* helper function for NLSF2A(..) */
static inline void silk_NLSF2A_find_poly(
    opus_int32          *out,      /* O    intermediate polynomial, QA [dd+1]        */
    const opus_int32    *cLSF,     /* I    vector of interleaved 2*cos(LSFs), QA [d] */
    opus_int            dd         /* I    polynomial order (= 1/2 * filter order)   */
)
{
    opus_int   k, n;
    opus_int32 ftmp;

    out[0] = SKP_LSHIFT( 1, QA );
    out[1] = -cLSF[0];
    for( k = 1; k < dd; k++ ) {
        ftmp = cLSF[2*k];            // QA
        out[k+1] = SKP_LSHIFT( out[k-1], 1 ) - (opus_int32)SKP_RSHIFT_ROUND64( SKP_SMULL( ftmp, out[k] ), QA );
        for( n = k; n > 1; n-- ) {
            out[n] += out[n-2] - (opus_int32)SKP_RSHIFT_ROUND64( SKP_SMULL( ftmp, out[n-1] ), QA );
        }
        out[1] -= ftmp;
    }
}

/* compute whitening filter coefficients from normalized line spectral frequencies */
void silk_NLSF2A(
    opus_int16        *a_Q12,            /* O    monic whitening filter coefficients in Q12,  [ d ]  */
    const opus_int16  *NLSF,             /* I    normalized line spectral frequencies in Q15, [ d ]  */
    const opus_int    d                  /* I    filter order (should be even)                       */
)
{
    opus_int   k, i, dd;
    opus_int32 cos_LSF_QA[ SILK_MAX_ORDER_LPC ];
    opus_int32 P[ SILK_MAX_ORDER_LPC / 2 + 1 ], Q[ SILK_MAX_ORDER_LPC / 2 + 1 ];
    opus_int32 Ptmp, Qtmp, f_int, f_frac, cos_val, delta;
    opus_int32 a32_QA1[ SILK_MAX_ORDER_LPC ];
    opus_int32 maxabs, absval, idx=0, sc_Q16, invGain_Q30;

    SKP_assert( LSF_COS_TAB_SZ_FIX == 128 );

    /* convert LSFs to 2*cos(LSF), using piecewise linear curve from table */
    for( k = 0; k < d; k++ ) {
        SKP_assert(NLSF[k] >= 0 );
        SKP_assert(NLSF[k] <= 32767 );

        /* f_int on a scale 0-127 (rounded down) */
        f_int = SKP_RSHIFT( NLSF[k], 15 - 7 );

        /* f_frac, range: 0..255 */
        f_frac = NLSF[k] - SKP_LSHIFT( f_int, 15 - 7 );

        SKP_assert(f_int >= 0);
        SKP_assert(f_int < LSF_COS_TAB_SZ_FIX );

        /* Read start and end value from table */
        cos_val = silk_LSFCosTab_FIX_Q12[ f_int ];                /* Q12 */
        delta   = silk_LSFCosTab_FIX_Q12[ f_int + 1 ] - cos_val;  /* Q12, with a range of 0..200 */

        /* Linear interpolation */
        cos_LSF_QA[k] = SKP_RSHIFT_ROUND( SKP_LSHIFT( cos_val, 8 ) + SKP_MUL( delta, f_frac ), 20 - QA ); /* QA */
    }

    dd = SKP_RSHIFT( d, 1 );

    /* generate even and odd polynomials using convolution */
    silk_NLSF2A_find_poly( P, &cos_LSF_QA[ 0 ], dd );
    silk_NLSF2A_find_poly( Q, &cos_LSF_QA[ 1 ], dd );

    /* convert even and odd polynomials to opus_int32 Q12 filter coefs */
    for( k = 0; k < dd; k++ ) {
        Ptmp = P[ k+1 ] + P[ k ];
        Qtmp = Q[ k+1 ] - Q[ k ];

        /* the Ptmp and Qtmp values at this stage need to fit in int32 */
        a32_QA1[ k ]     = -Qtmp - Ptmp;        /* QA+1 */
        a32_QA1[ d-k-1 ] =  Qtmp - Ptmp;        /* QA+1 */
    }

    /* Limit the maximum absolute value of the prediction coefficients, so that they'll fit in int16 */
    for( i = 0; i < 10; i++ ) {
        /* Find maximum absolute value and its index */
        maxabs = 0;
        for( k = 0; k < d; k++ ) {
            absval = SKP_abs( a32_QA1[k] );
            if( absval > maxabs ) {
                maxabs = absval;
                idx    = k;
            }
        }
        maxabs = SKP_RSHIFT_ROUND( maxabs, QA + 1 - 12 );       /* QA+1 -> Q12 */

        if( maxabs > SKP_int16_MAX ) {
            /* Reduce magnitude of prediction coefficients */
            maxabs = SKP_min( maxabs, 163838 );  /* ( SKP_int32_MAX >> 14 ) + SKP_int16_MAX = 163838 */
            sc_Q16 = SILK_FIX_CONST( 0.999, 16 ) - SKP_DIV32( SKP_LSHIFT( maxabs - SKP_int16_MAX, 14 ),
                                        SKP_RSHIFT32( SKP_MUL( maxabs, idx + 1), 2 ) );
            silk_bwexpander_32( a32_QA1, d, sc_Q16 );
        } else {
            break;
        }
    }

    if( i == 10 ) {
        /* Reached the last iteration, clip the coefficients */
        for( k = 0; k < d; k++ ) {
            a_Q12[ k ] = (opus_int16)SKP_SAT16( SKP_RSHIFT_ROUND( a32_QA1[ k ], QA + 1 - 12 ) ); /* QA+1 -> Q12 */
            a32_QA1[ k ] = SKP_LSHIFT( (opus_int32)a_Q12[ k ], QA + 1 - 12 );
        }
    } else {
        for( k = 0; k < d; k++ ) {
            a_Q12[ k ] = (opus_int16)SKP_RSHIFT_ROUND( a32_QA1[ k ], QA + 1 - 12 );       /* QA+1 -> Q12 */
        }
    }

    for( i = 1; i < MAX_LPC_STABILIZE_ITERATIONS; i++ ) {
        if( silk_LPC_inverse_pred_gain( &invGain_Q30, a_Q12, d ) == 1 ) {
            /* Prediction coefficients are (too close to) unstable; apply bandwidth expansion   */
            /* on the unscaled coefficients, convert to Q12 and measure again                   */
            silk_bwexpander_32( a32_QA1, d, 65536 - SKP_SMULBB( 9 + i, i ) );            /* 10_Q16 = 0.00015 */
            for( k = 0; k < d; k++ ) {
                a_Q12[ k ] = (opus_int16)SKP_RSHIFT_ROUND( a32_QA1[ k ], QA + 1 - 12 );  /* QA+1 -> Q12 */
            }
        } else {
            break;
        }
    }

    if( i == MAX_LPC_STABILIZE_ITERATIONS ) {
        /* Reached the last iteration, set coefficients to zero */
        for( k = 0; k < d; k++ ) {
            a_Q12[ k ] = 0;
        }
    }
}

