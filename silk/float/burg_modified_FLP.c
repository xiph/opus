/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "SigProc_FLP.h"
#include "tuning_parameters.h"
#include "define.h"

/* This code implements the method from https://www.opus-codec.org/docs/vos_fastburg.pdf */

/* Compute reflection coefficients from input signal */
silk_float silk_burg_modified_FLP(
    silk_float          af[],               /* O    prediction coefficients (length order)                      */
    const silk_float    x[],                /* I    input signal, length: nb_subfr*(D+L_sub)                    */
    const silk_float    minInvGain,         /* I    minimum inverse prediction gain                             */
    const opus_int      subfr_length,       /* I    input signal subframe length (incl. D preceding samples)    */
    const opus_int      nb_subfr,           /* I    number of subframes stacked in x                            */
    const opus_int      D                   /* I    order                                                       */
)
{
    opus_int         k, n, s, reached_max_gain;
    double           invGain, num, nrg, rc, tmp1, tmp2, x1, x2, atmp;
    const silk_float *x_ptr;
    double           c[ SILK_MAX_ORDER_LPC + 1 ];
    double           g[ SILK_MAX_ORDER_LPC + 1 ];
    double           a[ SILK_MAX_ORDER_LPC ];

    /* Compute autocorrelations, added over subframes */
    silk_memset( c, 0, (D + 1) * sizeof( double ) );
    for( s = 0; s < nb_subfr; s++ ) {
        x_ptr = x + s * subfr_length;
        for( n = 0; n < D + 1; n++ ) {
            c[ n ] += silk_inner_product_FLP( x_ptr, x_ptr + n, subfr_length - n );
        }
    }
    for( n = 0; n < D + 1; n++ ) {
        c[ n ] *= 2.0;
    }

    /* Initialize */
    c[ 0 ] += FIND_LPC_COND_FAC * c[ 0 ] + 1e-9f ;
    g[ 0 ] = c[ 0 ];
    tmp1 = 0.0f;
    for( s = 0; s < nb_subfr; s++ ) {
        x_ptr = x + s * subfr_length;
        x1 = x_ptr[ 0 ];
        x2 = x_ptr[ subfr_length - 1 ];
        tmp1 += x1 * x1 + x2 * x2;
    }
    g[ 0 ] -= tmp1;
    g[ 1 ] = c[ 1 ];
    rc = - g[ 1 ] / g[ 0 ];
    silk_assert( rc > -1.0 && rc < 1.0 );
    a[ 0 ] = rc;
    invGain = ( 1.0 - rc * rc );
    reached_max_gain = 0;
    for( n = 1; n < D; n++ ) {
        for( k = 0; k < (n >> 1) + 1; k++ ) {
            tmp1 = g[ k ];
            tmp2 = g[ n - k ];
            g[ k ]     = tmp1 + rc * tmp2;
            g[ n - k ] = tmp2 + rc * tmp1;
        }
        for( s = 0; s < nb_subfr; s++ ) {
            x_ptr = x + s * subfr_length;
            x1 = x_ptr[ n ];
            x2 = x_ptr[ subfr_length - n - 1 ];
            tmp1 = x1;
            tmp2 = x2;
            for( k = 0; k < n; k++ ) {
                atmp = a[ k ];
                c[ k + 1 ] -= x1 * x_ptr[ n - k - 1 ] + x2 * x_ptr[ subfr_length - n + k ];
                tmp1 += x_ptr[ n - k - 1 ] * atmp;
                tmp2 += x_ptr[ subfr_length - n + k ] * atmp;
            }
            for( k = 0; k <= n; k++ ) {
                g[ k ] -= tmp1 * x_ptr[ n - k ] + tmp2 * x_ptr[ subfr_length - n + k - 1 ];
            }
        }

        /* Calculate nominator and denominator for the next order reflection (parcor) coefficient */
        tmp1 = c[ n + 1 ];
        num = 0.0f;
        nrg = g[ 0 ];
        for( k = 0; k < n; k++ ) {
            atmp = a[ k ];
            tmp1 += c[ n - k ] * atmp;
            num  += g[ n - k ] * atmp;
            nrg  += g[ k + 1 ] * atmp;
        }
        g[ n + 1] = tmp1;
        num += tmp1;
        silk_assert( nrg > 0.0 );

        /* Calculate the next order reflection (parcor) coefficient */
        rc = -num / nrg;
        silk_assert( rc > -1.0 && rc < 1.0 );

        /* Update inverse prediction gain */
        tmp1 = invGain * ( 1.0 - rc * rc );
        if( tmp1 <= minInvGain ) {
            /* Max prediction gain exceeded; set reflection coefficient such that max prediction gain is exactly hit */
            rc = sqrt( 1.0 - minInvGain / invGain );
            if( num > 0 ) {
                /* Ensure adjusted reflection coefficient has the original sign */
                rc = -rc;
            }
            invGain = minInvGain;
            reached_max_gain = 1;
        } else {
            invGain = tmp1;
        }

        /* Update the AR coefficients */
        for( k = 0; k < (n + 1) >> 1; k++ ) {
            tmp1 = a[ k ];
            tmp2 = a[ n - k - 1 ];
            a[ k ]         = tmp1 + rc * tmp2;
            a[ n - k - 1 ] = tmp2 + rc * tmp1;
        }
        a[ n ] = rc;

        if( reached_max_gain ) {
            /* Reached max prediction gain; set remaining coefficients to zero and exit loop */
            for( k = n + 1; k < D; k++ ) {
                a[ k ] = 0.0;
            }
            break;
        }
    }

    /* Convert to silk_float */
    for( k = 0; k < D; k++ ) {
        af[ k ] = (silk_float)( -a[ k ] );
    }

    nrg = c[ 0 ] * 0.5 * (1.0 - FIND_LPC_COND_FAC);
    /* Subtract energy of preceding samples from C0 */
    for( s = 0; s < nb_subfr; s++ ) {
        nrg -= silk_energy_FLP( x + s * subfr_length, D );
    }
    /* Approximate residual energy */
    nrg *= invGain;

    /* Return approximate residual energy */
    return (silk_float)nrg;
}
