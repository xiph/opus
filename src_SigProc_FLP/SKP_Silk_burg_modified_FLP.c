/***********************************************************************
Copyright (c) 2006-2010, Skype Limited. All rights reserved. 
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

/*                                                                      *
 * SKP_Silk_burg_modified.c                                           *
 *                                                                      *
 * Calculates the reflection coefficients from the input vector         *
 * Input vector contains nb_subfr sub vectors of length L_sub + D       *
 *                                                                      *
 * Copyright 2009 (c), Skype Limited                                    *
 * Date: 091130                                                         *
 */

#include "SKP_Silk_SigProc_FLP.h"

#define MAX_FRAME_SIZE              544 // subfr_length * nb_subfr = ( 0.005 * 24000 + 16 ) * 4 = 544
#define MAX_NB_SUBFR                4

/* Compute reflection coefficients from input signal */
SKP_float SKP_Silk_burg_modified_FLP(     /* O    returns residual energy                                         */
    SKP_float       A[],                /* O    prediction coefficients (length order)                          */
    const SKP_float x[],                /* I    input signal, length: nb_subfr*(D+L_sub)                        */
    const SKP_int   subfr_length,       /* I    input signal subframe length (including D preceeding samples)   */
    const SKP_int   nb_subfr,           /* I    number of subframes stacked in x                                */
    const SKP_float WhiteNoiseFrac,     /* I    fraction added to zero-lag autocorrelation                      */
    const SKP_int   D                   /* I    order                                                           */
)
{
    SKP_int         k, n, s;
    double          C0, num, nrg_f, nrg_b, rc, Atmp, tmp1, tmp2;
    const SKP_float *x_ptr;
    double          C_first_row[ SKP_Silk_MAX_ORDER_LPC ], C_last_row[ SKP_Silk_MAX_ORDER_LPC ];
    double          CAf[ SKP_Silk_MAX_ORDER_LPC + 1 ], CAb[ SKP_Silk_MAX_ORDER_LPC + 1 ];
    double          Af[ SKP_Silk_MAX_ORDER_LPC ];

    SKP_assert( subfr_length * nb_subfr <= MAX_FRAME_SIZE );
    SKP_assert( nb_subfr <= MAX_NB_SUBFR );

    /* Compute autocorrelations, added over subframes */
    C0 = SKP_Silk_energy_FLP( x, nb_subfr * subfr_length );
    SKP_memset( C_first_row, 0, SKP_Silk_MAX_ORDER_LPC * sizeof( double ) );
    for( s = 0; s < nb_subfr; s++ ) {
        x_ptr = x + s * subfr_length;
        for( n = 1; n < D + 1; n++ ) {
            C_first_row[ n - 1 ] += SKP_Silk_inner_product_FLP( x_ptr, x_ptr + n, subfr_length - n );
        }
    }
    SKP_memcpy( C_last_row, C_first_row, SKP_Silk_MAX_ORDER_LPC * sizeof( double ) );

    /* Initialize */
    CAb[ 0 ] = CAf[ 0 ] = C0 + WhiteNoiseFrac * C0 + 1e-9f;

    for( n = 0; n < D; n++ ) {
        /* Update first row of correlation matrix (without first element) */
        /* Update last row of correlation matrix (without last element, stored in reversed order) */
        /* Update C * Af */
        /* Update C * flipud(Af) (stored in reversed order) */
        for( s = 0; s < nb_subfr; s++ ) {
            x_ptr = x + s * subfr_length;
            tmp1 = x_ptr[ n ];
            tmp2 = x_ptr[ subfr_length - n - 1 ];
            for( k = 0; k < n; k++ ) {
                C_first_row[ k ] -= x_ptr[ n ] * x_ptr[ n - k - 1 ];
                C_last_row[ k ]  -= x_ptr[ subfr_length - n - 1 ] * x_ptr[ subfr_length - n + k ];
                Atmp = Af[ k ];
                tmp1 += x_ptr[ n - k - 1 ] * Atmp;
                tmp2 += x_ptr[ subfr_length - n + k ] * Atmp;
            }
            for( k = 0; k <= n; k++ ) {
                CAf[ k ] -= tmp1 * x_ptr[ n - k ];
                CAb[ k ] -= tmp2 * x_ptr[ subfr_length - n + k - 1 ];
            }
        }
        tmp1 = C_first_row[ n ];
        tmp2 = C_last_row[ n ];
        for( k = 0; k < n; k++ ) {
            Atmp = Af[ k ];
            tmp1 += C_last_row[ n - k - 1 ]  * Atmp;
            tmp2 += C_first_row[ n - k - 1 ] * Atmp;
        }
        CAf[ n + 1 ] = tmp1;
        CAb[ n + 1 ] = tmp2;

        /* Calculate nominator and denominator for the next order reflection (parcor) coefficient */
        num = CAb[ n + 1 ];
        nrg_b = CAb[ 0 ];
        nrg_f = CAf[ 0 ];
        for( k = 0; k < n; k++ ) {
            Atmp = Af[ k ];
            num   += CAb[ n - k ] * Atmp;
            nrg_b += CAb[ k + 1 ] * Atmp;
            nrg_f += CAf[ k + 1 ] * Atmp;
        }
        SKP_assert( nrg_f > 0.0 );
        SKP_assert( nrg_b > 0.0 );

        /* Calculate the next order reflection (parcor) coefficient */
        rc = -2.0 * num / ( nrg_f + nrg_b );
        SKP_assert( rc > -1.0 && rc < 1.0 );

        /* Update the AR coefficients */
        for( k = 0; k < (n + 1) >> 1; k++ ) {
            tmp1 = Af[ k ];
            tmp2 = Af[ n - k - 1 ];
            Af[ k ]         = tmp1 + rc * tmp2;
            Af[ n - k - 1 ] = tmp2 + rc * tmp1;
        }
        Af[ n ] = rc;

        /* Update C * Af and C * Ab */
        for( k = 0; k <= n + 1; k++ ) {
            tmp1 = CAf[ k ];
            CAf[ k ]          += rc * CAb[ n - k + 1 ];
            CAb[ n - k + 1  ] += rc * tmp1;
        }
    }

    /* Return residual energy */
    nrg_f = CAf[ 0 ];
    tmp1 = 1.0;
    for( k = 0; k < D; k++ ) {
        Atmp = Af[ k ];
        nrg_f += CAf[ k + 1 ] * Atmp;
        tmp1  += Atmp * Atmp;
        A[ k ] = (SKP_float)(-Atmp);
    }
    nrg_f -= WhiteNoiseFrac * C0 * tmp1;

    return (SKP_float)nrg_f;
}
