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

#include "SKP_Silk_main_FLP.h"

/* compute weighted quantization errors for LPC_order element input vectors, over one codebook stage */
void SKP_Silk_NLSF_VQ_sum_error_FLP(
          SKP_float                 *err,               /* O    Weighted quantization errors [ N * K ]  */
    const SKP_float                 *in_NLSF_Q8,        /* I    Input vectors [ N * LPC_order ]         */
    const SKP_float                 *w,                 /* I    Weighting vectors [ N * LPC_order ]     */
    const SKP_int8                  *pCB_NLSF_Q8,       /* I    Codebook vectors [ K * LPC_order ]      */
    const SKP_int                   N,                  /* I    Number of input vectors                 */
    const SKP_int                   K,                  /* I    Number of codebook vectors              */
    const SKP_int                   LPC_order           /* I    LPC order                               */
)
{
    SKP_int        i, n;
    SKP_float      diff_Q8, sum_error_Q16;
    SKP_float      Wcpy[ MAX_LPC_ORDER ];
    const SKP_int8 *cb_vec_NLSF_Q8;

    /* Copy to local stack */
    SKP_memcpy( Wcpy, w, LPC_order * sizeof( SKP_float ) );

    if( LPC_order == 16 ) {
        /* Loop over input vectors */
        for( n = 0; n < N; n++ ) {
            /* Loop over codebook */
            cb_vec_NLSF_Q8 = pCB_NLSF_Q8;
            for( i = 0; i < K; i++ ) {
                /* Compute weighted squared quantization error */
                diff_Q8 = in_NLSF_Q8[ 0 ] - ( SKP_float )cb_vec_NLSF_Q8[ 0 ];
                sum_error_Q16  = Wcpy[ 0 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 1 ] - ( SKP_float )cb_vec_NLSF_Q8[ 1 ];
                sum_error_Q16 += Wcpy[ 1 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 2 ] - ( SKP_float )cb_vec_NLSF_Q8[ 2 ];
                sum_error_Q16 += Wcpy[ 2 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 3 ] - ( SKP_float )cb_vec_NLSF_Q8[ 3 ];
                sum_error_Q16 += Wcpy[ 3 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 4 ] - ( SKP_float )cb_vec_NLSF_Q8[ 4 ];
                sum_error_Q16 += Wcpy[ 4 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 5 ] - ( SKP_float )cb_vec_NLSF_Q8[ 5 ];
                sum_error_Q16 += Wcpy[ 5 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 6 ] - ( SKP_float )cb_vec_NLSF_Q8[ 6 ];
                sum_error_Q16 += Wcpy[ 6 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 7 ] - ( SKP_float )cb_vec_NLSF_Q8[ 7 ];
                sum_error_Q16 += Wcpy[ 7 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 8 ] - ( SKP_float )cb_vec_NLSF_Q8[ 8 ];
                sum_error_Q16 += Wcpy[ 8 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 9 ] - ( SKP_float )cb_vec_NLSF_Q8[ 9 ];
                sum_error_Q16 += Wcpy[ 9 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 10 ] - ( SKP_float )cb_vec_NLSF_Q8[ 10 ];
                sum_error_Q16 += Wcpy[ 10 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 11 ] - ( SKP_float )cb_vec_NLSF_Q8[ 11 ];
                sum_error_Q16 += Wcpy[ 11 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 12 ] - ( SKP_float )cb_vec_NLSF_Q8[ 12 ];
                sum_error_Q16 += Wcpy[ 12 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 13 ] - ( SKP_float )cb_vec_NLSF_Q8[ 13 ];
                sum_error_Q16 += Wcpy[ 13 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 14 ] - ( SKP_float )cb_vec_NLSF_Q8[ 14 ];
                sum_error_Q16 += Wcpy[ 14 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 15 ] - ( SKP_float )cb_vec_NLSF_Q8[ 15 ];
                sum_error_Q16 += Wcpy[ 15 ] * diff_Q8 * diff_Q8;

                err[ i ] = ( 1.0f / 65536.0f ) * sum_error_Q16;
                cb_vec_NLSF_Q8 += 16;
            }
            err        += K;
            in_NLSF_Q8 += 16;
        }
    } else {
        SKP_assert( LPC_order == 10 );

        /* Loop over input vectors */
        for( n = 0; n < N; n++ ) {
            /* Loop over codebook */
            cb_vec_NLSF_Q8 = pCB_NLSF_Q8;
            for( i = 0; i < K; i++ ) {
                /* Compute weighted squared quantization error */
                diff_Q8 = in_NLSF_Q8[ 0 ] - ( SKP_float )cb_vec_NLSF_Q8[ 0 ];
                sum_error_Q16  = Wcpy[ 0 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 1 ] - ( SKP_float )cb_vec_NLSF_Q8[ 1 ];
                sum_error_Q16 += Wcpy[ 1 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 2 ] - ( SKP_float )cb_vec_NLSF_Q8[ 2 ];
                sum_error_Q16 += Wcpy[ 2 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 3 ] - ( SKP_float )cb_vec_NLSF_Q8[ 3 ];
                sum_error_Q16 += Wcpy[ 3 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 4 ] - ( SKP_float )cb_vec_NLSF_Q8[ 4 ];
                sum_error_Q16 += Wcpy[ 4 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 5 ] - ( SKP_float )cb_vec_NLSF_Q8[ 5 ];
                sum_error_Q16 += Wcpy[ 5 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 6 ] - ( SKP_float )cb_vec_NLSF_Q8[ 6 ];
                sum_error_Q16 += Wcpy[ 6 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 7 ] - ( SKP_float )cb_vec_NLSF_Q8[ 7 ];
                sum_error_Q16 += Wcpy[ 7 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 8 ] - ( SKP_float )cb_vec_NLSF_Q8[ 8 ];
                sum_error_Q16 += Wcpy[ 8 ] * diff_Q8 * diff_Q8;
                diff_Q8 = in_NLSF_Q8[ 9 ] - ( SKP_float )cb_vec_NLSF_Q8[ 9 ];
                sum_error_Q16 += Wcpy[ 9 ] * diff_Q8 * diff_Q8;

                err[ i ] = ( 1.0f / 65536.0f ) * sum_error_Q16;
                cb_vec_NLSF_Q8 += 10;
            }
            err        += K;
            in_NLSF_Q8 += 10;
        }
    }
}
