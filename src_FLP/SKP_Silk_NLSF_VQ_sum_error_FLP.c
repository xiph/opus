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

#include "SKP_Silk_main_FLP.h"

/* compute weighted quantization errors for LPC_order element input vectors, over one codebook stage */
void SKP_Silk_NLSF_VQ_sum_error_FLP(
          SKP_float                 *err,               /* O    Weighted quantization errors [ N * K ]  */
    const SKP_float                 *in,                /* I    Input vectors [ N * LPC_order ]         */
    const SKP_float                 *w,                 /* I    Weighting vectors [ N * LPC_order ]     */
    const SKP_float                 *pCB,               /* I    Codebook vectors [ K * LPC_order ]      */
    const SKP_int                   N,                  /* I    Number of input vectors                 */
    const SKP_int                   K,                  /* I    Number of codebook vectors              */
    const SKP_int                   LPC_order           /* I    LPC order                               */
)
{
    SKP_int     i, n;
    SKP_float   diff, sum_error;
    SKP_float   Wcpy[ MAX_LPC_ORDER ];
    const SKP_float *cb_vec;

    /* Copy to local stack */
    SKP_memcpy( Wcpy, w, LPC_order * sizeof( SKP_float ) );

    if( LPC_order == 16 ) {
        /* Loop over input vectors */
        for( n = 0; n < N; n++ ) {
            /* Loop over codebook */
            cb_vec = pCB;
            for( i = 0; i < K; i++ ) {
                /* Compute weighted squared quantization error */
                diff = in[ 0 ] - cb_vec[ 0 ];
                sum_error  = Wcpy[ 0 ] * diff * diff;
                diff = in[ 1 ] - cb_vec[ 1 ];
                sum_error += Wcpy[ 1 ] * diff * diff;
                diff = in[ 2 ] - cb_vec[ 2 ];
                sum_error += Wcpy[ 2 ] * diff * diff;
                diff = in[ 3 ] - cb_vec[ 3 ];
                sum_error += Wcpy[ 3 ] * diff * diff;
                diff = in[ 4 ] - cb_vec[ 4 ];
                sum_error += Wcpy[ 4 ] * diff * diff;
                diff = in[ 5 ] - cb_vec[ 5 ];
                sum_error += Wcpy[ 5 ] * diff * diff;
                diff = in[ 6 ] - cb_vec[ 6 ];
                sum_error += Wcpy[ 6 ] * diff * diff;
                diff = in[ 7 ] - cb_vec[ 7 ];
                sum_error += Wcpy[ 7 ] * diff * diff;
                diff = in[ 8 ] - cb_vec[ 8 ];
                sum_error += Wcpy[ 8 ] * diff * diff;
                diff = in[ 9 ] - cb_vec[ 9 ];
                sum_error += Wcpy[ 9 ] * diff * diff;
                diff = in[ 10 ] - cb_vec[ 10 ];
                sum_error += Wcpy[ 10 ] * diff * diff;
                diff = in[ 11 ] - cb_vec[ 11 ];
                sum_error += Wcpy[ 11 ] * diff * diff;
                diff = in[ 12 ] - cb_vec[ 12 ];
                sum_error += Wcpy[ 12 ] * diff * diff;
                diff = in[ 13 ] - cb_vec[ 13 ];
                sum_error += Wcpy[ 13 ] * diff * diff;
                diff = in[ 14 ] - cb_vec[ 14 ];
                sum_error += Wcpy[ 14 ] * diff * diff;
                diff = in[ 15 ] - cb_vec[ 15 ];
                sum_error += Wcpy[ 15 ] * diff * diff;

                err[ i ] = sum_error;
                cb_vec += 16;
            }
            err += K;
            in  += 16;
        }
    } else {
        SKP_assert( LPC_order == 10 );

        /* Loop over input vectors */
        for( n = 0; n < N; n++ ) {
            /* Loop over codebook */
            cb_vec = pCB;
            for( i = 0; i < K; i++ ) {
                /* Compute weighted squared quantization error */
                diff = in[ 0 ] - cb_vec[ 0 ];
                sum_error  = Wcpy[ 0 ] * diff * diff;
                diff = in[ 1 ] - cb_vec[ 1 ];
                sum_error += Wcpy[ 1 ] * diff * diff;
                diff = in[ 2 ] - cb_vec[ 2 ];
                sum_error += Wcpy[ 2 ] * diff * diff;
                diff = in[ 3 ] - cb_vec[ 3 ];
                sum_error += Wcpy[ 3 ] * diff * diff;
                diff = in[ 4 ] - cb_vec[ 4 ];
                sum_error += Wcpy[ 4 ] * diff * diff;
                diff = in[ 5 ] - cb_vec[ 5 ];
                sum_error += Wcpy[ 5 ] * diff * diff;
                diff = in[ 6 ] - cb_vec[ 6 ];
                sum_error += Wcpy[ 6 ] * diff * diff;
                diff = in[ 7 ] - cb_vec[ 7 ];
                sum_error += Wcpy[ 7 ] * diff * diff;
                diff = in[ 8 ] - cb_vec[ 8 ];
                sum_error += Wcpy[ 8 ] * diff * diff;
                diff = in[ 9 ] - cb_vec[ 9 ];
                sum_error += Wcpy[ 9 ] * diff * diff;

                err[ i ] = sum_error;
                cb_vec += 10;
            }
            err += K;
            in  += 10;
        }
    }
}
