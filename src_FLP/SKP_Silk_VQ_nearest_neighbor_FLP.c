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

/* entropy constrained MATRIX-weighted VQ, for a single input data vector */
void SKP_Silk_VQ_WMat_EC_FLP(
          SKP_int                   *ind,               /* O    Index of best codebook vector           */
          SKP_float                 *rate_dist,         /* O    Best weighted quant. error + mu * rate  */
    const SKP_float                 *in,                /* I    Input vector to be quantized            */
    const SKP_float                 *W,                 /* I    Weighting matrix                        */
    const SKP_int16                 *cb,                /* I    Codebook                                */
    const SKP_int16                 *cl_Q6,             /* I    Code length for each codebook vector    */
    const SKP_float                 mu,                 /* I    Tradeoff between WSSE and rate          */
    const SKP_int                   L                   /* I    Number of vectors in codebook           */
)
{
    SKP_int   k;
    SKP_float sum1;
    SKP_float diff[ 5 ];
    const SKP_int16 *cb_row;

    /* Loop over codebook */
    *rate_dist = SKP_float_MAX;
    cb_row = cb;
    for( k = 0; k < L; k++ ) {
        /* Calc difference between in vector and cbk vector */
        diff[ 0 ] = in[ 0 ] - ( SKP_float )cb_row[ 0 ] * Q14_CONVERSION_FAC;
        diff[ 1 ] = in[ 1 ] - ( SKP_float )cb_row[ 1 ] * Q14_CONVERSION_FAC;
        diff[ 2 ] = in[ 2 ] - ( SKP_float )cb_row[ 2 ] * Q14_CONVERSION_FAC;
        diff[ 3 ] = in[ 3 ] - ( SKP_float )cb_row[ 3 ] * Q14_CONVERSION_FAC;
        diff[ 4 ] = in[ 4 ] - ( SKP_float )cb_row[ 4 ] * Q14_CONVERSION_FAC;

        /* Weighted rate */
        sum1 = mu * cl_Q6[ k ] / 64.0f;

        /* Add weighted quantization error, assuming W is symmetric */
        /* first row of W */
        sum1 += diff[ 0 ] * ( W[ 0 ] * diff[ 0 ] + 
                     2.0f * ( W[ 1 ] * diff[ 1 ] + 
                              W[ 2 ] * diff[ 2 ] + 
                              W[ 3 ] * diff[ 3 ] + 
                              W[ 4 ] * diff[ 4 ] ) );

        /* second row of W */
        sum1 += diff[ 1 ] * ( W[ 6 ] * diff[ 1 ] + 
                     2.0f * ( W[ 7 ] * diff[ 2 ] + 
                              W[ 8 ] * diff[ 3 ] + 
                              W[ 9 ] * diff[ 4 ] ) );

        /* third row of W */
        sum1 += diff[ 2 ] * ( W[ 12 ] * diff[ 2 ] + 
                    2.0f *  ( W[ 13 ] * diff[ 3 ] + 
                              W[ 14 ] * diff[ 4 ] ) );

        /* fourth row of W */
        sum1 += diff[ 3 ] * ( W[ 18 ] * diff[ 3 ] + 
                     2.0f * ( W[ 19 ] * diff[ 4 ] ) );

        /* last row of W */
        sum1 += diff[ 4 ] * ( W[ 24 ] * diff[ 4 ] );

        /* find best */
        if( sum1 < *rate_dist ) {
            *rate_dist = sum1;
            *ind = k;
        }

        /* Go to next cbk vector */
        cb_row += LTP_ORDER;
    }
}
