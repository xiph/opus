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

#include "SKP_Silk_main.h"

/* Find least-squares prediction gain for one signal based on another and quantize it */
SKP_int32 SKP_Silk_stereo_find_predictor(               /* O    Returns predictor in Q13                    */
    const SKP_int16     x[],                            /* I    Basis signal                                */
    const SKP_int16     y[],                            /* I    Target signal                               */
    SKP_int             length                          /* I    Number of samples                           */
)
{
    SKP_int   scale, scale1, scale2;
    SKP_int32 nrg1, nrg2, corr, pred_Q13;

    /* Find  predictor */
    SKP_Silk_sum_sqr_shift( &nrg1, &scale1, x, length );
    SKP_Silk_sum_sqr_shift( &nrg2, &scale2, y, length );
    if( scale1 > scale2 ) {
        scale = scale1;
    } else {
        scale = scale2;
        nrg1 = SKP_RSHIFT32( nrg1, scale2 - scale1 );
    }
    corr = SKP_Silk_inner_prod_aligned_scale( x, y, scale, length );
    pred_Q13 = SKP_DIV32_varQ( corr, SKP_max( nrg1, 1 ), 13 );
    pred_Q13 = SKP_LIMIT( pred_Q13, -SKP_FIX_CONST( 10, 13 ), SKP_FIX_CONST( 10, 13 ) );

    return pred_Q13;
}
