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

#include "SKP_Silk_SigProc_FLP.h"

/* 
R. Laroia, N. Phamdo and N. Farvardin, "Robust and Efficient Quantization of Speech LSP
Parameters Using Structured Vector Quantization", Proc. IEEE Int. Conf. Acoust., Speech,
Signal Processing, pp. 641-644, 1991.
*/

#define MIN_NDELTA                  1e-4f

/* Laroia low complexity NLSF weights */
void SKP_Silk_NLSF_VQ_weights_laroia_FLP( 
          SKP_float     *pXW,           /* 0: Pointer to input vector weights           [D x 1] */
    const SKP_float     *pX,            /* I: Pointer to input vector                   [D x 1] */ 
    const SKP_int        D              /* I: Input vector dimension                            */
)
{
    SKP_int   k;
    SKP_float tmp1, tmp2;
    
    /* Safety checks */
    SKP_assert( D > 0 );
    SKP_assert( ( D & 1 ) == 0 );
    
    /* First value */
    tmp1 = 1.0f / SKP_max_float( pX[ 0 ],           MIN_NDELTA );
    tmp2 = 1.0f / SKP_max_float( pX[ 1 ] - pX[ 0 ], MIN_NDELTA );
    pXW[ 0 ] = tmp1 + tmp2;
    
    /* Main loop */
    for( k = 1; k < D - 1; k += 2 ) {
        tmp1 = 1.0f / SKP_max_float( pX[ k + 1 ] - pX[ k ], MIN_NDELTA );
        pXW[ k ] = tmp1 + tmp2;

        tmp2 = 1.0f / SKP_max_float( pX[ k + 2 ] - pX[ k + 1 ], MIN_NDELTA );
        pXW[ k + 1 ] = tmp1 + tmp2;
    }
    
    /* Last value */
    tmp1 = 1.0f / SKP_max_float( 1.0f - pX[ D - 1 ], MIN_NDELTA );
    pXW[ D - 1 ] = tmp1 + tmp2;
}
