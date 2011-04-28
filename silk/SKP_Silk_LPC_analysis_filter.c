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

#include "SKP_Silk_SigProc_FIX.h"

/*******************************************/
/* LPC analysis filter                     */
/* NB! State is kept internally and the    */
/* filter always starts with zero state    */
/* first Order output samples are not set  */
/*******************************************/

void SKP_Silk_LPC_analysis_filter(
    SKP_int16            *out,           /* O:   Output signal                               */
    const SKP_int16      *in,            /* I:   Input signal                                */
    const SKP_int16      *B,             /* I:   MA prediction coefficients, Q12 [order]     */
    const SKP_int32      len,            /* I:   Signal length                               */
    const SKP_int32      d               /* I:   Filter order                                */
)
{
    SKP_int         ix, j;
    SKP_int32       out32_Q12, out32;
    const SKP_int16 *in_ptr;

    SKP_assert( d >= 6 );
    SKP_assert( (d & 1) == 0 );
    SKP_assert( d <= len );

    for ( ix = d; ix < len; ix++) {
        in_ptr = &in[ ix - 1 ];

        out32_Q12 = SKP_SMULBB(            in_ptr[  0 ], B[ 0 ] );
        out32_Q12 = SKP_SMLABB( out32_Q12, in_ptr[ -1 ], B[ 1 ] );
        out32_Q12 = SKP_SMLABB( out32_Q12, in_ptr[ -2 ], B[ 2 ] );
        out32_Q12 = SKP_SMLABB( out32_Q12, in_ptr[ -3 ], B[ 3 ] );
        out32_Q12 = SKP_SMLABB( out32_Q12, in_ptr[ -4 ], B[ 4 ] );
        out32_Q12 = SKP_SMLABB( out32_Q12, in_ptr[ -5 ], B[ 5 ] );
        for( j = 6; j < d; j += 2 ) {
            out32_Q12 = SKP_SMLABB( out32_Q12, in_ptr[ -j     ], B[ j     ] );
            out32_Q12 = SKP_SMLABB( out32_Q12, in_ptr[ -j - 1 ], B[ j + 1 ] );
        }

        /* Subtract prediction */
        out32_Q12 = SKP_SUB32( SKP_LSHIFT( (SKP_int32)in_ptr[ 1 ], 12 ), out32_Q12 );

        /* Scale to Q0 */
        out32 = SKP_RSHIFT_ROUND( out32_Q12, 12 );

        /* Saturate output */
        out[ ix ] = ( SKP_int16 )SKP_SAT16( out32 );
    }

    /* Set first LPC d samples to zero instead of undefined */
    SKP_memset( out, 0, d * sizeof( SKP_int16 ) );
}
