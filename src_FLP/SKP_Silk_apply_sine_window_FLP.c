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

/* Apply sine window to signal vector.                                                                  */
/* Window types:                                                                                        */
/*  0 -> sine window from 0 to pi                                                                       */
/*  1 -> sine window from 0 to pi/2                                                                     */
/*  2 -> sine window from pi/2 to pi                                                                    */
void SKP_Silk_apply_sine_window_FLP(
          SKP_float                 px_win[],           /* O    Pointer to windowed signal              */
    const SKP_float                 px[],               /* I    Pointer to input signal                 */
    const SKP_int                   win_type,           /* I    Selects a window type                   */
    const SKP_int                   length              /* I    Window length, multiple of 4            */
)
{
    SKP_int   k;
    SKP_float freq, c, S0, S1;

    /* Length must be multiple of 4 */
    SKP_assert( ( length & 3 ) == 0 );

    freq = PI / ( length + 1 );
    if( win_type == 0 ) {
        freq = 2.0f * freq;
    }

    /* Approximation of 2 * cos(f) */
    c = 2.0f - freq * freq;

    /* Initialize state */
    if( win_type < 2 ) {
        /* Start from 0 */
        S0 = 0.0f;
        /* Approximation of sin(f) */
        S1 = freq;
    } else {
        /* Start from 1 */
        S0 = 1.0f;
        /* Approximation of cos(f) */
        S1 = 0.5f * c;
    }

    /* Uses the recursive equation:   sin(n*f) = 2 * cos(f) * sin((n-1)*f) - sin((n-2)*f)   */
    /* 4 samples at a time */
    for( k = 0; k < length; k += 4 ) {
        px_win[ k + 0 ] = px[ k + 0 ] * 0.5f * ( S0 + S1 );
        px_win[ k + 1 ] = px[ k + 1 ] * S1;
        S0 = c * S1 - S0;
        px_win[ k + 2 ] = px[ k + 2 ] * 0.5f * ( S1 + S0 );
        px_win[ k + 3 ] = px[ k + 3 ] * S0;
        S1 = c * S0 - S1;
    }
}
