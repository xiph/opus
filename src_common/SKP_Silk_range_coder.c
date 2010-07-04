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

#include "SKP_Silk_main.h"

/* Range decoder for one symbol */
void SKP_Silk_range_decoder(
    SKP_int                         data[],             /* O    uncompressed data                           */
    ec_dec                          *psRangeDec,        /* I/O  Compressor data structure                   */
    const SKP_uint16                prob[],             /* I    cumulative density function                 */
    SKP_int                         probIx              /* I    initial (middle) entry of cdf               */
)
{
    SKP_uint32 low_Q16, high_Q16;

    SKP_uint32 low_Q16_returned;
    SKP_int    temp;

    if( prob[ 2 ] == 65535 ) {
        /* Instead of detection, we could add a separate function and call when we know that output is a bit */
        *data = ec_dec_bit_prob( psRangeDec, 65536 - prob[ 1 ] );
    } else {
        low_Q16_returned = ec_decode_bin( psRangeDec, 16 );

#if 1
        temp = 0;
        while( low_Q16_returned >= prob[ ++temp ] ) {}
        *data = temp - 1;
#else
        temp = probIx;
        if( low_Q16_returned >= prob[ temp ] ){
            while( low_Q16_returned >= prob[ temp ] ) {
                temp++;
            }
            temp = temp - 1;
        } else {
            /* search down */
            while( low_Q16_returned < prob[ temp ] ) {
                temp--;
            }
        }
        *data = temp;
#endif

        low_Q16  = prob[ *data ];
        high_Q16 = prob[ *data + 1 ];

#ifdef SAVE_ALL_INTERNAL_DATA
        DEBUG_STORE_DATA( dec_lr.dat, &low_Q16_returned,  sizeof( SKP_uint32 ) );
        DEBUG_STORE_DATA( dec_l.dat,  &low_Q16,           sizeof( SKP_uint32 ) );
        DEBUG_STORE_DATA( dec_h.dat,  &high_Q16,          sizeof( SKP_uint32 ) );
#endif  
        ec_dec_update( psRangeDec, low_Q16, high_Q16,( 1 << 16 ) );
    }
#ifdef SAVE_ALL_INTERNAL_DATA
    DEBUG_STORE_DATA( dec.dat, data, sizeof( SKP_int ) );
#endif
}
