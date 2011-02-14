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

/*******************************************/
/* Encode LBRR side info and excitation    */
/*******************************************/
void SKP_Silk_LBRR_embed(
    SKP_Silk_encoder_state      *psEncC,            /* I/O  Encoder state                               */
    ec_enc                      *psRangeEnc         /* I/O  Compressor data structure                   */
)
{
    SKP_int   i;
    SKP_int32 LBRR_symbol;

    /* Encode LBRR flags */
    LBRR_symbol = 0;
    for( i = 0; i < psEncC->nFramesPerPacket; i++ ) {
        LBRR_symbol |= SKP_LSHIFT( psEncC->LBRR_flags[ i ], i );
    }
    psEncC->LBRR_flag = LBRR_symbol > 0 ? 1 : 0;
    if( LBRR_symbol && psEncC->nFramesPerPacket > 1 ) {
        ec_enc_icdf( psRangeEnc, LBRR_symbol - 1, SKP_Silk_LBRR_flags_iCDF_ptr[ psEncC->nFramesPerPacket - 2 ], 8 );
    }

    /* Code indices and excitation signals */
    for( i = 0; i < psEncC->nFramesPerPacket; i++ ) {
        if( psEncC->LBRR_flags[ i ] ) {
            SKP_Silk_encode_indices( psEncC, psRangeEnc, i, 1 );
            SKP_Silk_encode_pulses( psRangeEnc, psEncC->indices_LBRR[i].signalType, 
                psEncC->indices_LBRR[i].quantOffsetType, psEncC->pulses_LBRR[ i ], psEncC->frame_length );
        }
    }
}
