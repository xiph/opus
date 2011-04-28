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
#include "SKP_Silk_tuning_parameters.h"

/* Control SNR of redidual quantizer */
SKP_int SKP_Silk_control_SNR(
    SKP_Silk_encoder_state      *psEncC,            /* I/O  Pointer to Silk encoder state               */
    SKP_int32                   TargetRate_bps      /* I    Target max bitrate (bps)                    */
)
{
    SKP_int k, ret = SKP_SILK_NO_ERROR;
    SKP_int32 frac_Q6;
    const SKP_int32 *rateTable;

    /* Set bitrate/coding quality */
    TargetRate_bps = SKP_LIMIT( TargetRate_bps, MIN_TARGET_RATE_BPS, MAX_TARGET_RATE_BPS );
    if( TargetRate_bps != psEncC->TargetRate_bps ) {
        psEncC->TargetRate_bps = TargetRate_bps;

        /* If new TargetRate_bps, translate to SNR_dB value */
        if( psEncC->fs_kHz == 8 ) {
            rateTable = TargetRate_table_NB;
        } else if( psEncC->fs_kHz == 12 ) {
            rateTable = TargetRate_table_MB;
        } else if( psEncC->fs_kHz == 16 ) {
            rateTable = TargetRate_table_WB;
        } else {
            SKP_assert( 0 );
        }

        /* Reduce bitrate for 10 ms modes in these calculations */
        if( psEncC->nb_subfr == 2 ) {
            TargetRate_bps -= REDUCE_BITRATE_10_MS_BPS;
        }

        /* Find bitrate interval in table and interpolate */
        for( k = 1; k < TARGET_RATE_TAB_SZ; k++ ) {
            if( TargetRate_bps <= rateTable[ k ] ) {
                frac_Q6 = SKP_DIV32( SKP_LSHIFT( TargetRate_bps - rateTable[ k - 1 ], 6 ), 
                                                 rateTable[ k ] - rateTable[ k - 1 ] );
                psEncC->SNR_dB_Q7 = SKP_LSHIFT( SNR_table_Q1[ k - 1 ], 6 ) + SKP_MUL( frac_Q6, SNR_table_Q1[ k ] - SNR_table_Q1[ k - 1 ] );
                break;
            }
        }
    }

    return ret;
}
