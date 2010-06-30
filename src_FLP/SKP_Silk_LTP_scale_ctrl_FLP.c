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

#define NB_THRESHOLDS           11
/* Table containing trained thresholds for LTP scaling */
static const SKP_float LTPScaleThresholds[ NB_THRESHOLDS ] = 
{
    0.95f, 0.8f, 0.50f, 0.400f, 0.3f, 0.2f,
    0.15f, 0.1f, 0.08f, 0.075f, 0.0f
};


void SKP_Silk_LTP_scale_ctrl_FLP(
    SKP_Silk_encoder_state_FLP      *psEnc,             /* I/O  Encoder state FLP                       */
    SKP_Silk_encoder_control_FLP    *psEncCtrl          /* I/O  Encoder control FLP                     */
)
{
    SKP_int round_loss, frames_per_packet;
    SKP_float g_out, g_limit, thrld1, thrld2;

    /* 1st order high-pass filter */
    //g_HP(n) = g(n) - g(n-1) + 0.5 * g_HP(n-1);       // tune the 0.5: higher means longer impact of jump
    psEnc->HPLTPredCodGain = SKP_max_float( psEncCtrl->LTPredCodGain - psEnc->prevLTPredCodGain, 0.0f ) 
                            + 0.5f * psEnc->HPLTPredCodGain;
    
    psEnc->prevLTPredCodGain = psEncCtrl->LTPredCodGain;

    /* combine input and filtered input */
    g_out = 0.5f * psEncCtrl->LTPredCodGain + ( 1.0f - 0.5f ) * psEnc->HPLTPredCodGain;
    g_limit = SKP_sigmoid( 0.5f * ( g_out - 6 ) );
    
    
    /* Default is minimum scaling */
    psEncCtrl->sCmn.LTP_scaleIndex = 0;

    /* Round the loss measure to whole pct */
    round_loss = ( SKP_int )( psEnc->sCmn.PacketLoss_perc );
    round_loss = SKP_max( 0, round_loss );

    /* Only scale if first frame in packet 0% */
    if( psEnc->sCmn.nFramesInPayloadBuf == 0 ){
        
        frames_per_packet = psEnc->sCmn.PacketSize_ms / ( SUB_FRAME_LENGTH_MS * psEnc->sCmn.nb_subfr );

        round_loss += ( frames_per_packet - 1 );
        thrld1 = LTPScaleThresholds[ SKP_min_int( round_loss,     NB_THRESHOLDS - 1 ) ];
        thrld2 = LTPScaleThresholds[ SKP_min_int( round_loss + 1, NB_THRESHOLDS - 1 ) ];
    
        if( g_limit > thrld1 ) {
            /* High Scaling */
            psEncCtrl->sCmn.LTP_scaleIndex = 2;
        } else if( g_limit > thrld2 ) {
            /* Middle Scaling */
            psEncCtrl->sCmn.LTP_scaleIndex = 1;
        }
    }
    psEncCtrl->LTP_scale = ( SKP_float)SKP_Silk_LTPScales_table_Q14[ psEncCtrl->sCmn.LTP_scaleIndex ] / 16384.0f;
}
