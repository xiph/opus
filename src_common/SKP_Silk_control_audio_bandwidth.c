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

/* Control internal sampling rate */
SKP_int SKP_Silk_control_audio_bandwidth(
    SKP_Silk_encoder_state      *psEncC,            /* I/O  Pointer to Silk encoder state               */
    SKP_int32                   TargetRate_bps      /* I    Target max bitrate (bps)                    */
)
{
    SKP_int fs_kHz;

    fs_kHz = psEncC->fs_kHz;

    /* Reduce bitrate for 10 ms modes in these calculations */
    if( psEncC->nb_subfr == 2 ) {
        TargetRate_bps -= REDUCE_BITRATE_10_MS_BPS;
    }

    if( fs_kHz == 0 ) {
        /* Encoder has just been initialized */
        if( TargetRate_bps >= WB2MB_BITRATE_BPS ) {
            fs_kHz = 16;
        } else if( TargetRate_bps >= MB2NB_BITRATE_BPS ) {
            fs_kHz = 12;
        } else {
            fs_kHz = 8;
        }
        /* Make sure internal rate is not higher than external rate or maximum allowed, or lower than minimum allowed */
        fs_kHz = SKP_min( fs_kHz, SKP_DIV32_16( psEncC->API_fs_Hz, 1000 ) );
        fs_kHz = SKP_min( fs_kHz, psEncC->maxInternal_fs_kHz );
        fs_kHz = SKP_max( fs_kHz, psEncC->minInternal_fs_kHz );
    } else if( SKP_SMULBB( fs_kHz, 1000 ) > psEncC->API_fs_Hz || fs_kHz > psEncC->maxInternal_fs_kHz || fs_kHz < psEncC->minInternal_fs_kHz ) {
        /* Make sure internal rate is not higher than external rate or maximum allowed, or lower than minimum allowed */
        fs_kHz = SKP_DIV32_16( psEncC->API_fs_Hz, 1000 );
        fs_kHz = SKP_min( fs_kHz, psEncC->maxInternal_fs_kHz );
        fs_kHz = SKP_max( fs_kHz, psEncC->minInternal_fs_kHz );
    } else {
        /* State machine for the internal sampling rate switching */
        if( psEncC->API_fs_Hz > 8000 && psEncC->prevSignalType == TYPE_NO_VOICE_ACTIVITY ) { /* Low speech activity */
            /* Check if we should switch down */
            if( ( psEncC->fs_kHz == 12 && TargetRate_bps < MB2NB_BITRATE_BPS && psEncC->minInternal_fs_kHz <=  8 ) ||
                ( psEncC->fs_kHz == 16 && TargetRate_bps < WB2MB_BITRATE_BPS && psEncC->minInternal_fs_kHz <= 12 ) ) 
            {
                /* Switch down */
                if( SWITCH_TRANSITION_FILTERING && psEncC->sLP.mode == 0 ) {
                    /* New transition */
                    psEncC->sLP.transition_frame_no = TRANSITION_FRAMES;

                    /* Reset transition filter state */
                    SKP_memset( psEncC->sLP.In_LP_State, 0, sizeof( psEncC->sLP.In_LP_State ) );
                } 
                if( psEncC->sLP.transition_frame_no <= 0 ) {
                    /* Stop transition phase */
                    psEncC->sLP.mode = 0;

                    /* Switch to a lower sample frequency */
                    fs_kHz = psEncC->fs_kHz == 16 ? 12 : 8;
                } else {
                    /* Direction: down (at double speed) */
                    psEncC->sLP.mode = -2;
                } 
            } 
            else
            if( ( psEncC->fs_kHz ==  8 && TargetRate_bps > NB2MB_BITRATE_BPS && psEncC->maxInternal_fs_kHz >= 12 && psEncC->API_fs_Hz >= 12000 ) ||
                ( psEncC->fs_kHz == 12 && TargetRate_bps > MB2WB_BITRATE_BPS && psEncC->maxInternal_fs_kHz >= 16 && psEncC->API_fs_Hz >= 16000 ) ) 
            {
                /* Switch up */
                if( SWITCH_TRANSITION_FILTERING && psEncC->sLP.mode == 0 ) {
                    /* Switch to a higher sample frequency */
                    fs_kHz = psEncC->fs_kHz == 8 ? 12 : 16;

                    /* New transition */
                    psEncC->sLP.transition_frame_no = 0;
                } 
                if( psEncC->sLP.transition_frame_no >= TRANSITION_FRAMES ) {
                    /* Stop transition phase */
                    psEncC->sLP.mode = 0;
                } else {
                    /* Direction: up */
                    psEncC->sLP.mode = 1;
                }
            }
        }
    }

#ifdef FORCE_INTERNAL_FS_KHZ
    fs_kHz = FORCE_INTERNAL_FS_KHZ;
#endif

    return fs_kHz;
}
