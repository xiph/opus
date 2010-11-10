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
#include "SKP_Silk_PLC.h"

/****************/
/* Decode frame */
/****************/
SKP_int SKP_Silk_decode_frame(
    SKP_Silk_decoder_state      *psDec,             /* I/O  Pointer to Silk decoder state               */
    ec_dec                      *psRangeDec,        /* I/O  Compressor data structure                   */
    SKP_int16                   pOut[],             /* O    Pointer to output speech frame              */
    SKP_int32                   *pN,                /* O    Pointer to size of output frame             */
    const SKP_int               nBytes,             /* I    Payload length                              */
    SKP_int                     action,             /* I    Action from Jitter Buffer                   */
    SKP_int                     *decBytes           /* O    Used bytes to decode this frame             */
)
{
    SKP_Silk_decoder_control sDecCtrl;
    SKP_int         L, fs_Khz_old, nb_subfr_old, mv_len, ret = 0;
    SKP_int         Pulses[ MAX_FRAME_LENGTH ];

TIC(decode_frame)

    L = psDec->frame_length;
    sDecCtrl.LTP_scale_Q14 = 0;
    
    /* Safety checks */
    SKP_assert( L > 0 && L <= MAX_FRAME_LENGTH );

    /********************************************/
    /* Decode Frame if packet is not lost  */
    /********************************************/
    *decBytes = 0;
    if( action == 0 ) {
        /********************************************/
        /* Initialize arithmetic coder              */
        /********************************************/
        fs_Khz_old    = psDec->fs_kHz;
        nb_subfr_old  = psDec->nb_subfr;
        if( psDec->nFramesDecoded == 0 ) {
            SKP_Silk_decode_indices( psDec, psRangeDec );
        }

        /********************************************/
        /* Decode parameters and pulse signal       */
        /********************************************/
TIC(decode_params)
        SKP_Silk_decode_parameters( psDec, &sDecCtrl, psRangeDec, Pulses, 1 );
TOC(decode_params)

        if( 0 ) { //psDec->sRC.error ) {
            psDec->nBytesLeft = 0;

            action = 1;                         /* PLC operation */
            psDec->nb_subfr = nb_subfr_old;
            SKP_Silk_decoder_set_fs( psDec, fs_Khz_old );

            /* Avoid crashing */
            *decBytes = psRangeDec->buf->storage; 
            /*
            if( psDec->sRC.error == RANGE_CODER_DEC_PAYLOAD_TOO_LONG ) {
                ret = SKP_SILK_DEC_PAYLOAD_TOO_LARGE;
            } else {
                ret = SKP_SILK_DEC_PAYLOAD_ERROR;
            }
            */
        } else {
            *decBytes = psRangeDec->buf->storage - psDec->nBytesLeft;
            psDec->nFramesDecoded++;
        
            /* Update lengths. Sampling frequency could have changed */
            L = psDec->frame_length;

            /********************************************************/
            /* Run inverse NSQ                                      */
            /********************************************************/
TIC(decode_core)
            SKP_Silk_decode_core( psDec, &sDecCtrl, pOut, Pulses );
TOC(decode_core)

            /********************************************************/
            /* Update PLC state                                     */
            /********************************************************/
            SKP_Silk_PLC( psDec, &sDecCtrl, pOut, L, action );

            psDec->lossCnt = 0;
            psDec->prev_sigtype = sDecCtrl.sigtype;

            /* A frame has been decoded without errors */
            psDec->first_frame_after_reset = 0;
        }
    }
    /*************************************************************/
    /* Generate Concealment frame if packet is lost, or corrupt  */
    /*************************************************************/
    if( action == 1 ) {
        /* Handle packet loss by extrapolation */
        SKP_Silk_PLC( psDec, &sDecCtrl, pOut, L, action );
    }

    /*************************/
    /* Update output buffer. */
    /*************************/
    SKP_assert( psDec->ltp_mem_length >= psDec->frame_length );
    mv_len = psDec->ltp_mem_length - psDec->frame_length;
    SKP_memmove( psDec->outBuf, &psDec->outBuf[ psDec->ltp_mem_length - mv_len ], 
        mv_len * sizeof(SKP_int16) );
    SKP_memcpy( &psDec->outBuf[ mv_len ], pOut, 
        psDec->frame_length * sizeof( SKP_int16 ) );

    /****************************************************************/
    /* Ensure smooth connection of extrapolated and good frames     */
    /****************************************************************/
    SKP_Silk_PLC_glue_frames( psDec, &sDecCtrl, pOut, L );

    /************************************************/
    /* Comfort noise generation / estimation        */
    /************************************************/
    SKP_Silk_CNG( psDec, &sDecCtrl, pOut , L );

    /********************************************/
    /* HP filter output                            */
    /********************************************/
    SKP_assert( ( ( psDec->fs_kHz == 12 ) && ( L % 3 ) == 0 ) || 
                ( ( psDec->fs_kHz != 12 ) && ( L % 2 ) == 0 ) );
TIC(HP_out)
    SKP_Silk_biquad( pOut, psDec->HP_B, psDec->HP_A, psDec->HPState, pOut, L );
TOC(HP_out)

    /********************************************/
    /* set output frame length                    */
    /********************************************/
    *pN = ( SKP_int16 )L;

    /* Update some decoder state variables */
    psDec->lagPrev = sDecCtrl.pitchL[ MAX_NB_SUBFR - 1 ];

TOC(decode_frame)

    return ret;
}
