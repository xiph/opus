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
#include "SKP_Silk_PLC.h"

#define DECODE_NORMAL       0
#define PACKET_LOST         1
#define DECODE_LBRR         2

/****************/
/* Decode frame */
/****************/
SKP_int SKP_Silk_decode_frame(
    SKP_Silk_decoder_state      *psDec,             /* I/O  Pointer to Silk decoder state               */
    ec_dec                      *psRangeDec,        /* I/O  Compressor data structure                   */
    SKP_int16                   pOut[],             /* O    Pointer to output speech frame              */
    SKP_int32                   *pN,                /* O    Pointer to size of output frame             */
    const SKP_int               nBytes,             /* I    Payload length                              */
    SKP_int                     lostFlag            /* I    0: no loss, 1 loss, 2 decode fec            */
)
{
    SKP_Silk_decoder_control sDecCtrl;
    SKP_int         i, L, mv_len, ret = 0;
    SKP_int8        flags;
    SKP_int32       LBRR_symbol;
    SKP_int         pulses[ MAX_FRAME_LENGTH ];

TIC(DECODE_FRAME)

    L = psDec->frame_length;
    sDecCtrl.LTP_scale_Q14 = 0;

    /* Safety checks */
    SKP_assert( L > 0 && L <= MAX_FRAME_LENGTH );

    /********************************************/
    /* Decode Frame if packet is not lost       */
    /********************************************/
    if( lostFlag != PACKET_LOST && psDec->nFramesDecoded == 0 ) {
        /* First decoder call for this payload */
        /* Decode VAD flags and LBRR flag */
        flags = SKP_RSHIFT( psRangeDec->buf[ 0 ], 7 - psDec->nFramesPerPacket ) & 
            ( SKP_LSHIFT( 1, psDec->nFramesPerPacket + 1 ) - 1 );
        psDec->LBRR_flag = flags & 1;
        for( i = psDec->nFramesPerPacket - 1; i >= 0 ; i-- ) {
            flags = SKP_RSHIFT( flags, 1 );
            psDec->VAD_flags[ i ] = flags & 1;
        }
        for( i = 0; i < psDec->nFramesPerPacket + 1; i++ ) {
            ec_dec_icdf( psRangeDec, SKP_Silk_uniform2_iCDF, 8 );
        }
       
        /* Decode LBRR flags */
        SKP_memset( psDec->LBRR_flags, 0, sizeof( psDec->LBRR_flags ) );
        if( psDec->LBRR_flag ) {
            if( psDec->nFramesPerPacket == 1 ) {
                psDec->LBRR_flags[ 0 ] = 1;
            } else {
                LBRR_symbol = ec_dec_icdf( psRangeDec, SKP_Silk_LBRR_flags_iCDF_ptr[ psDec->nFramesPerPacket - 2 ], 8 ) + 1;
                for( i = 0; i < psDec->nFramesPerPacket; i++ ) {
                    psDec->LBRR_flags[ i ] = SKP_RSHIFT( LBRR_symbol, i ) & 1;
                }
            }
        }

        if( lostFlag == DECODE_NORMAL ) {
            /* Regular decoding: skip all LBRR data */
            for( i = 0; i < psDec->nFramesPerPacket; i++ ) {
                if( psDec->LBRR_flags[ i ] ) {
                    SKP_Silk_decode_indices( psDec, psRangeDec, i, 1 );
                    SKP_Silk_decode_pulses( psRangeDec, pulses, psDec->indices.signalType, 
                        psDec->indices.quantOffsetType, psDec->frame_length );
                }
            }
        }

    }

    if( lostFlag == DECODE_LBRR && psDec->LBRR_flags[ psDec->nFramesDecoded ] == 0 ) {
        /* Treat absent LBRR data as lost frame */
        lostFlag = PACKET_LOST;
        psDec->nFramesDecoded++;
    }

    if( lostFlag != PACKET_LOST ) {
        /*********************************************/
        /* Decode quantization indices of side info  */
        /*********************************************/
TIC(decode_indices)
        SKP_Silk_decode_indices( psDec, psRangeDec, psDec->nFramesDecoded, lostFlag );
TOC(decode_indices)

        /*********************************************/
        /* Decode quantization indices of excitation */
        /*********************************************/
TIC(decode_pulses)
        SKP_Silk_decode_pulses( psRangeDec, pulses, psDec->indices.signalType, 
                psDec->indices.quantOffsetType, psDec->frame_length );
TOC(decode_pulses)

        /********************************************/
        /* Decode parameters and pulse signal       */
        /********************************************/
TIC(decode_params)
        SKP_Silk_decode_parameters( psDec, &sDecCtrl );
TOC(decode_params)

        /* Update length. Sampling frequency may have changed */
        L = psDec->frame_length;

        /********************************************************/
        /* Run inverse NSQ                                      */
        /********************************************************/
TIC(decode_core)
        SKP_Silk_decode_core( psDec, &sDecCtrl, pOut, pulses );
TOC(decode_core)

        /********************************************************/
        /* Update PLC state                                     */
        /********************************************************/
        SKP_Silk_PLC( psDec, &sDecCtrl, pOut, L, 0 );

        psDec->lossCnt = 0;
        psDec->prevSignalType = psDec->indices.signalType;
        SKP_assert( psDec->prevSignalType >= 0 && psDec->prevSignalType <= 2 );

        /* A frame has been decoded without errors */
        psDec->first_frame_after_reset = 0;
        psDec->nFramesDecoded++;
    } else {
        /* Handle packet loss by extrapolation */
        SKP_Silk_PLC( psDec, &sDecCtrl, pOut, L, 1 );
    }

    /*************************/
    /* Update output buffer. */
    /*************************/
    SKP_assert( psDec->ltp_mem_length >= psDec->frame_length );
    mv_len = psDec->ltp_mem_length - psDec->frame_length;
    SKP_memmove( psDec->outBuf, &psDec->outBuf[ psDec->frame_length ], mv_len * sizeof(SKP_int16) );
    SKP_memcpy( &psDec->outBuf[ mv_len ], pOut, psDec->frame_length * sizeof( SKP_int16 ) );

    /****************************************************************/
    /* Ensure smooth connection of extrapolated and good frames     */
    /****************************************************************/
    SKP_Silk_PLC_glue_frames( psDec, &sDecCtrl, pOut, L );

    /************************************************/
    /* Comfort noise generation / estimation        */
    /************************************************/
    SKP_Silk_CNG( psDec, &sDecCtrl, pOut, L );

    /********************************************/
    /* HP filter output                            */
    /********************************************/
TIC(HP_out)
    SKP_Silk_biquad_alt( pOut, psDec->HP_B, psDec->HP_A, psDec->HPState, pOut, L );
TOC(HP_out)

    /* Update some decoder state variables */
    psDec->lagPrev = sDecCtrl.pitchL[ psDec->nb_subfr - 1 ];

    /********************************************/
    /* set output frame length                    */
    /********************************************/
    *pN = ( SKP_int16 )L;

TOC(DECODE_FRAME)

    return ret;
}
    