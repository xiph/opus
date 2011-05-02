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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "SKP_Silk_SDK_API.h"
#include "SKP_Silk_main.h"

/************************/
/* Decoder Super Struct */
/************************/
typedef struct {
    SKP_Silk_decoder_state          channel_state[ DECODER_NUM_CHANNELS ];
    stereo_state                    sStereo;
    SKP_int                         nChannels;
} SKP_Silk_decoder;

/*********************/
/* Decoder functions */
/*********************/

SKP_int SKP_Silk_SDK_Get_Decoder_Size( SKP_int32 *decSizeBytes ) 
{
    SKP_int ret = SKP_SILK_NO_ERROR;

    *decSizeBytes = sizeof( SKP_Silk_decoder );

    return ret;
}

/* Reset decoder state */
SKP_int SKP_Silk_SDK_InitDecoder(
    void* decState                                      /* I/O: State                                          */
)
{
    SKP_int n, ret = SKP_SILK_NO_ERROR;
    SKP_Silk_decoder_state *channel_state = ((SKP_Silk_decoder *)decState)->channel_state;

    for( n = 0; n < DECODER_NUM_CHANNELS; n++ ) {
        ret  = SKP_Silk_init_decoder( &channel_state[ n ] );
    }

    return ret;
}

/* Decode a frame */
SKP_int SKP_Silk_SDK_Decode(
    void*                               decState,       /* I/O: State                                           */
    SKP_SILK_SDK_DecControlStruct*      decControl,     /* I/O: Control Structure                               */
    SKP_int                             lostFlag,       /* I:   0: no loss, 1 loss, 2 decode FEC                */
    SKP_int                             newPacketFlag,  /* I:   Indicates first decoder call for this packet    */
    ec_dec                              *psRangeDec,    /* I/O  Compressor data structure                       */
    SKP_int16                           *samplesOut,    /* O:   Decoded output speech vector                    */
    SKP_int32                           *nSamplesOut    /* O:   Number of samples decoded                       */
)
{
    SKP_int   i, n, prev_fs_kHz, doResample, flags, nFlags, ret = SKP_SILK_NO_ERROR;
    SKP_int32 nSamplesOutDec, LBRR_symbol;
    SKP_int16 samplesOut1_tmp[ 2 * MAX_FS_KHZ * MAX_FRAME_LENGTH_MS ];
    SKP_int16 samplesOut2_tmp[ MAX_API_FS_KHZ * MAX_FRAME_LENGTH_MS ];
    SKP_int   MS_pred_Q14[ 2 ] = { 0 };
    SKP_int16 *dec_out_ptr, *resample_out_ptr;
    SKP_Silk_decoder *psDec = ( SKP_Silk_decoder * )decState;
    SKP_Silk_decoder_state *channel_state = psDec->channel_state;

    /**********************************/
    /* Test if first frame in payload */
    /**********************************/
    if( newPacketFlag ) {
        for( n = 0; n < decControl->nChannels; n++ ) {
            channel_state[ n ].nFramesDecoded = 0;  /* Used to count frames in packet */
        }
    }

    /* Save previous sample frequency */
    prev_fs_kHz = channel_state[ 0 ].fs_kHz;

    if( decControl->nChannels > psDec->nChannels ) {
        /* Mono -> Stereo transition: init state of second channel and stereo state */
        SKP_memset( &psDec->sStereo, 0, sizeof( psDec->sStereo ) );
        ret += SKP_Silk_init_decoder( &channel_state[ 1 ] );
    }
    psDec->nChannels = decControl->nChannels;

    for( n = 0; n < decControl->nChannels; n++ ) {
        if( channel_state[ n ].nFramesDecoded == 0 ) {
            SKP_int fs_kHz_dec;
            if( decControl->payloadSize_ms == 10 ) {
                channel_state[ n ].nFramesPerPacket = 1;
                channel_state[ n ].nb_subfr = 2;
            } else if( decControl->payloadSize_ms == 20 ) {
                channel_state[ n ].nFramesPerPacket = 1;
                channel_state[ n ].nb_subfr = 4;
            } else if( decControl->payloadSize_ms == 40 ) {
                channel_state[ n ].nFramesPerPacket = 2;
                channel_state[ n ].nb_subfr = 4;
            } else if( decControl->payloadSize_ms == 60 ) {
                channel_state[ n ].nFramesPerPacket = 3;
                channel_state[ n ].nb_subfr = 4;
            } else {
                SKP_assert( 0 );
                return SKP_SILK_DEC_INVALID_FRAME_SIZE;
            } 
            fs_kHz_dec = ( decControl->internalSampleRate >> 10 ) + 1;
            if( fs_kHz_dec != 8 && fs_kHz_dec != 12 && fs_kHz_dec != 16 ) {
                SKP_assert( 0 );
                return SKP_SILK_DEC_INVALID_SAMPLING_FREQUENCY;
            }
            SKP_Silk_decoder_set_fs( &channel_state[ n ], fs_kHz_dec );
        }
    }
    
    if( decControl->API_sampleRate > MAX_API_FS_KHZ * 1000 || decControl->API_sampleRate < 8000 ) {
        ret = SKP_SILK_DEC_INVALID_SAMPLING_FREQUENCY;
        return( ret );
    }

    doResample = SKP_SMULBB( channel_state[ 0 ].fs_kHz, 1000 ) != decControl->API_sampleRate;

    /* Set up pointers to temp buffers */
    if( doResample || decControl->nChannels == 2 ) { 
        dec_out_ptr = samplesOut1_tmp;
    } else {
        dec_out_ptr = samplesOut;
    }
    if( decControl->nChannels == 2 ) {
        resample_out_ptr = samplesOut2_tmp;
    } else {
        resample_out_ptr = samplesOut;
    }

    if( lostFlag != FLAG_PACKET_LOST && channel_state[ 0 ].nFramesDecoded == 0 ) {
        /* First decoder call for this payload */
        nFlags = SKP_SMULBB( decControl->nChannels, channel_state[ 0 ].nFramesPerPacket + 1 );
        flags = SKP_RSHIFT( psRangeDec->buf[ 0 ], 8 - nFlags ) & ( SKP_LSHIFT( 1, nFlags ) - 1 );
        for( i = 0; i < nFlags; i++ ) {
            ec_dec_icdf( psRangeDec, SKP_Silk_uniform2_iCDF, 8 );
        }
        /* Decode VAD flags and LBRR flag */
        for( n = decControl->nChannels - 1; n >= 0; n-- ) {
            channel_state[ n ].LBRR_flag = flags & 1;
            flags = SKP_RSHIFT( flags, 1 );
            for( i = channel_state[ n ].nFramesPerPacket - 1; i >= 0 ; i-- ) {
                channel_state[ n ].VAD_flags[ i ] = flags & 1;
                flags = SKP_RSHIFT( flags, 1 );
            }
        }       
        /* Decode LBRR flags */
        for( n = 0; n < decControl->nChannels; n++ ) {
            SKP_memset( channel_state[ n ].LBRR_flags, 0, sizeof( channel_state[ n ].LBRR_flags ) );
            if( channel_state[ n ].LBRR_flag ) {
                if( channel_state[ n ].nFramesPerPacket == 1 ) {
                    channel_state[ n ].LBRR_flags[ 0 ] = 1;
                } else {
                    LBRR_symbol = ec_dec_icdf( psRangeDec, SKP_Silk_LBRR_flags_iCDF_ptr[ channel_state[ n ].nFramesPerPacket - 2 ], 8 ) + 1;
                    for( i = 0; i < channel_state[ n ].nFramesPerPacket; i++ ) {
                        channel_state[ n ].LBRR_flags[ i ] = SKP_RSHIFT( LBRR_symbol, i ) & 1;
                    }
                }
            }
        }

        if( lostFlag == FLAG_DECODE_NORMAL ) {
            /* Regular decoding: skip all LBRR data */
            for( i = 0; i < channel_state[ 0 ].nFramesPerPacket; i++ ) {
                for( n = 0; n < decControl->nChannels; n++ ) {
                    if( channel_state[ n ].LBRR_flags[ i ] ) {
                        SKP_int pulses[ MAX_FRAME_LENGTH ];
                        SKP_Silk_decode_indices( &channel_state[ n ], psRangeDec, i, 1 );
                        SKP_Silk_decode_pulses( psRangeDec, pulses, channel_state[ n ].indices.signalType, 
                            channel_state[ n ].indices.quantOffsetType, channel_state[ n ].frame_length );
                    }
                }
            }
        }
    }

    /* Get MS predictor index */
    if( decControl->nChannels == 2 ) {
        SKP_Silk_stereo_decode_pred( psRangeDec, MS_pred_Q14 );
    }

    /* Call decoder for one frame */
    for( n = 0; n < decControl->nChannels; n++ ) {
        ret += SKP_Silk_decode_frame( &channel_state[ n ], psRangeDec, &dec_out_ptr[ n * MAX_FS_KHZ * MAX_FRAME_LENGTH_MS ], &nSamplesOutDec, lostFlag );
    }

    /* Convert Mid/Side to Left/Right */
    if( decControl->nChannels == 2 ) {
        SKP_Silk_stereo_MS_to_LR( &psDec->sStereo, dec_out_ptr, &dec_out_ptr[ MAX_FS_KHZ * MAX_FRAME_LENGTH_MS ], MS_pred_Q14, channel_state[ 0 ].fs_kHz, nSamplesOutDec );
    }

    /* Number of output samples */
    if( doResample ) {
        *nSamplesOut = SKP_DIV32( nSamplesOutDec * decControl->API_sampleRate, SKP_SMULBB( channel_state[ 0 ].fs_kHz, 1000 ) );
    } else {
        *nSamplesOut = nSamplesOutDec;
    }

    for( n = 0; n < decControl->nChannels; n++ ) {
        /* Resample if needed */
        if( doResample ) {
            /* Initialize resampler when switching internal or external sampling frequency */
            if( prev_fs_kHz != channel_state[ n ].fs_kHz || channel_state[ n ].prev_API_sampleRate != decControl->API_sampleRate ) {
                ret = SKP_Silk_resampler_init( &channel_state[ n ].resampler_state, SKP_SMULBB( channel_state[ n ].fs_kHz, 1000 ), decControl->API_sampleRate );
            }

            /* Resample the output to API_sampleRate */
            ret += SKP_Silk_resampler( &channel_state[ n ].resampler_state, resample_out_ptr, &dec_out_ptr[ n * MAX_FS_KHZ * MAX_FRAME_LENGTH_MS ], nSamplesOutDec );
        } else {
            resample_out_ptr = &dec_out_ptr[ n * MAX_FS_KHZ * MAX_FRAME_LENGTH_MS ];
        }

        /* Interleave if needed */
        if( decControl->nChannels == 2 ) {
            for( i = 0; i < *nSamplesOut; i++ ) {
                samplesOut[ n + 2 * i ] = resample_out_ptr[ i ];
            }
        }
        
        channel_state[ n ].prev_API_sampleRate = decControl->API_sampleRate;
    }

    /* Copy parameters to control stucture */
    decControl->frameSize        = ( SKP_int )*nSamplesOut;
    decControl->framesPerPayload = ( SKP_int )channel_state[ n ].nFramesPerPacket;

    return ret;
}

/* Getting table of contents for a packet */
SKP_int SKP_Silk_SDK_get_TOC(
    const SKP_uint8                     *payload,           /* I    Payload data                                */
    const SKP_int                       nBytesIn,           /* I:   Number of input bytes                       */
    const SKP_int                       nFramesPerPayload,  /* I:   Number of SILK frames per payload           */
    SKP_Silk_TOC_struct                 *Silk_TOC           /* O:   Type of content                             */
)
{
    SKP_int i, flags, ret = SKP_SILK_NO_ERROR;

    if( nBytesIn < 1 ) {
        return -1;
    }
    if( nFramesPerPayload < 0 || nFramesPerPayload > 3 ) {
        return -1;
    }

    SKP_memset( Silk_TOC, 0, sizeof( Silk_TOC ) );

    /* For stereo, extract the flags for the mid channel */
    flags = SKP_RSHIFT( payload[ 0 ], 7 - nFramesPerPayload ) & ( SKP_LSHIFT( 1, nFramesPerPayload + 1 ) - 1 );

    Silk_TOC->inbandFECFlag = flags & 1;
    for( i = nFramesPerPayload - 1; i >= 0 ; i-- ) {
        flags = SKP_RSHIFT( flags, 1 );
        Silk_TOC->VADFlags[ i ] = flags & 1;
        Silk_TOC->VADFlag |= flags & 1;
    }

    return ret;
}
