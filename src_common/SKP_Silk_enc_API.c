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


#include "SKP_Silk_define.h"
#include "SKP_Silk_SDK_API.h"
#include "SKP_Silk_control.h"
#include "SKP_Silk_typedef.h"
#include "SKP_Silk_structs.h"
#include "SKP_Silk_tuning_parameters.h"
#if FIXED_POINT
#include "SKP_Silk_main_FIX.h"
#define SKP_Silk_encoder_state_Fxx      SKP_Silk_encoder_state_FIX
#define SKP_Silk_encode_frame_Fxx       SKP_Silk_encode_frame_FIX
#else
#include "SKP_Silk_main_FLP.h"
#define SKP_Silk_encoder_state_Fxx      SKP_Silk_encoder_state_FLP
#define SKP_Silk_encode_frame_Fxx       SKP_Silk_encode_frame_FLP
#endif

/* Encoder Super Struct */
typedef struct {
    SKP_Silk_encoder_state_Fxx          state_Fxx[ ENCODER_NUM_CHANNELS ];
    stereo_state                        sStereo;
    SKP_int32                           nBitsExceeded;
    SKP_int                             nChannels;
    SKP_int                             timeSinceSwitchAllowed_ms;
    SKP_int                             allowBandwidthSwitch;
} SKP_Silk_encoder;

/****************************************/
/* Encoder functions                    */
/****************************************/

SKP_int SKP_Silk_SDK_Get_Encoder_Size( SKP_int32 *encSizeBytes )
{
    SKP_int ret = SKP_SILK_NO_ERROR;
    
    *encSizeBytes = sizeof( SKP_Silk_encoder );
    
    return ret;
}

/*************************/
/* Init or Reset encoder */
/*************************/
SKP_int SKP_Silk_SDK_InitEncoder(
    void                            *encState,          /* I/O: State                                           */
    SKP_SILK_SDK_EncControlStruct   *encStatus          /* O:   Control structure                               */
)
{
    SKP_Silk_encoder *psEnc;
    SKP_int n, ret = SKP_SILK_NO_ERROR;

    psEnc = (SKP_Silk_encoder *)encState;
    
    /* Reset encoder */
    for( n = 0; n < ENCODER_NUM_CHANNELS; n++ ) {
        if( ret += SKP_Silk_init_encoder( &psEnc->state_Fxx[ n ] ) ) {
            SKP_assert( 0 );
        }
    }
    SKP_memset( &psEnc->sStereo, 0, sizeof( psEnc->sStereo ) );

    psEnc->nBitsExceeded = 0;
    psEnc->nChannels = 1;

    /* Read control structure */
    if( ret += SKP_Silk_SDK_QueryEncoder( encState, encStatus ) ) {
        SKP_assert( 0 );
    }

    return ret;
}

/***************************************/
/* Read control structure from encoder */
/***************************************/
SKP_int SKP_Silk_SDK_QueryEncoder(
    const void *encState,                       /* I:   State Vector                                    */
    SKP_SILK_SDK_EncControlStruct *encStatus    /* O:   Control Structure                               */
)
{
    SKP_Silk_encoder_state_Fxx *state_Fxx;
    SKP_int ret = SKP_SILK_NO_ERROR;

    state_Fxx = ((SKP_Silk_encoder *)encState)->state_Fxx;

    encStatus->API_sampleRate            = state_Fxx->sCmn.API_fs_Hz;
    encStatus->maxInternalSampleRate     = state_Fxx->sCmn.maxInternal_fs_Hz;
    encStatus->minInternalSampleRate     = state_Fxx->sCmn.minInternal_fs_Hz;
    encStatus->desiredInternalSampleRate = state_Fxx->sCmn.desiredInternal_fs_Hz;
    encStatus->payloadSize_ms            = state_Fxx->sCmn.PacketSize_ms;
    encStatus->bitRate                   = state_Fxx->sCmn.TargetRate_bps;
    encStatus->packetLossPercentage      = state_Fxx->sCmn.PacketLoss_perc;
    encStatus->complexity                = state_Fxx->sCmn.Complexity;
    encStatus->useInBandFEC              = state_Fxx->sCmn.useInBandFEC;
    encStatus->useDTX                    = state_Fxx->sCmn.useDTX;
    encStatus->useCBR                    = state_Fxx->sCmn.useCBR;
    encStatus->internalSampleRate        = SKP_SMULBB( state_Fxx->sCmn.fs_kHz, 1000 );

    return ret;
}

/**************************/
/* Encode frame with Silk */
/**************************/
SKP_int SKP_Silk_SDK_Encode( 
    void                                *encState,      /* I/O: State                                           */
    SKP_SILK_SDK_EncControlStruct       *encControl,    /* I:   Control structure                               */
    const SKP_int16                     *samplesIn,     /* I:   Speech sample input vector                      */
    SKP_int                             nSamplesIn,     /* I:   Number of samples in input vector               */
    ec_enc                              *psRangeEnc,    /* I/O  Compressor data structure                       */
    SKP_int32                           *nBytesOut,     /* I/O: Number of bytes in payload (input: Max bytes)   */
    const SKP_int                       prefillFlag     /* I:   Flag to indicate prefilling buffers; no coding  */
)
{
    SKP_int   n, i, nBits, flags, tmp_payloadSize_ms, tmp_complexity, MS_predictorIx = 0, ret = 0;
    SKP_int   nSamplesToBuffer, nBlocksOf10ms, nSamplesFromInput = 0;
    SKP_int   speech_act_thr_for_switch_Q8;
    SKP_int32 TargetRate_bps, channelRate_bps, LBRR_symbol;
    SKP_Silk_encoder *psEnc = ( SKP_Silk_encoder * )encState;
    SKP_int16 buf[ MAX_FRAME_LENGTH_MS * MAX_API_FS_KHZ ];

    /* Check values in encoder control structure */
    if( ( ret = check_control_input( encControl ) != 0 ) ) {
        SKP_assert( 0 );
        return ret;
    }

    if( encControl->nChannels > psEnc->nChannels ) {
        /* Mono -> Stereo transition: init state of second channel and stereo state */
        SKP_memset( &psEnc->sStereo, 0, sizeof( psEnc->sStereo ) );
        ret += SKP_Silk_init_encoder( &psEnc->state_Fxx[ 1 ] );
    }
    psEnc->nChannels = encControl->nChannels;

    nBlocksOf10ms = SKP_DIV32( 100 * nSamplesIn, encControl->API_sampleRate );
    if( prefillFlag ) {
        /* Only accept input length of 10 ms */
        if( nBlocksOf10ms != 1 ) {
            ret = SKP_SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
            SKP_assert( 0 );
            return ret;
        }
        /* Reset Encoder */
        for( n = 0; n < encControl->nChannels; n++ ) {
            if( ret = SKP_Silk_init_encoder( &psEnc->state_Fxx[ n ] ) ) {
                SKP_assert( 0 );
            }
        }
        tmp_payloadSize_ms = encControl->payloadSize_ms;
        encControl->payloadSize_ms = 10;
        tmp_complexity = encControl->complexity;
        encControl->complexity = 0;
        for( n = 0; n < encControl->nChannels; n++ ) {
            psEnc->state_Fxx[ n ].sCmn.controlled_since_last_payload = 0;
            psEnc->state_Fxx[ n ].sCmn.prefillFlag = 1;
        }
    } else {
        /* Only accept input lengths that are a multiple of 10 ms */
        if( nBlocksOf10ms * encControl->API_sampleRate != 100 * nSamplesIn || nSamplesIn < 0 ) {
            ret = SKP_SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
            SKP_assert( 0 );
            return ret;
        }
        /* Make sure no more than one packet can be produced */
        if( 1000 * (SKP_int32)nSamplesIn > encControl->payloadSize_ms * encControl->API_sampleRate ) {
            ret = SKP_SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
            SKP_assert( 0 );
            return ret;
        }
    }

    TargetRate_bps = SKP_RSHIFT32( encControl->bitRate, encControl->nChannels - 1 );
    for( n = 0; n < encControl->nChannels; n++ ) {
        if( ( ret = SKP_Silk_control_encoder( &psEnc->state_Fxx[ n ], encControl, TargetRate_bps, psEnc->allowBandwidthSwitch ) ) != 0 ) {
            SKP_assert( 0 );
            return ret;
        }
    }
    SKP_assert( encControl->nChannels == 1 || psEnc->state_Fxx[ 0 ].sCmn.fs_kHz == psEnc->state_Fxx[ 1 ].sCmn.fs_kHz );

    /* Input buffering/resampling and encoding */
    while( 1 ) {
        nSamplesToBuffer  = psEnc->state_Fxx[ 0 ].sCmn.frame_length - psEnc->state_Fxx[ 0 ].sCmn.inputBufIx;
        nSamplesToBuffer  = SKP_min( nSamplesToBuffer, 10 * nBlocksOf10ms * psEnc->state_Fxx[ 0 ].sCmn.fs_kHz );
        nSamplesFromInput = SKP_DIV32_16( nSamplesToBuffer * psEnc->state_Fxx[ 0 ].sCmn.API_fs_Hz, psEnc->state_Fxx[ 0 ].sCmn.fs_kHz * 1000 );
        /* Resample and write to buffer */
        if( encControl->nChannels == 2 ) {
            for( n = 0; n < nSamplesFromInput; n++ ) {
                buf[ n ] = samplesIn[ 2 * n ];
            }
            ret += SKP_Silk_resampler( &psEnc->state_Fxx[ 0 ].sCmn.resampler_state, 
                &psEnc->state_Fxx[ 0 ].sCmn.inputBuf[ psEnc->state_Fxx[ 0 ].sCmn.inputBufIx ], buf, nSamplesFromInput );
            psEnc->state_Fxx[ 0 ].sCmn.inputBufIx += nSamplesToBuffer;

            nSamplesToBuffer  = psEnc->state_Fxx[ 1 ].sCmn.frame_length - psEnc->state_Fxx[ 1 ].sCmn.inputBufIx;
            nSamplesToBuffer  = SKP_min( nSamplesToBuffer, 10 * nBlocksOf10ms * psEnc->state_Fxx[ 1 ].sCmn.fs_kHz );
            for( n = 0; n < nSamplesFromInput; n++ ) {
                buf[ n ] = samplesIn[ 2 * n + 1 ];
            }
            ret += SKP_Silk_resampler( &psEnc->state_Fxx[ 1 ].sCmn.resampler_state, 
                &psEnc->state_Fxx[ 1 ].sCmn.inputBuf[ psEnc->state_Fxx[ 1 ].sCmn.inputBufIx ], buf, nSamplesFromInput );
            psEnc->state_Fxx[ 1 ].sCmn.inputBufIx += nSamplesToBuffer;
        } else {
            SKP_assert( encControl->nChannels == 1 );
            ret += SKP_Silk_resampler( &psEnc->state_Fxx[ 0 ].sCmn.resampler_state, 
                &psEnc->state_Fxx[ 0 ].sCmn.inputBuf[ psEnc->state_Fxx[ 0 ].sCmn.inputBufIx ], samplesIn, nSamplesFromInput );
            psEnc->state_Fxx[ 0 ].sCmn.inputBufIx += nSamplesToBuffer;
        }
        samplesIn  += nSamplesFromInput * encControl->nChannels;
        nSamplesIn -= nSamplesFromInput;

        /* Default */
        psEnc->allowBandwidthSwitch = 0;

        /* Silk encoder */
        if( psEnc->state_Fxx[ 0 ].sCmn.inputBufIx >= psEnc->state_Fxx[ 0 ].sCmn.frame_length ) {
            /* Enough data in input buffer, so encode */
            SKP_assert( psEnc->state_Fxx[ 0 ].sCmn.inputBufIx == psEnc->state_Fxx[ 0 ].sCmn.frame_length );
            SKP_assert( encControl->nChannels == 1 || psEnc->state_Fxx[ 1 ].sCmn.inputBufIx == psEnc->state_Fxx[ 1 ].sCmn.frame_length );

            /* Deal with LBRR data */
            if( psEnc->state_Fxx[ 0 ].sCmn.nFramesAnalyzed == 0 && !prefillFlag ) {
                /* Create space at start of payload for VAD and FEC flags */
                SKP_uint8 iCDF[ 2 ] = { 0, 0 };
                iCDF[ 0 ] = 256 - SKP_RSHIFT( 256, ( psEnc->state_Fxx[ 0 ].sCmn.nFramesPerPacket + 1 ) * encControl->nChannels );
                ec_enc_icdf( psRangeEnc, 0, iCDF, 8 );

                /* Encode any LBRR data from previous packet */
                /* Encode LBRR flags */
                for( n = 0; n < encControl->nChannels; n++ ) {
                    LBRR_symbol = 0;
                    for( i = 0; i < psEnc->state_Fxx[ n ].sCmn.nFramesPerPacket; i++ ) {
                        LBRR_symbol |= SKP_LSHIFT( psEnc->state_Fxx[ n ].sCmn.LBRR_flags[ i ], i );
                    }
                    psEnc->state_Fxx[ n ].sCmn.LBRR_flag = LBRR_symbol > 0 ? 1 : 0;
                    if( LBRR_symbol && psEnc->state_Fxx[ n ].sCmn.nFramesPerPacket > 1 ) {
                        ec_enc_icdf( psRangeEnc, LBRR_symbol - 1, SKP_Silk_LBRR_flags_iCDF_ptr[ psEnc->state_Fxx[ n ].sCmn.nFramesPerPacket - 2 ], 8 );
                    }
                }

                /* Code LBRR indices and excitation signals */
                for( i = 0; i < psEnc->state_Fxx[ 0 ].sCmn.nFramesPerPacket; i++ ) {
                    for( n = 0; n < encControl->nChannels; n++ ) {                
                        if( psEnc->state_Fxx[ n ].sCmn.LBRR_flags[ i ] ) {
                            SKP_Silk_encode_indices( &psEnc->state_Fxx[ n ].sCmn, psRangeEnc, i, 1 );
                            SKP_Silk_encode_pulses( psRangeEnc, psEnc->state_Fxx[ n ].sCmn.indices_LBRR[i].signalType, psEnc->state_Fxx[ n ].sCmn.indices_LBRR[i].quantOffsetType, 
                                psEnc->state_Fxx[ n ].sCmn.pulses_LBRR[ i ], psEnc->state_Fxx[ n ].sCmn.frame_length );
                        }
                    }
                }

                /* Reset LBRR flags */
                for( n = 0; n < encControl->nChannels; n++ ) {                
                    SKP_memset( psEnc->state_Fxx[ n ].sCmn.LBRR_flags, 0, sizeof( psEnc->state_Fxx[ n ].sCmn.LBRR_flags ) );
                }
            }

            /* Convert Left/Right to Mid/Side */
            if( encControl->nChannels == 2 ) {
                SKP_Silk_stereo_LR_to_MS( &psEnc->sStereo, psEnc->state_Fxx[ 0 ].sCmn.inputBuf, psEnc->state_Fxx[ 1 ].sCmn.inputBuf, 
                    &MS_predictorIx, psEnc->state_Fxx[ 0 ].sCmn.fs_kHz, psEnc->state_Fxx[ 0 ].sCmn.frame_length );
                ec_enc_icdf( psRangeEnc, MS_predictorIx, SKP_Silk_stereo_predictor_iCDF, 8 );
            }


            /* Total target bits for packet */
            nBits = SKP_DIV32_16( SKP_MUL( encControl->bitRate, encControl->payloadSize_ms ), 1000 );
            /* Subtract bits already used */
            nBits -= ec_tell( psRangeEnc );
            /* Divide by number of uncoded frames left in packet */
            nBits = SKP_DIV32_16( nBits, psEnc->state_Fxx[ 0 ].sCmn.nFramesPerPacket - psEnc->state_Fxx[ 0 ].sCmn.nFramesAnalyzed );
            /* Convert to bits/second */
            if( encControl->payloadSize_ms == 10 ) {
                TargetRate_bps = SKP_SMULBB( nBits, 100 );
            } else {
                TargetRate_bps = SKP_SMULBB( nBits, 50 );
            }
            /* Subtract fraction of bits in excess of target in previous packets */
            TargetRate_bps -= SKP_DIV32_16( SKP_MUL( psEnc->nBitsExceeded, 1000 ), BITRESERVOIR_DECAY_TIME_MS );
            /* Don't exceed input bitrate */
            TargetRate_bps = SKP_min( TargetRate_bps, encControl->bitRate );

            /* Encode */
            for( n = 0; n < encControl->nChannels; n++ ) {
                /* For stereo coding, allocate 60% of the bitrate to mid and 40% to side */
                if( encControl->nChannels == 1 ) {
                    channelRate_bps = TargetRate_bps;
                } else if( n == 0 ) {
                    channelRate_bps = SKP_SMULWW( TargetRate_bps, SKP_FIX_CONST( 0.6, 16 ) );
                } else {
                    channelRate_bps = SKP_SMULWB( TargetRate_bps, SKP_FIX_CONST( 0.4, 16 ) );
                }
                SKP_Silk_control_SNR( &psEnc->state_Fxx[ n ].sCmn, channelRate_bps );
                //SKP_Silk_control_SNR( &psEnc->state_Fxx[ n ].sCmn, TargetRate_bps / 2 );
                if( ( ret = SKP_Silk_encode_frame_Fxx( &psEnc->state_Fxx[ n ], nBytesOut, psRangeEnc ) ) != 0 ) {
                    SKP_assert( 0 );
                }
                psEnc->state_Fxx[ n ].sCmn.controlled_since_last_payload = 0;
                psEnc->state_Fxx[ n ].sCmn.inputBufIx = 0;
            }

            /* Insert VAD and FEC flags at beginning of bitstream */
            if( *nBytesOut > 0 ) {
                flags = 0;
                for( n = 0; n < encControl->nChannels; n++ ) {
                    for( i = 0; i < psEnc->state_Fxx[ n ].sCmn.nFramesPerPacket; i++ ) {
                        flags  = SKP_LSHIFT( flags, 1 );
                        flags |= psEnc->state_Fxx[ n ].sCmn.VAD_flags[ i ];
                    }
                    flags  = SKP_LSHIFT( flags, 1 );
                    flags |= psEnc->state_Fxx[ n ].sCmn.LBRR_flag;
                }
                ec_enc_patch_initial_bits( psRangeEnc, flags, ( psEnc->state_Fxx[ 0 ].sCmn.nFramesPerPacket + 1 ) * encControl->nChannels );

                /* Return zero bytes if DTXed */
                if( psEnc->state_Fxx[ 0 ].sCmn.inDTX && ( encControl->nChannels == 1 || psEnc->state_Fxx[ 1 ].sCmn.inDTX ) ) {
                    *nBytesOut = 0;
                }

                psEnc->nBitsExceeded += *nBytesOut * 8;
                psEnc->nBitsExceeded -= SKP_DIV32_16( SKP_MUL( encControl->bitRate, encControl->payloadSize_ms ), 1000 );
                psEnc->nBitsExceeded  = SKP_LIMIT( psEnc->nBitsExceeded, 0, 10000 );

                /* Update flag indicating if bandwidth switching is allowed */
                speech_act_thr_for_switch_Q8 = SKP_SMLAWB( SKP_FIX_CONST( SPEECH_ACTIVITY_DTX_THRES, 8 ), 
                    SKP_FIX_CONST( ( 1 - SPEECH_ACTIVITY_DTX_THRES ) / MAX_BANDWIDTH_SWITCH_DELAY_MS, 16 + 8 ), psEnc->timeSinceSwitchAllowed_ms );
                if( psEnc->state_Fxx[ 0 ].sCmn.speech_activity_Q8 < speech_act_thr_for_switch_Q8 ) {
                    psEnc->allowBandwidthSwitch = 1;
                    psEnc->timeSinceSwitchAllowed_ms = 0;
                } else {
                    psEnc->allowBandwidthSwitch = 0;
                    psEnc->timeSinceSwitchAllowed_ms += encControl->payloadSize_ms;
                }
            }

            if( nSamplesIn == 0 ) {
                break;
            }
        } else {
            break;
        }
    }

    encControl->allowBandwidthSwitch = psEnc->allowBandwidthSwitch;
    encControl->inWBmodeWithoutVariableLP = ( psEnc->state_Fxx[ 0 ].sCmn.fs_kHz == 16 ) && ( psEnc->state_Fxx[ 0 ].sCmn.sLP.mode == 0 );
    encControl->internalSampleRate = SKP_SMULBB( psEnc->state_Fxx[ 0 ].sCmn.fs_kHz, 1000 );
    if( prefillFlag ) {
        encControl->payloadSize_ms = tmp_payloadSize_ms;
        encControl->complexity = tmp_complexity;
        for( n = 0; n < encControl->nChannels; n++ ) {
            psEnc->state_Fxx[ n ].sCmn.controlled_since_last_payload = 0;
            psEnc->state_Fxx[ n ].sCmn.prefillFlag = 0;
        }
    }

    return ret;
}

