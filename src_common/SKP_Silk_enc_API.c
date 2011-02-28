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
#if FIXED_POINT
#include "SKP_Silk_main_FIX.h"
#define SKP_Silk_encoder_state_Fxx      SKP_Silk_encoder_state_FIX
#define SKP_Silk_init_encoder_Fxx       SKP_Silk_init_encoder_FIX
#define SKP_Silk_control_encoder_Fxx    SKP_Silk_control_encoder_FIX
#define SKP_Silk_encode_frame_Fxx       SKP_Silk_encode_frame_FIX
#else
#include "SKP_Silk_main_FLP.h"
#define SKP_Silk_encoder_state_Fxx      SKP_Silk_encoder_state_FLP
#define SKP_Silk_init_encoder_Fxx       SKP_Silk_init_encoder_FLP
#define SKP_Silk_control_encoder_Fxx    SKP_Silk_control_encoder_FLP
#define SKP_Silk_encode_frame_Fxx       SKP_Silk_encode_frame_FLP
#endif
#define SKP_Silk_EncodeControlStruct    SKP_SILK_SDK_EncControlStruct

/**************************/
/* Encode frame with Silk */
/**************************/
static SKP_int process_enc_control_struct( 
    SKP_Silk_encoder_state_Fxx          *psEnc,         /* I/O: State                                           */
    SKP_Silk_EncodeControlStruct        *encControl     /* I:   Control structure                               */
)
{
    SKP_int   max_internal_fs_kHz, min_internal_fs_kHz, Complexity, PacketSize_ms, PacketLoss_perc, UseInBandFEC, ret = SKP_SILK_NO_ERROR;
    SKP_int32 TargetRate_bps, API_fs_Hz;

    SKP_assert( encControl != NULL );

    /* Check sampling frequency first, to avoid divide by zero later */
    if( ( ( encControl->API_sampleRate        !=  8000 ) &&
          ( encControl->API_sampleRate        != 12000 ) &&
          ( encControl->API_sampleRate        != 16000 ) &&
          ( encControl->API_sampleRate        != 24000 ) && 
          ( encControl->API_sampleRate        != 32000 ) &&
          ( encControl->API_sampleRate        != 44100 ) &&
          ( encControl->API_sampleRate        != 48000 ) ) ||
        ( ( encControl->maxInternalSampleRate !=  8000 ) &&
          ( encControl->maxInternalSampleRate != 12000 ) &&
          ( encControl->maxInternalSampleRate != 16000 ) ) ||
        ( ( encControl->minInternalSampleRate !=  8000 ) &&
          ( encControl->minInternalSampleRate != 12000 ) &&
          ( encControl->minInternalSampleRate != 16000 ) ) ||
          ( encControl->minInternalSampleRate > encControl->maxInternalSampleRate ) ) {
        ret = SKP_SILK_ENC_FS_NOT_SUPPORTED;
        SKP_assert( 0 );
        return( ret );
    }
    if( encControl->useDTX < 0 || encControl->useDTX > 1 ) {
        ret = SKP_SILK_ENC_INVALID_DTX_SETTING;
    }
	if( encControl->useCBR < 0 || encControl->useCBR > 1 ) {
        ret = SKP_SILK_ENC_INVALID_CBR_SETTING;
    }

    /* Set encoder parameters from control structure */
    API_fs_Hz           =            encControl->API_sampleRate;
    max_internal_fs_kHz = (SKP_int)( encControl->maxInternalSampleRate >> 10 ) + 1;   /* convert Hz -> kHz */
    min_internal_fs_kHz = (SKP_int)( encControl->minInternalSampleRate >> 10 ) + 1;   /* convert Hz -> kHz */
    PacketSize_ms       =            encControl->payloadSize_ms;
    TargetRate_bps      =            encControl->bitRate;
    PacketLoss_perc     =            encControl->packetLossPercentage;
    UseInBandFEC        =            encControl->useInBandFEC;
    Complexity          =            encControl->complexity;
    psEnc->sCmn.useDTX  =            encControl->useDTX;
	psEnc->sCmn.useCBR  =			 encControl->useCBR;

    /* Save values in state */
    psEnc->sCmn.API_fs_Hz          = API_fs_Hz;
    psEnc->sCmn.maxInternal_fs_kHz = max_internal_fs_kHz;
    psEnc->sCmn.minInternal_fs_kHz = min_internal_fs_kHz;
    psEnc->sCmn.useInBandFEC       = UseInBandFEC;

    TargetRate_bps = SKP_LIMIT( TargetRate_bps, MIN_TARGET_RATE_BPS, MAX_TARGET_RATE_BPS );
    if( ( ret = SKP_Silk_control_encoder_Fxx( psEnc, PacketSize_ms, TargetRate_bps, 
                        PacketLoss_perc, Complexity) ) != 0 ) {
        SKP_assert( 0 );
        return( ret );
    }

    encControl->internalSampleRate = SKP_SMULBB( psEnc->sCmn.fs_kHz, 1000 );
    
    return ret;
}

/****************************************/
/* Encoder functions                    */
/****************************************/

SKP_int SKP_Silk_SDK_Get_Encoder_Size( SKP_int32 *encSizeBytes )
{
    SKP_int ret = SKP_SILK_NO_ERROR;
    
    *encSizeBytes = sizeof( SKP_Silk_encoder_state_Fxx );
    
    return ret;
}

/*************************/
/* Init or Reset encoder */
/*************************/
SKP_int SKP_Silk_SDK_InitEncoder(
    void                            *encState,          /* I/O: State                                           */
    SKP_Silk_EncodeControlStruct    *encStatus          /* O:   Control structure                               */
)
{
    SKP_Silk_encoder_state_Fxx *psEnc;
    SKP_int ret = SKP_SILK_NO_ERROR;

        
    psEnc = ( SKP_Silk_encoder_state_Fxx* )encState;

    /* Reset Encoder */
    if( ret += SKP_Silk_init_encoder_Fxx( psEnc ) ) {
        SKP_assert( 0 );
    }

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
    SKP_Silk_EncodeControlStruct *encStatus     /* O:   Control Structure                               */
)
{
    SKP_Silk_encoder_state_Fxx *psEnc;
    SKP_int ret = SKP_SILK_NO_ERROR;

    psEnc = ( SKP_Silk_encoder_state_Fxx* )encState;

    encStatus->API_sampleRate        = psEnc->sCmn.API_fs_Hz;
    encStatus->maxInternalSampleRate = SKP_SMULBB( psEnc->sCmn.maxInternal_fs_kHz, 1000 );
    encStatus->minInternalSampleRate = SKP_SMULBB( psEnc->sCmn.minInternal_fs_kHz, 1000 );
    encStatus->payloadSize_ms        = psEnc->sCmn.PacketSize_ms;
    encStatus->bitRate               = psEnc->sCmn.TargetRate_bps;
    encStatus->packetLossPercentage  = psEnc->sCmn.PacketLoss_perc;
    encStatus->complexity            = psEnc->sCmn.Complexity;
    encStatus->useInBandFEC          = psEnc->sCmn.useInBandFEC;
    encStatus->useDTX                = psEnc->sCmn.useDTX;
    encStatus->useCBR                = psEnc->sCmn.useCBR;
    encStatus->internalSampleRate    = SKP_SMULBB( psEnc->sCmn.fs_kHz, 1000 );
    return ret;
}

/**************************/
/* Encode frame with Silk */
/**************************/
SKP_int SKP_Silk_SDK_Encode( 
    void                                *encState,      /* I/O: State                                           */
    SKP_Silk_EncodeControlStruct        *encControl,    /* I:   Control structure                               */
    const SKP_int16                     *samplesIn,     /* I:   Speech sample input vector                      */
    SKP_int                             nSamplesIn,     /* I:   Number of samples in input vector               */
    ec_enc                              *psRangeEnc,    /* I/O  Compressor data structure                       */
    SKP_int32                           *nBytesOut,     /* I/O: Number of bytes in payload (input: Max bytes)   */
    const SKP_int                       prefillFlag     /* I:   Flag to indicate prefilling buffers no coding   */
)
{
    SKP_int   tmp_payloadSize_ms, tmp_complexity, ret = 0;
    SKP_int   nSamplesToBuffer, nBlocksOf10ms, nSamplesFromInput = 0;
    SKP_Silk_encoder_state_Fxx *psEnc = ( SKP_Silk_encoder_state_Fxx* )encState;

    ret = process_enc_control_struct( psEnc, encControl );

    nBlocksOf10ms = SKP_DIV32( 100 * nSamplesIn, psEnc->sCmn.API_fs_Hz );
    if( prefillFlag ) {
        /* Only accept input length of 10 ms */
        if( nBlocksOf10ms != 1 ) {
            ret = SKP_SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
            SKP_assert( 0 );
            return( ret );
        }
        /* Reset Encoder */
        if( ret = SKP_Silk_init_encoder_Fxx( psEnc ) ) {
            SKP_assert( 0 );
        }
        tmp_payloadSize_ms = encControl->payloadSize_ms;
        encControl->payloadSize_ms = 10;
        tmp_complexity = encControl->complexity;
        encControl->complexity = 0;
        ret = process_enc_control_struct( psEnc, encControl );
        psEnc->sCmn.prefillFlag = 1;
    } else {
        /* Only accept input lengths that are a multiple of 10 ms */
        if( nBlocksOf10ms * psEnc->sCmn.API_fs_Hz != 100 * nSamplesIn || nSamplesIn < 0 ) {
            ret = SKP_SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
            SKP_assert( 0 );
            return( ret );
        }
        /* Make sure no more than one packet can be produced */
        if( 1000 * (SKP_int32)nSamplesIn > psEnc->sCmn.PacketSize_ms * psEnc->sCmn.API_fs_Hz ) {
            ret = SKP_SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
            SKP_assert( 0 );
            return( ret );
        }
    }

    /* Input buffering/resampling and encoding */
    while( 1 ) {
        nSamplesToBuffer = psEnc->sCmn.frame_length - psEnc->sCmn.inputBufIx;
        if( psEnc->sCmn.API_fs_Hz == SKP_SMULBB( 1000, psEnc->sCmn.fs_kHz ) ) { 
            nSamplesToBuffer  = SKP_min_int( nSamplesToBuffer, nSamplesIn );
            nSamplesFromInput = nSamplesToBuffer;
            /* Copy to buffer */
            SKP_memcpy( &psEnc->sCmn.inputBuf[ psEnc->sCmn.inputBufIx ], samplesIn, nSamplesFromInput * sizeof( SKP_int16 ) );
        } else {  
            nSamplesToBuffer  = SKP_min( nSamplesToBuffer, 10 * nBlocksOf10ms * psEnc->sCmn.fs_kHz );
            nSamplesFromInput = SKP_DIV32_16( nSamplesToBuffer * psEnc->sCmn.API_fs_Hz, psEnc->sCmn.fs_kHz * 1000 );
            /* Resample and write to buffer */
            ret += SKP_Silk_resampler( &psEnc->sCmn.resampler_state, &psEnc->sCmn.inputBuf[ psEnc->sCmn.inputBufIx ], samplesIn, nSamplesFromInput );
        } 
        samplesIn              += nSamplesFromInput;
        nSamplesIn             -= nSamplesFromInput;
        psEnc->sCmn.inputBufIx += nSamplesToBuffer;

        /* Silk encoder */
        if( psEnc->sCmn.inputBufIx >= psEnc->sCmn.frame_length ) {
            SKP_assert( psEnc->sCmn.inputBufIx == psEnc->sCmn.frame_length );

            /* Enough data in input buffer, so encode */
            if( ( ret = SKP_Silk_encode_frame_Fxx( psEnc, nBytesOut, psRangeEnc ) ) != 0 ) {
                SKP_assert( 0 );
            }
            psEnc->sCmn.inputBufIx = 0;
            psEnc->sCmn.controlled_since_last_payload = 0;

            if( nSamplesIn == 0 ) {
                break;
            }
        } else {
            break;
        }
    }

    if( prefillFlag ) {
        encControl->payloadSize_ms = tmp_payloadSize_ms;
        encControl->complexity = tmp_complexity;
        ret = process_enc_control_struct( psEnc, encControl );
        psEnc->sCmn.prefillFlag = 0;
    }

    return ret;
}

