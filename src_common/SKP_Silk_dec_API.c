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

#include "SKP_Silk_SDK_API.h"
#include "SKP_Silk_main.h"

/*********************/
/* Decoder functions */
/*********************/

SKP_int SKP_Silk_SDK_Get_Decoder_Size( SKP_int32 *decSizeBytes ) 
{
    SKP_int ret = SKP_SILK_NO_ERROR;

    *decSizeBytes = sizeof( SKP_Silk_decoder_state );

    return ret;
}

/* Reset decoder state */
SKP_int SKP_Silk_SDK_InitDecoder(
    void* decState                                      /* I/O: State                                          */
)
{
    SKP_int ret = SKP_SILK_NO_ERROR;
    SKP_Silk_decoder_state *struc;

    struc = (SKP_Silk_decoder_state *)decState;

    ret  = SKP_Silk_init_decoder( struc );

    return ret;
}

/* Prefill LPC synthesis buffer, HP filter and upsampler. Input must be exactly 10 ms of audio. */
SKP_int SKP_Silk_SDK_Decoder_prefill_buffers(           /* O:   Returns error code                              */
    void*                               decState,       /* I/O: State                                           */
    SKP_SILK_SDK_DecControlStruct*      decControl,     /* I/O: Control Structure                               */
    const SKP_int16                     *samplesIn,     /* I:   Speech sample input vector  (10 ms)             */
    SKP_int                             nSamplesIn      /* I:   Number of samples in input vector               */
)
{
    SKP_int   i, nSamples, ret = 0;
    SKP_Silk_decoder_state *psDec = ( SKP_Silk_decoder_state *)decState;
    SKP_Silk_resampler_state_struct resampler_state;
    SKP_int16 buf[ 10 * MAX_FS_KHZ ];
    SKP_int16 buf_out[ 10 * MAX_API_FS_KHZ ];
    const SKP_int16 *in_ptr;

    /* Compute some numbers at API sampling rate */
    if( nSamplesIn != SKP_DIV32_16( decControl->API_sampleRate, 100 ) ) {
        return -1;
    }

    /* Resample input if necessary */
    if( decControl->API_sampleRate != SKP_SMULBB( 1000, psDec->fs_kHz ) ) { 
        ret += SKP_Silk_resampler_init( &resampler_state, decControl->API_sampleRate, SKP_SMULBB( 1000, psDec->fs_kHz ) );
        ret += SKP_Silk_resampler( &resampler_state, buf, samplesIn, nSamplesIn );
        in_ptr = buf;
        nSamples = SKP_SMULBB( 10, psDec->fs_kHz );
    } else {
        in_ptr = samplesIn;
        nSamples = nSamplesIn;
    }

    /* Set synthesis filter state */
    for( i = 0; i < psDec->LPC_order; i++ ) {
        psDec->sLPC_Q14[ MAX_LPC_ORDER - i ] = SKP_LSHIFT( SKP_SMULWB( psDec->prev_inv_gain_Q16, in_ptr[ nSamples - i ] ), 14 );
    }

    /* HP filter */
    SKP_Silk_biquad_alt( in_ptr, psDec->HP_B, psDec->HP_A, psDec->HPState, buf, nSamples );

    /* Output upsampler */
    SKP_Silk_resampler( &psDec->resampler_state, buf_out, buf, nSamples );

    /* Avoid using LSF interpolation or pitch prediction in first next frame */
    psDec->first_frame_after_reset = 1;

    return ret;
}

/* Decode a frame */
SKP_int SKP_Silk_SDK_Decode(
    void*                               decState,       /* I/O: State                                           */
    SKP_SILK_SDK_DecControlStruct*      decControl,     /* I/O: Control Structure                               */
    SKP_int                             lostFlag,       /* I:   0: no loss, 1 loss, 2 decode fec                */
    SKP_int                             newPacketFlag,  /* I:   Indicates first decoder call for this packet    */
    ec_dec                              *psRangeDec,    /* I/O  Compressor data structure                       */
    const SKP_int                       nBytesIn,       /* I:   Number of input bytes                           */
    SKP_int16                           *samplesOut,    /* O:   Decoded output speech vector                    */
    SKP_int32                           *nSamplesOut    /* O:   Number of samples decoded                       */
)
{
    SKP_int ret = SKP_SILK_NO_ERROR, prev_fs_kHz;
    SKP_Silk_decoder_state *psDec;

    psDec = (SKP_Silk_decoder_state *)decState;

    /**********************************/
    /* Test if first frame in payload */
    /**********************************/
    if( newPacketFlag ) {
        /* First Frame in Payload */
        psDec->nFramesDecoded = 0;  /* Used to count frames in packet */
    }

    /* Save previous sample frequency */
    prev_fs_kHz = psDec->fs_kHz;

    if( psDec->nFramesDecoded == 0 ) {
        SKP_int fs_kHz_dec;
        if( decControl->payloadSize_ms == 10 ) {
            psDec->nFramesPerPacket = 1;
            psDec->nb_subfr = 2;
        } else if( decControl->payloadSize_ms == 20 ) {
            psDec->nFramesPerPacket = 1;
            psDec->nb_subfr = 4;
        } else if( decControl->payloadSize_ms == 40 ) {
            psDec->nFramesPerPacket = 2;
            psDec->nb_subfr = 4;
        } else if( decControl->payloadSize_ms == 60 ) {
            psDec->nFramesPerPacket = 3;
            psDec->nb_subfr = 4;
        } else {
            SKP_assert( 0 );
            return SKP_SILK_DEC_INVALID_FRAME_SIZE;
        } 
        fs_kHz_dec = ( decControl->internalSampleRate >> 10 ) + 1;
        if( fs_kHz_dec != 8 && fs_kHz_dec != 12 && fs_kHz_dec != 16 ) {
            SKP_assert( 0 );
            return SKP_SILK_DEC_INVALID_SAMPLING_FREQUENCY;
        }
        SKP_Silk_decoder_set_fs( psDec, fs_kHz_dec );
    }
    
    /* Call decoder for one frame */
    ret += SKP_Silk_decode_frame( psDec, psRangeDec, samplesOut, nSamplesOut, nBytesIn, lostFlag );
    
    if( decControl->API_sampleRate > MAX_API_FS_KHZ * 1000 || decControl->API_sampleRate < 8000 ) {
        ret = SKP_SILK_DEC_INVALID_SAMPLING_FREQUENCY;
        return( ret );
    }

    /* Resample if needed */
    if( SKP_SMULBB( psDec->fs_kHz, 1000 ) != decControl->API_sampleRate ) { 
        SKP_int16 samplesOut_tmp[ MAX_API_FS_KHZ * MAX_FRAME_LENGTH_MS ];
        SKP_assert( psDec->fs_kHz <= MAX_API_FS_KHZ );

        /* Copy to a tmp buffer as the resampling writes to samplesOut */
        SKP_memcpy( samplesOut_tmp, samplesOut, *nSamplesOut * sizeof( SKP_int16 ) );

        /* (Re-)initialize resampler state when switching internal sampling frequency */
        if( prev_fs_kHz != psDec->fs_kHz || psDec->prev_API_sampleRate != decControl->API_sampleRate ) {
            ret = SKP_Silk_resampler_init( &psDec->resampler_state, SKP_SMULBB( psDec->fs_kHz, 1000 ), decControl->API_sampleRate );
        }

        /* Resample the output to API_sampleRate */
        ret += SKP_Silk_resampler( &psDec->resampler_state, samplesOut, samplesOut_tmp, *nSamplesOut );

        /* Update the number of output samples */
        *nSamplesOut = SKP_DIV32( ( SKP_int32 )*nSamplesOut * decControl->API_sampleRate, SKP_SMULBB( psDec->fs_kHz, 1000 ) );
    }

    psDec->prev_API_sampleRate = decControl->API_sampleRate;

    /* Copy all parameters that are needed out of internal structure to the control stucture */
    decControl->frameSize                 = ( SKP_int )*nSamplesOut;
    decControl->framesPerPayload          = ( SKP_int )psDec->nFramesPerPacket;

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

    flags = SKP_RSHIFT( payload[ 0 ], 7 - nFramesPerPayload ) & ( SKP_LSHIFT( 1, nFramesPerPayload + 1 ) - 1 );

    Silk_TOC->inbandFECFlag = flags & 1;
    for( i = nFramesPerPayload - 1; i >= 0 ; i-- ) {
        flags = SKP_RSHIFT( flags, 1 );
        Silk_TOC->VADFlags[ i ] = flags & 1;
        Silk_TOC->VADFlag |= flags & 1;
    }

    return ret;
}
