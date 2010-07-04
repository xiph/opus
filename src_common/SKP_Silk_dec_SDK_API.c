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

/* Decode a frame */
SKP_int SKP_Silk_SDK_Decode(
    void*                               decState,       /* I/O: State                                           */
    SKP_SILK_SDK_DecControlStruct*      decControl,     /* I/O: Control Structure                               */
    SKP_int                             lostFlag,       /* I:   0: no loss, 1 loss                              */
    ec_dec                              *psRangeDec,    /* I/O  Compressor data structure                       */
    const SKP_int                       nBytesIn,       /* I:   Number of input bytes                           */
    SKP_int16                           *samplesOut,    /* O:   Decoded output speech vector                    */
    SKP_int16                           *nSamplesOut    /* I/O: Number of samples (vector/decoded)              */
)
{
    SKP_int ret = SKP_SILK_NO_ERROR, used_bytes, prev_fs_kHz;
    SKP_Silk_decoder_state *psDec;

    psDec = (SKP_Silk_decoder_state *)decState;

    /**********************************/
    /* Test if first frame in payload */
    /**********************************/
    if( psDec->moreInternalDecoderFrames == 0 ) {
        /* First Frame in Payload */
        psDec->nFramesDecoded = 0;  /* Used to count frames in packet */
    }

    if( psDec->moreInternalDecoderFrames == 0 &&    /* First frame in packet    */
        lostFlag == 0 &&                            /* Not packet loss          */
        nBytesIn > MAX_ARITHM_BYTES ) {             /* Too long payload         */
            /* Avoid trying to decode a too large packet */
            lostFlag = 1;
            ret = SKP_SILK_DEC_PAYLOAD_TOO_LARGE;
    }
            
    /* Save previous sample frequency */
    prev_fs_kHz = psDec->fs_kHz;
    
    /* Call decoder for one frame */
    ret += SKP_Silk_decode_frame( psDec, psRangeDec, samplesOut, nSamplesOut, nBytesIn, lostFlag, &used_bytes );
    
    if( used_bytes ) { /* Only Call if not a packet loss */
        if( psDec->nBytesLeft > 0 && psDec->FrameTermination == SKP_SILK_MORE_FRAMES && psDec->nFramesDecoded < 5 ) {
            /* We have more frames in the Payload */
            psDec->moreInternalDecoderFrames = 1;
        } else {
            /* Last frame in Payload */
            psDec->moreInternalDecoderFrames = 0;
            psDec->nFramesInPacket = psDec->nFramesDecoded;
        
            /* Track inband FEC usage */
            if( psDec->vadFlag == VOICE_ACTIVITY ) {
                if( psDec->FrameTermination == SKP_SILK_LAST_FRAME ) {
                    psDec->no_FEC_counter++;
                    if( psDec->no_FEC_counter > NO_LBRR_THRES ) {
                        psDec->inband_FEC_offset = 0;
                    }
                } else if( psDec->FrameTermination == SKP_SILK_LBRR_VER1 ) {
                    psDec->inband_FEC_offset = 1; /* FEC info with 1 packet delay */
                    psDec->no_FEC_counter    = 0;
                } else if( psDec->FrameTermination == SKP_SILK_LBRR_VER2 ) {
                    psDec->inband_FEC_offset = 2; /* FEC info with 2 packets delay */
                    psDec->no_FEC_counter    = 0;
                }
            }
        }
    }

    if( MAX_API_FS_KHZ * 1000 < decControl->API_sampleRate ||
        8000       > decControl->API_sampleRate ) {
        ret = SKP_SILK_DEC_INVALID_SAMPLING_FREQUENCY;
        return( ret );
    }

    /* Resample if needed */
    if( psDec->fs_kHz * 1000 != decControl->API_sampleRate ) { 
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
        *nSamplesOut = SKP_DIV32( ( SKP_int32 )*nSamplesOut * decControl->API_sampleRate, psDec->fs_kHz * 1000 );
    }

    psDec->prev_API_sampleRate = decControl->API_sampleRate;

    /* Copy all parameters that are needed out of internal structure to the control stucture */
    decControl->frameSize                 = ( SKP_int )psDec->frame_length;
    decControl->framesPerPacket           = ( SKP_int )psDec->nFramesInPacket;
    decControl->inBandFECOffset           = ( SKP_int )psDec->inband_FEC_offset;
    decControl->moreInternalDecoderFrames = ( SKP_int )psDec->moreInternalDecoderFrames;

    return ret;
}

#if 0
/* Function to find LBRR information in a packet */
void SKP_Silk_SDK_search_for_LBRR(
    const SKP_uint8                     *inData,        /* I:   Encoded input vector                            */
    const SKP_int16                     nBytesIn,       /* I:   Number of input Bytes                           */
    SKP_int                             lost_offset,    /* I:   Offset from lost packet                         */
    SKP_uint8                           *LBRRData,      /* O:   LBRR payload                                    */
    SKP_int16                           *nLBRRBytes     /* O:   Number of LBRR Bytes                            */
)
{
    SKP_int ret = SKP_SILK_NO_ERROR;
    SKP_Silk_decoder_state   sDec; // Local decoder state to avoid interfering with running decoder */
    SKP_Silk_decoder_control sDecCtrl;
    SKP_int i, TempQ[ MAX_FRAME_LENGTH ];

    if( lost_offset < 1 || lost_offset > MAX_LBRR_DELAY ) {
        /* No useful FEC in this packet */
        *nLBRRBytes = 0;
        return;
    }

    sDec.nFramesDecoded = 0;
    sDec.fs_kHz         = 0; /* Force update parameters LPC_order etc */
    SKP_memset( sDec.prevNLSF_Q15, 0, MAX_LPC_ORDER * sizeof( SKP_int ) );

    /* Decode all parameter indices for the whole packet*/
    SKP_Silk_decode_indices( &sDec, psRangeDec );

    /* Is there usable LBRR in this packet */
    *nLBRRBytes = 0;
    if( ( sDec.FrameTermination - 1 ) & lost_offset && sDec.FrameTermination > 0 && sDec.nBytesLeft >= 0 ) {
        /* The wanted FEC is present in the packet */
        for( i = 0; i < sDec.nFramesInPacket; i++ ) {
            SKP_Silk_decode_parameters( &sDec, &sDecCtrl, psRangeDec, TempQ, 0 );
            
            if( sDec.nBytesLeft <= 0 || sDec.sRC.error ) {
                /* Corrupt stream */
                LBRRData = NULL;
                *nLBRRBytes = 0;
                break;
            } else {
                sDec.nFramesDecoded++;
            }
        }
    
        if( LBRRData != NULL ) {
            /* The wanted FEC is present in the packet */
            *nLBRRBytes = sDec.nBytesLeft;
            SKP_memcpy( LBRRData, &inData[ nBytesIn - sDec.nBytesLeft ], sDec.nBytesLeft * sizeof( SKP_uint8 ) );
        }
    }
}
#endif

#if 0  // todo: clean up, make efficient
/* Getting type of content for a packet */
void SKP_Silk_SDK_get_TOC(
    ec_dec                              *psRangeDec,    /* I/O  Compressor data structure                   */
    const SKP_int16                     nBytesIn,       /* I:   Number of input bytes                           */
    SKP_Silk_TOC_struct                 *Silk_TOC       /* O:   Type of content                                 */
)
{
    SKP_Silk_decoder_state      sDec; // Local Decoder state to avoid interfering with running decoder */
    SKP_int i, ret = SKP_SILK_NO_ERROR;

    sDec.nFramesDecoded = 0;
    sDec.fs_kHz         = 0; /* Force update parameters LPC_order etc */

    /* Decode all parameter indices for the whole packet*/
    SKP_Silk_decode_indices( &sDec );
    
    if( sDec.nFramesInPacket > SILK_MAX_FRAMES_PER_PACKET || sDec.sRC.error ) {
        /* Corrupt packet */
        SKP_memset( Silk_TOC, 0, sizeof( SKP_Silk_TOC_struct ) );
        Silk_TOC->corrupt = 1;
    } else {
        Silk_TOC->corrupt = 0;
        Silk_TOC->framesInPacket = sDec.nFramesInPacket;
        Silk_TOC->fs_kHz         = sDec.fs_kHz;
        if( sDec.FrameTermination == SKP_SILK_LAST_FRAME ) {
            Silk_TOC->inbandLBRR = sDec.FrameTermination;
        } else {
            Silk_TOC->inbandLBRR = sDec.FrameTermination - 1;
        }
        /* Copy data */
        for( i = 0; i < sDec.nFramesInPacket; i++ ) {
            Silk_TOC->vadFlags[ i ]     = sDec.vadFlagBuf[ i ];
            Silk_TOC->sigtypeFlags[ i ] = sDec.sigtype[ i ];
        }
    }
}
#endif

/**************************/
/* Get the version number */
/**************************/
/* Return a pointer to string specifying the version */ 
const char *SKP_Silk_SDK_get_version()
{
    static const char version[] = "1.0.4";
    return version;
}