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

#ifndef SKP_SILK_SDK_API_H
#define SKP_SILK_SDK_API_H

#include "SKP_Silk_control.h"
#include "SKP_Silk_typedef.h"
#include "SKP_Silk_errors.h"
#include "entenc.h"
#include "entdec.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define SILK_MAX_FRAMES_PER_PACKET  3

/* Struct for TOC (Table of Contents) */
typedef struct {
    SKP_int     VADFlag;                                /* Voice activity for packet                            */
    SKP_int     VADFlags[ SILK_MAX_FRAMES_PER_PACKET ]; /* Voice activity for each frame in packet              */
    SKP_int     inbandFECFlag;                          /* Flag indicating if packet contains in-band FEC       */
} SKP_Silk_TOC_struct;

/****************************************/
/* Encoder functions                    */
/****************************************/

/***********************************************/
/* Get size in bytes of the Silk encoder state */
/***********************************************/
SKP_int SKP_Silk_SDK_Get_Encoder_Size(                  /* O:   Returns error code                              */
    SKP_int32                           *encSizeBytes   /* O:   Number of bytes in SILK encoder state           */
);

/*************************/
/* Init or reset encoder */
/*************************/
SKP_int SKP_Silk_SDK_InitEncoder(                       /* O:   Returns error code                              */
    void                                *encState,      /* I/O: State                                           */
    SKP_SILK_SDK_EncControlStruct       *encStatus      /* O:   Encoder Status                                  */
);

/***************************************/
/* Read control structure from encoder */
/***************************************/
SKP_int SKP_Silk_SDK_QueryEncoder(                      /* O:   Returns error code                              */
    const void                          *encState,      /* I:   State                                           */
    SKP_SILK_SDK_EncControlStruct       *encStatus      /* O:   Encoder Status                                  */
);

/**************************/
/* Encode frame with Silk */
/**************************/
/* Note: if prefillFlag is set, the input must contain 10 ms of audio, irrespective of what 					*/
/* encControl->payloadSize_ms is set to 																		*/
SKP_int SKP_Silk_SDK_Encode(                            /* O:   Returns error code                              */
    void                                *encState,      /* I/O: State                                           */
    SKP_SILK_SDK_EncControlStruct       *encControl,    /* I:   Control status                                  */
    const SKP_int16                     *samplesIn,     /* I:   Speech sample input vector                      */
    SKP_int                             nSamplesIn,     /* I:   Number of samples in input vector               */
    ec_enc                              *psRangeEnc,    /* I/O  Compressor data structure                       */
    SKP_int32                           *nBytesOut,     /* I/O: Number of bytes in payload (input: Max bytes)   */
    const SKP_int                       prefillFlag     /* I:   Flag to indicate prefilling buffers no coding   */
);

/****************************************/
/* Decoder functions                    */
/****************************************/

/***********************************************/
/* Get size in bytes of the Silk decoder state */
/***********************************************/
SKP_int SKP_Silk_SDK_Get_Decoder_Size(                  /* O:   Returns error code                              */
    SKP_int32                           *decSizeBytes   /* O:   Number of bytes in SILK decoder state           */
);

/*************************/
/* Init or Reset decoder */
/*************************/
SKP_int SKP_Silk_SDK_InitDecoder(                       /* O:   Returns error code                              */
    void                                *decState       /* I/O: State                                           */
);

/************************************************************************************************/
/* Prefill LPC synthesis buffer, HP filter and upsampler. Input must be exactly 10 ms of audio. */
/************************************************************************************************/
SKP_int SKP_Silk_SDK_Decoder_prefill_buffers(           /* O:   Returns error code                              */
    void*                               decState,       /* I/O: State                                           */
    SKP_SILK_SDK_DecControlStruct*      decControl,     /* I/O: Control Structure                               */
    const SKP_int16                     *samplesIn,     /* I:   Speech sample input vector  (10 ms)             */
    SKP_int                             nSamplesIn      /* I:   Number of samples in input vector               */
);

/******************/
/* Decode a frame */
/******************/
SKP_int SKP_Silk_SDK_Decode(                            /* O:   Returns error code                              */
    void*                               decState,       /* I/O: State                                           */
    SKP_SILK_SDK_DecControlStruct*      decControl,     /* I/O: Control Structure                               */
    SKP_int                             lostFlag,       /* I:   0: no loss, 1 loss, 2 decode fec                */
    SKP_int                             newPacketFlag,  /* I:   Indicates first decoder call for this packet    */
    ec_dec                              *psRangeDec,    /* I/O  Compressor data structure                       */
    const SKP_int                       nBytesIn,       /* I:   Number of input bytes                           */
    SKP_int16                           *samplesOut,    /* O:   Decoded output speech vector                    */
    SKP_int32                           *nSamplesOut    /* O:   Number of samples decoded                       */
);

/***************************************************************/
/* Find Low Bit Rate Redundancy (LBRR) information in a packet */
/***************************************************************/
void SKP_Silk_SDK_search_for_LBRR(
    const SKP_uint8                     *inData,        /* I:   Encoded input vector                            */
    const SKP_int16                     nBytesIn,       /* I:   Number of input bytes                           */
    SKP_int                             lost_offset,    /* I:   Offset from lost packet                         */
    SKP_uint8                           *LBRRData,      /* O:   LBRR payload                                    */
    SKP_int32                           *nLBRRBytes     /* O:   Number of LBRR Bytes                            */
);

/**************************************/
/* Get table of contents for a packet */
/**************************************/
SKP_int SKP_Silk_SDK_get_TOC(
    const SKP_uint8                     *payload,           /* I    Payload data                                */
    const SKP_int                       nBytesIn,           /* I:   Number of input bytes                       */
    const SKP_int                       nFramesPerPayload,  /* I:   Number of SILK frames per payload           */
    SKP_Silk_TOC_struct                 *Silk_TOC           /* O:   Type of content                             */
);

#ifdef __cplusplus
}
#endif

#endif
