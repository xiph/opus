/* (C) 2007-2008 Jean-Marc Valin, CSIRO
*/
/**
  @file celt.h
  @brief Contains all the functions for encoding and decoding audio streams
 */

/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CELT_H
#define CELT_H

#include "celt_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
/** No error */
#define CELT_OK                0
/** An (or more) invalid argument (e.g. out of range) */
#define CELT_BAD_ARG          -1
/** The mode struct passed is invalid */
#define CELT_INVALID_MODE     -2
/** An internal error was detected */
#define CELT_INTERNAL_ERROR   -3
/** The data passed (e.g. compressed data to decoder) is corrupted */
#define CELT_CORRUPTED_DATA   -4

/* Requests */
/** GET the frame size used in the current mode */
#define CELT_GET_FRAME_SIZE   1000
/** GET the lookahead used in the current mode */
#define CELT_GET_LOOKAHEAD    1001
/** GET the number of channels used in the current mode */
#define CELT_GET_NB_CHANNELS  1002

/** GET the bit-stream version for compatibility check */
#define CELT_GET_BITSTREAM_VERSION 2000


/** Contains the state of an encoder. One encoder state is needed for each 
    stream. It is initialised once at the beginning of the stream. Do *not*
    re-initialise the state for every frame.
   @brief Encoder state
 */
typedef struct CELTEncoder CELTEncoder;

/** State of the decoder. One decoder state is needed for each stream. It is
    initialised once at the beginning of the stream. Do *not* re-initialise
    the state for every frame */
typedef struct CELTDecoder CELTDecoder;

/** The mode contains all the information necessary to create an encoder. Both
    the encoder and decoder need to be initialised with exactly the same mode,
    otherwise the quality will be very bad */
typedef struct CELTMode CELTMode;


/** \defgroup codec Encoding and decoding */
/*  @{ */

/* Mode calls */

/** Creates a new mode struct. This will be passed to an encoder or decoder.
    The mode MUST NOT BE DESTROYED until the encoders and decoders that use it
    are destroyed as well.
 @param Fs Sampling rate (32000 to 64000 Hz)
 @param channels Number of channels
 @param frame_size Number of samples (per channel) to encode in each packet (64 - 256)
 @param lookahead Extra latency (in samples per channel) in addition to the frame size (between 32 and frame_size). The larger that value, the better the quality (at the expense of latency)
 @param error Returned error code (if NULL, no error will be returned)
 @return A newly created mode
*/
CELTMode *celt_mode_create(celt_int32_t Fs, int channels, int frame_size, int lookahead, int *error);

/** Destroys a mode struct. Only call this after all encoders and decoders
    using this mode are destroyed as well.
 @param mode Mode to be destroyed
*/
void celt_mode_destroy(CELTMode *mode);

/** Query information from a mode */
int celt_mode_info(const CELTMode *mode, int request, celt_int32_t *value);


/* Encoder stuff */


/** Creates a new encoder state. Each stream needs its own encoder state (can't
    be shared across simultaneous streams).
 @param mode Contains all the information about the characteristics of the stream
             (must be the same characteristics as used for the decoder)
 @return Newly created encoder state.
*/
CELTEncoder *celt_encoder_create(const CELTMode *mode);

/** Destroys a an encoder state.
 @param st Encoder state to be destroyed
 */
void celt_encoder_destroy(CELTEncoder *st);

/** Encodes a frame of audio.
 @param st Encoder state
 @param pcm PCM audio in signed 16-bit format (native endian). There must be 
            exactly frame_size samples per channel. The input data is 
            overwritten by a copy of what the remote decoder would decode.
 @param compressed The compressed data is written here
 @param nbCompressedBytes Number of bytes to use for compressing the frame
                          (can change from one frame to another)
 @return Number of bytes written to "compressed". Should be the same as 
         "nbCompressedBytes" unless the stream is VBR. If negative, an error
         has occured (see error codes). It is IMPORTANT that the length returned
         be somehow transmitted to the decoder. Otherwise, no decoding is possible.
*/
int celt_encode(CELTEncoder *st, celt_int16_t *pcm, unsigned char *compressed, int nbCompressedBytes);

/* Decoder stuff */


/** Creates a new decoder state. Each stream needs its own decoder state (can't
    be shared across simultaneous streams).
 @param mode Contains all the information about the characteristics of the
             stream (must be the same characteristics as used for the encoder)
 @return Newly created decoder state.
 */
CELTDecoder *celt_decoder_create(const CELTMode *mode);

/** Destroys a a decoder state.
 @param st Decoder state to be destroyed
 */
void celt_decoder_destroy(CELTDecoder *st);

/** Decodes a frame of audio.
 @param st Decoder state
 @param data Compressed data produced by an encoder
 @param len Number of bytes to read from "data". This MUST be exactly the number
            of bytes returned by the encoder. Using a larger value WILL NOT WORK.
 @param pcm One frame (frame_size samples per channel) of decoded PCM will be
            returned here. 
 @return Error code.
   */
int celt_decode(CELTDecoder *st, unsigned char *data, int len, celt_int16_t *pcm);

/*  @} */


#ifdef __cplusplus
}
#endif

#endif /*CELT_H */
