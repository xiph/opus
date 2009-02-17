/* (C) 2007-2008 Jean-Marc Valin, CSIRO
   (C) 2008 Gregory Maxwell */
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

#if defined(__GNUC__) && defined(CELT_BUILD)
#define EXPORT __attribute__ ((visibility ("default")))
#elif defined(WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define _celt_check_int(x) (((void)((x) == (int)0)), (int)(x))

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
/** Invalid/unsupported request number */
#define CELT_UNIMPLEMENTED    -5

/* Requests */
#define CELT_SET_COMPLEXITY_REQUEST    2
/** Controls the complexity from 0-10 (int) */
#define CELT_SET_COMPLEXITY(x) CELT_SET_COMPLEXITY_REQUEST, _celt_check_int(x)
#define CELT_SET_LTP_REQUEST    3
/** Activate or deactivate the use of the long term predictor (PITCH) from 0 or 1 (int) */
#define CELT_SET_LTP(x) CELT_SET_LTP_REQUEST, _celt_check_int(x)

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
 @param Fs Sampling rate (32000 to 96000 Hz)
 @param channels Number of channels
 @param frame_size Number of samples (per channel) to encode in each packet (even values; 64 - 512)
 @param lookahead Extra latency (in samples per channel) in addition to the frame size (between 32 and frame_size). 
 @param error Returned error code (if NULL, no error will be returned)
 @return A newly created mode
*/
EXPORT CELTMode *celt_mode_create(celt_int32_t Fs, int channels, int frame_size, int *error);

/** Destroys a mode struct. Only call this after all encoders and decoders
    using this mode are destroyed as well.
 @param mode Mode to be destroyed
*/
EXPORT void celt_mode_destroy(CELTMode *mode);

/** Query information from a mode */
EXPORT int celt_mode_info(const CELTMode *mode, int request, celt_int32_t *value);

/* Encoder stuff */


/** Creates a new encoder state. Each stream needs its own encoder state (can't
    be shared across simultaneous streams).
 @param mode Contains all the information about the characteristics of the stream
             (must be the same characteristics as used for the decoder)
 @return Newly created encoder state.
*/
EXPORT CELTEncoder *celt_encoder_create(const CELTMode *mode);

/** Destroys a an encoder state.
 @param st Encoder state to be destroyed
 */
EXPORT void celt_encoder_destroy(CELTEncoder *st);

/** Encodes a frame of audio.
 @param st Encoder state
 @param pcm PCM audio in float format, with a normal range of ±1.0. 
 *          Samples with a range beyond ±1.0 are supported but will be clipped by 
 *          decoders using the integer API and should only be used if it is known that
 *          the far end supports extended dynmaic range. There must be exactly
 *          frame_size samples per channel. 
 @param optional_synthesis If not NULL, the encoder copies the audio signal that
 *                         the decoder would decode. It is the same as calling the
 *                         decoder on the compressed data, just faster.
 *                         This may alias pcm. 
 @param compressed The compressed data is written here. This may not alias pcm or
 *                         optional_synthesis.
 @param nbCompressedBytes Maximum number of bytes to use for compressing the frame
 *                        (can change from one frame to another)
 @return Number of bytes written to "compressed". Will be the same as 
 *       "nbCompressedBytes" unless the stream is VBR and will never be larger.
 *       If negative, an error has occurred (see error codes). It is IMPORTANT that
 *       the length returned be somehow transmitted to the decoder. Otherwise, no
 *       decoding is possible.
*/
EXPORT int celt_encode_float(CELTEncoder *st, const float *pcm, float *optional_synthesis, unsigned char *compressed, int nbCompressedBytes);

/** Encodes a frame of audio.
 @param st Encoder state
 @param pcm PCM audio in signed 16-bit format (native endian). There must be 
 *          exactly frame_size samples per channel. 
 @param optional_synthesis If not NULL, the encoder copies the audio signal that
 *                         the decoder would decode. It is the same as calling the
 *                         decoder on the compressed data, just faster.
 *                         This may alias pcm. 
 @param compressed The compressed data is written here. This may not alias pcm or
 *                         optional_synthesis.
 @param nbCompressedBytes Maximum number of bytes to use for compressing the frame
 *                        (can change from one frame to another)
 @return Number of bytes written to "compressed". Will be the same as 
 *       "nbCompressedBytes" unless the stream is VBR and will never be larger.
 *       If negative, an error has occurred (see error codes). It is IMPORTANT that
 *       the length returned be somehow transmitted to the decoder. Otherwise, no
 *       decoding is possible.
 */
EXPORT int celt_encode(CELTEncoder *st, const celt_int16_t *pcm, celt_int16_t *optional_synthesis, unsigned char *compressed, int nbCompressedBytes);

/** Query and set encoder parameters 
 @param st Encoder state
 @param request Parameter to change or query
 @param value Pointer to a 32-bit int value
 @return Error code
*/
EXPORT int celt_encoder_ctl(CELTEncoder * st, int request, ...);

/* Decoder stuff */


/** Creates a new decoder state. Each stream needs its own decoder state (can't
    be shared across simultaneous streams).
 @param mode Contains all the information about the characteristics of the
             stream (must be the same characteristics as used for the encoder)
 @return Newly created decoder state.
 */
EXPORT CELTDecoder *celt_decoder_create(const CELTMode *mode);

/** Destroys a a decoder state.
 @param st Decoder state to be destroyed
 */
EXPORT void celt_decoder_destroy(CELTDecoder *st);

/** Decodes a frame of audio.
 @param st Decoder state
 @param data Compressed data produced by an encoder
 @param len Number of bytes to read from "data". This MUST be exactly the number
            of bytes returned by the encoder. Using a larger value WILL NOT WORK.
 @param pcm One frame (frame_size samples per channel) of decoded PCM will be
            returned here in float format. 
 @return Error code.
   */
EXPORT int celt_decode_float(CELTDecoder *st, unsigned char *data, int len, float *pcm);

/** Decodes a frame of audio.
 @param st Decoder state
 @param data Compressed data produced by an encoder
 @param len Number of bytes to read from "data". This MUST be exactly the number
            of bytes returned by the encoder. Using a larger value WILL NOT WORK.
 @param pcm One frame (frame_size samples per channel) of decoded PCM will be
            returned here in 16-bit PCM format (native endian). 
 @return Error code.
 */
EXPORT int celt_decode(CELTDecoder *st, unsigned char *data, int len, celt_int16_t *pcm);

/*  @} */


#ifdef __cplusplus
}
#endif

#endif /*CELT_H */
