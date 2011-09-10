/* Copyright (c) 2010-2011 Xiph.Org Foundation, Skype Limited
   Written by Jean-Marc Valin and Koen Vos */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

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

/**
 * @file opus.h
 * @brief Opus reference implementation API
 */

#ifndef OPUS_H
#define OPUS_H

#include "opus_types.h"
#include "opus_defines.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup opusencoder Opus Encoder
  * @{
  */

/** Opus encoder state.
  * This contains the complete state of an Opus encoder.
  * It is position independent and can be freely copied.
  * @see opus_encoder_create,opus_encoder_init
  */
typedef struct OpusEncoder OpusEncoder;

OPUS_EXPORT int opus_encoder_get_size(int channels);

/**
 */

/** Allocates and initializes an encoder state.
 * There are three coding modes:
 * OPUS_APPLICATION_VOIP gives best quality at a given bitrate for voice
 *    signals. It enhances the  input signal by high-pass filtering and
 *    emphasizing formants and harmonics. Optionally  it includes in-band
 *    forward error correction to protect against packet loss. Use this
 *    mode for typical VoIP applications. Because of the enhancement,
 *    even at high bitrates the output may sound different from the input.
 * OPUS_APPLICATION_AUDIO gives best quality at a given bitrate for most
 *    non-voice signals like music. Use this mode for music and mixed
 *    (music/voice) content, broadcast, and applications requiring less
 *    than 15 ms of coding delay.
 * OPUS_APPLICATION_RESTRICTED_LOWDELAY configures low-delay mode that
 *    disables the speech-optimized mode in exchange for slightly reduced delay.
 * This is useful when the caller knows that the speech-optimized modes will not be needed (use with caution).
 * @param [in] Fs <tt>opus_int32</tt>: Sampling rate of input signal (Hz)
 * @param [in] channels <tt>int</tt>: Number of channels (1/2) in input signal
 * @param [in] application <tt>int</tt>: Coding mode (OPUS_APPLICATION_VOIP/OPUS_APPLICATION_AUDIO/OPUS_APPLICATION_RESTRICTED_LOWDELAY)
 * @param [out] error <tt>int*</tt>: Error code
 */
OPUS_EXPORT OpusEncoder *opus_encoder_create(
    opus_int32 Fs,
    int channels,
    int application,
    int *error
);

/** Initializes a previously allocated encoder state
  * The memory pointed to by st must be the size returned by opus_encoder_get_size.
  * This is intended for applications which use their own allocator instead of malloc. @see opus_encoder_create,opus_encoder_get_size
  * To reset a previously initialized state use the OPUS_RESET_STATE CTL.
  * @param [in] st <tt>OpusEncoder*</tt>: Encoder state
  * @param [in] Fs <tt>opus_int32</tt>: Sampling rate of input signal (Hz)
  * @param [in] channels <tt>int</tt>: Number of channels (1/2) in input signal
  * @param [in] application <tt>int</tt>: Coding mode (OPUS_APPLICATION_VOIP/OPUS_APPLICATION_AUDIO/OPUS_APPLICATION_RESTRICTED_LOWDELAY)
  * @retval OPUS_OK Success.
  */
OPUS_EXPORT int opus_encoder_init(
    OpusEncoder *st,
    opus_int32 Fs,
    int channels,
    int application
);

/** Encodes an Opus frame.
  * The passed frame_size must an opus frame size for the encoder's sampling rate.
  * For example, at 48kHz the permitted values are 120, 240, 480, or 960.
  * Passing in a duration of less than 10ms (480 samples at 48kHz) will
  * prevent the encoder from using the LPC or hybrid modes.
  * @param [in] st <tt>OpusEncoder*</tt>: Encoder state
  * @param [in] pcm <tt>opus_int16*</tt>: Input signal (interleaved if 2 channels). length is frame_size*channels*sizeof(opus_int16)
  * @param [in] frame_size <tt>int</tt>: Number of samples per frame of input signal
  * @param [out] data <tt>char*</tt>: Output payload (at least max_data_bytes long)
  * @param [in] max_data_bytes <tt>int</tt>: Allocated memory for payload; don't use for controlling bitrate
  * @returns length of the data payload (in bytes)
  */
OPUS_EXPORT int opus_encode(
    OpusEncoder *st,
    const opus_int16 *pcm,
    int frame_size,
    unsigned char *data,
    int max_data_bytes
);

/** Encodes an Opus frame from floating point input.
  * The passed frame_size must an opus frame size for the encoder's sampling rate.
  * For example, at 48kHz the permitted values are 120, 240, 480, or 960.
  * Passing in a duration of less than 10ms (480 samples at 48kHz) will
  * prevent the encoder from using the LPC or hybrid modes.
  * @param [in] st <tt>OpusEncoder*</tt>: Encoder state
  * @param [in] pcm <tt>float*</tt>: Input signal (interleaved if 2 channels). length is frame_size*channels*sizeof(float)
  * @param [in] frame_size <tt>int</tt>: Number of samples per frame of input signal
  * @param [out] data <tt>char*</tt>: Output payload (at least max_data_bytes long)
  * @param [in] max_data_bytes <tt>int</tt>: Allocated memory for payload; don't use for controlling bitrate
  * @returns length of the data payload (in bytes)
  */
OPUS_EXPORT int opus_encode_float(
    OpusEncoder *st,
    const float *pcm,
    int frame_size,
    unsigned char *data,
    int max_data_bytes
);

/** Frees an OpusEncoder allocated by opus_encoder_create.
  * @param[in] st <tt>OpusEncoder*</tt>: State to be freed.
  */
OPUS_EXPORT void opus_encoder_destroy(OpusEncoder *st);

/** Perform a CTL function on an Opus encoder.
  * @see encoderctls
  */
OPUS_EXPORT int opus_encoder_ctl(OpusEncoder *st, int request, ...);
/**@}*/

/** @defgroup opusdecoder Opus Decoder
  * @{
  */

/** Opus decoder state.
  * This contains the complete state of an Opus decoder.
  * It is position independent and can be freely copied.
  * @see opus_decoder_create,opus_decoder_init
  */
typedef struct OpusDecoder OpusDecoder;

/** Gets the size of an OpusDecoder structure.
  * @param [in] channels <tt>int</tt>: Number of channels
  * @returns size
  */
OPUS_EXPORT int opus_decoder_get_size(int channels);

/** Allocates and initializes a decoder state.
  * @param [in] Fs <tt>opus_int32</tt>: Sampling rate of input signal (Hz)
  * @param [in] channels <tt>int</tt>: Number of channels (1/2) in input signal
  * @param [out] error <tt>int*</tt>: Error code
  */
OPUS_EXPORT OpusDecoder *opus_decoder_create(
    opus_int32 Fs,
    int channels,
    int *error
);

/** Initializes a previously allocated decoder state.
  * The state must be the size returned by opus_decoder_get_size.
  * This is intended for applications which use their own allocator instead of malloc. @see opus_decoder_create,opus_decoder_get_size
  * To reset a previously initialized state use the OPUS_RESET_STATE CTL.
  * @param [in] st <tt>OpusDecoder*</tt>: Decoder state.
  * @param [in] Fs <tt>opus_int32</tt>: Sampling rate of input signal (Hz)
  * @param [in] channels <tt>int</tt>: Number of channels (1/2) in input signal
  * @retval OPUS_OK Success.
  */
OPUS_EXPORT int opus_decoder_init(
    OpusDecoder *st,
    opus_int32 Fs,
    int channels
);

/** Decode an Opus frame
  * @param [in] st <tt>OpusDecoder*</tt>: Decoder state
  * @param [in] data <tt>char*</tt>: Input payload. Use a NULL pointer to indicate packet loss
  * @param [in] len <tt>int</tt>: Number of bytes in payload*
  * @param [out] pcm <tt>opus_int16*</tt>: Output signal (interleaved if 2 channels). length
  *  is frame_size*channels*sizeof(opus_int16)
  * @param [in] frame_size Number of samples per channel of available space in *pcm,
  *  if less than the maximum frame size (120ms) some frames can not be decoded
  * @param [in] decode_fec <tt>int</tt>: Flag (0/1) to request that any in-band forward error correction data be
  *  decoded. If no such data is available the frame is decoded as if it were lost.
  * @returns Number of decoded samples
  */
OPUS_EXPORT int opus_decode(
    OpusDecoder *st,
    const unsigned char *data,
    int len,
    opus_int16 *pcm,
    int frame_size,
    int decode_fec
);

/** Decode an opus frame with floating point output
  * @param [in] st <tt>OpusDecoder*</tt>: Decoder state
  * @param [in] data <tt>char*</tt>: Input payload. Use a NULL pointer to indicate packet loss
  * @param [in] len <tt>int</tt>: Number of bytes in payload
  * @param [out] pcm <tt>float*</tt>: Output signal (interleaved if 2 channels). length
  *  is frame_size*channels*sizeof(float)
  * @param [in] frame_size Number of samples per channel of available space in *pcm,
  *  if less than the maximum frame size (120ms) some frames can not be decoded
  * @param [in] decode_fec <tt>int</tt>: Flag (0/1) to request that any in-band forward error correction data be
  *  decoded. If no such data is available the frame is decoded as if it were lost.
  * @returns Number of decoded samples
  */
OPUS_EXPORT int opus_decode_float(
    OpusDecoder *st,
    const unsigned char *data,
    int len,
    float *pcm,
    int frame_size,
    int decode_fec
);

/** Perform a CTL function on an Opus decoder.
  * @see decoderctls
  */
OPUS_EXPORT int opus_decoder_ctl(OpusDecoder *st, int request, ...);

/** Frees an OpusDecoder allocated by opus_decoder_create.
  * @param[in] st <tt>OpusDecoder*</tt>: State to be freed.
  */
OPUS_EXPORT void opus_decoder_destroy(OpusDecoder *st);

/** Parse an opus packet into one or more frames.
  * Opus_decode will perform this operation internally so most applications do
  * not need to use this function.
  * This function does not copy the frames, the returned pointers are pointers into
  * the input packet.
  * @param [in] data <tt>char*</tt>: Opus packet to be parsed
  * @param [in] len <tt>int</tt>: size of data
  * @param [out] out_toc <tt>char*</tt>: TOC pointer
  * @param [out] frames <tt>char*[48]</tt> encapsulated frames
  * @param [out] size <tt>short[48]</tt> sizes of the encapsulated frames
  * @param [out] payload_offset <tt>int*</tt>: returns the position of the payload within the packet (in bytes)
  * @returns number of frames
  */
OPUS_EXPORT int opus_packet_parse(
   const unsigned char *data,
   int len,
   unsigned char *out_toc,
   const unsigned char *frames[48],
   short size[48],
   int *payload_offset
);

/** Gets the bandwidth of an Opus packet.
  * @param [in] data <tt>char*</tt>: Opus packet
  * @retval OPUS_BANDWIDTH_NARROWBAND Narrowband (4kHz bandpass)
  * @retval OPUS_BANDWIDTH_MEDIUMBAND Mediumband (6kHz bandpass)
  * @retval OPUS_BANDWIDTH_WIDEBAND Wideband (8kHz bandpass)
  * @retval OPUS_BANDWIDTH_SUPERWIDEBAND Superwideband (12kHz bandpass)
  * @retval OPUS_BANDWIDTH_FULLBAND Fullband (20kHz bandpass)
  * @retval OPUS_INVALID_PACKET The compressed data passed is corrupted or of an unsupported type
  */
OPUS_EXPORT int opus_packet_get_bandwidth(const unsigned char *data);

/** Gets the number of samples per frame from an Opus packet.
  * @param [in] data <tt>char*</tt>: Opus packet
  * @param [in] Fs <tt>opus_int32</tt>: Sampling rate in Hz
  * @returns Number of samples per frame
  * @retval OPUS_INVALID_PACKET The compressed data passed is corrupted or of an unsupported type
  */
OPUS_EXPORT int opus_packet_get_samples_per_frame(const unsigned char *data, opus_int32 Fs);

/** Gets the number of channels from an Opus packet.
  * @param [in] data <tt>char*</tt>: Opus packet
  * @returns Number of channels
  * @retval OPUS_INVALID_PACKET The compressed data passed is corrupted or of an unsupported type
  */
OPUS_EXPORT int opus_packet_get_nb_channels(const unsigned char *data);

/** Gets the number of frame in an Opus packet.
  * @param [in] packet <tt>char*</tt>: Opus packet
  * @param [in] len <tt>int</tt>: Length of packet
  * @returns Number of frames
  * @retval OPUS_INVALID_PACKET The compressed data passed is corrupted or of an unsupported type
  */
OPUS_EXPORT int opus_packet_get_nb_frames(const unsigned char packet[], int len);

/** Gets the number of samples of an Opus packet.
  * @param [in] dec <tt>OpusDecoder*</tt>: Decoder state
  * @param [in] packet <tt>char*</tt>: Opus packet
  * @param [in] len <tt>int</tt>: Length of packet
  * @returns Number of samples
  * @retval OPUS_INVALID_PACKET The compressed data passed is corrupted or of an unsupported type
  */
OPUS_EXPORT int opus_decoder_get_nb_samples(const OpusDecoder *dec, const unsigned char packet[], int len);
/**@}*/

/** @defgroup repacketizer Repacketizer
  * @{
  */

typedef struct OpusRepacketizer OpusRepacketizer;

OPUS_EXPORT int opus_repacketizer_get_size(void);

OPUS_EXPORT OpusRepacketizer *opus_repacketizer_init(OpusRepacketizer *rp);

OPUS_EXPORT OpusRepacketizer *opus_repacketizer_create(void);

OPUS_EXPORT void opus_repacketizer_destroy(OpusRepacketizer *rp);

OPUS_EXPORT int opus_repacketizer_cat(OpusRepacketizer *rp, const unsigned char *data, int len);

OPUS_EXPORT int opus_repacketizer_out_range(OpusRepacketizer *rp, int begin, int end, unsigned char *data, int maxlen);

OPUS_EXPORT int opus_repacketizer_get_nb_frames(OpusRepacketizer *rp);

OPUS_EXPORT int opus_repacketizer_out(OpusRepacketizer *rp, unsigned char *data, int maxlen);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* OPUS_H */
