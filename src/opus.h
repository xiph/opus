/* Copyright (c) 2010 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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

#ifndef OPUS_H
#define OPUS_H

#include "opus_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) && defined(OPUS_BUILD)
#define OPUS_EXPORT __attribute__ ((visibility ("default")))
#elif defined(WIN32)
#define OPUS_EXPORT __declspec(dllexport)
#else
#define OPUS_EXPORT
#endif

#define __check_int(x) (((void)((x) == (int)0)), (int)(x))
#define __check_int_ptr(ptr) ((ptr) + ((ptr) - (int*)(ptr)))

/* Error codes */
/** No error */
#define OPUS_OK                0
/** An (or more) invalid argument (e.g. out of range) */
#define OPUS_BAD_ARG          -1
/** The mode struct passed is invalid */
#define OPUS_BUFFER_TOO_SMALL -2
/** An internal error was detected */
#define OPUS_INTERNAL_ERROR   -3
/** The data passed (e.g. compressed data to decoder) is corrupted */
#define OPUS_CORRUPTED_DATA   -4
/** Invalid/unsupported request number */
#define OPUS_UNIMPLEMENTED    -5
/** An encoder or decoder structure is invalid or already freed */
#define OPUS_INVALID_STATE    -6
/** Memory allocation has failed */
#define OPUS_ALLOC_FAIL       -7

#define OPUS_BITRATE_AUTO       -1

#define OPUS_APPLICATION_VOIP        2000
#define OPUS_APPLICATION_AUDIO       2001

#define OPUS_SIGNAL_AUTO             3000
#define OPUS_SIGNAL_VOICE            3001
#define OPUS_SIGNAL_MUSIC            3002

#define MODE_SILK_ONLY          1000
#define MODE_HYBRID             1001
#define MODE_CELT_ONLY          1002

#define OPUS_BANDWIDTH_AUTO          1100
#define OPUS_BANDWIDTH_NARROWBAND    1101
#define OPUS_BANDWIDTH_MEDIUMBAND    1102
#define OPUS_BANDWIDTH_WIDEBAND      1103
#define OPUS_BANDWIDTH_SUPERWIDEBAND 1104
#define OPUS_BANDWIDTH_FULLBAND      1105



#define OPUS_SET_MODE_REQUEST 0
#define OPUS_SET_MODE(x) OPUS_SET_MODE_REQUEST, __check_int(x)
#define OPUS_GET_MODE_REQUEST 1
#define OPUS_GET_MODE(x) OPUS_GET_MODE_REQUEST, __check_int_ptr(x)

#define OPUS_SET_BITRATE_REQUEST 2
#define OPUS_SET_BITRATE(x) OPUS_SET_BITRATE_REQUEST, __check_int(x)
#define OPUS_GET_BITRATE_REQUEST 3
#define OPUS_GET_BITRATE(x) OPUS_GET_BITRATE_REQUEST, __check_int_ptr(x)

#define OPUS_SET_VBR_FLAG_REQUEST 6
#define OPUS_SET_VBR_FLAG(x) OPUS_SET_VBR_FLAG_REQUEST, __check_int(x)
#define OPUS_GET_VBR_FLAG_REQUEST 7
#define OPUS_GET_VBR_FLAG(x) OPUS_GET_VBR_FLAG_REQUEST, __check_int_ptr(x)

#define OPUS_SET_BANDWIDTH_REQUEST 8
#define OPUS_SET_BANDWIDTH(x) OPUS_SET_BANDWIDTH_REQUEST, __check_int(x)
#define OPUS_GET_BANDWIDTH_REQUEST 9
#define OPUS_GET_BANDWIDTH(x) OPUS_GET_BANDWIDTH_REQUEST, __check_int_ptr(x)

#define OPUS_SET_COMPLEXITY_REQUEST 10
#define OPUS_SET_COMPLEXITY(x) OPUS_SET_COMPLEXITY_REQUEST, __check_int(x)
#define OPUS_GET_COMPLEXITY_REQUEST 11
#define OPUS_GET_COMPLEXITY(x) OPUS_GET_COMPLEXITY_REQUEST, __check_int_ptr(x)

#define OPUS_SET_INBAND_FEC_FLAG_REQUEST 12
#define OPUS_SET_INBAND_FEC_FLAG(x) OPUS_SET_INBAND_FEC_FLAG_REQUEST, __check_int(x)
#define OPUS_GET_INBAND_FEC_FLAG_REQUEST 13
#define OPUS_GET_INBAND_FEC_FLAG(x) OPUS_GET_INBAND_FEC_FLAG_REQUEST, __check_int_ptr(x)

#define OPUS_SET_PACKET_LOSS_PERC_REQUEST 14
#define OPUS_SET_PACKET_LOSS_PERC(x) OPUS_SET_PACKET_LOSS_PERC_REQUEST, __check_int(x)
#define OPUS_GET_PACKET_LOSS_PERC_REQUEST 15
#define OPUS_GET_PACKET_LOSS_PERC(x) OPUS_GET_PACKET_LOSS_PERC_REQUEST, __check_int_ptr(x)

#define OPUS_SET_DTX_FLAG_REQUEST 16
#define OPUS_SET_DTX_FLAG(x) OPUS_SET_DTX_FLAG_REQUEST, __check_int(x)
#define OPUS_GET_DTX_FLAG_REQUEST 17
#define OPUS_GET_DTX_FLAG(x) OPUS_GET_DTX_FLAG_REQUEST, __check_int_ptr(x)

#define OPUS_SET_VOICE_RATIO_REQUEST 18
#define OPUS_SET_VOICE_RATIO(x) OPUS_SET_VOICE_RATIO_REQUEST, __check_int(x)
#define OPUS_GET_VOICE_RATIO_REQUEST 19
#define OPUS_GET_VOICE_RATIO(x) OPUS_GET_VOICE_RATIO_REQUEST, __check_int_ptr(x)

#define OPUS_SET_VBR_CONSTRAINT_REQUEST 20
#define OPUS_SET_VBR_CONSTRAINT(x) OPUS_SET_VBR_CONSTRAINT_REQUEST, __check_int(x)
#define OPUS_GET_VBR_CONSTRAINT_REQUEST 21
#define OPUS_GET_VBR_CONSTRAINT(x) OPUS_GET_VBR_CONSTRAINT_REQUEST, __check_int_ptr(x)

#define OPUS_SET_FORCE_MONO_REQUEST 22
#define OPUS_SET_FORCE_MONO(x) OPUS_SET_FORCE_MONO_REQUEST, __check_int(x)
#define OPUS_GET_FORCE_MONO_REQUEST 23
#define OPUS_GET_FORCE_MONO(x) OPUS_GET_FORCE_MONO_REQUEST, __check_int_ptr(x)

#define OPUS_SET_SIGNAL_REQUEST 24
#define OPUS_SET_SIGNAL(x) OPUS_SET_SIGNAL_REQUEST, __check_int(x)
#define OPUS_GET_SIGNAL_REQUEST 25
#define OPUS_GET_SIGNAL(x) OPUS_GET_SIGNAL_REQUEST, __check_int_ptr(x)

#define OPUS_GET_LOOKAHEAD_REQUEST 27
#define OPUS_GET_LOOKAHEAD(x) OPUS_GET_LOOKAHEAD_REQUEST, __check_int_ptr(x)

typedef struct OpusEncoder OpusEncoder;
typedef struct OpusDecoder OpusDecoder;

/*
 * There are two coding modes:
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
 */

/* Returns initialized encoder state */
OPUS_EXPORT OpusEncoder *opus_encoder_create(
    int Fs,                     /* Sampling rate of input signal (Hz) */
    int channels,               /* Number of channels (1/2) in input signal */
    int application             /* Coding mode (OPUS_APPLICATION_VOIP/OPUS_APPLICATION_AUDIO) */
);

OPUS_EXPORT OpusEncoder *opus_encoder_init(
    OpusEncoder *st,            /* Encoder state */
    int Fs,                     /* Sampling rate of input signal (Hz) */
    int channels,               /* Number of channels (1/2) in input signal */
    int application             /* Coding mode (OPUS_APPLICATION_VOIP/OPUS_APPLICATION_AUDIO) */
);

/* Returns length of the data payload (in bytes) */
OPUS_EXPORT int opus_encode(
    OpusEncoder *st,            /* Encoder state */
    const opus_int16 *pcm,      /* Input signal (interleaved if 2 channels). length is frame_size*channels */
    int frame_size,             /* Number of samples per frame of input signal */
    unsigned char *data,        /* Output payload (no more than max_data_bytes long) */
    int max_data_bytes          /* Allocated memory for payload; don't use for controlling bitrate */
);

OPUS_EXPORT void opus_encoder_destroy(OpusEncoder *st);

OPUS_EXPORT int opus_encoder_ctl(OpusEncoder *st, int request, ...);

OPUS_EXPORT OpusDecoder *opus_decoder_create(
    int Fs,                     /* Sampling rate of output signal (Hz) */
    int channels                /* Number of channels (1/2) in output signal */
);

OPUS_EXPORT OpusDecoder *opus_decoder_init(OpusDecoder *st,
    int Fs,                     /* Sampling rate of output signal (Hz) */
    int channels                /* Number of channels (1/2) in output signal */
);

/* Returns the number of samples decoded or a negative error code */
OPUS_EXPORT int opus_decode(
    OpusDecoder *st,            /* Decoder state */
    const unsigned char *data,  /* Input payload. Use a NULL pointer to indicate packet loss */
    int len,                    /* Number of bytes in payload */
    opus_int16 *pcm,            /* Output signal (interleaved if 2 channels). length is frame_size*channels */
    int frame_size,             /* Number of samples per frame of input signal */
    int decode_fec              /* Flag (0/1) to request that any in-band forward error correction data be */
                                /* decoded. If no such data is available the frame is decoded as if it were lost. */
);

OPUS_EXPORT int opus_decoder_ctl(OpusDecoder *st, int request, ...);

OPUS_EXPORT void opus_decoder_destroy(OpusDecoder *st);

OPUS_EXPORT int opus_packet_get_bandwidth(const unsigned char *data);
OPUS_EXPORT int opus_packet_get_samples_per_frame(const unsigned char *data, int Fs);
OPUS_EXPORT int opus_packet_get_nb_channels(const unsigned char *data);
OPUS_EXPORT int opus_packet_get_nb_frames(const unsigned char packet[], int len);
OPUS_EXPORT int opus_decoder_get_nb_samples(const OpusDecoder *dec, const unsigned char packet[], int len);

OPUS_EXPORT const char *opus_strerror(int error);

OPUS_EXPORT const char *opus_get_version_string(void);

/* For testing purposes: the encoder and decoder state should
   always be identical after coding a payload */
OPUS_EXPORT int opus_encoder_get_final_range(OpusEncoder *st);
OPUS_EXPORT int opus_decoder_get_final_range(OpusDecoder *st);


#ifdef __cplusplus
}
#endif

#endif /* OPUS_H */
