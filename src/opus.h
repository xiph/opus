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

#ifndef OPUS_H
#define OPUS_H

#include "opus_types.h"
#include "opus_defines.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OPUS_BITRATE_AUTO       -1

#define OPUS_APPLICATION_VOIP        2000
#define OPUS_APPLICATION_AUDIO       2001

#define OPUS_SIGNAL_AUTO             3000
#define OPUS_SIGNAL_VOICE            3001
#define OPUS_SIGNAL_MUSIC            3002

#define OPUS_BANDWIDTH_AUTO          1100
#define OPUS_BANDWIDTH_NARROWBAND    1101
#define OPUS_BANDWIDTH_MEDIUMBAND    1102
#define OPUS_BANDWIDTH_WIDEBAND      1103
#define OPUS_BANDWIDTH_SUPERWIDEBAND 1104
#define OPUS_BANDWIDTH_FULLBAND      1105



typedef struct OpusEncoder OpusEncoder;
typedef struct OpusDecoder OpusDecoder;

OPUS_EXPORT int opus_encoder_get_size(int channels);

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
    opus_int32 Fs,              /* Sampling rate of input signal (Hz) */
    int channels,               /* Number of channels (1/2) in input signal */
    int application,            /* Coding mode (OPUS_APPLICATION_VOIP/OPUS_APPLICATION_AUDIO) */
    int *error                  /* Error code */
);

OPUS_EXPORT int opus_encoder_init(
    OpusEncoder *st,            /* Encoder state */
    opus_int32 Fs,              /* Sampling rate of input signal (Hz) */
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

/* Returns length of the data payload (in bytes) */
OPUS_EXPORT int opus_encode_float(
    OpusEncoder *st,            /* Encoder state */
    const float *pcm,           /* Input signal (interleaved if 2 channels). length is frame_size*channels 0dbFS range of +/-1.0*/
    int frame_size,             /* Number of samples per frame of input signal */
    unsigned char *data,        /* Output payload (no more than max_data_bytes long) */
    int max_data_bytes          /* Allocated memory for payload; don't use for controlling bitrate */
);

OPUS_EXPORT void opus_encoder_destroy(OpusEncoder *st);

OPUS_EXPORT int opus_encoder_ctl(OpusEncoder *st, int request, ...);



OPUS_EXPORT int opus_decoder_get_size(int channels);

OPUS_EXPORT OpusDecoder *opus_decoder_create(
    opus_int32 Fs,              /* Sampling rate of output signal (Hz) */
    int channels,               /* Number of channels (1/2) in output signal */
    int *error                  /* Error code*/
);

OPUS_EXPORT int opus_decoder_init(OpusDecoder *st,
    opus_int32 Fs,              /* Sampling rate of output signal (Hz) */
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

/* Returns the number of samples decoded or a negative error code */
OPUS_EXPORT int opus_decode_float(
    OpusDecoder *st,            /* Decoder state */
    const unsigned char *data,  /* Input payload. Use a NULL pointer to indicate packet loss */
    int len,                    /* Number of bytes in payload */
    float *pcm,                 /* Output signal (interleaved if 2 channels). length is frame_size*channels 0dbFS range of -/+1.0*/
    int frame_size,             /* Number of samples per frame of input signal */
    int decode_fec              /* Flag (0/1) to request that any in-band forward error correction data be */
                                /* decoded. If no such data is available the frame is decoded as if it were lost. */
);

OPUS_EXPORT int opus_decoder_ctl(OpusDecoder *st, int request, ...);

OPUS_EXPORT void opus_decoder_destroy(OpusDecoder *st);

OPUS_EXPORT int opus_packet_parse(const unsigned char *data, int len,
      unsigned char *out_toc, const unsigned char *frames[48],
      short size[48], int *payload_offset);

OPUS_EXPORT int opus_packet_get_bandwidth(const unsigned char *data);
OPUS_EXPORT int opus_packet_get_samples_per_frame(const unsigned char *data, opus_int32 Fs);
OPUS_EXPORT int opus_packet_get_nb_channels(const unsigned char *data);
OPUS_EXPORT int opus_packet_get_nb_frames(const unsigned char packet[], int len);
OPUS_EXPORT int opus_decoder_get_nb_samples(const OpusDecoder *dec, const unsigned char packet[], int len);


/* Repacketizer */
typedef struct OpusRepacketizer OpusRepacketizer;

OPUS_EXPORT int opus_repacketizer_get_size(void);

OPUS_EXPORT OpusRepacketizer *opus_repacketizer_init(OpusRepacketizer *rp);

OPUS_EXPORT OpusRepacketizer *opus_repacketizer_create(void);

OPUS_EXPORT void opus_repacketizer_destroy(OpusRepacketizer *rp);

OPUS_EXPORT int opus_repacketizer_cat(OpusRepacketizer *rp, const unsigned char *data, int len);

OPUS_EXPORT int opus_repacketizer_out_range(OpusRepacketizer *rp, int begin, int end, unsigned char *data, int maxlen);

OPUS_EXPORT int opus_repacketizer_get_nb_frames(OpusRepacketizer *rp);

OPUS_EXPORT int opus_repacketizer_out(OpusRepacketizer *rp, unsigned char *data, int maxlen);

#ifdef __cplusplus
}
#endif

#endif /* OPUS_H */
