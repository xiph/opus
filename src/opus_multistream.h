/* Copyright (c) 2011 Xiph.Org Foundation
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


#ifndef OPUS_MULTISTREAM_H
#define OPUS_MULTISTREAM_H

#include "opus.h"

typedef struct OpusMSEncoder OpusMSEncoder;
typedef struct OpusMSDecoder OpusMSDecoder;

OPUS_EXPORT OpusMSEncoder *opus_multistream_encoder_create(
      int Fs,                     /* Sampling rate of input signal (Hz) */
      int channels,               /* Number of channels (1/2) in input signal */
      int streams,
      int coupled_streams,
      unsigned char *mapping,
      int application,            /* Coding mode (OPUS_APPLICATION_VOIP/OPUS_APPLICATION_AUDIO) */
      int *error                  /* Error code */
);

OPUS_EXPORT int opus_multistream_encoder_init(
      OpusMSEncoder *st,            /* Encoder state */
      int Fs,                     /* Sampling rate of input signal (Hz) */
      int channels,               /* Number of channels (1/2) in input signal */
      int streams,
      int coupled_streams,
      unsigned char *mapping,
      int application             /* Coding mode (OPUS_APPLICATION_VOIP/OPUS_APPLICATION_AUDIO) */
);

/* Returns length of the data payload (in bytes) */
OPUS_EXPORT int opus_multistream_encode(
    OpusMSEncoder *st,            /* Encoder state */
    const opus_int16 *pcm,      /* Input signal (interleaved if 2 channels). length is frame_size*channels */
    int frame_size,             /* Number of samples per frame of input signal */
    unsigned char *data,        /* Output payload (no more than max_data_bytes long) */
    int max_data_bytes          /* Allocated memory for payload; don't use for controlling bitrate */
);

/* Returns length of the data payload (in bytes) */
OPUS_EXPORT int opus_multistream_encode_float(
      OpusMSEncoder *st,            /* Encoder state */
      const float *pcm,      /* Input signal (interleaved if 2 channels). length is frame_size*channels */
      int frame_size,             /* Number of samples per frame of input signal */
      unsigned char *data,        /* Output payload (no more than max_data_bytes long) */
      int max_data_bytes          /* Allocated memory for payload; don't use for controlling bitrate */
  );

OPUS_EXPORT void opus_multistream_encoder_destroy(OpusMSEncoder *st);

OPUS_EXPORT int opus_multistream_encoder_ctl(OpusMSEncoder *st, int request, ...);

OPUS_EXPORT OpusMSDecoder *opus_multistream_decoder_create(
      int Fs,                     /* Sampling rate of input signal (Hz) */
      int channels,               /* Number of channels (1/2) in input signal */
      int streams,
      int coupled_streams,
      unsigned char *mapping,
      int *error                  /* Error code */
);

OPUS_EXPORT int opus_multistream_decoder_init(
      OpusMSDecoder *st,            /* Encoder state */
      int Fs,                     /* Sampling rate of input signal (Hz) */
      int channels,               /* Number of channels (1/2) in input signal */
      int streams,
      int coupled_streams,
      unsigned char *mapping
);

/* Returns the number of samples decoded or a negative error code */
OPUS_EXPORT int opus_multistream_decode(
    OpusMSDecoder *st,            /* Decoder state */
    const unsigned char *data,  /* Input payload. Use a NULL pointer to indicate packet loss */
    int len,                    /* Number of bytes in payload */
    opus_int16 *pcm,            /* Output signal (interleaved if 2 channels). length is frame_size*channels */
    int frame_size,             /* Number of samples per frame of input signal */
    int decode_fec              /* Flag (0/1) to request that any in-band forward error correction data be */
                                /* decoded. If no such data is available the frame is decoded as if it were lost. */
);

/* Returns the number of samples decoded or a negative error code */
OPUS_EXPORT int opus_multistream_decode_float(
    OpusMSDecoder *st,            /* Decoder state */
    const unsigned char *data,  /* Input payload. Use a NULL pointer to indicate packet loss */
    int len,                    /* Number of bytes in payload */
    float *pcm,                 /* Output signal (interleaved if 2 channels). length is frame_size*channels */
    int frame_size,             /* Number of samples per frame of input signal */
    int decode_fec              /* Flag (0/1) to request that any in-band forward error correction data be */
                                /* decoded. If no such data is available the frame is decoded as if it were lost. */
);

OPUS_EXPORT int opus_multistream_decoder_ctl(OpusMSDecoder *st, int request, ...);

OPUS_EXPORT void opus_multistream_decoder_destroy(OpusMSDecoder *st);

#endif /* OPUS_MULTISTREAM_H */
