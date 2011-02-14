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

#define __check_int(x) (((void)((x) == (int)0)), (int)(x))
#define __check_int_ptr(ptr) ((ptr) + ((ptr) - (int*)(ptr)))

#define OPUS_TEST_RANGE_CODER_STATE     1

#define MODE_SILK_ONLY 1000
#define MODE_HYBRID    1001
#define MODE_CELT_ONLY 1002

#define BANDWIDTH_NARROWBAND    1100
#define BANDWIDTH_MEDIUMBAND    1101
#define BANDWIDTH_WIDEBAND      1102
#define BANDWIDTH_SUPERWIDEBAND 1103
#define BANDWIDTH_FULLBAND      1104



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

typedef struct OpusEncoder OpusEncoder;
typedef struct OpusDecoder OpusDecoder;

OpusEncoder *opus_encoder_create(int Fs, int channels);

/* returns length of data payload (in bytes) */
int opus_encode(OpusEncoder *st, const short *pcm, int frame_size,
		unsigned char *data, int max_data_bytes);

void opus_encoder_destroy(OpusEncoder *st);

void opus_encoder_ctl(OpusEncoder *st, int request, ...);

OpusDecoder *opus_decoder_create(int Fs, int channels);

/* returns (CELT) error code */
int opus_decode(OpusDecoder *st, const unsigned char *data, int len,
		short *pcm, int frame_size, int decode_fec);

void opus_decoder_ctl(OpusDecoder *st, int request, ...);

void opus_decoder_destroy(OpusDecoder *st);

#if OPUS_TEST_RANGE_CODER_STATE
int opus_encoder_get_final_range(OpusEncoder *st);
int opus_decoder_get_final_range(OpusDecoder *st);
#endif


#ifdef __cplusplus
}
#endif

#endif /* OPUS_H */
