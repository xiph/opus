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

#ifndef HYBRID_H
#define HYBRID_H

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

#define MODE_SILK_ONLY 1000
#define MODE_HYBRID    1001
#define MODE_CELT_ONLY 1002

#define BANDWIDTH_NARROWBAND    1100
#define BANDWIDTH_MEDIUMBAND    1101
#define BANDWIDTH_WIDEBAND      1102
#define BANDWIDTH_SUPERWIDEBAND 1103
#define BANDWIDTH_FULLBAND      1104



#define HYBRID_SET_MODE_REQUEST 0
#define HYBRID_SET_MODE(x) HYBRID_SET_MODE_REQUEST, __check_int(x)
#define HYBRID_GET_MODE_REQUEST 1
#define HYBRID_GET_MODE(x) HYBRID_GET_MODE_REQUEST, __check_int_ptr(x)

#define HYBRID_SET_BANDWIDTH_REQUEST 2
#define HYBRID_SET_BANDWIDTH(x) HYBRID_SET_BANDWIDTH_REQUEST, __check_int(x)
#define HYBRID_GET_BANDWIDTH_REQUEST 3
#define HYBRID_GET_BANDWIDTH(x) HYBRID_GET_BANDWIDTH_REQUEST, __check_int_ptr(x)

#define HYBRID_SET_VBR_RATE_REQUEST 4
#define HYBRID_SET_VBR_RATE(x) HYBRID_SET_VBR_RATE_REQUEST, __check_int(x)
#define HYBRID_GET_VBR_RATE_REQUEST 5
#define HYBRID_GET_VBR_RATE(x) HYBRID_GET_VBR_RATE_REQUEST, __check_int_ptr(x)

typedef struct HybridEncoder HybridEncoder;
typedef struct HybridDecoder HybridDecoder;

HybridEncoder *hybrid_encoder_create(int Fs);

int hybrid_encode(HybridEncoder *st, const short *pcm, int frame_size,
		unsigned char *data, int bytes_per_packet);

void hybrid_encoder_destroy(HybridEncoder *st);

void hybrid_encoder_ctl(HybridEncoder *st, int request, ...);

HybridDecoder *hybrid_decoder_create(int Fs);

int hybrid_decode(HybridDecoder *st, const unsigned char *data, int len,
		short *pcm, int frame_size);

void hybrid_decoder_ctl(HybridDecoder *st, int request, ...);

void hybrid_decoder_destroy(HybridDecoder *st);

#ifdef __cplusplus
}
#endif

#endif /* HYBRID_H */
