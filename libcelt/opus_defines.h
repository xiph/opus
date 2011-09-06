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

#ifndef OPUS_DEFINES_H
#define OPUS_DEFINES_H

#include "opus_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) && defined(OPUS_BUILD)

# define OPUS_EXPORT __attribute__ ((visibility ("default")))

#elif defined(WIN32)

# ifdef OPUS_BUILD
#   define OPUS_EXPORT __declspec(dllexport)
# else
#   define OPUS_EXPORT __declspec(dllimport)
# endif

#else

# define OPUS_EXPORT

#endif

#define __opus_check_int(x) (((void)((x) == (opus_int32)0)), (opus_int32)(x))
#define __opus_check_int_ptr(ptr) ((ptr) + ((ptr) - (opus_int32*)(ptr)))
#define __opus_check_uint_ptr(ptr) ((ptr) + ((ptr) - (opus_uint32*)(ptr)))

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


#define OPUS_BITRATE_AUTO       -2
#define OPUS_BITRATE_MAX       -1

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


/* OPUS_APPLICATION_VOIP or OPUS_APPLICATION_AUDIO */
#define OPUS_SET_APPLICATION_REQUEST 4000
#define OPUS_SET_APPLICATION(x) OPUS_SET_APPLICATION_REQUEST, __opus_check_int(x)
#define OPUS_GET_APPLICATION_REQUEST 4001
#define OPUS_GET_APPLICATION(x) OPUS_GET_APPLICATION_REQUEST, __opus_check_int_ptr(x)

/* Coding bit-rate in bit/second */
#define OPUS_SET_BITRATE_REQUEST 4002
#define OPUS_SET_BITRATE(x) OPUS_SET_BITRATE_REQUEST, __opus_check_int(x)
#define OPUS_GET_BITRATE_REQUEST 4003
#define OPUS_GET_BITRATE(x) OPUS_GET_BITRATE_REQUEST, __opus_check_int_ptr(x)

/* 0 for CBR, 1 for VBR */
#define OPUS_SET_VBR_REQUEST 4006
#define OPUS_SET_VBR(x) OPUS_SET_VBR_REQUEST, __opus_check_int(x)
#define OPUS_GET_VBR_REQUEST 4007
#define OPUS_GET_VBR(x) OPUS_GET_VBR_REQUEST, __opus_check_int_ptr(x)

#define OPUS_SET_BANDWIDTH_REQUEST 4008
#define OPUS_SET_BANDWIDTH(x) OPUS_SET_BANDWIDTH_REQUEST, __opus_check_int(x)
#define OPUS_GET_BANDWIDTH_REQUEST 4009
#define OPUS_GET_BANDWIDTH(x) OPUS_GET_BANDWIDTH_REQUEST, __opus_check_int_ptr(x)

#define OPUS_SET_COMPLEXITY_REQUEST 4010
#define OPUS_SET_COMPLEXITY(x) OPUS_SET_COMPLEXITY_REQUEST, __opus_check_int(x)
#define OPUS_GET_COMPLEXITY_REQUEST 4011
#define OPUS_GET_COMPLEXITY(x) OPUS_GET_COMPLEXITY_REQUEST, __opus_check_int_ptr(x)

#define OPUS_SET_INBAND_FEC_REQUEST 4012
#define OPUS_SET_INBAND_FEC(x) OPUS_SET_INBAND_FEC_REQUEST, __opus_check_int(x)
#define OPUS_GET_INBAND_FEC_REQUEST 4013
#define OPUS_GET_INBAND_FEC(x) OPUS_GET_INBAND_FEC_REQUEST, __opus_check_int_ptr(x)

#define OPUS_SET_PACKET_LOSS_PERC_REQUEST 4014
#define OPUS_SET_PACKET_LOSS_PERC(x) OPUS_SET_PACKET_LOSS_PERC_REQUEST, __opus_check_int(x)
#define OPUS_GET_PACKET_LOSS_PERC_REQUEST 4015
#define OPUS_GET_PACKET_LOSS_PERC(x) OPUS_GET_PACKET_LOSS_PERC_REQUEST, __opus_check_int_ptr(x)

#define OPUS_SET_DTX_REQUEST 4016
#define OPUS_SET_DTX(x) OPUS_SET_DTX_REQUEST, __opus_check_int(x)
#define OPUS_GET_DTX_REQUEST 4017
#define OPUS_GET_DTX(x) OPUS_GET_DTX_REQUEST, __opus_check_int_ptr(x)

#define OPUS_SET_VOICE_RATIO_REQUEST 4018
#define OPUS_SET_VOICE_RATIO(x) OPUS_SET_VOICE_RATIO_REQUEST, __opus_check_int(x)
#define OPUS_GET_VOICE_RATIO_REQUEST 4019
#define OPUS_GET_VOICE_RATIO(x) OPUS_GET_VOICE_RATIO_REQUEST, __opus_check_int_ptr(x)

#define OPUS_SET_VBR_CONSTRAINT_REQUEST 4020
#define OPUS_SET_VBR_CONSTRAINT(x) OPUS_SET_VBR_CONSTRAINT_REQUEST, __opus_check_int(x)
#define OPUS_GET_VBR_CONSTRAINT_REQUEST 4021
#define OPUS_GET_VBR_CONSTRAINT(x) OPUS_GET_VBR_CONSTRAINT_REQUEST, __opus_check_int_ptr(x)

#define OPUS_SET_FORCE_MONO_REQUEST 4022
#define OPUS_SET_FORCE_MONO(x) OPUS_SET_FORCE_MONO_REQUEST, __opus_check_int(x)
#define OPUS_GET_FORCE_MONO_REQUEST 4023
#define OPUS_GET_FORCE_MONO(x) OPUS_GET_FORCE_MONO_REQUEST, __opus_check_int_ptr(x)

#define OPUS_SET_SIGNAL_REQUEST 4024
#define OPUS_SET_SIGNAL(x) OPUS_SET_SIGNAL_REQUEST, __opus_check_int(x)
#define OPUS_GET_SIGNAL_REQUEST 4025
#define OPUS_GET_SIGNAL(x) OPUS_GET_SIGNAL_REQUEST, __opus_check_int_ptr(x)

#define OPUS_GET_LOOKAHEAD_REQUEST 4027
#define OPUS_GET_LOOKAHEAD(x) OPUS_GET_LOOKAHEAD_REQUEST, __opus_check_int_ptr(x)

#define OPUS_RESET_STATE 4028

/* For testing purposes: the encoder and decoder state should
   always be identical after coding a payload */
#define OPUS_GET_FINAL_RANGE_REQUEST 4031
#define OPUS_GET_FINAL_RANGE(x) OPUS_GET_FINAL_RANGE_REQUEST, __opus_check_uint_ptr(x)



OPUS_EXPORT const char *opus_strerror(int error);

OPUS_EXPORT const char *opus_get_version_string(void);


#ifdef __cplusplus
}
#endif

#endif /* OPUS_H */
