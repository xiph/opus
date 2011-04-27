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

#ifndef OPUS_ENCODER_H
#define OPUS_ENCODER_H

#include "celt.h"
#include "opus.h"
#include "SKP_Silk_SDK_API.h"

/* FIXME: This is only valid for 48 kHz */
#define MAX_ENCODER_BUFFER 480

struct OpusEncoder {
	CELTEncoder *celt_enc;
	SKP_SILK_SDK_EncControlStruct silk_mode;
	void        *silk_enc;
	int          channels;
	int          stream_channels;

    int          mode;
    int          user_mode;
    int          prev_mode;
	int          bandwidth;
	int          user_bandwidth;
	int          voice_ratio;
    /* Sampling rate (at the API level) */
    int          Fs;
    int          use_vbr;
    int          vbr_constraint;
    int          bitrate_bps;
    int          encoder_buffer;
    int          delay_compensation;
    int          bandwidth_change;
    short        delay_buffer[MAX_ENCODER_BUFFER*2];

#ifdef OPUS_TEST_RANGE_CODER_STATE
    int          rangeFinal;
#endif
};


#endif /* OPUS_ENCODER_H */

