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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include "hybrid_encoder.h"
#include "celt/libcelt/entenc.h"
#include "celt/libcelt/modes.h"
#include "SKP_Silk_SDK_API.h"

HybridEncoder *hybrid_encoder_create()
{
	HybridEncoder *st;
	int ret, encSizeBytes;

	st = malloc(sizeof(HybridEncoder));

    /* Create SILK encoder */
    ret = SKP_Silk_SDK_Get_Encoder_Size( &encSizeBytes );
    if( ret ) {
    	/* Handle error */
    }
	st->silk_enc = malloc(encSizeBytes);

    ret = SKP_Silk_SDK_InitEncoder( st->silk_enc, &st->encControl );
    if( ret ) {
        /* Handle error */
    }
    /* Set Encoder parameters */
    st->encControl.API_sampleRate        = 48000;
    st->encControl.maxInternalSampleRate = 16000;
    st->encControl.packetSize            = 960;
    st->encControl.packetLossPercentage  = 0;
    st->encControl.useInBandFEC          = 0;
    st->encControl.useDTX                = 0;
    st->encControl.complexity            = 2;
    st->encControl.bitRate               = 20000;

    /* Create CELT encoder */
	/* We should not have to create a CELT mode for each encoder state */
	st->celt_mode = celt_mode_create(48000, 960, NULL);
	/* Initialize CELT encoder */
	st->celt_enc = celt_encoder_create(st->celt_mode, 1, NULL);

	return st;
}

int hybrid_encode(HybridEncoder *st, const short *pcm, int frame_size,
		unsigned char *data, int bytes_per_packet)
{
	int silk_ret, celt_ret;
	SKP_int16 nBytes;
	ec_enc enc;
	ec_byte_buffer buf;

	ec_byte_writeinit_buffer(&buf, data, bytes_per_packet);
	ec_enc_init(&enc,&buf);

	/* FIXME: Call SILK encoder for the low band */
	silk_ret = SKP_Silk_SDK_Encode( st->silk_enc, &st->encControl, pcm, 960, &enc, &nBytes );
    if( silk_ret ) {
        /* Handle error */
    }

	/* This should be adjusted based on the SILK bandwidth */
	celt_encoder_ctl(st->celt_enc, CELT_SET_START_BAND(13));

	/* Encode high band with CELT */
	celt_ret = celt_encode_with_ec(st->celt_enc, pcm, NULL, frame_size, data, bytes_per_packet, &enc);

	return celt_ret;
}

void hybrid_encoder_destroy(HybridEncoder *st)
{
	free(st->silk_enc);

	celt_encoder_destroy(st->celt_enc);
	celt_mode_destroy(st->celt_mode);

	free(st);
}
