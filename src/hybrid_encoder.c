/* Copyright (c) 2010 Xiph.Org Foundation, Skype Limited
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
#include <stdio.h>
#include <stdarg.h>
#include "hybrid_encoder.h"
#include "entenc.h"
#include "modes.h"
#include "SKP_Silk_SDK_API.h"

HybridEncoder *hybrid_encoder_create(int Fs)
{
	HybridEncoder *st;
	int ret, encSizeBytes;

	st = calloc(sizeof(HybridEncoder), 1);

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

    st->Fs = Fs;

    /* Set Encoder parameters */
    st->encControl.API_sampleRate        = Fs;
    st->encControl.maxInternalSampleRate = 16000;
    st->encControl.packetSize            = Fs/50;
    st->encControl.packetLossPercentage  = 0;
    st->encControl.useInBandFEC          = 0;
    st->encControl.useDTX                = 0;
    st->encControl.complexity            = 2;
    st->encControl.bitRate               = 18000;

    /* Create CELT encoder */
	/* We should not have to create a CELT mode for each encoder state */
	st->celt_mode = celt_mode_create(Fs, Fs/50, NULL);
	/* Initialize CELT encoder */
	st->celt_enc = celt_encoder_create(st->celt_mode, 1, NULL);

	st->mode = MODE_HYBRID;
	st->bandwidth = BANDWIDTH_FULLBAND;
	st->vbr_rate = 0;

	return st;
}

int hybrid_encode(HybridEncoder *st, const short *pcm, int frame_size,
		unsigned char *data, int bytes_per_packet)
{
    int i;
	int ret=0;
	SKP_int16 nBytes;
	ec_enc enc;
	ec_byte_buffer buf;

	ec_byte_writeinit_buffer(&buf, data, bytes_per_packet);
	ec_enc_init(&enc,&buf);

	if (st->mode != MODE_CELT_ONLY)
	{
	    st->encControl.bitRate = (bytes_per_packet*50*8+6000)/2;
	    if (st->Fs / frame_size == 100)
	        st->encControl.bitRate += 5000;
	    st->encControl.packetSize = frame_size;

	    if (st->bandwidth == BANDWIDTH_NARROWBAND)
	        st->encControl.maxInternalSampleRate = 8000;
	    else if (st->bandwidth == BANDWIDTH_MEDIUMBAND)
            st->encControl.maxInternalSampleRate = 12000;
	    else
	        st->encControl.maxInternalSampleRate = 16000;

	    /* Call SILK encoder for the low band */
	    nBytes = bytes_per_packet;
	    ret = SKP_Silk_SDK_Encode( st->silk_enc, &st->encControl, pcm, frame_size, &enc, &nBytes );
	    if( ret ) {
	        fprintf (stderr, "SILK encode error\n");
	        /* Handle error */
	    }
	    ret = (ec_enc_tell(&enc, 0)+7)>>3;
	}

	if (st->mode == MODE_HYBRID)
	{
	    /* This should be adjusted based on the SILK bandwidth */
	    celt_encoder_ctl(st->celt_enc, CELT_SET_START_BAND(17));
	} else {
        celt_encoder_ctl(st->celt_enc, CELT_SET_START_BAND(0));
	}

	if (st->mode != MODE_SILK_ONLY && st->bandwidth > BANDWIDTH_WIDEBAND)
	{
	    short buf[960];

        if (st->bandwidth == BANDWIDTH_SUPERWIDEBAND)
            celt_encoder_ctl(st->celt_enc, CELT_SET_END_BAND(20));
        else
            celt_encoder_ctl(st->celt_enc, CELT_SET_END_BAND(21));

	    for (i=0;i<ENCODER_DELAY_COMPENSATION;i++)
	        buf[i] = st->delay_buffer[i];
        for (;i<frame_size;i++)
            buf[i] = pcm[i-ENCODER_DELAY_COMPENSATION];

        celt_encoder_ctl(st->celt_enc, CELT_SET_PREDICTION(1));
	    /* Encode high band with CELT */
	    ret = celt_encode_with_ec(st->celt_enc, buf, NULL, frame_size, data, bytes_per_packet, &enc);
	    for (i=0;i<ENCODER_DELAY_COMPENSATION;i++)
	        st->delay_buffer[i] = pcm[frame_size-ENCODER_DELAY_COMPENSATION+i];
	} else {
	    ec_enc_done(&enc);
	}

	return ret;
}

void hybrid_encoder_ctl(HybridEncoder *st, int request, ...)
{
    va_list ap;

    va_start(ap, request);

    switch (request)
    {
        case HYBRID_SET_MODE_REQUEST:
        {
            int value = va_arg(ap, int);
            st->mode = value;
        }
        break;
        case HYBRID_GET_MODE_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->mode;
        }
        break;
        case HYBRID_SET_BANDWIDTH_REQUEST:
        {
            int value = va_arg(ap, int);
            st->bandwidth = value;
        }
        break;
        case HYBRID_GET_BANDWIDTH_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->bandwidth;
        }
        break;
        case HYBRID_SET_VBR_RATE_REQUEST:
        {
            int value = va_arg(ap, int);
            st->vbr_rate = value;
        }
        break;
        case HYBRID_GET_VBR_RATE_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->vbr_rate;
        }
        break;
        default:
            fprintf(stderr, "unknown hybrid_encoder_ctl() request: %d", request);
            break;
    }
}

void hybrid_encoder_destroy(HybridEncoder *st)
{
	free(st->silk_enc);

	celt_encoder_destroy(st->celt_enc);
	celt_mode_destroy(st->celt_mode);

	free(st);
}

