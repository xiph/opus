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
#include "opus_encoder.h"
#include "entenc.h"
#include "modes.h"
#include "SKP_Silk_SDK_API.h"

OpusEncoder *opus_encoder_create(int Fs, int channels)
{
    char *raw_state;
	OpusEncoder *st;
	int ret, silkEncSizeBytes, celtEncSizeBytes;
    SKP_SILK_SDK_EncControlStruct encControl;

    /* Create SILK encoder */
    ret = SKP_Silk_SDK_Get_Encoder_Size( &silkEncSizeBytes );
    if( ret ) {
    	/* Handle error */
    }
    celtEncSizeBytes = celt_encoder_get_size(channels);
    raw_state = calloc(sizeof(OpusEncoder)+silkEncSizeBytes+celtEncSizeBytes, 1);
    st = (OpusEncoder*)raw_state;
    st->silk_enc = (void*)(raw_state+sizeof(OpusEncoder));
    st->celt_enc = (CELTEncoder*)(raw_state+sizeof(OpusEncoder)+silkEncSizeBytes);
    st->stream_channels = st->channels = channels;

    st->Fs = Fs;

    /*encControl.API_sampleRate        = st->Fs;
    encControl.packetLossPercentage  = 0;
    encControl.useInBandFEC          = 0;
    encControl.useDTX                = 0;
    encControl.complexity            = 2;*/
    ret = SKP_Silk_SDK_InitEncoder( st->silk_enc, &encControl );
    if( ret ) {
        /* Handle error */
    }

    /* Create CELT encoder */
	/* Initialize CELT encoder */
	st->celt_enc = celt_encoder_init(st->celt_enc, Fs, channels, NULL);

	st->mode = MODE_HYBRID;
	st->bandwidth = BANDWIDTH_FULLBAND;
	st->vbr_rate = 0;

	return st;
}

int opus_encode(OpusEncoder *st, const short *pcm, int frame_size,
		unsigned char *data, int bytes_per_packet)
{
    int i;
	int ret=0;
	SKP_int32 nBytes;
	ec_enc enc;
	ec_byte_buffer buf;
	SKP_SILK_SDK_EncControlStruct encControl;
	int framerate, period;

	bytes_per_packet -= 1;
	data += 1;
	ec_byte_writeinit_buffer(&buf, data, bytes_per_packet);
	ec_enc_init(&enc,&buf);

	if (st->mode != MODE_CELT_ONLY)
	{
	    /* Set Encoder parameters */
	    encControl.API_sampleRate        = st->Fs;
	    encControl.packetLossPercentage  = 2;
	    encControl.useInBandFEC          = 0;
	    encControl.useDTX                = 0;
	    encControl.complexity            = 2;

	    if (st->vbr_rate != 0)
            encControl.bitRate = (st->vbr_rate+6000)/2;
	    else {
	        encControl.bitRate = (bytes_per_packet*8*(celt_int32)st->Fs/frame_size+6000)/2;
	        if (st->Fs  == 100 * frame_size)
	            encControl.bitRate -= 5000;
	    }
	    encControl.payloadSize_ms = 1000 * frame_size / st->Fs;

	    if (st->mode == MODE_HYBRID)
	    	encControl.minInternalSampleRate = 16000;
	    else
	    	encControl.minInternalSampleRate = 8000;

	    if (st->bandwidth == BANDWIDTH_NARROWBAND)
	        encControl.maxInternalSampleRate = 8000;
	    else if (st->bandwidth == BANDWIDTH_MEDIUMBAND)
	    	encControl.maxInternalSampleRate = 12000;
	    else
	    	encControl.maxInternalSampleRate = 16000;

	    /* Call SILK encoder for the low band */
	    nBytes = bytes_per_packet;
	    ret = SKP_Silk_SDK_Encode( st->silk_enc, &encControl, pcm, frame_size, &enc, &nBytes );
	    if( ret ) {
	        fprintf (stderr, "SILK encode error %d\n", ret);
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
		int endband;
	    short pcm_buf[960*2];

	    switch(st->bandwidth)
	    {
	    case BANDWIDTH_NARROWBAND:
	    	endband = 13;
	    	break;
	    case BANDWIDTH_WIDEBAND:
	    	endband = 17;
	    	break;
	    case BANDWIDTH_SUPERWIDEBAND:
	    	endband = 19;
	    	break;
	    case BANDWIDTH_FULLBAND:
	    	endband = 21;
	    	break;
	    }
	    celt_encoder_ctl(st->celt_enc, CELT_SET_END_BAND(endband));
	    celt_encoder_ctl(st->celt_enc, CELT_SET_CHANNELS(st->stream_channels));

	    for (i=0;i<ENCODER_DELAY_COMPENSATION*st->channels;i++)
	        pcm_buf[i] = st->delay_buffer[i];
        for (;i<frame_size*st->channels;i++)
            pcm_buf[i] = pcm[i-ENCODER_DELAY_COMPENSATION*st->channels];

        celt_encoder_ctl(st->celt_enc, CELT_SET_PREDICTION(1));

        if (st->vbr_rate != 0)
        {
            int tmp;

            tmp = (st->mode == MODE_HYBRID) ? (st->vbr_rate-6000)/2 : st->vbr_rate;
            tmp = ((ec_enc_tell(&enc, 0)+4)>>3) + tmp * frame_size/(8*st->Fs);
            if (tmp <= bytes_per_packet)
                bytes_per_packet = tmp;
            ec_byte_shrink(&buf, bytes_per_packet);
        }
	    /* Encode high band with CELT */
	    ret = celt_encode_with_ec(st->celt_enc, pcm_buf, frame_size, NULL, bytes_per_packet, &enc);
	    for (i=0;i<ENCODER_DELAY_COMPENSATION*st->channels;i++)
	        st->delay_buffer[i] = pcm[frame_size*st->channels-ENCODER_DELAY_COMPENSATION*st->channels+i];
	} else {
	    ec_enc_done(&enc);
	}

	/* Signalling the mode in the first byte */
	data--;
	framerate = st->Fs/frame_size;
	period = 0;
	while (framerate < 400)
	{
	    framerate <<= 1;
	    period++;
	}
    if (st->mode == MODE_SILK_ONLY)
    {
        data[0] = (st->bandwidth-BANDWIDTH_NARROWBAND)<<5;
        data[0] |= (period-2)<<3;
    } else if (st->mode == MODE_CELT_ONLY)
    {
        int tmp = st->bandwidth-BANDWIDTH_MEDIUMBAND;
        if (tmp < 0)
            tmp = 0;
        data[0] = 0x80;
        data[0] |= tmp << 5;
        data[0] |= period<<3;
    } else /* Opus */
    {
        data[0] = 0x60;
        data[0] |= (st->bandwidth-BANDWIDTH_SUPERWIDEBAND)<<4;
        data[0] |= (period-2)<<3;
    }
    data[0] |= (st->stream_channels==2)<<2;
    /*printf ("%x\n", (int)data[0]);*/

    return ret+1;
}

void opus_encoder_ctl(OpusEncoder *st, int request, ...)
{
    va_list ap;

    va_start(ap, request);

    switch (request)
    {
        case OPUS_SET_MODE_REQUEST:
        {
            int value = va_arg(ap, int);
            st->mode = value;
        }
        break;
        case OPUS_GET_MODE_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->mode;
        }
        break;
        case OPUS_SET_BANDWIDTH_REQUEST:
        {
            int value = va_arg(ap, int);
            st->bandwidth = value;
        }
        break;
        case OPUS_GET_BANDWIDTH_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->bandwidth;
        }
        break;
        case OPUS_SET_VBR_RATE_REQUEST:
        {
            int value = va_arg(ap, int);
            st->vbr_rate = value;
        }
        break;
        case OPUS_GET_VBR_RATE_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->vbr_rate;
        }
        break;
        default:
            fprintf(stderr, "unknown opus_encoder_ctl() request: %d", request);
            break;
    }

    va_end(ap);
}

void opus_encoder_destroy(OpusEncoder *st)
{
	free(st);
}

