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

    ret = SKP_Silk_SDK_InitEncoder( st->silk_enc, &st->silk_mode );
    if( ret ) {
        /* Handle error */
    }

    /* default SILK parameters */
    st->silk_mode.API_sampleRate        = st->Fs;
    st->silk_mode.maxInternalSampleRate = 16000;
    st->silk_mode.minInternalSampleRate = 8000;
    st->silk_mode.payloadSize_ms        = 20;
    st->silk_mode.packetLossPercentage  = 0;
    st->silk_mode.useInBandFEC          = 0;
    st->silk_mode.useDTX                = 0;
    st->silk_mode.complexity            = 10;

    /* Create CELT encoder */
	/* Initialize CELT encoder */
	st->celt_enc = celt_encoder_init(st->celt_enc, Fs, channels, NULL);

	st->mode = MODE_HYBRID;
	st->bandwidth = BANDWIDTH_FULLBAND;
	st->use_vbr = 0;
	st->bitrate_bps = 32000;

	return st;
}

int opus_encode(OpusEncoder *st, const short *pcm, int frame_size,
		unsigned char *data, int max_data_bytes)
{
    int i;
	int ret=0;
	SKP_int32 nBytes;
	ec_enc enc;
	int framerate, period;
    int silk_internal_bandwidth;
    int bytes_target;

	bytes_target = st->bitrate_bps * frame_size / (st->Fs * 8) - 1;

	data += 1;
	ec_enc_init(&enc, data, max_data_bytes-1);

	/* SILK processing */
    if (st->mode != MODE_CELT_ONLY) {
        st->silk_mode.bitRate = st->bitrate_bps - 8*st->Fs/frame_size;
        if( st->mode == MODE_HYBRID ) {
            if( st->bandwidth == BANDWIDTH_SUPERWIDEBAND ) {
                if( st->Fs == 100 * frame_size ) {
                    /* 24 kHz, 10 ms */
                    st->silk_mode.bitRate = ( ( st->silk_mode.bitRate + 2000 + st->use_vbr * 1000 ) * 2 ) / 3;
                } else {
                    /* 24 kHz, 20 ms */
                    st->silk_mode.bitRate = ( ( st->silk_mode.bitRate + 1000 + st->use_vbr * 1000 ) * 2 ) / 3;
                }
            } else {
                if( st->Fs == 100 * frame_size ) {
                    /* 48 kHz, 10 ms */
                    st->silk_mode.bitRate = ( st->silk_mode.bitRate + 8000 + st->use_vbr * 3000 ) / 2;
                } else {
                    /* 48 kHz, 20 ms */
                    st->silk_mode.bitRate = ( st->silk_mode.bitRate + 9000 + st->use_vbr * 1000 ) / 2;
                }
            }
        }

        st->silk_mode.payloadSize_ms = 1000 * frame_size / st->Fs;
        if (st->bandwidth == BANDWIDTH_NARROWBAND) {
            st->silk_mode.maxInternalSampleRate = 8000;
        } else if (st->bandwidth == BANDWIDTH_MEDIUMBAND) {
            st->silk_mode.maxInternalSampleRate = 12000;
        } else {
            SKP_assert( st->mode == MODE_HYBRID || st->bandwidth == BANDWIDTH_WIDEBAND );
            st->silk_mode.maxInternalSampleRate = 16000;
        }
        if( st->mode == MODE_HYBRID ) {
            /* Don't allow bandwidth reduction at lowest bitrates in hybrid mode */
            st->silk_mode.minInternalSampleRate = 16000;
        } else {
            st->silk_mode.minInternalSampleRate = 8000;
        }

        /* Call SILK encoder for the low band */
        nBytes = max_data_bytes-1;
        ret = SKP_Silk_SDK_Encode( st->silk_enc, &st->silk_mode, pcm, frame_size, &enc, &nBytes );
        if( ret ) {
            fprintf (stderr, "SILK encode error: %d\n", ret);
            /* Handle error */
        }
        /* Extract SILK internal bandwidth for signaling in first byte */
        if( st->mode == MODE_SILK_ONLY ) {
            if( st->silk_mode.internalSampleRate == 8000 ) {
                silk_internal_bandwidth = BANDWIDTH_NARROWBAND;
            } else if( st->silk_mode.internalSampleRate == 12000 ) {
                silk_internal_bandwidth = BANDWIDTH_MEDIUMBAND;
            } else if( st->silk_mode.internalSampleRate == 16000 ) {
                silk_internal_bandwidth = BANDWIDTH_WIDEBAND;
            }
        }
    }

    /* CELT processing */
	if (st->mode != MODE_SILK_ONLY)
	{
		int endband;
	    short pcm_buf[960*2];
	    int nb_compr_bytes;

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

        celt_encoder_ctl(st->celt_enc, CELT_SET_VBR(0));
        celt_encoder_ctl(st->celt_enc, CELT_SET_BITRATE(510000));
        if (st->mode == MODE_HYBRID)
        {
            int len;
            celt_encoder_ctl(st->celt_enc, CELT_SET_START_BAND(17));

            len = (ec_tell(&enc)+7)>>3;
            if( st->use_vbr ) {
                nb_compr_bytes = len + bytes_target - (st->silk_mode.bitRate * frame_size) / (8 * st->Fs);
            } else {
                /* check if SILK used up too much */
                nb_compr_bytes = len > bytes_target ? len : bytes_target;
            }
        } else {
            celt_encoder_ctl(st->celt_enc, CELT_SET_START_BAND(0));
            if (st->use_vbr)
            {
                celt_encoder_ctl(st->celt_enc, CELT_SET_VBR(1));
                celt_encoder_ctl(st->celt_enc, CELT_SET_BITRATE(st->bitrate_bps));
                nb_compr_bytes = max_data_bytes-1;
            } else {
                nb_compr_bytes = bytes_target;
            }
        }

        for (i=0;i<IMIN(frame_size, ENCODER_DELAY_COMPENSATION)*st->channels;i++)
            pcm_buf[i] = st->delay_buffer[i];
        for (;i<frame_size*st->channels;i++)
            pcm_buf[i] = pcm[i-ENCODER_DELAY_COMPENSATION*st->channels];

        ec_enc_shrink(&enc, nb_compr_bytes);

	    /* Encode high band with CELT */
	    ret = celt_encode_with_ec(st->celt_enc, pcm_buf, frame_size, NULL, nb_compr_bytes, &enc);

	    if (frame_size>ENCODER_DELAY_COMPENSATION)
	    {
	        for (i=0;i<ENCODER_DELAY_COMPENSATION*st->channels;i++)
	            st->delay_buffer[i] = pcm[(frame_size-ENCODER_DELAY_COMPENSATION)*st->channels+i];
	    } else {
	        int tmp = ENCODER_DELAY_COMPENSATION-frame_size;
	        for (i=0;i<tmp*st->channels;i++)
	            st->delay_buffer[i] = st->delay_buffer[i+frame_size*st->channels];
	        for (i=0;i<frame_size*st->channels;i++)
	            st->delay_buffer[tmp*st->channels+i] = pcm[i];
	    }
	} else {
	    ret = (ec_tell(&enc)+7)>>3;
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
        data[0] = (silk_internal_bandwidth-BANDWIDTH_NARROWBAND)<<5;
        data[0] |= (period-2)<<3;
    } else if (st->mode == MODE_CELT_ONLY)
    {
        int tmp = st->bandwidth-BANDWIDTH_MEDIUMBAND;
        if (tmp < 0)
            tmp = 0;
        data[0] = 0x80;
        data[0] |= tmp << 5;
        data[0] |= period<<3;
    } else /* Hybrid */
    {
        data[0] = 0x60;
        data[0] |= (st->bandwidth-BANDWIDTH_SUPERWIDEBAND)<<4;
        data[0] |= (period-2)<<3;
    }
    data[0] |= (st->stream_channels==2)<<2;
    /*printf ("%x\n", (int)data[0]);*/

#if OPUS_TEST_RANGE_CODER_STATE
    st->rangeFinal = enc.rng;
#endif

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
        case OPUS_SET_BITRATE_REQUEST:
        {
            int value = va_arg(ap, int);
            st->bitrate_bps = value;
        }
        break;
        case OPUS_GET_BITRATE_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->bitrate_bps;
        }
        break;
        case OPUS_SET_BANDWIDTH_REQUEST:
        {
            int value = va_arg(ap, int);
            st->bandwidth = value;
            if (st->bandwidth == BANDWIDTH_NARROWBAND) {
                st->silk_mode.maxInternalSampleRate = 8000;
            } else if (st->bandwidth == BANDWIDTH_MEDIUMBAND) {
                st->silk_mode.maxInternalSampleRate = 12000;
            } else {
                st->silk_mode.maxInternalSampleRate = 16000;
            }
        }
        break;
        case OPUS_GET_BANDWIDTH_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->bandwidth;
        }
        break;
        case OPUS_SET_DTX_FLAG_REQUEST:
        {
            int value = va_arg(ap, int);
            st->silk_mode.useDTX = value;
        }
        break;
        case OPUS_GET_DTX_FLAG_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->silk_mode.useDTX;
        }
        break;
        case OPUS_SET_COMPLEXITY_REQUEST:
        {
            int value = va_arg(ap, int);
            st->silk_mode.complexity = value;
            celt_encoder_ctl(st->celt_enc, CELT_SET_COMPLEXITY(value));
        }
        break;
        case OPUS_GET_COMPLEXITY_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->silk_mode.complexity;
        }
        break;
        case OPUS_SET_INBAND_FEC_FLAG_REQUEST:
        {
            int value = va_arg(ap, int);
            st->silk_mode.useInBandFEC = value;
        }
        break;
        case OPUS_GET_INBAND_FEC_FLAG_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->silk_mode.useInBandFEC;
        }
        break;
        case OPUS_SET_PACKET_LOSS_PERC_REQUEST:
        {
            int value = va_arg(ap, int);
            st->silk_mode.packetLossPercentage = value;
        }
        break;
        case OPUS_GET_PACKET_LOSS_PERC_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->silk_mode.packetLossPercentage;
        }
        break;
        case OPUS_SET_VBR_FLAG_REQUEST:
        {
            int value = va_arg(ap, int);
            st->use_vbr = value;
            st->silk_mode.useCBR = 1-value;
        }
        break;
        case OPUS_GET_VBR_FLAG_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->use_vbr;
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

#if OPUS_TEST_RANGE_CODER_STATE
int opus_encoder_get_final_range(OpusEncoder *st)
{
    return st->rangeFinal;
}
#endif
