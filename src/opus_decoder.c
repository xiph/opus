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
#include "celt.h"
#include "opus_decoder.h"
#include "entdec.h"
#include "modes.h"
#include "SKP_Silk_SDK_API.h"


OpusDecoder *opus_decoder_create(int Fs, int channels)
{
    char *raw_state;
	int ret, silkDecSizeBytes, celtDecSizeBytes;
	OpusDecoder *st;

	/* Initialize SILK encoder */
    ret = SKP_Silk_SDK_Get_Decoder_Size( &silkDecSizeBytes );
    if( ret ) {
        /* Handle error */
    }
    celtDecSizeBytes = celt_decoder_get_size(channels);
    raw_state = calloc(sizeof(OpusDecoder)+silkDecSizeBytes+celtDecSizeBytes, 1);
    st = (OpusDecoder*)raw_state;
    st->silk_dec = (void*)(raw_state+sizeof(OpusDecoder));
    st->celt_dec = (CELTDecoder*)(raw_state+sizeof(OpusDecoder)+silkDecSizeBytes);
    st->stream_channels = st->channels = channels;

    st->Fs = Fs;

    /* Reset decoder */
    ret = SKP_Silk_SDK_InitDecoder( st->silk_dec );
    if( ret ) {
        /* Handle error */
    }

	/* Initialize CELT decoder */
	st->celt_dec = celt_decoder_init(st->celt_dec, Fs, channels, NULL);
    celt_decoder_ctl(st->celt_dec, CELT_SET_SIGNALLING(0));

	st->prev_mode = 0;
	return st;
}

static void smooth_fade(const short *in1, const short *in2, short *out,
        int overlap, int channels, const celt_word16 *window, int Fs)
{
	int i, c;
	int inc = 48000/Fs;
	for (c=0;c<channels;c++)
	{
		for (i=0;i<overlap;i++)
		{
		    celt_word16 w = MULT16_16_Q15(window[i*inc], window[i*inc]);
		    out[i*channels+c] = SHR32(MAC16_16(MULT16_16(w,in2[i*channels+c]),
		            Q15ONE-w, in1[i*channels+c]), 15);
		}
	}
}

static int opus_packet_get_mode(const unsigned char *data)
{
	int mode;
    if (data[0]&0x80)
    {
        mode = MODE_CELT_ONLY;
    } else if ((data[0]&0x60) == 0x60)
    {
        mode = MODE_HYBRID;
    } else {

        mode = MODE_SILK_ONLY;
    }
    return mode;
}

static int opus_decode_frame(OpusDecoder *st, const unsigned char *data,
		int len, short *pcm, int frame_size, int decode_fec)
{
	int i, silk_ret=0, celt_ret=0;
	ec_dec dec;
    SKP_SILK_SDK_DecControlStruct DecControl;
    SKP_int32 silk_frame_size;
    short pcm_celt[960*2];
    short pcm_transition[960*2];
    int audiosize;
    int mode;
    int transition=0;
    int start_band;
    int redundancy=0;
    int redundancy_bytes = 0;
    int celt_to_silk=0;
    short redundant_audio[240*2];
    int c;
    int F2_5, F5, F10;
    const celt_word16 *window;

    F10 = st->Fs/100;
    F5 = F10>>1;
    F2_5 = F5>>1;
    /* Payloads of 1 (2 including ToC) or 0 trigger the PLC/DTX */
    if (len<=1)
    	data = NULL;

	audiosize = st->frame_size;
    if (data != NULL)
    {
    	mode = st->mode;
        ec_dec_init(&dec,(unsigned char*)data,len);
    } else {
        mode = st->prev_mode;
    }

    if (st->stream_channels > st->channels)
        return OPUS_CORRUPTED_DATA;

    if (data!=NULL && !st->prev_redundancy && mode != st->prev_mode && st->prev_mode > 0
    		&& !(mode == MODE_SILK_ONLY && st->prev_mode == MODE_HYBRID)
    		&& !(mode == MODE_HYBRID && st->prev_mode == MODE_SILK_ONLY))
    {
    	transition = 1;
    	if (mode == MODE_CELT_ONLY)
    	    opus_decode_frame(st, NULL, 0, pcm_transition, IMAX(F10, audiosize), 0);
    }
    if (audiosize > frame_size)
    {
        fprintf(stderr, "PCM buffer too small: %d vs %d (mode = %d)\n", audiosize, frame_size, mode);
        return OPUS_BAD_ARG;
    } else {
        frame_size = audiosize;
    }

    /* SILK processing */
    if (mode != MODE_CELT_ONLY)
    {
        int lost_flag, decoded_samples;
        SKP_int16 *pcm_ptr = pcm;

        if (st->prev_mode==MODE_CELT_ONLY)
        	SKP_Silk_SDK_InitDecoder( st->silk_dec );

        DecControl.API_sampleRate = st->Fs;
        DecControl.payloadSize_ms = 1000 * audiosize / st->Fs;
        if( mode == MODE_SILK_ONLY ) {
            if( st->bandwidth == BANDWIDTH_NARROWBAND ) {
                DecControl.internalSampleRate = 8000;
            } else if( st->bandwidth == BANDWIDTH_MEDIUMBAND ) {
                DecControl.internalSampleRate = 12000;
            } else if( st->bandwidth == BANDWIDTH_WIDEBAND ) {
                DecControl.internalSampleRate = 16000;
            } else {
            	DecControl.internalSampleRate = 16000;
                SKP_assert( 0 );
            }
        } else {
            /* Hybrid mode */
            DecControl.internalSampleRate = 16000;
        }
        DecControl.nChannels = st->channels;

        lost_flag = data == NULL ? 1 : 2 * decode_fec;
        decoded_samples = 0;
        do {
            /* Call SILK decoder */
            int first_frame = decoded_samples == 0;
            silk_ret = SKP_Silk_SDK_Decode( st->silk_dec, &DecControl, 
                lost_flag, first_frame, &dec, pcm_ptr, &silk_frame_size );
            if( silk_ret ) {
                fprintf (stderr, "SILK decode error\n");
                /* Handle error */
            }
            pcm_ptr += silk_frame_size * st->channels;
            decoded_samples += silk_frame_size;
        } while( decoded_samples < frame_size );
    } else {
        for (i=0;i<frame_size*st->channels;i++)
            pcm[i] = 0;
    }

    start_band = 0;
    if (mode != MODE_CELT_ONLY && data != NULL)
    {
        /* Check if we have a redundant 0-8 kHz band */
        redundancy = ec_dec_bit_logp(&dec, 12);
        if (redundancy)
        {
            celt_to_silk = ec_dec_bit_logp(&dec, 1);
            if (mode == MODE_HYBRID)
            	redundancy_bytes = 2 + ec_dec_uint(&dec, 256);
            else
            	redundancy_bytes = len - ((ec_tell(&dec)+7)>>3);
            len -= redundancy_bytes;
            if (len<0)
                return CELT_CORRUPTED_DATA;
            /* Shrink decoder because of raw bits */
            dec.storage -= redundancy_bytes;
        }
    }
    if (mode != MODE_CELT_ONLY)
    	start_band = 17;

    if (mode != MODE_SILK_ONLY)
    {
        int endband=21;

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
        celt_decoder_ctl(st->celt_dec, CELT_SET_END_BAND(endband));
        celt_decoder_ctl(st->celt_dec, CELT_SET_CHANNELS(st->stream_channels));
    }

    if (redundancy)
        transition = 0;

    if (transition && mode != MODE_CELT_ONLY)
        opus_decode_frame(st, NULL, 0, pcm_transition, IMAX(F10, audiosize), 0);

    /* 5 ms redundant frame for CELT->SILK*/
    if (redundancy && celt_to_silk)
    {
        celt_decode(st->celt_dec, data+len, redundancy_bytes, redundant_audio, F5);
        celt_decoder_ctl(st->celt_dec, CELT_RESET_STATE);
    }

    /* MUST be after PLC */
    celt_decoder_ctl(st->celt_dec, CELT_SET_START_BAND(start_band));

    if (transition)
    	celt_decoder_ctl(st->celt_dec, CELT_RESET_STATE);

    if (mode != MODE_SILK_ONLY)
    {
        /* Decode CELT */
        celt_ret = celt_decode_with_ec(st->celt_dec, decode_fec?NULL:data, len, pcm_celt, frame_size, &dec);
        for (i=0;i<frame_size*st->channels;i++)
            pcm[i] = ADD_SAT16(pcm[i], pcm_celt[i]);
    }


    {
        const CELTMode *celt_mode;
        celt_decoder_ctl(st->celt_dec, CELT_GET_MODE(&celt_mode));
        window = celt_mode->window;
    }

    /* 5 ms redundant frame for SILK->CELT */
    if (redundancy && !celt_to_silk)
    {
        celt_decoder_ctl(st->celt_dec, CELT_RESET_STATE);
        celt_decoder_ctl(st->celt_dec, CELT_SET_START_BAND(0));

        celt_decode(st->celt_dec, data+len, redundancy_bytes, redundant_audio, F5);
        smooth_fade(pcm+st->channels*(frame_size-F2_5), redundant_audio+st->channels*F2_5,
        		pcm+st->channels*(frame_size-F2_5), F2_5, st->channels, window, st->Fs);
    }
    if (redundancy && celt_to_silk)
    {
        for (c=0;c<st->channels;c++)
        {
            for (i=0;i<F2_5;i++)
                pcm[st->channels*i+c] = redundant_audio[st->channels*i];
        }
        smooth_fade(redundant_audio+st->channels*F2_5, pcm+st->channels*F2_5,
                pcm+st->channels*F2_5, F2_5, st->channels, window, st->Fs);
    }
    if (transition)
    {
    	for (i=0;i<F2_5;i++)
    		pcm[i] = pcm_transition[i];
    	if (audiosize >= F5)
    	    smooth_fade(pcm_transition+F2_5, pcm+F2_5, pcm+F2_5, F2_5,
    	            st->channels, window, st->Fs);
    }
#if OPUS_TEST_RANGE_CODER_STATE
    st->rangeFinal = dec.rng;
#endif

    st->prev_mode = mode;
    st->prev_redundancy = redundancy;
	return celt_ret<0 ? celt_ret : audiosize;

}

static int parse_size(const unsigned char *data, int len, short *size)
{
	if (len<1)
	{
		*size = -1;
		return -1;
	} else if (data[0]<252)
	{
		*size = data[0];
		return 1;
	} else if (len<2)
	{
		*size = -1;
		return -1;
	} else {
		*size = 4*data[1] + data[0];
		return 2;
	}
}

int opus_decode(OpusDecoder *st, const unsigned char *data,
		int len, short *pcm, int frame_size, int decode_fec)
{
	int i, bytes, nb_samples;
	int count;
	unsigned char ch, toc;
	/* 48 x 2.5 ms = 120 ms */
	short size[48];
	if (len==0 || data==NULL)
	    return opus_decode_frame(st, NULL, 0, pcm, frame_size, 0);
	else if (len<0)
		return CELT_BAD_ARG;
	st->mode = opus_packet_get_mode(data);
	st->bandwidth = opus_packet_get_bandwidth(data);
	st->frame_size = opus_packet_get_samples_per_frame(data, st->Fs);
	st->stream_channels = opus_packet_get_nb_channels(data);
	toc = *data++;
	len--;
	switch (toc&0x3)
	{
	/* One frame */
	case 0:
		count=1;
		size[0] = len;
		break;
		/* Two CBR frames */
	case 1:
		count=2;
		if (len&0x1)
			return OPUS_CORRUPTED_DATA;
		size[0] = size[1] = len/2;
		break;
		/* Two VBR frames */
	case 2:
		count = 2;
		bytes = parse_size(data, len, size);
		len -= bytes;
		if (size[0]<0 || size[0] > len)
			return OPUS_CORRUPTED_DATA;
		data += bytes;
		size[1] = len-size[0];
		break;
		/* Multiple CBR/VBR frames (from 0 to 120 ms) */
	case 3:
		if (len<1)
			return OPUS_CORRUPTED_DATA;
		/* Number of frames encoded in bits 0 to 5 */
		ch = *data++;
		count = ch&0x3F;
		if (st->frame_size*count*25 > 3*st->Fs)
		    return OPUS_CORRUPTED_DATA;
		len--;
		/* Padding bit */
		if (ch&0x40)
		{
			int padding=0;
			int p;
			do {
				if (len<=0)
					return OPUS_CORRUPTED_DATA;
				p = *data++;
				len--;
				padding += p==255 ? 254: p;
			} while (p==255);
			len -= padding;
		}
		if (len<0)
			return OPUS_CORRUPTED_DATA;
		/* Bit 7 is VBR flag (bit 6 is ignored) */
		if (ch&0x80)
		{
			/* VBR case */
			int last_size=len;
			for (i=0;i<count-1;i++)
			{
				bytes = parse_size(data, len, size+i);
				len -= bytes;
				if (size[i]<0 || size[i] > len)
					return OPUS_CORRUPTED_DATA;
				data += bytes;
				last_size -= bytes+size[i];
			}
			if (last_size<0)
				return OPUS_CORRUPTED_DATA;
			if (count)
				size[count-1]=last_size;
		} else {
			/* CBR case */
			int sz = count != 0 ? len/count : 0;
			if (sz*count!=len)
				return OPUS_CORRUPTED_DATA;
			for (i=0;i<count;i++)
				size[i] = sz;
		}
		break;
	}
	if (count*st->frame_size > frame_size)
		return OPUS_BAD_ARG;
	nb_samples=0;
	for (i=0;i<count;i++)
	{
		int ret;
		ret = opus_decode_frame(st, data, len, pcm, frame_size-nb_samples, decode_fec);
		if (ret<0)
			return ret;
		data += size[i];
		pcm += ret;
		nb_samples += ret;
	}
	return nb_samples;
}
int opus_decoder_ctl(OpusDecoder *st, int request, ...)
{
    va_list ap;

    va_start(ap, request);

    switch (request)
    {
        case OPUS_GET_MODE_REQUEST:
        {
            int *value = va_arg(ap, int*);
            *value = st->prev_mode;
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
        default:
            fprintf(stderr, "unknown opus_decoder_ctl() request: %d", request);
            break;
    }

    va_end(ap);
    return OPUS_OK;
}

void opus_decoder_destroy(OpusDecoder *st)
{
	free(st);
}

#if OPUS_TEST_RANGE_CODER_STATE
int opus_decoder_get_final_range(OpusDecoder *st)
{
    return st->rangeFinal;
}
#endif


int opus_packet_get_bandwidth(const unsigned char *data)
{
	int bandwidth;
    if (data[0]&0x80)
    {
        bandwidth = BANDWIDTH_MEDIUMBAND + ((data[0]>>5)&0x3);
        if (bandwidth == BANDWIDTH_MEDIUMBAND)
            bandwidth = BANDWIDTH_NARROWBAND;
    } else if ((data[0]&0x60) == 0x60)
    {
        bandwidth = (data[0]&0x10) ? BANDWIDTH_FULLBAND : BANDWIDTH_SUPERWIDEBAND;
    } else {

        bandwidth = BANDWIDTH_NARROWBAND + ((data[0]>>5)&0x3);
    }
    return bandwidth;
}

int opus_packet_get_samples_per_frame(const unsigned char *data, int Fs)
{
	int audiosize;
    if (data[0]&0x80)
    {
        audiosize = ((data[0]>>3)&0x3);
        audiosize = (Fs<<audiosize)/400;
    } else if ((data[0]&0x60) == 0x60)
    {
        audiosize = (data[0]&0x08) ? Fs/50 : Fs/100;
    } else {

        audiosize = ((data[0]>>3)&0x3);
        if (audiosize == 3)
            audiosize = Fs*60/1000;
        else
            audiosize = (Fs<<audiosize)/100;
    }
    return audiosize;
}

int opus_packet_get_nb_channels(const unsigned char *data)
{
    return (data[0]&0x4) ? 2 : 1;
}

int opus_packet_get_nb_frames(const unsigned char packet[], int len)
{
	int count;
	if (len<1)
		return OPUS_BAD_ARG;
	count = packet[0]&0x3;
	if (count==0)
		return 1;
	else if (count!=3)
		return 2;
	else if (len<2)
		return OPUS_CORRUPTED_DATA;
	else
		return packet[1]&0x3F;
}

int opus_decoder_get_nb_samples(const OpusDecoder *dec, const unsigned char packet[], int len)
{
	int samples;
	int count = opus_packet_get_nb_frames(packet, len);
	samples = count*opus_packet_get_samples_per_frame(packet, dec->Fs);
	/* Can't have more than 120 ms */
	if (samples*25 > dec->Fs*3)
		return OPUS_CORRUPTED_DATA;
	else
		return samples;
}

