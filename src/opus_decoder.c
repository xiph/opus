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
#include "opus_decoder.h"
#include "entdec.h"
#include "modes.h"
#include "SKP_Silk_SDK_API.h"


OpusDecoder *opus_decoder_create(int Fs)
{
    char *raw_state;
	int ret, silkDecSizeBytes, celtDecSizeBytes;
	CELTMode *celtMode;
	OpusDecoder *st;

    /* We should not have to create a CELT mode for each encoder state */
    celtMode = celt_mode_create(Fs, Fs/50, NULL);

	/* Initialize SILK encoder */
    ret = SKP_Silk_SDK_Get_Decoder_Size( &silkDecSizeBytes );
    if( ret ) {
        /* Handle error */
    }
    celtDecSizeBytes = celt_decoder_get_size(celtMode, 1);
    raw_state = calloc(sizeof(OpusDecoder)+silkDecSizeBytes+celtDecSizeBytes, 1);
    st = (OpusDecoder*)raw_state;
    st->silk_dec = (void*)(raw_state+sizeof(OpusDecoder));
    st->celt_dec = (CELTDecoder*)(raw_state+sizeof(OpusDecoder)+silkDecSizeBytes);

    st->Fs = Fs;
    st->celt_mode = celtMode;

    /* Reset decoder */
    ret = SKP_Silk_SDK_InitDecoder( st->silk_dec );
    if( ret ) {
        /* Handle error */
    }

	/* Initialize CELT decoder */
	st->celt_dec = celt_decoder_init(st->celt_dec, st->celt_mode, 1, NULL);

	return st;

}
int opus_decode(OpusDecoder *st, const unsigned char *data,
		int len, short *pcm, int frame_size)
{
	int i, silk_ret=0, celt_ret=0;
	ec_dec dec;
	ec_byte_buffer buf;
    SKP_SILK_SDK_DecControlStruct DecControl;
    SKP_int16 silk_frame_size;
    short pcm_celt[960];
    int audiosize;

    if (data != NULL)
    {
        /* Decoding mode/bandwidth/framesize from first byte */
        if (data[0]&0x80)
        {
            st->mode = MODE_CELT_ONLY;
            st->bandwidth = BANDWIDTH_MEDIUMBAND + ((data[0]>>5)&0x3);
            if (st->bandwidth == BANDWIDTH_MEDIUMBAND)
                st->bandwidth = BANDWIDTH_NARROWBAND;
            audiosize = ((data[0]>>3)&0x3);
            audiosize = (st->Fs<<audiosize)/400;
        } else if ((data[0]&0x60) == 0x60)
        {
            st->mode = MODE_HYBRID;
            st->bandwidth = (data[0]&0x10) ? BANDWIDTH_FULLBAND : BANDWIDTH_SUPERWIDEBAND;
            audiosize = (data[0]&0x08) ? st->Fs/50 : st->Fs/100;
        } else {

            st->mode = MODE_SILK_ONLY;
            st->bandwidth = BANDWIDTH_NARROWBAND + ((data[0]>>5)&0x3);
            audiosize = ((data[0]>>3)&0x3);
            if (audiosize == 3)
                audiosize = st->Fs*60/1000;
            else
                audiosize = (st->Fs<<audiosize)/100;
        }
        /*printf ("%d %d %d\n", st->mode, st->bandwidth, audiosize);*/

        len -= 1;
        data += 1;
        ec_byte_readinit(&buf,(unsigned char*)data,len);
        ec_dec_init(&dec,&buf);
    }

    if (st->mode != MODE_CELT_ONLY)
    {
        DecControl.API_sampleRate = st->Fs;

        /* We Should eventually have to set the bandwidth here */

        /* Call SILK encoder for the low band */
        silk_ret = SKP_Silk_SDK_Decode( st->silk_dec, &DecControl, data == NULL, &dec, len, pcm, &silk_frame_size );
        if (silk_ret)
        {
            fprintf (stderr, "SILK decode error\n");
            /* Handle error */
        }
    } else {
        for (i=0;i<frame_size;i++)
            pcm[i] = 0;
    }

    if (st->mode == MODE_HYBRID)
    {
        /* This should be adjusted based on the SILK bandwidth */
        celt_decoder_ctl(st->celt_dec, CELT_SET_START_BAND(17));
    } else {
        celt_decoder_ctl(st->celt_dec, CELT_SET_START_BAND(0));
    }

    if (st->mode != MODE_SILK_ONLY && st->bandwidth > BANDWIDTH_WIDEBAND)
    {
        if (st->bandwidth == BANDWIDTH_SUPERWIDEBAND)
            celt_decoder_ctl(st->celt_dec, CELT_SET_END_BAND(20));
        else
            celt_decoder_ctl(st->celt_dec, CELT_SET_END_BAND(21));
        /* Encode high band with CELT */
        celt_ret = celt_decode_with_ec(st->celt_dec, data, len, pcm_celt, frame_size, &dec);
        for (i=0;i<frame_size;i++)
            pcm[i] += pcm_celt[i];
    }
	return celt_ret;

}

void opus_decoder_ctl(OpusDecoder *st, int request, ...)
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
        default:
            fprintf(stderr, "unknown opus_decoder_ctl() request: %d", request);
            break;
    }

    va_end(ap);
}

void opus_decoder_destroy(OpusDecoder *st)
{
	celt_mode_destroy(st->celt_mode);

	free(st);
}
