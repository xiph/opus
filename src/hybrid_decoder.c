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
#include "hybrid_decoder.h"
#include "entdec.h"
#include "modes.h"
#include "SKP_Silk_SDK_API.h"


HybridDecoder *hybrid_decoder_create(int Fs)
{
	int ret, decSizeBytes;
	HybridDecoder *st;

	st = malloc(sizeof(HybridDecoder));

	st->Fs = Fs;

	/* Initialize SILK encoder */
    ret = SKP_Silk_SDK_Get_Decoder_Size( &decSizeBytes );
    if( ret ) {
        /* Handle error */
    }
    st->silk_dec = malloc( decSizeBytes );

    /* Reset decoder */
    ret = SKP_Silk_SDK_InitDecoder( st->silk_dec );
    if( ret ) {
        /* Handle error */
    }

	/* We should not have to create a CELT mode for each encoder state */
	st->celt_mode = celt_mode_create(Fs, Fs/50, NULL);
	/* Initialize CELT encoder */
	st->celt_dec = celt_decoder_create(st->celt_mode, 1, NULL);

	return st;

}
int hybrid_decode(HybridDecoder *st, const unsigned char *data,
		int len, short *pcm, int frame_size)
{
	int i, silk_ret=0, celt_ret=0;
	ec_dec dec;
	ec_byte_buffer buf;
    SKP_SILK_SDK_DecControlStruct DecControl;
    SKP_int16 silk_frame_size;
    short pcm_celt[960];

    if (data != NULL)
    {
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

void hybrid_decoder_ctl(HybridDecoder *st, int request, ...)
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
        default:
            fprintf(stderr, "unknown hybrid_decoder_ctl() request: %d", request);
            break;
    }

    va_end(ap);
}

void hybrid_decoder_destroy(HybridDecoder *st)
{
	free(st->silk_dec);
	celt_decoder_destroy(st->celt_dec);
	celt_mode_destroy(st->celt_mode);

	free(st);
}
