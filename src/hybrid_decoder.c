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
#include "hybrid_decoder.h"
#include "celt/libcelt/entdec.h"


HybridDecoder *hybrid_decoder_create()
{
	HybridDecoder *st;

	st = malloc(sizeof(HybridDecoder));

	/* FIXME: Initialize SILK encoder here */
	st->silk_dec = NULL;

	/* We should not have to create a CELT mode for each encoder state */
	st->celt_mode = celt_mode_create(48000, 960, NULL);
	/* Initialize CELT encoder */
	st->celt_dec = celt_decoder_create(st->celt_mode, 1, NULL);

	return st;

}
int hybrid_decode(HybridDecoder *st, const unsigned char *data,
		int len, short *pcm, int frame_size)
{
	int celt_ret;
	ec_dec dec;
	ec_byte_buffer buf;

	ec_byte_readinit(&buf,(unsigned char*)data,len);
	ec_dec_init(&dec,&buf);

	/* FIXME: Call SILK encoder for the low band */

	/* This should be adjusted based on the SILK bandwidth */
	celt_decoder_ctl(st->celt_dec, CELT_SET_START_BAND(13));

	/* Encode high band with CELT */
	celt_ret = celt_decode_with_ec(st->celt_dec, data, len, pcm, frame_size, &dec);

	return celt_ret;

}

void hybrid_decoder_destroy(HybridDecoder *st)
{
	/* FIXME: Destroy SILK encoder state */

	celt_decoder_destroy(st->celt_dec);
	celt_mode_destroy(st->celt_mode);

	free(st);
}
