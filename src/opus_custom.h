/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Copyright (c) 2008 Gregory Maxwell 
   Written by Jean-Marc Valin and Gregory Maxwell */
/**
  @file celt.h
  @brief Contains all the functions for encoding and decoding audio
 */

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

#ifndef OPUS_CUSTOM_H
#define OPUS_CUSTOM_H

#ifdef ENABLE_OPUS_CUSTOM

#include "celt.h"
#include "opus.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OpusCustomEncoder CELTEncoder
#define OpusCustomDecoder CELTDecoder
#define OpusCustomMode CELTMode



#define opus_custom_mode_create celt_mode_create
#define opus_custom_mode_destroy celt_mode_destroy

#define opus_custom_encoder_get_size celt_encoder_get_size_custom
#define opus_custom_encoder_create celt_encoder_create_custom
#define opus_custom_encoder_init celt_encoder_init_custom


#define opus_custom_encoder_destroy celt_encoder_destroy

#define opus_custom_encode_float celt_encode_float
#define opus_custom_encode celt_encode
#define opus_custom_encoder_ctl celt_encoder_ctl
#define opus_custom_decoder_get_size celt_decoder_get_size_custom
#define opus_custom_decoder_create celt_decoder_create_custom
#define opus_custom_decoder_init celt_decoder_init_custom
#define opus_custom_decoder_destroy celt_decoder_destroy
#define opus_custom_decode_float celt_decode_float
#define opus_custom_decode celt_decode
#define opus_custom_decoder_ctl celt_decoder_ctl


#ifdef __cplusplus
}
#endif

#endif /* ENABLE_OPUS_CUSTOM */

#endif /* OPUS_CUSTOM_H */
