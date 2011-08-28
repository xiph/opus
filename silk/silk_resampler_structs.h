/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, (subject to the limitations in the disclaimer below)
are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Skype Limited, nor the names of specific
contributors, may be used to endorse or promote products derived from
this software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED
BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifndef SILK_RESAMPLER_STRUCTS_H
#define SILK_RESAMPLER_STRUCTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Flag to enable support for input/output sampling rates above 48 kHz. Turn off for embedded devices */
#define RESAMPLER_SUPPORT_ABOVE_48KHZ                1

#define SILK_RESAMPLER_MAX_FIR_ORDER                 16
#define SILK_RESAMPLER_MAX_IIR_ORDER                 6


typedef struct _silk_resampler_state_struct{
    opus_int32       sIIR[ SILK_RESAMPLER_MAX_IIR_ORDER ];        /* this must be the first element of this struct */
    opus_int32       sFIR[ SILK_RESAMPLER_MAX_FIR_ORDER ];
    opus_int32       sDown2[ 2 ];
    void            (*resampler_function)( void *, opus_int16 *, const opus_int16 *, opus_int32 );
    void            (*up2_function)(  opus_int32 *, opus_int16 *, const opus_int16 *, opus_int32 );
    opus_int32       batchSize;
    opus_int32       invRatio_Q16;
    opus_int32       FIR_Fracs;
    opus_int32       input2x;
    const opus_int16    *Coefs;
#if RESAMPLER_SUPPORT_ABOVE_48KHZ
    opus_int32       sDownPre[ 2 ];
    opus_int32       sUpPost[ 2 ];
    void            (*down_pre_function)( opus_int32 *, opus_int16 *, const opus_int16 *, opus_int32 );
    void            (*up_post_function)(  opus_int32 *, opus_int16 *, const opus_int16 *, opus_int32 );
    opus_int32       batchSizePrePost;
    opus_int32       ratio_Q16;
    opus_int32       nPreDownsamplers;
    opus_int32       nPostUpsamplers;
#endif
    opus_int32 magic_number;
} silk_resampler_state_struct;

#ifdef __cplusplus
}
#endif
#endif /* SILK_RESAMPLER_STRUCTS_H */

