/***********************************************************************
Copyright (C) 2014 Vidyo
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <arm_neon.h>
#include "main.h"
#include "stack_alloc.h"
#include "NSQ.h"
#include "celt/cpu_support.h"
#include "celt/arm/armcpu.h"

opus_int32 silk_noise_shape_quantizer_short_prediction_neon(const opus_int32 *buf32, const opus_int32 *coef32, opus_int order)
{
    int32x4_t coef0 = vld1q_s32(coef32);
    int32x4_t coef1 = vld1q_s32(coef32 + 4);
    int32x4_t coef2 = vld1q_s32(coef32 + 8);
    int32x4_t coef3 = vld1q_s32(coef32 + 12);

    int32x4_t a0 = vld1q_s32(buf32 - 15);
    int32x4_t a1 = vld1q_s32(buf32 - 11);
    int32x4_t a2 = vld1q_s32(buf32 - 7);
    int32x4_t a3 = vld1q_s32(buf32 - 3);

    int32x4_t b0 = vqdmulhq_s32(coef0, a0);
    int32x4_t b1 = vqdmulhq_s32(coef1, a1);
    int32x4_t b2 = vqdmulhq_s32(coef2, a2);
    int32x4_t b3 = vqdmulhq_s32(coef3, a3);

    int32x4_t c0 = vaddq_s32(b0, b1);
    int32x4_t c1 = vaddq_s32(b2, b3);

    int32x4_t d = vaddq_s32(c0, c1);

    int64x2_t e = vpaddlq_s32(d);

    int64x1_t f = vadd_s64(vget_low_s64(e), vget_high_s64(e));

    opus_int32 out = vget_lane_s32(vreinterpret_s32_s64(f), 0);

    out += silk_RSHIFT( order, 1 );

    return out;
}
