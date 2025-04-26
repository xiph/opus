/***********************************************************************
Copyright (C) 2025 Xiph.Org Foundation and contributors.
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

#ifndef SILK_MACROS_LX7_H
#define SILK_MACROS_LX7_H

/* This macro only avoids the undefined behaviour from a left shift of
   a negative value. It should only be used in macros that can't include
   SigProc_FIX.h. In other cases, use silk_LSHIFT32(). */
#define SAFE_SHL(a, b) ((opus_int32)((opus_uint32)(a) << (b)))

/* (a32 * (opus_int32)((opus_int16)(b32))) >> 16 output have to be 32bit int */
#undef silk_SMULWB
static OPUS_INLINE opus_int32 silk_SMULWB_lx7(opus_int32 a32, opus_int32 b32)
{
    opus_int32 res;
    __asm__(
        "mulsh %0, %1, %2\n\t"
        : "=r"(res)
        : "r"(a32), "r"(SAFE_SHL(b32, 16))
    );
    return res;
}
#define silk_SMULWB(a32, b32) (silk_SMULWB_lx7(a32, b32))

/* a32 + (b32 * (opus_int32)((opus_int16)(c32))) >> 16 output have to be 32bit int */
#undef silk_SMLAWB
#define silk_SMLAWB(a32, b32, c32) ((a32) + silk_SMULWB(b32, c32))

/* (a32 * (b32 >> 16)) >> 16 */
#undef silk_SMULWT
static OPUS_INLINE opus_int32 silk_SMULWT_lx7(opus_int32 a32, opus_int32 b32)
{
    opus_int32 res;
    __asm__(
        "mulsh %0, %1, %2\n\t"
        : "=r"(res)
        : "r"(a32), "r"(SAFE_SHL(b32 >> 16, 16))
    );
    return res;
}

/* a32 + (b32 * (c32 >> 16)) >> 16 */
#undef silk_SMLAWT
#define silk_SMLAWT(a32, b32, c32) ((a32) + silk_SMULWT_lx7(b32, c32))

#undef silk_CLZ32
static OPUS_INLINE opus_int32 silk_CLZ32_lx7(opus_int32 in32)
{
    return __builtin_clz(in32);
}
#define silk_CLZ32(in32) (silk_CLZ32_lx7(in32))

#endif
