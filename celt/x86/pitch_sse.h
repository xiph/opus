/* Copyright (c) 2013 Xiph.Org Foundation
   Written by Jean-Marc Valin */
/**
   @file pitch_sse.h
   @brief Pitch analysis
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
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef PITCH_SSE_H
#define PITCH_SSE_H

#include <xmmintrin.h>
#include "arch.h"

#define OVERRIDE_XCORR_KERNEL

static inline void xcorr_kernel(const opus_val16 * _x, const opus_val16 * _y, opus_val32 _sum[4], int len)
{
   int j;
   __m128 sum;
   __m128 x;
   __m128 y;
   __m128 y2;
   __m128 y1;
   __m128 y3;
   __m128 tmp;
   sum = _mm_loadu_ps(_sum);

   x = _mm_loadu_ps(_x);
   y = _mm_loadu_ps(_y);
   y1 = _mm_loadu_ps(_y+1);
   for (j=0;j<len-3;j+=4)
   {
      _x+=4;
      _y+=4;
      y2 = _mm_loadu_ps(_y);
      y3 = _mm_loadu_ps(_y+1);
      tmp = _mm_shuffle_ps(x, x, 0x00);
      sum = _mm_add_ps(sum, _mm_mul_ps(tmp, y));
      tmp = _mm_shuffle_ps(x, x, 0x55);
      sum = _mm_add_ps(sum, _mm_mul_ps(tmp, y1));
      tmp = _mm_shuffle_ps(x, x, 0xaa);
      y = _mm_shuffle_ps(y, y2, 0x4e);
      sum = _mm_add_ps(sum, _mm_mul_ps(tmp, y));
      tmp = _mm_shuffle_ps(x, x, 0xff);
      y = _mm_shuffle_ps(y1, y3, 0x4e);
      sum = _mm_add_ps(sum, _mm_mul_ps(tmp, y));
      x = _mm_loadu_ps(_x);
      y = y2;
      y1 = y3;
   }
   _y++;
   if (j++<len)
   {
      tmp = _mm_shuffle_ps(x, x, 0x00);
      sum = _mm_add_ps(sum, _mm_mul_ps(tmp, y));
   }
   if (j++<len)
   {
      tmp = _mm_shuffle_ps(x, x, 0x55);
      y = _mm_loadu_ps(_y++);
      sum = _mm_add_ps(sum, _mm_mul_ps(tmp, y));
   }
   if (j++<len)
   {
      tmp = _mm_shuffle_ps(x, x, 0xaa);
      y = _mm_loadu_ps(_y++);
      sum = _mm_add_ps(sum, _mm_mul_ps(tmp, y));
   }
   _mm_storeu_ps(_sum, sum);
}

#endif
