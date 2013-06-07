/* Copyright (c) 2013 Jean-Marc Valin and John Ridges */
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
static inline void xcorr_kernel(const opus_val16 *x, const opus_val16 *y, opus_val32 sum[4], int len)
{
   int j;
   __m128 xsum1, xsum2;
   xsum1 = _mm_loadu_ps(sum);
   xsum2 = _mm_setzero_ps();

   for (j = 0; j < len-3; j += 4)
   {
      __m128 x0 = _mm_loadu_ps(x+j);
      __m128 y0 = _mm_loadu_ps(y+j);
      __m128 y3 = _mm_loadu_ps(y+j+3);

      xsum1 = _mm_add_ps(xsum1,_mm_mul_ps(_mm_shuffle_ps(x0,x0,0x00),y0));
      xsum2 = _mm_add_ps(xsum2,_mm_mul_ps(_mm_shuffle_ps(x0,x0,0x55),
                                          _mm_shuffle_ps(y0,y3,0x49)));
      xsum1 = _mm_add_ps(xsum1,_mm_mul_ps(_mm_shuffle_ps(x0,x0,0xaa),
                                          _mm_shuffle_ps(y0,y3,0x9e)));
      xsum2 = _mm_add_ps(xsum2,_mm_mul_ps(_mm_shuffle_ps(x0,x0,0xff),y3));
   }
   if (j < len)
   {
      xsum1 = _mm_add_ps(xsum1,_mm_mul_ps(_mm_load1_ps(x+j),_mm_loadu_ps(y+j)));
      if (++j < len)
      {
         xsum2 = _mm_add_ps(xsum2,_mm_mul_ps(_mm_load1_ps(x+j),_mm_loadu_ps(y+j)));
         if (++j < len)
         {
            xsum1 = _mm_add_ps(xsum1,_mm_mul_ps(_mm_load1_ps(x+j),_mm_loadu_ps(y+j)));
         }
      }
   }
   _mm_storeu_ps(sum,_mm_add_ps(xsum1,xsum2));
}

#endif
