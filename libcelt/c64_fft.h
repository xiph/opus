/* (c) Copyright 2008/2009 Xiph.Org Foundation */
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

#ifndef _dsp_fft_h_
#define _dsp_fft_h_

#include "config.h"

#include "arch.h"
#include "os_support.h"
#include "mathops.h"
#include "stack_alloc.h"

typedef struct {
  int nfft;
  int shift;
  celt_int32 *twiddle;
  celt_int32 *itwiddle;
} c64_fft_t;

extern c64_fft_t *c64_fft16_alloc(int length, int x, int y);
extern void c64_fft16_free(c64_fft_t *state);
extern void c64_fft16_inplace(c64_fft_t *state, celt_int16 *X);
extern void c64_ifft16(c64_fft_t *state, const celt_int16 *X, celt_int16 *Y);

extern c64_fft_t *c64_fft32_alloc(int length, int x, int y);
extern void c64_fft32_free(c64_fft_t *state);
extern void c64_fft32(c64_fft_t *state, const celt_int32 *X, celt_int32 *Y);
extern void c64_ifft32(c64_fft_t *state, const celt_int32 *X, celt_int32 *Y);

#endif
