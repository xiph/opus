/* (C) 2007-2008 Timothy Terriberry */
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

#ifndef CWRS_H
#define CWRS_H

#include "arch.h"
#include "stack_alloc.h"
#include "entenc.h"
#include "entdec.h"

/* 32-bit versions */
celt_uint32_t ncwrs_u32(int _n,int _m,celt_uint32_t *_u);

void cwrsi32(int _n,int _m,celt_uint32_t _i,int *_x,int *_s,
 celt_uint32_t *_u);

celt_uint32_t icwrs32(int _n,int _m,const int *_x,const int *_s,
 celt_uint32_t *_u);

/* 64-bit versions */
celt_uint64_t ncwrs_u64(int _n,int _m,celt_uint64_t *_u);

celt_uint64_t ncwrs_unext64(int _n,celt_uint64_t *_u);

void cwrsi64(int _n,int _m,celt_uint64_t _i,int *_x,int *_s,
 celt_uint64_t *_u);

celt_uint64_t icwrs64(int _n,int _m,const int *_x,const int *_s,
 celt_uint64_t *_u);


void comb2pulse(int _n,int _m,int * restrict _y,const int *_x,const int *_s);

void pulse2comb(int _n,int _m,int *_x,int *_s,const int *_y);

void encode_pulses(int *_y, int N, int K, ec_enc *enc);

void decode_pulses(int *_y, int N, int K, ec_dec *dec);

#endif /* CWRS_H */
