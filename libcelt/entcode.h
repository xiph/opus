/* Copyright (c) 2001-2008 Timothy B. Terriberry
   Copyright (c) 2008-2009 Xiph.Org Foundation */
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

#include "celt_types.h"

#if !defined(_entcode_H)
# define _entcode_H (1)
# include <limits.h>
# include "ecintrin.h"



typedef celt_int32 ec_int32;
typedef celt_uint32 ec_uint32;
typedef struct ec_byte_buffer ec_byte_buffer;



/*The number of bits to code at a time when coding bits directly.*/
# define EC_UNIT_BITS  (8)
/*The mask for the given bits.*/
# define EC_UNIT_MASK  ((1U<<EC_UNIT_BITS)-1)



/*Simple libogg1-style buffer.*/
struct ec_byte_buffer{
  unsigned char *buf;
  unsigned char *ptr;
  unsigned char *end_ptr;
  long           storage;
};

/*Encoding functions.*/
void ec_byte_writeinit_buffer(ec_byte_buffer *_b, unsigned char *_buf, long _size);
void ec_byte_shrink(ec_byte_buffer *_b, long _size);
void ec_byte_writeinit(ec_byte_buffer *_b);
void ec_byte_writetrunc(ec_byte_buffer *_b,long _bytes);
void ec_byte_write1(ec_byte_buffer *_b,unsigned _value);
void ec_byte_write_at_end(ec_byte_buffer *_b,unsigned _value);
void ec_byte_write4(ec_byte_buffer *_b,ec_uint32 _value);
void ec_byte_writecopy(ec_byte_buffer *_b,void *_source,long _bytes);
void ec_byte_writeclear(ec_byte_buffer *_b);
/*Decoding functions.*/
void ec_byte_readinit(ec_byte_buffer *_b,unsigned char *_buf,long _bytes);
int ec_byte_look1(ec_byte_buffer *_b);
unsigned char ec_byte_look_at_end(ec_byte_buffer *_b);
int ec_byte_look4(ec_byte_buffer *_b,ec_uint32 *_val);
void ec_byte_adv1(ec_byte_buffer *_b);
void ec_byte_adv4(ec_byte_buffer *_b);
int ec_byte_read1(ec_byte_buffer *_b);
int ec_byte_read4(ec_byte_buffer *_b,ec_uint32 *_val);
/*Shared functions.*/
static inline void ec_byte_reset(ec_byte_buffer *_b){
   _b->ptr=_b->buf;
}

static inline long ec_byte_bytes(ec_byte_buffer *_b){
   return _b->ptr-_b->buf;
}

static inline unsigned char *ec_byte_get_buffer(ec_byte_buffer *_b){
   return _b->buf;
}

int ec_ilog(ec_uint32 _v);

#endif
