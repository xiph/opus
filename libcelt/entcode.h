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
# include <stddef.h>
# include "ecintrin.h"



typedef celt_int32            ec_int32;
typedef celt_uint32           ec_uint32;
typedef size_t                ec_window;
typedef struct ec_byte_buffer ec_byte_buffer;



/*This must be at least 32 bits.*/
# define EC_WINDOW_SIZE ((int)sizeof(ec_window)*CHAR_BIT)

/*The number of bits to use for the range-coded part of unsigned integers.*/
# define EC_UINT_BITS   (8)



/*Simple libogg1-style buffer.*/
struct ec_byte_buffer{
  unsigned char *buf;
  ec_uint32      offs;
  ec_uint32      end_offs;
  ec_uint32      storage;
};

/*Encoding functions.*/
void ec_byte_writeinit_buffer(ec_byte_buffer *_b, unsigned char *_buf, ec_uint32 _size);
void ec_byte_shrink(ec_byte_buffer *_b, ec_uint32 _size);
int ec_byte_write(ec_byte_buffer *_b,unsigned _value);
int ec_byte_write_at_end(ec_byte_buffer *_b,unsigned _value);
int ec_byte_write_done(ec_byte_buffer *_b,int _start_bits_available,
 unsigned _end_byte,int _end_bits_used);
/*Decoding functions.*/
void ec_byte_readinit(ec_byte_buffer *_b,unsigned char *_buf,ec_uint32 _bytes);
int ec_byte_read(ec_byte_buffer *_b);
int ec_byte_read_from_end(ec_byte_buffer *_b);
/*Shared functions.*/
static inline void ec_byte_reset(ec_byte_buffer *_b){
  _b->offs=_b->end_offs=0;
}

static inline ec_uint32 ec_byte_bytes(ec_byte_buffer *_b){
  return _b->offs;
}

static inline unsigned char *ec_byte_get_buffer(ec_byte_buffer *_b){
  return _b->buf;
}

int ec_ilog(ec_uint32 _v);

#endif
