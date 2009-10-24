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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "os_support.h"
#include "entenc.h"
#include "arch.h"


#define EC_BUFFER_INCREMENT (256)

void ec_byte_writeinit_buffer(ec_byte_buffer *_b, unsigned char *_buf, long _size){
  _b->ptr=_b->buf=_buf;
  _b->end_ptr=_b->buf+_size-1;
  _b->storage=_size;
}

void ec_byte_shrink(ec_byte_buffer *_b, long _size){
   _b->end_ptr=_b->buf+_size-1;
   _b->storage=_size;
}

void ec_byte_write1(ec_byte_buffer *_b,unsigned _value){
  ptrdiff_t endbyte;
  endbyte=_b->ptr-_b->buf;
  if(endbyte>=_b->storage){
    celt_fatal("range encoder overflow\n");
  }
  *(_b->ptr++)=(unsigned char)_value;
}

void ec_byte_write_at_end(ec_byte_buffer *_b,unsigned _value){
  if (_b->end_ptr < _b->ptr)
  {
    celt_fatal("byte buffer collision");
  }
  *(_b->end_ptr--)=(unsigned char)_value;
}


void ec_enc_bits(ec_enc *_this,ec_uint32 _fl,int _ftb){
  unsigned fl;
  unsigned ft;
  while(_ftb>EC_UNIT_BITS){
    _ftb-=EC_UNIT_BITS;
    fl=(unsigned)(_fl>>_ftb)&EC_UNIT_MASK;
    ec_encode_raw(_this,fl,fl+1,EC_UNIT_BITS);
  }
  ft=1<<_ftb;
  fl=(unsigned)_fl&ft-1;
  ec_encode_raw(_this,fl,fl+1,_ftb);
}

void ec_enc_uint(ec_enc *_this,ec_uint32 _fl,ec_uint32 _ft){
  unsigned  ft;
  unsigned  fl;
  int       ftb;
  /*In order to optimize EC_ILOG(), it is undefined for the value 0.*/
  celt_assert(_ft>1);
  _ft--;
  ftb=EC_ILOG(_ft);
  if(ftb>EC_UNIT_BITS){
    ftb-=EC_UNIT_BITS;
    ft=(_ft>>ftb)+1;
    fl=(unsigned)(_fl>>ftb);
    ec_encode(_this,fl,fl+1,ft);
    ec_enc_bits(_this,_fl,ftb);
  } else {
    ec_encode(_this,_fl,_fl+1,_ft+1);
  }
}

