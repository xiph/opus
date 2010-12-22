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

#if defined(HAVE_CONFIG_H)
# include "config.h"
#endif
#include "os_support.h"
#include "entenc.h"
#include "arch.h"


void ec_byte_writeinit_buffer(ec_byte_buffer *_b,
 unsigned char *_buf,ec_uint32 _size){
  _b->buf=_buf;
  _b->end_offs=_b->offs=0;
  _b->storage=_size;
}

void ec_byte_shrink(ec_byte_buffer *_b,ec_uint32 _size){
  celt_assert(_b->offs+_b->end_offs<=_size);
  CELT_MOVE(_b->buf+_size-_b->end_offs,
   _b->buf+_b->storage-_b->end_offs,_b->end_offs);
  _b->storage=_size;
}

int ec_byte_write(ec_byte_buffer *_b,unsigned _value){
  if(_b->offs+_b->end_offs>=_b->storage)return -1;
  _b->buf[_b->offs++]=(unsigned char)_value;
  return 0;
}

int ec_byte_write_at_end(ec_byte_buffer *_b,unsigned _value){
  if(_b->offs+_b->end_offs>=_b->storage)return -1;
  _b->buf[_b->storage-++(_b->end_offs)]=(unsigned char)_value;
  return 0;
}

int ec_byte_write_done(ec_byte_buffer *_b,int _start_bits_available,
 unsigned _end_byte,int _end_bits_used){
  int ret;
  CELT_MEMSET(_b->buf+_b->offs,0,_b->storage-_b->offs-_b->end_offs);
  ret=0;
  if(_end_bits_used>0){
    if(_b->offs+_b->end_offs>=_b->storage){
      /*If there's no range coder data at all, give up.*/
      if(_b->end_offs>=_b->storage)return -1;
      /*If we've busted, don't add too many extra bits to the last byte; it
         would corrupt the range coder data, and that's more important.*/
      if(_start_bits_available<_end_bits_used){
        _end_bits_used=_start_bits_available;
        _end_byte&=(1<<_start_bits_available)-1;
        ret=-1;
      }
    }
    _b->buf[_b->storage-_b->end_offs-1]|=_end_byte;
  }
  return ret;
}

void ec_enc_uint(ec_enc *_this,ec_uint32 _fl,ec_uint32 _ft){
  unsigned  ft;
  unsigned  fl;
  int       ftb;
  /*In order to optimize EC_ILOG(), it is undefined for the value 0.*/
  celt_assert(_ft>1);
  _ft--;
  ftb=EC_ILOG(_ft);
  if(ftb>EC_UINT_BITS){
    ftb-=EC_UINT_BITS;
    ft=(_ft>>ftb)+1;
    fl=(unsigned)(_fl>>ftb);
    ec_encode(_this,fl,fl+1,ft);
    ec_enc_bits(_this,_fl&((ec_uint32)1<<ftb)-1,ftb);
  }
  else ec_encode(_this,_fl,_fl+1,_ft+1);
}

int ec_enc_get_error(ec_enc *_this){
  return _this->error;
}
