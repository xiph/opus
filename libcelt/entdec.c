/* (C) 2001-2008 Timothy B. Terriberry
   (C) 2008 Jean-Marc Valin */
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

#include <stddef.h>
#include "entdec.h"
#include "os_support.h"
#include "arch.h"


void ec_byte_readinit(ec_byte_buffer *_b,unsigned char *_buf,long _bytes){
  _b->buf=_b->ptr=_buf;
  _b->storage=_bytes;
}

int ec_byte_look1(ec_byte_buffer *_b){
  ptrdiff_t endbyte;
  endbyte=_b->ptr-_b->buf;
  if(endbyte>=_b->storage)return -1;
  else return _b->ptr[0];
}

int ec_byte_look4(ec_byte_buffer *_b,ec_uint32 *_val){
  ptrdiff_t endbyte;
  endbyte=_b->ptr-_b->buf;
  if(endbyte+4>_b->storage){
    if(endbyte<_b->storage){
      *_val=_b->ptr[0];
      endbyte++;
      if(endbyte<_b->storage){
        *_val|=(ec_uint32)_b->ptr[1]<<8;
        endbyte++;
        if(endbyte<_b->storage)*_val|=(ec_uint32)_b->ptr[2]<<16;
      }
    }
    return -1;
  }
  else{
    *_val=_b->ptr[0];
    *_val|=(ec_uint32)_b->ptr[1]<<8;
    *_val|=(ec_uint32)_b->ptr[2]<<16;
    *_val|=(ec_uint32)_b->ptr[3]<<24;
  }
  return 0;
}

void ec_byte_adv1(ec_byte_buffer *_b){
  _b->ptr++;
}

void ec_byte_adv4(ec_byte_buffer *_b){
  _b->ptr+=4;
}

int ec_byte_read1(ec_byte_buffer *_b){
  ptrdiff_t endbyte;
  endbyte=_b->ptr-_b->buf;
  if(endbyte>=_b->storage)return -1;
  else return *(_b->ptr++);
}

int ec_byte_read4(ec_byte_buffer *_b,ec_uint32 *_val){
  unsigned char *end;
  end=_b->buf+_b->storage;
  if(_b->ptr+4>end){
    if(_b->ptr<end){
      *_val=*(_b->ptr++);
      if(_b->ptr<end){
        *_val|=(ec_uint32)*(_b->ptr++)<<8;
        if(_b->ptr<end)*_val|=(ec_uint32)*(_b->ptr++)<<16;
      }
    }
    return -1;
  }
  else{
    *_val=(*_b->ptr++);
    *_val|=(ec_uint32)*(_b->ptr++)<<8;
    *_val|=(ec_uint32)*(_b->ptr++)<<16;
    *_val|=(ec_uint32)*(_b->ptr++)<<24;
  }
  return 0;
}



ec_uint32 ec_dec_bits(ec_dec *_this,int _ftb){
  ec_uint32 t;
  unsigned  s;
  unsigned  ft;
  t=0;
  while(_ftb>EC_UNIT_BITS){
    s=ec_decode_bin(_this,EC_UNIT_BITS);
    ec_dec_update(_this,s,s+1,EC_UNIT_MASK+1);
    t=t<<EC_UNIT_BITS|s;
    _ftb-=EC_UNIT_BITS;
  }
  ft=1U<<_ftb;
  s=ec_decode_bin(_this,_ftb);
  ec_dec_update(_this,s,s+1,ft);
  t=t<<_ftb|s;
  return t;
}

ec_uint32 ec_dec_uint(ec_dec *_this,ec_uint32 _ft){
  ec_uint32 t;
  unsigned  ft;
  unsigned  s;
  int       ftb;
  t=0;
  /*In order to optimize EC_ILOG(), it is undefined for the value 0.*/
  celt_assert(_ft>1);
  _ft--;
  ftb=EC_ILOG(_ft);
  if(ftb>EC_UNIT_BITS){
    ftb-=EC_UNIT_BITS;
    ft=(unsigned)(_ft>>ftb)+1;
    s=ec_decode(_this,ft);
    ec_dec_update(_this,s,s+1,ft);
    t=t<<EC_UNIT_BITS|s;
    t = t<<ftb|ec_dec_bits(_this,ftb);
    if (t>_ft)
    {
       celt_notify("uint decode error");
       t = _ft;
    }
    return t;
  } else {
    _ft++;
    s=ec_decode(_this,(unsigned)_ft);
    ec_dec_update(_this,s,s+1,(unsigned)_ft);
    t=t<<ftb|s;
    return t;
  }
}
