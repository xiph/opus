#include <stdlib.h>
#include <string.h>
#include "entenc.h"



#define EC_BUFFER_INCREMENT (256)

void ec_byte_writeinit(ec_byte_buffer *_b){
  _b->ptr=_b->buf=malloc(EC_BUFFER_INCREMENT);
  _b->storage=EC_BUFFER_INCREMENT;
}

void ec_byte_writetrunc(ec_byte_buffer *_b,long _bytes){
  _b->ptr=_b->buf+_bytes;
}

void ec_byte_write1(ec_byte_buffer *_b,unsigned _value){
  ptrdiff_t endbyte;
  endbyte=_b->ptr-_b->buf;
  if(endbyte>=_b->storage){
    _b->buf=realloc(_b->buf,_b->storage+EC_BUFFER_INCREMENT);
    _b->storage+=EC_BUFFER_INCREMENT;
    _b->ptr=_b->buf+endbyte;
  }
  *(_b->ptr++)=(unsigned char)_value;
}

void ec_byte_write4(ec_byte_buffer *_b,ec_uint32 _value){
  ptrdiff_t endbyte;
  endbyte=_b->ptr-_b->buf;
  if(endbyte+4>_b->storage){
    _b->buf=realloc(_b->buf,_b->storage+EC_BUFFER_INCREMENT);
    _b->storage+=EC_BUFFER_INCREMENT;
    _b->ptr=_b->buf+endbyte;
  }
  *(_b->ptr++)=(unsigned char)_value;
  _value>>=8;
  *(_b->ptr++)=(unsigned char)_value;
  _value>>=8;
  *(_b->ptr++)=(unsigned char)_value;
  _value>>=8;
  *(_b->ptr++)=(unsigned char)_value;
}

void ec_byte_writecopy(ec_byte_buffer *_b,void *_source,long _bytes){
  ptrdiff_t endbyte;
  endbyte=_b->ptr-_b->buf;
  if(endbyte+_bytes>_b->storage){
    _b->storage=endbyte+_bytes+EC_BUFFER_INCREMENT;
    _b->buf=realloc(_b->buf,_b->storage);
    _b->ptr=_b->buf+endbyte;
  }
  memmove(_b->ptr,_source,_bytes);
  _b->ptr+=_bytes;
}

void ec_byte_writeclear(ec_byte_buffer *_b){
  free(_b->buf);
}



void ec_enc_bits(ec_enc *_this,ec_uint32 _fl,int _ftb){
  unsigned fl;
  unsigned ft;
  while(_ftb>EC_UNIT_BITS){
    _ftb-=EC_UNIT_BITS;
    fl=(unsigned)(_fl>>_ftb)&EC_UNIT_MASK;
    ec_encode(_this,fl,fl+1,EC_UNIT_MASK+1);
  }
  ft=1<<_ftb;
  fl=_fl&ft-1;
  ec_encode(_this,fl,fl+1,ft);
}

void ec_enc_uint(ec_enc *_this,ec_uint32 _fl,ec_uint32 _ft){
  ec_uint32 mask;
  ec_uint32 ft;
  unsigned  fl;
  int       ftb;
  _ft--;
  ftb=EC_ILOG(_ft);
  while(ftb>EC_UNIT_BITS){
    ftb-=EC_UNIT_BITS;
    ft=(_ft>>ftb)+1;
    fl=(unsigned)(_fl>>ftb);
    ec_encode(_this,fl,fl+1,ft);
    if(fl<ft-1){
      ec_enc_bits(_this,_fl,ftb);
      return;
    }
    mask=((ec_uint32)1<<ftb)-1;
    _fl=_fl&mask;
    _ft=_ft&mask;
  }
  ec_encode(_this,_fl,_fl+1,_ft+1);
}
