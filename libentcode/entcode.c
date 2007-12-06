#include "entcode.h"



void ec_byte_reset(ec_byte_buffer *_b){
  _b->ptr=_b->buf;
}

long ec_byte_bytes(ec_byte_buffer *_b){
  return _b->ptr-_b->buf;
}

unsigned char *ec_byte_get_buffer(ec_byte_buffer *_b){
  return _b->buf;
}



int ec_ilog(ec_uint32 _v){
#if defined(EC_CLZ)
  return EC_CLZ0-EC_CLZ(_v)&-!!_v;
#else
  /*On a Pentium M, this branchless version tested as the fastest on
     1,000,000,000 random 32-bit integers, edging out a similar version with
     branches, and a 256-entry LUT version.*/
  int ret;
  int m;
  ret=!!_v;
  m=!!(_v&0xFFFF0000)<<4;
  _v>>=m;
  ret|=m;
  m=!!(_v&0xFF00)<<3;
  _v>>=m;
  ret|=m;
  m=!!(_v&0xF0)<<2;
  _v>>=m;
  ret|=m;
  m=!!(_v&0xC)<<1;
  _v>>=m;
  ret|=m;
  ret+=!!(_v&0x2);
  return ret;
#endif
}
