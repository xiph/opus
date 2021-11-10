/*Daala video codec
Copyright (c) 2012 Daala project contributors.  All rights reserved.
Author: Timothy B. Terriberry

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "kiss99.h"

void kiss99_srand(kiss99_ctx *_this,const unsigned char *_data,int _ndata){
  int i;
  _this->z=362436069;
  _this->w=521288629;
  _this->jsr=123456789;
  _this->jcong=380116160;
  for(i=3;i<_ndata;i+=4){
    _this->z^=_data[i-3];
    _this->w^=_data[i-2];
    _this->jsr^=_data[i-1];
    _this->jcong^=_data[i];
    kiss99_rand(_this);
  }
  if(i-3<_ndata)_this->z^=_data[i-3];
  if(i-2<_ndata)_this->w^=_data[i-2];
  if(i-1<_ndata)_this->jsr^=_data[i-1];
}

uint32_t kiss99_rand(kiss99_ctx *_this){
  uint32_t znew;
  uint32_t wnew;
  uint32_t mwc;
  uint32_t shr3;
  uint32_t cong;
  znew=36969*(_this->z&0xFFFF)+(_this->z>>16);
  wnew=18000*(_this->w&0xFFFF)+(_this->w>>16);
  mwc=(znew<<16)+wnew;
  shr3=_this->jsr^(_this->jsr<<17);
  shr3^=shr3>>13;
  shr3^=shr3<<5;
  cong=69069*_this->jcong+1234567;
  _this->z=znew;
  _this->w=wnew;
  _this->jsr=shr3;
  _this->jcong=cong;
  return (mwc^cong)+shr3;
}
