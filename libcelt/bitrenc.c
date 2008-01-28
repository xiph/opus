#include "bitrenc.h"

void ec_bitree_update(unsigned *_this,int _sz,int _sym,int _val){
  do{
    _this[_sym]+=_val;
    _sym+=_sym+1&-(_sym+1);
  }
  while(_sym<_sz);
}
