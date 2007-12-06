#include "bitree.h"

int ec_bitree_find_and_update(unsigned *_this,int _sz,int _split,
 unsigned _freq,unsigned *_fl,int _val){
  int base;
  int test;
  int fl;
  base=-1;
  fl=0;
  while(_split>0){
    test=base+_split;
    if(test<_sz){
      if(_freq>=_this[test]){
        _freq-=_this[test];
        fl+=_this[test];
        base=test;
      }
      else _this[test]+=_val;
    }
    _split>>=1;
  }
  *_fl=fl;
  return base+1;
}
