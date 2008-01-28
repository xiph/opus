#include <stdlib.h>
#include <string.h>
#include "probmod.h"
#include "bitree.h"

void ec_probmod_init(ec_probmod *_this,unsigned _sz){
  ec_probmod_init_full(_this,_sz,1U,1U<<23,NULL);
}

void ec_probmod_init_from_counts(ec_probmod *_this,unsigned _sz,
 const unsigned *_counts){
  ec_probmod_init_full(_this,_sz,1U,1U<<23,_counts);
}

void ec_probmod_init_full(ec_probmod *_this,unsigned _sz,unsigned _inc,
 unsigned _thresh,const unsigned *_counts){
  unsigned s;
  _this->sz=_sz;
  for(s=1;s<=_this->sz;s<<=1);
  _this->split=s>>1;
  _this->inc=_inc;
  _this->thresh=_thresh;
  _this->bitree=(unsigned *)malloc(_sz*sizeof(*_this->bitree));
  if(_counts!=NULL)memcpy(_this->bitree,_counts,_sz*sizeof(*_this->bitree));
  else for(s=0;s<_this->sz;s++)_this->bitree[s]=1;
  ec_bitree_from_counts(_this->bitree,_sz);
  _this->ft=ec_bitree_get_cumul(_this->bitree,_sz);
}

void ec_probmod_clear(ec_probmod *_this){
  free(_this->bitree);
}
