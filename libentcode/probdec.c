#include <stdlib.h>
#include <string.h>
#include "probdec.h"
#include "bitrdec.h"



/*Gets the cumulative frequency count between _lo and _hi, as well as the
   cumulative frequency count below _lo.*/
static unsigned ec_probmod_get_total(ec_probmod *_this,unsigned *_fl,
 unsigned _lo,unsigned _hi){
  *_fl=ec_bitree_get_cumul(_this->bitree,_lo);
  return ec_bitree_get_cumul(_this->bitree,_hi)-*_fl;
}

static int ec_probmod_find_and_update(ec_probmod *_this,ec_probsamp *_samp,
 unsigned _freq){
  int sym;
  sym=ec_bitree_find_and_update(_this->bitree,_this->sz,_this->split,
   _freq,&_samp->fl,_this->inc);
  _samp->fs=ec_bitree_get_freq(_this->bitree,sym)-_this->inc;
  _this->ft+=_this->inc;
  if(_this->ft>_this->thresh){
    ec_bitree_halve(_this->bitree,_this->sz,_this->split);
    _this->ft=ec_bitree_get_cumul(_this->bitree,_this->sz);
  }
  return sym;
}



int ec_probmod_read(ec_probmod *_this,ec_dec *_dec){
  ec_probsamp samp;
  unsigned    freq;
  int         sym;
  samp.ft=_this->ft;
  freq=ec_decode(_dec,samp.ft);
  sym=ec_probmod_find_and_update(_this,&samp,freq);
  ec_dec_update(_dec,samp.fl,samp.fl+samp.fs,samp.ft);
  return sym;
}

int ec_probmod_read_range(ec_probmod *_this,ec_dec *_dec,int _lo,int _hi){
  ec_probsamp samp;
  unsigned    freq;
  unsigned    base;
  int         sz;
  int         sym;
  sz=_this->sz;
  _lo=EC_MINI(_lo,sz);
  _hi=EC_MINI(_hi,sz);
  if(_hi<=_lo)return -1;
  samp.ft=ec_probmod_get_total(_this,&base,_lo,_hi);
  freq=ec_decode(_dec,samp.ft);
  sym=ec_probmod_find_and_update(_this,&samp,freq+base);
  samp.fl-=base;
  ec_dec_update(_dec,samp.fl,samp.fl+samp.fs,samp.ft);
  return sym;
}
