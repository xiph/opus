#include <string.h>
#include "probenc.h"
#include "bitrenc.h"



static void ec_probmod_samp_and_update(ec_probmod *_this,ec_probsamp *_samp,
 unsigned _sym){
  unsigned sz;
  sz=_this->sz;
  _samp->fs=ec_bitree_get_freq(_this->bitree,_sym);
  _samp->fl=ec_bitree_get_cumul(_this->bitree,_sym);
  _samp->ft=_this->ft;
  ec_bitree_update(_this->bitree,sz,_sym,_this->inc);
  _this->ft+=_this->inc;
  if(_this->ft>_this->thresh){
    ec_bitree_halve(_this->bitree,sz,_this->split);
    _this->ft=ec_bitree_get_cumul(_this->bitree,sz);
  }
}

static void ec_probmod_samp_and_update_range(ec_probmod *_this,
 ec_probsamp *_samp,int _sym,int _lo,int _hi){
  unsigned base;
  int      sz;
  sz=_this->sz;
  base=ec_bitree_get_cumul(_this->bitree,_lo);
  _samp->fs=ec_bitree_get_freq(_this->bitree,_sym);
  _samp->fl=ec_bitree_get_cumul(_this->bitree,_sym)-base;
  _samp->ft=ec_bitree_get_cumul(_this->bitree,_hi)-base;
  ec_bitree_update(_this->bitree,sz,_sym,_this->inc);
  _this->ft+=_this->inc;
  if(_this->ft>_this->thresh){
    ec_bitree_halve(_this->bitree,sz,_this->split);
    _this->ft=ec_bitree_get_cumul(_this->bitree,sz);
  }
}

void ec_probmod_write(ec_probmod *_this,ec_enc *_enc,int _sym){
  ec_probsamp samp;
  ec_probmod_samp_and_update(_this,&samp,_sym);
  ec_encode(_enc,samp.fl,samp.fl+samp.fs,samp.ft);
}

void ec_probmod_write_range(ec_probmod *_this,ec_enc *_enc,int _sym,
 int _lo,int _hi){
  ec_probsamp samp;
  ec_probmod_samp_and_update_range(_this,&samp,_sym,_lo,_hi);
  ec_encode(_enc,samp.fl,samp.fl+samp.fs,samp.ft);
}
