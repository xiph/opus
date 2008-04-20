/* (C) 2007-2008 Timothy B. Terriberry
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

/* Functions for encoding and decoding pulse vectors.
   These are based on the function
     U(n,m) = U(n-1,m) + U(n,m-1) + U(n-1,m-1),
     U(n,1) = U(1,m) = 2,
    which counts the number of ways of placing m pulses in n dimensions, where
     at least one pulse lies in dimension 0.
   For more details, see: http://people.xiph.org/~tterribe/notes/cwrs.html
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "os_support.h"
#include <stdlib.h>
#include <string.h>
#include "cwrs.h"
#include "mathops.h"

/*Computes the next row/column of any recurrence that obeys the relation
   u[i][j]=u[i-1][j]+u[i][j-1]+u[i-1][j-1].
  _ui0 is the base case for the new row/column.*/
static inline void unext32(celt_uint32_t *_ui,int _len,celt_uint32_t _ui0){
  celt_uint32_t ui1;
  int           j;
  /* doing a do-while would overrun the array if we had less than 2 samples */
  j=1; do {
    ui1=_ui[j]+_ui[j-1]+_ui0;
    _ui[j-1]=_ui0;
    _ui0=ui1;
  } while (++j<_len);
  _ui[j-1]=_ui0;
}

static inline void unext64(celt_uint64_t *_ui,int _len,celt_uint64_t _ui0){
  celt_uint64_t ui1;
  int           j;
  /* doing a do-while would overrun the array if we had less than 2 samples */
  j=1; do {
    ui1=_ui[j]+_ui[j-1]+_ui0;
    _ui[j-1]=_ui0;
    _ui0=ui1;
  } while (++j<_len);
  _ui[j-1]=_ui0;
}

/*Computes the previous row/column of any recurrence that obeys the relation
   u[i-1][j]=u[i][j]-u[i][j-1]-u[i-1][j-1].
  _ui0 is the base case for the new row/column.*/
static inline void uprev32(celt_uint32_t *_ui,int _n,celt_uint32_t _ui0){
  celt_uint32_t ui1;
  int           j;
  /* doing a do-while would overrun the array if we had less than 2 samples */
  j=1; do {
    ui1=_ui[j]-_ui[j-1]-_ui0;
    _ui[j-1]=_ui0;
    _ui0=ui1;
  } while (++j<_n);
  _ui[j-1]=_ui0;
}

static inline void uprev64(celt_uint64_t *_ui,int _n,celt_uint64_t _ui0){
  celt_uint64_t ui1;
  int           j;
  /* doing a do-while would overrun the array if we had less than 2 samples */
  j=1; do {
    ui1=_ui[j]-_ui[j-1]-_ui0;
    _ui[j-1]=_ui0;
    _ui0=ui1;
  } while (++j<_n);
  _ui[j-1]=_ui0;
}

/*Returns the number of ways of choosing _m elements from a set of size _n with
   replacement when a sign bit is needed for each unique element.
  On input, _u should be initialized to column (_m-1) of U(n,m).
  On exit, _u will be initialized to column _m of U(n,m).*/
celt_uint32_t ncwrs_unext32(int _n,celt_uint32_t *_ui){
  celt_uint32_t ret;
  celt_uint32_t ui0;
  celt_uint32_t ui1;
  int           j;
  ret=ui0=2;
  celt_assert(_n>=2);
  j=1; do {
    ui1=_ui[j]+_ui[j-1]+ui0;
    _ui[j-1]=ui0;
    ui0=ui1;
    ret+=ui0;
  } while (++j<_n);
  _ui[j-1]=ui0;
  return ret;
}

celt_uint64_t ncwrs_unext64(int _n,celt_uint64_t *_ui){
  celt_uint64_t ret;
  celt_uint64_t ui0;
  celt_uint64_t ui1;
  int           j;
  ret=ui0=2;
  celt_assert(_n>=2);
  j=1; do {
    ui1=_ui[j]+_ui[j-1]+ui0;
    _ui[j-1]=ui0;
    ui0=ui1;
    ret+=ui0;
  } while (++j<_n);
  _ui[j-1]=ui0;
  return ret;
}

/*Returns the number of ways of choosing _m elements from a set of size _n with
   replacement when a sign bit is needed for each unique element.
  On exit, _u will be initialized to column _m of U(n,m).*/
celt_uint32_t ncwrs_u32(int _n,int _m,celt_uint32_t *_u){
  int k;
  CELT_MEMSET(_u,0,_n);
  if(_m<=0)return 1;
  if(_n<=0)return 0;
  for(k=1;k<_m;k++)unext32(_u,_n,2);
  return ncwrs_unext32(_n,_u);
}

celt_uint64_t ncwrs_u64(int _n,int _m,celt_uint64_t *_u){
  int k;
  CELT_MEMSET(_u,0,_n);
  if(_m<=0)return 1;
  if(_n<=0)return 0;
  for(k=1;k<_m;k++)unext64(_u,_n,2);
  return ncwrs_unext64(_n,_u);
}

/*Returns the _i'th combination of _m elements chosen from a set of size _n
   with associated sign bits.
  _x: Returns the combination with elements sorted in ascending order.
  _s: Returns the associated sign bits.
  _u: Temporary storage already initialized to column _m of U(n,m).
      Its contents will be overwritten.*/
void cwrsi32(int _n,int _m,celt_uint32_t _i,int *_x,int *_s,celt_uint32_t *_u){
  int j;
  int k;
  for(k=j=0;k<_m;k++){
    celt_uint32_t p;
    celt_uint32_t t;
    p=_u[_n-j-1];
    if(k>0){
      t=p>>1;
      if(t<=_i||_s[k-1])_i+=t;
    }
    while(p<=_i){
      _i-=p;
      j++;
      p=_u[_n-j-1];
    }
    t=p>>1;
    _s[k]=_i>=t;
    _x[k]=j;
    if(_s[k])_i-=t;
    uprev32(_u,_n-j,2);
  }
}

void cwrsi64(int _n,int _m,celt_uint64_t _i,int *_x,int *_s,celt_uint64_t *_u){
  int j;
  int k;
  for(k=j=0;k<_m;k++){
    celt_uint64_t p;
    celt_uint64_t t;
    p=_u[_n-j-1];
    if(k>0){
      t=p>>1;
      if(t<=_i||_s[k-1])_i+=t;
    }
    while(p<=_i){
      _i-=p;
      j++;
      p=_u[_n-j-1];
    }
    t=p>>1;
    _s[k]=_i>=t;
    _x[k]=j;
    if(_s[k])_i-=t;
    uprev64(_u,_n-j,2);
  }
}

/*Returns the index of the given combination of _m elements chosen from a set
   of size _n with associated sign bits.
  _x: The combination with elements sorted in ascending order.
  _s: The associated sign bits.
  _u: Temporary storage already initialized to column _m of U(n,m).
      Its contents will be overwritten.*/
celt_uint32_t icwrs32(int _n,int _m,const int *_x,const int *_s,
 celt_uint32_t *_u){
  celt_uint32_t i;
  int           j;
  int           k;
  i=0;
  for(k=j=0;k<_m;k++){
    celt_uint32_t p;
    p=_u[_n-j-1];
    if(k>0)p>>=1;
    while(j<_x[k]){
      i+=p;
      j++;
      p=_u[_n-j-1];
    }
    if((k==0||_x[k]!=_x[k-1])&&_s[k])i+=p>>1;
    uprev32(_u,_n-j,2);
  }
  return i;
}

celt_uint64_t icwrs64(int _n,int _m,const int *_x,const int *_s,
 celt_uint64_t *_u){
  celt_uint64_t i;
  int           j;
  int           k;
  i=0;
  for(k=j=0;k<_m;k++){
    celt_uint64_t p;
    p=_u[_n-j-1];
    if(k>0)p>>=1;
    while(j<_x[k]){
      i+=p;
      j++;
      p=_u[_n-j-1];
    }
    if((k==0||_x[k]!=_x[k-1])&&_s[k])i+=p>>1;
    uprev64(_u,_n-j,2);
  }
  return i;
}

/*Converts a combination _x of _m unit pulses with associated sign bits _s into
   a pulse vector _y of length _n.
  _y: Returns the vector of pulses.
  _x: The combination with elements sorted in ascending order. _x[_m] = -1
  _s: The associated sign bits.*/
void comb2pulse(int _n,int _m,int * restrict _y,const int *_x,const int *_s){
  int k;
  const int signs[2]={1,-1};
  CELT_MEMSET(_y, 0, _n);
  k=0; do {
    _y[_x[k]]+=signs[_s[k]];
  } while (++k<_m);
}

/*Converts a pulse vector vector _y of length _n into a combination of _m unit
   pulses with associated sign bits _s.
  _x: Returns the combination with elements sorted in ascending order.
  _s: Returns the associated sign bits.
  _y: The vector of pulses, whose sum of absolute values must be _m.*/
void pulse2comb(int _n,int _m,int *_x,int *_s,const int *_y){
  int j;
  int k;
  for(k=j=0;j<_n;j++){
    if(_y[j]){
      int n;
      int s;
      n=abs(_y[j]);
      s=_y[j]<0;
      do {
        _x[k]=j;
        _s[k]=s;
        k++;
      } while (--n>0);
    }
  }
}

static inline void encode_comb32(int _n,int _m,const int *_x,const int *_s,
 ec_enc *_enc){
  VARDECL(celt_uint32_t,u);
  celt_uint32_t nc;
  celt_uint32_t i;
  SAVE_STACK;
  ALLOC(u,_n,celt_uint32_t);
  nc=ncwrs_u32(_n,_m,u);
  i=icwrs32(_n,_m,_x,_s,u);
  ec_enc_uint(_enc,i,nc);
  RESTORE_STACK;
}

static inline void encode_comb64(int _n,int _m,const int *_x,const int *_s,
 ec_enc *_enc){
  VARDECL(celt_uint64_t,u);
  celt_uint64_t nc;
  celt_uint64_t i;
  SAVE_STACK;
  ALLOC(u,_n,celt_uint64_t);
  nc=ncwrs_u64(_n,_m,u);
  i=icwrs64(_n,_m,_x,_s,u);
  ec_enc_uint64(_enc,i,nc);
  RESTORE_STACK;
}

void encode_pulses(int *_y, int N, int K, ec_enc *enc)
{
   VARDECL(int, comb);
   VARDECL(int, signs);
   SAVE_STACK;

   ALLOC(comb, K, int);
   ALLOC(signs, K, int);

   pulse2comb(N, K, comb, signs, _y);
   /* Simple heuristic to figure out whether it fits in 32 bits */
   if((N+4)*(K+4)<250 || (celt_ilog2(N)+1)*K<31)
   {
      encode_comb32(N, K, comb, signs, enc);
   } else {
      encode_comb64(N, K, comb, signs, enc);
   }
   RESTORE_STACK;
}

static inline void decode_comb32(int _n,int _m,int *_x,int *_s,ec_dec *_dec){
  VARDECL(celt_uint32_t,u);
  SAVE_STACK;
  ALLOC(u,_n,celt_uint32_t);
  cwrsi32(_n,_m,ec_dec_uint(_dec,ncwrs_u32(_n,_m,u)),_x,_s,u);
  RESTORE_STACK;
}

static inline void decode_comb64(int _n,int _m,int *_x,int *_s,ec_dec *_dec){
  VARDECL(celt_uint64_t,u);
  SAVE_STACK;
  ALLOC(u,_n,celt_uint64_t);
  cwrsi64(_n,_m,ec_dec_uint64(_dec,ncwrs_u64(_n,_m,u)),_x,_s,u);
  RESTORE_STACK;
}

void decode_pulses(int *_y, int N, int K, ec_dec *dec)
{
   VARDECL(int, comb);
   VARDECL(int, signs);
   SAVE_STACK;

   ALLOC(comb, K, int);
   ALLOC(signs, K, int);
   /* Simple heuristic to figure out whether it fits in 32 bits */
   if((N+4)*(K+4)<250 || (celt_ilog2(N)+1)*K<31)
   {
      decode_comb32(N, K, comb, signs, dec);
   } else {
      decode_comb64(N, K, comb, signs, dec);
   }
   comb2pulse(N, K, _y, comb, signs);
   RESTORE_STACK;
}
