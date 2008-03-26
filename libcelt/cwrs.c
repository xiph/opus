/* (C) 2007 Timothy B. Terriberry
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

/* Functions for encoding and decoding pulse vectors. For more details, see:
   http://people.xiph.org/~tterribe/notes/cwrs.html
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include "cwrs.h"

/* Knowing ncwrs() for a fixed number of pulses m and for all vector sizes n,
   compute ncwrs() for m+1, for all n. Could also be used when m and n are
   swapped just by changing nc */
static void next_ncwrs32(celt_uint32_t *nc, int len, int nc0)
{
   int i;
   celt_uint32_t mem;
   
   mem = nc[0];
   nc[0] = nc0;
   for (i=1;i<len;i++)
   {
      celt_uint32_t tmp = nc[i]+nc[i-1]+mem;
      mem = nc[i];
      nc[i] = tmp;
   }
}

/* Knowing ncwrs() for a fixed number of pulses m and for all vector sizes n,
   compute ncwrs() for m-1, for all n. Could also be used when m and n are
   swapped just by changing nc */
static void prev_ncwrs32(celt_uint32_t *nc, int len, int nc0)
{
   int i;
   celt_uint32_t mem;
   
   mem = nc[0];
   nc[0] = nc0;
   for (i=1;i<len;i++)
   {
      celt_uint32_t tmp = nc[i]-nc[i-1]-mem;
      mem = nc[i];
      nc[i] = tmp;
   }
}

static void next_ncwrs64(celt_uint64_t *nc, int len, int nc0)
{
   int i;
   celt_uint64_t mem;
   
   mem = nc[0];
   nc[0] = nc0;
   for (i=1;i<len;i++)
   {
      celt_uint64_t tmp = nc[i]+nc[i-1]+mem;
      mem = nc[i];
      nc[i] = tmp;
   }
}

static void prev_ncwrs64(celt_uint64_t *nc, int len, int nc0)
{
   int i;
   celt_uint64_t mem;
   
   mem = nc[0];
   nc[0] = nc0;
   for (i=1;i<len;i++)
   {
      celt_uint64_t tmp = nc[i]-nc[i-1]-mem;
      mem = nc[i];
      nc[i] = tmp;
   }
}

/*Returns the numer of ways of choosing _m elements from a set of size _n with
   replacement when a sign bit is needed for each unique element.*/
celt_uint32_t ncwrs(int _n,int _m)
{
   int i;
   celt_uint32_t ret;
   VARDECL(celt_uint32_t, nc);
   SAVE_STACK;
   ALLOC(nc,_n+1, celt_uint32_t);
   for (i=0;i<_n+1;i++)
      nc[i] = 1;
   for (i=0;i<_m;i++)
      next_ncwrs32(nc, _n+1, 0);
   ret = nc[_n];
   RESTORE_STACK;
   return ret;
}

/*Returns the numer of ways of choosing _m elements from a set of size _n with
   replacement when a sign bit is needed for each unique element.*/
celt_uint64_t ncwrs64(int _n,int _m)
{
   int i;
   celt_uint64_t ret;
   VARDECL(celt_uint64_t, nc);
   SAVE_STACK;
   ALLOC(nc,_n+1, celt_uint64_t);
   for (i=0;i<_n+1;i++)
      nc[i] = 1;
   for (i=0;i<_m;i++)
      next_ncwrs64(nc, _n+1, 0);
   ret = nc[_n];
   RESTORE_STACK;
   return ret;
}


/*Returns the _i'th combination of _m elements chosen from a set of size _n
   with associated sign bits.
  _x:      Returns the combination with elements sorted in ascending order.
  _s:      Returns the associated sign bits.*/
void cwrsi(int _n,int _m,celt_uint32_t _i,int * restrict _x,int * restrict _s){
  int j;
  int k;
  VARDECL(celt_uint32_t, nc);
  SAVE_STACK;
  ALLOC(nc,_n+1, celt_uint32_t);
  for (j=0;j<_n+1;j++)
    nc[j] = 1;
  for (k=0;k<_m-1;k++)
    next_ncwrs32(nc, _n+1, 0);
  for(k=j=0;k<_m;k++){
    celt_uint32_t pn, p, t;
    /*p=ncwrs(_n-j,_m-k-1);
    pn=ncwrs(_n-j-1,_m-k-1);*/
    p=nc[_n-j];
    pn=nc[_n-j-1];
    p+=pn;
    if(k>0){
      t=p>>1;
      if(t<=_i||_s[k-1])_i+=t;
    }
    while(p<=_i){
      _i-=p;
      j++;
      p=pn;
      /*pn=ncwrs(_n-j-1,_m-k-1);*/
      pn=nc[_n-j-1];
      p+=pn;
    }
    t=p>>1;
    _s[k]=_i>=t;
    _x[k]=j;
    if(_s[k])_i-=t;
    if (k<_m-2)
      prev_ncwrs32(nc, _n-j+1, 0);
    else
      prev_ncwrs32(nc, _n-j+1, 1);
  }
  RESTORE_STACK;
}

/*Returns the index of the given combination of _m elements chosen from a set
   of size _n with associated sign bits.
  _x:      The combination with elements sorted in ascending order.
  _s:      The associated sign bits.*/
celt_uint32_t icwrs(int _n,int _m,const int *_x,const int *_s, celt_uint32_t *bound){
  celt_uint32_t i;
  int      j;
  int      k;
  VARDECL(celt_uint32_t, nc);
  SAVE_STACK;
  ALLOC(nc,_n+1, celt_uint32_t);
  for (j=0;j<_n+1;j++)
    nc[j] = 1;
  for (k=0;k<_m;k++)
    next_ncwrs32(nc, _n+1, 0);
  if (bound)
    *bound = nc[_n];
  i=0;
  for(k=j=0;k<_m;k++){
    celt_uint32_t pn;
    celt_uint32_t p;
    if (k<_m-1)
      prev_ncwrs32(nc, _n-j+1, 0);
    else
      prev_ncwrs32(nc, _n-j+1, 1);
    /*p=ncwrs(_n-j,_m-k-1);
    pn=ncwrs(_n-j-1,_m-k-1);*/
    p=nc[_n-j];
    pn=nc[_n-j-1];
    p+=pn;
    if(k>0)p>>=1;
    while(j<_x[k]){
      i+=p;
      j++;
      p=pn;
      /*pn=ncwrs(_n-j-1,_m-k-1);*/
      pn=nc[_n-j-1];
      p+=pn;
    }
    if((k==0||_x[k]!=_x[k-1])&&_s[k])i+=p>>1;
  }
  RESTORE_STACK;
  return i;
}

/*Returns the _i'th combination of _m elements chosen from a set of size _n
   with associated sign bits.
  _x:      Returns the combination with elements sorted in ascending order.
  _s:      Returns the associated sign bits.*/
void cwrsi64(int _n,int _m,celt_uint64_t _i,int * restrict _x,int * restrict _s){
  int j;
  int k;
  VARDECL(celt_uint64_t, nc);
  SAVE_STACK;
  ALLOC(nc,_n+1, celt_uint64_t);
  for (j=0;j<_n+1;j++)
    nc[j] = 1;
  for (k=0;k<_m-1;k++)
    next_ncwrs64(nc, _n+1, 0);
  for(k=j=0;k<_m;k++){
    celt_uint64_t pn, p, t;
    /*p=ncwrs64(_n-j,_m-k-1);
    pn=ncwrs64(_n-j-1,_m-k-1);*/
    p=nc[_n-j];
    pn=nc[_n-j-1];
    p+=pn;
    if(k>0){
      t=p>>1;
      if(t<=_i||_s[k-1])_i+=t;
    }
    while(p<=_i){
      _i-=p;
      j++;
      p=pn;
      /*pn=ncwrs64(_n-j-1,_m-k-1);*/
      pn=nc[_n-j-1];
      p+=pn;
    }
    t=p>>1;
    _s[k]=_i>=t;
    _x[k]=j;
    if(_s[k])_i-=t;
    if (k<_m-2)
      prev_ncwrs64(nc, _n-j+1, 0);
    else
      prev_ncwrs64(nc, _n-j+1, 1);
  }
  RESTORE_STACK;
}

/*Returns the index of the given combination of _m elements chosen from a set
   of size _n with associated sign bits.
  _x:      The combination with elements sorted in ascending order.
  _s:      The associated sign bits.*/
celt_uint64_t icwrs64(int _n,int _m,const int *_x,const int *_s, celt_uint64_t *bound){
  celt_uint64_t i;
  int           j;
  int           k;
  VARDECL(celt_uint64_t, nc);
  SAVE_STACK;
  ALLOC(nc,_n+1, celt_uint64_t);
  for (j=0;j<_n+1;j++)
    nc[j] = 1;
  for (k=0;k<_m;k++)
    next_ncwrs64(nc, _n+1, 0);
  if (bound)
     *bound = nc[_n];
  i=0;
  for(k=j=0;k<_m;k++){
    celt_uint64_t pn;
    celt_uint64_t p;
    if (k<_m-1)
      prev_ncwrs64(nc, _n-j+1, 0);
    else
      prev_ncwrs64(nc, _n-j+1, 1);
    /*p=ncwrs64(_n-j,_m-k-1);
    pn=ncwrs64(_n-j-1,_m-k-1);*/
    p=nc[_n-j];
    pn=nc[_n-j-1];
    p+=pn;
    if(k>0)p>>=1;
    while(j<_x[k]){
      i+=p;
      j++;
      p=pn;
      /*pn=ncwrs64(_n-j-1,_m-k-1);*/
      pn=nc[_n-j-1];
      p+=pn;
    }
    if((k==0||_x[k]!=_x[k-1])&&_s[k])i+=p>>1;
  }
  RESTORE_STACK;
  return i;
}

/*Converts a combination _x of _m unit pulses with associated sign bits _s into
   a pulse vector _y of length _n.
  _y: Returns the vector of pulses.
  _x: The combination with elements sorted in ascending order.
  _s: The associated sign bits.*/
void comb2pulse(int _n,int _m,int *_y,const int *_x,const int *_s){
  int j;
  int k;
  int n;
  for(k=j=0;k<_m;k+=n){
    for(n=1;k+n<_m&&_x[k+n]==_x[k];n++);
    while(j<_x[k])_y[j++]=0;
    _y[j++]=_s[k]?-n:n;
  }
  while(j<_n)_y[j++]=0;
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
      for(;n-->0;k++){
        _x[k]=j;
        _s[k]=s;
      }
    }
  }
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
   if((N+4)*(K+4)<250 || EC_ILOG(N)*K<31)
   {
      celt_uint32_t bound, id;
      id = icwrs(N, K, comb, signs, &bound);
      ec_enc_uint(enc,id,bound);
   } else {
      celt_uint64_t bound, id;
      id = icwrs64(N, K, comb, signs, &bound);
      ec_enc_uint64(enc,id,bound);
   }
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
   if((N+4)*(K+4)<250 || EC_ILOG(N)*K<31)
   {
      cwrsi(N, K, ec_dec_uint(dec, ncwrs(N, K)), comb, signs);
      comb2pulse(N, K, _y, comb, signs);
   } else {
      cwrsi64(N, K, ec_dec_uint64(dec, ncwrs64(N, K)), comb, signs);
      comb2pulse(N, K, _y, comb, signs);
   }
   RESTORE_STACK;
}

