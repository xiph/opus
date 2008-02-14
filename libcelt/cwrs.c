/* (C) 2007 Timothy B. Terriberry */
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
#include <stdlib.h>
#include "cwrs.h"

static celt_uint64_t update_ncwrs64(celt_uint64_t *nc, int len, int nc0)
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

static celt_uint64_t reverse_ncwrs64(celt_uint64_t *nc, int len, int nc0)
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

/* Optional implementation of ncwrs64 using update_ncwrs64(). It's slightly
   slower than the standard ncwrs64(), but it could still be useful.
celt_uint64_t ncwrs64_opt(int _n,int _m)
{
   int i;
   celt_uint64_t ret;
   celt_uint64_t nc[_n+1];
   for (i=0;i<_n+1;i++)
      nc[i] = 1;
   for (i=0;i<_m;i++)
      update_ncwrs64(nc, _n+1, 0);
   return nc[_n];
}*/

/*Returns the numer of ways of choosing _m elements from a set of size _n with
   replacement when a sign bit is needed for each unique element.*/
#if 0
static celt_uint32_t ncwrs(int _n,int _m){
  static celt_uint32_t c[32][32];
  if(_n<0||_m<0)return 0;
  if(!c[_n][_m]){
    if(_m<=0)c[_n][_m]=1;
    else if(_n>0)c[_n][_m]=ncwrs(_n-1,_m)+ncwrs(_n,_m-1)+ncwrs(_n-1,_m-1);
  }
  return c[_n][_m];
}
#else
celt_uint32_t ncwrs(int _n,int _m){
  celt_uint32_t ret;
  celt_uint32_t f;
  celt_uint32_t d;
  int      i;
  if(_n<0||_m<0)return 0;
  if(_m==0)return 1;
  if(_n==0)return 0;
  ret=0;
  f=_n;
  d=1;
  for(i=1;i<=_m;i++){
    ret+=f*d<<i;
    f=(f*(_n-i))/(i+1);
    d=(d*(_m-i))/i;
  }
  return ret;
}
#endif

#if 0
celt_uint64_t ncwrs64(int _n,int _m){
  static celt_uint64_t c[101][101];
  if(_n<0||_m<0)return 0;
  if(!c[_n][_m]){
    if(_m<=0)c[_n][_m]=1;
    else if(_n>0)c[_n][_m]=ncwrs64(_n-1,_m)+ncwrs64(_n,_m-1)+ncwrs64(_n-1,_m-1);
}
  return c[_n][_m];
}
#else
celt_uint64_t ncwrs64(int _n,int _m){
  celt_uint64_t ret;
  celt_uint64_t f;
  celt_uint64_t d;
  int           i;
  if(_n<0||_m<0)return 0;
  if(_m==0)return 1;
  if(_n==0)return 0;
  ret=0;
  f=_n;
  d=1;
  for(i=1;i<=_m;i++){
    ret+=f*d<<i;
    f=(f*(_n-i))/(i+1);
    d=(d*(_m-i))/i;
  }
  return ret;
}
#endif

/*Returns the _i'th combination of _m elements chosen from a set of size _n
   with associated sign bits.
  _x:      Returns the combination with elements sorted in ascending order.
  _s:      Returns the associated sign bits.*/
void cwrsi(int _n,int _m,celt_uint32_t _i,int *_x,int *_s){
  int j;
  int k;
  for(k=j=0;k<_m;k++){
    celt_uint32_t pn;
    celt_uint32_t p;
    celt_uint32_t t;
    p=ncwrs(_n-j,_m-k-1);
    pn=ncwrs(_n-j-1,_m-k-1);
    p+=pn;
    if(k>0){
      t=p>>1;
      if(t<=_i||_s[k-1])_i+=t;
    }
    while(p<=_i){
      _i-=p;
      j++;
      p=pn;
      pn=ncwrs(_n-j-1,_m-k-1);
      p+=pn;
    }
    t=p>>1;
    _s[k]=_i>=t;
    _x[k]=j;
    if(_s[k])_i-=t;
  }
}

/*Returns the index of the given combination of _m elements chosen from a set
   of size _n with associated sign bits.
  _x:      The combination with elements sorted in ascending order.
  _s:      The associated sign bits.*/
celt_uint32_t icwrs(int _n,int _m,const int *_x,const int *_s){
  celt_uint32_t i;
  int      j;
  int      k;
  i=0;
  for(k=j=0;k<_m;k++){
    celt_uint32_t pn;
    celt_uint32_t p;
    p=ncwrs(_n-j,_m-k-1);
    pn=ncwrs(_n-j-1,_m-k-1);
    p+=pn;
    if(k>0)p>>=1;
    while(j<_x[k]){
      i+=p;
      j++;
      p=pn;
      pn=ncwrs(_n-j-1,_m-k-1);
      p+=pn;
    }
    if((k==0||_x[k]!=_x[k-1])&&_s[k])i+=p>>1;
  }
  return i;
}

/*Returns the _i'th combination of _m elements chosen from a set of size _n
   with associated sign bits.
  _x:      Returns the combination with elements sorted in ascending order.
  _s:      Returns the associated sign bits.*/
void cwrsi64(int _n,int _m,celt_uint64_t _i,int *_x,int *_s){
  int j;
  int k;
  celt_uint64_t nc[_n+1];
  for (j=0;j<_n+1;j++)
    nc[j] = 1;
  for (k=0;k<_m-1;k++)
    update_ncwrs64(nc, _n+1, 0);
  for(k=j=0;k<_m;k++){
    celt_uint64_t pn;
    celt_uint64_t p;
    celt_uint64_t t;
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
      reverse_ncwrs64(nc, _n+1, 0);
    else
      reverse_ncwrs64(nc, _n+1, 1);
  }
}

/*Returns the index of the given combination of _m elements chosen from a set
   of size _n with associated sign bits.
  _x:      The combination with elements sorted in ascending order.
  _s:      The associated sign bits.*/
celt_uint64_t icwrs64(int _n,int _m,const int *_x,const int *_s){
  celt_uint64_t i;
  int           j;
  int           k;
  celt_uint64_t nc[_n+1];
  for (j=0;j<_n+1;j++)
    nc[j] = 1;
  for (k=0;k<_m-1;k++)
    update_ncwrs64(nc, _n+1, 0);
  i=0;
  for(k=j=0;k<_m;k++){
    celt_uint64_t pn;
    celt_uint64_t p;
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
    if (k<_m-2)
      reverse_ncwrs64(nc, _n+1, 0);
    else
      reverse_ncwrs64(nc, _n+1, 1);
  }
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

