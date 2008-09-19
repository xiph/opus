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

#if 0
int log2_frac(ec_uint32 val, int frac)
{
   int i;
   /* EC_ILOG() actually returns log2()+1, go figure */
   int L = EC_ILOG(val)-1;
   /*printf ("in: %d %d ", val, L);*/
   if (L>14)
      val >>= L-14;
   else if (L<14)
      val <<= 14-L;
   L <<= frac;
   /*printf ("%d\n", val);*/
   for (i=0;i<frac;i++)
{
      val = (val*val) >> 15;
      /*printf ("%d\n", val);*/
      if (val > 16384)
         L |= (1<<(frac-i-1));
      else   
         val <<= 1;
}
   return L;
}
#endif

int log2_frac64(ec_uint64 val, int frac)
{
   int i;
   /* EC_ILOG64() actually returns log2()+1, go figure */
   int L = EC_ILOG64(val)-1;
   /*printf ("in: %d %d ", val, L);*/
   if (L>14)
      val >>= L-14;
   else if (L<14)
      val <<= 14-L;
   L <<= frac;
   /*printf ("%d\n", val);*/
   for (i=0;i<frac;i++)
   {
      val = (val*val) >> 15;
      /*printf ("%d\n", val);*/
      if (val > 16384)
         L |= (1<<(frac-i-1));
      else   
         val <<= 1;
   }
   return L;
}

int fits_in32(int _n, int _m)
{
   static const celt_int16_t maxN[15] = {
      255, 255, 255, 255, 255, 109,  60,  40,
       29,  24,  20,  18,  16,  14,  13};
   static const celt_int16_t maxM[15] = {
      255, 255, 255, 255, 255, 238,  95,  53,
       36,  27,  22,  18,  16,  15,  13};
   if (_n>=14)
   {
      if (_m>=14)
         return 0;
      else
         return _n <= maxN[_m];
   } else {
      return _m <= maxM[_n];
   }   
}
int fits_in64(int _n, int _m)
{
   static const celt_int16_t maxN[28] = {
      255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 178, 129, 100,  81,  68,  58,
       51,  46,  42,  38,  36,  33,  31,  30,
       28, 27, 26, 25};
   static const celt_int16_t maxM[28] = {
      255, 255, 255, 255, 255, 255, 255, 255, 
      255, 255, 245, 166, 122,  94,  77,  64, 
       56,  49,  44,  40,  37,  34,  32,  30,
       29,  27,  26,  25};
   if (_n>=27)
   {
      if (_m>=27)
         return 0;
      else
         return _n <= maxN[_m];
   } else {
      return _m <= maxM[_n];
   }
}

#define MASK32 (0xFFFFFFFF)

/*INV_TABLE[i] holds the multiplicative inverse of (2*i-1) mod 2**32.*/
static const unsigned INV_TABLE[128]={
  0x00000001,0xAAAAAAAB,0xCCCCCCCD,0xB6DB6DB7,
  0x38E38E39,0xBA2E8BA3,0xC4EC4EC5,0xEEEEEEEF,
  0xF0F0F0F1,0x286BCA1B,0x3CF3CF3D,0xE9BD37A7,
  0xC28F5C29,0x684BDA13,0x4F72C235,0xBDEF7BDF,
  0x3E0F83E1,0x8AF8AF8B,0x914C1BAD,0x96F96F97,
  0xC18F9C19,0x2FA0BE83,0xA4FA4FA5,0x677D46CF,
  0x1A1F58D1,0xFAFAFAFB,0x8C13521D,0x586FB587,
  0xB823EE09,0xA08AD8F3,0xC10C9715,0xBEFBEFBF,
  0xC0FC0FC1,0x07A44C6B,0xA33F128D,0xE327A977,
  0xC7E3F1F9,0x962FC963,0x3F2B3885,0x613716AF,
  0x781948B1,0x2B2E43DB,0xFCFCFCFD,0x6FD0EB67,
  0xFA3F47E9,0xD2FD2FD3,0x3F4FD3F5,0xD4E25B9F,
  0x5F02A3A1,0xBF5A814B,0x7C32B16D,0xD3431B57,
  0xD8FD8FD9,0x8D28AC43,0xDA6C0965,0xDB195E8F,
  0x0FDBC091,0x61F2A4BB,0xDCFDCFDD,0x46FDD947,
  0x56BE69C9,0xEB2FDEB3,0x26E978D5,0xEFDFBF7F,
  0x0FE03F81,0xC9484E2B,0xE133F84D,0xE1A8C537,
  0x077975B9,0x70586723,0xCD29C245,0xFAA11E6F,
  0x0FE3C071,0x08B51D9B,0x8CE2CABD,0xBF937F27,
  0xA8FE53A9,0x592FE593,0x2C0685B5,0x2EB11B5F,
  0xFCD1E361,0x451AB30B,0x72CFE72D,0xDB35A717,
  0xFB74A399,0xE80BFA03,0x0D516325,0x1BCB564F,
  0xE02E4851,0xD962AE7B,0x10F8ED9D,0x95AEDD07,
  0xE9DC0589,0xA18A4473,0xEA53FA95,0xEE936F3F,
  0x90948F41,0xEAFEAFEB,0x3D137E0D,0xEF46C0F7,
  0x028C1979,0x791064E3,0xC04FEC05,0xE115062F,
  0x32385831,0x6E68575B,0xA10D387D,0x6FECF2E7,
  0x3FB47F69,0xED4BFB53,0x74FED775,0xDB43BB1F,
  0x87654321,0x9BA144CB,0x478BBCED,0xBFB912D7,
  0x1FDCD759,0x14B2A7C3,0xCB125CE5,0x437B2E0F,
  0x10FEF011,0xD2B3183B,0x386CAB5D,0xEF6AC0C7,
  0x0E64C149,0x9A020A33,0xE6B41C55,0xFEFEFEFF
};

/*Computes (_a*_b-_c)/(2*_d-1) when the quotient is known to be exact.
  _a, _b, _c, and _d may be arbitrary so long as the arbitrary precision result
   fits in 32 bits, but currently the table for multiplicative inverses is only
   valid for _d<128.*/
static inline celt_uint32_t imusdiv32odd(celt_uint32_t _a,celt_uint32_t _b,
 celt_uint32_t _c,celt_uint32_t _d){
  return (_a*_b-_c)*INV_TABLE[_d]&MASK32;
}

/*Computes (_a*_b-_c)/_d when the quotient is known to be exact.
  _d does not actually have to be even, but imusdiv32odd will be faster when
   it's odd, so you should use that instead.
  _a and _d are assumed to be small (e.g., _a*_d fits in 32 bits; currently the
   table for multiplicative inverses is only valid for _d<256).
  _b and _c may be arbitrary so long as the arbitrary precision reuslt fits in
   32 bits.*/
static inline celt_uint32_t imusdiv32even(celt_uint32_t _a,celt_uint32_t _b,
 celt_uint32_t _c,celt_uint32_t _d){
  unsigned inv;
  int      mask;
  int      shift;
  int      one;
  shift=EC_ILOG(_d^_d-1);
  inv=INV_TABLE[_d-1>>shift];
  shift--;
  one=1<<shift;
  mask=one-1;
  return (_a*(_b>>shift)-(_c>>shift)+
   (_a*(_b&mask)+one-(_c&mask)>>shift)-1)*inv&MASK32;
}

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
  ret=ui0=1;
  celt_assert(_n>=2);
  j=1; do {
    ui1=_ui[j]+_ui[j-1]+ui0;
    _ui[j-1]=ui0;
    ui0=ui1;
    ret+=ui0;
  } while (++j<_n);
  _ui[j-1]=ui0;
  return ret<<=1;
}

/*Returns the number of ways of choosing _m elements from a set of size _n with
   replacement when a sign bit is needed for each unique element.
  _u: On exit, _u[i] contains U(i+1,_m).*/
celt_uint32_t ncwrs_u32(int _n,int _m,celt_uint32_t *_u){
  celt_uint32_t ret;
  celt_uint32_t um2;
  int           k;
  /*If _m==0, _u[] should be set to zero and the return should be 1.*/
  celt_assert(_m>0);
  /*We'll overflow our buffer unless _n>=2.*/
  celt_assert(_n>=2);
  um2=_u[0]=1;
  if(_m<=6){
    if(_m<2){
      k=1;
      do _u[k]=1;
      while(++k<_n);
    }
    else{
      k=1;
      do _u[k]=(k<<1)+1;
      while(++k<_n);
      for(k=2;k<_m;k++)unext32(_u,_n,1);
    }
  }
  else{
    celt_uint32_t um1;
    celt_uint32_t n2m1;
    _u[1]=n2m1=um1=(_m<<1)-1;
    for(k=2;k<_n;k++){
      /*U(n,m) = ((2*n-1)*U(n,m-1)-U(n,m-2))/(m-1) + U(n,m-2)*/
      _u[k]=um2=imusdiv32even(n2m1,um1,um2,k)+um2;
      if(++k>=_n)break;
      _u[k]=um1=imusdiv32odd(n2m1,um2,um1,k>>1)+um1;
    }
  }
  ret=1;
  k=1;
  do ret+=_u[k];
  while(++k<_n);
  return ret<<1;
}

celt_uint64_t ncwrs_u64(int _n,int _m,celt_uint64_t *_u){
  int k;
  CELT_MEMSET(_u,0,_n);
  if(_m<=0)return 1;
  if(_n<=0)return 0;
  for(k=1;k<_m;k++)unext64(_u,_n,1);
  return ncwrs_unext64(_n,_u);
}


/*Returns the _i'th combination of _m elements chosen from a set of size _n
   with associated sign bits.
  _y: Returns the vector of pulses.
  _u: Must contain entries [1..._n] of column _m of U() on input.
      Its contents will be destructively modified.*/
void cwrsi32(int _n,int _m,celt_uint32_t _i,celt_uint32_t _nc,int *_y,
 celt_uint32_t *_u){
  celt_uint32_t p;
  celt_uint32_t q;
  int           j;
  int           k;
  celt_assert(_n>0);
  p=_nc;
  q=0;
  j=0;
  k=_m;
  do{
    int s;
    int yj;
    p-=q;
    q=_u[_n-j-1];
    p-=q;
    s=_i>=p;
    if(s)_i-=p;
    yj=k;
    while(q>_i){
      uprev32(_u,_n-j,--k>0);
      p=q;
      q=_u[_n-j-1];
    }
    _i-=q;
    yj-=k;
    _y[j]=yj-(yj<<1&-s);
  }
  while(++j<_n);
}

/*Returns the _i'th combination of _m elements chosen from a set of size _n
   with associated sign bits.
  _y: Returns the vector of pulses.
  _u: Must contain entries [1..._n] of column _m of U() on input.
      Its contents will be destructively modified.*/
void cwrsi64(int _n,int _m,celt_uint64_t _i,celt_uint64_t _nc,int *_y,
 celt_uint64_t *_u){
  celt_uint64_t p;
  celt_uint64_t q;
  int           j;
  int           k;
  celt_assert(_n>0);
  p=_nc;
  q=0;
  j=0;
  k=_m;
  do{
    int s;
    int yj;
    p-=q;
    q=_u[_n-j-1];
    p-=q;
    s=_i>=p;
    if(s)_i-=p;
    yj=k;
    while(q>_i){
      uprev64(_u,_n-j,--k>0);
      p=q;
      q=_u[_n-j-1];
    }
    _i-=q;
    yj-=k;
    _y[j]=yj-(yj<<1&-s);
  }
  while(++j<_n);
}

/*Returns the index of the given combination of _m elements chosen from a set
   of size _n with associated sign bits.
  _y:  The vector of pulses, whose sum of absolute values must be _m.
  _nc: Returns V(_n,_m).*/
celt_uint32_t icwrs32(int _n,int _m,celt_uint32_t *_nc,const int *_y,
 celt_uint32_t *_u){
  celt_uint32_t nc;
  celt_uint32_t i;
  int           j;
  int           k;
  /*We can't unroll the first two iterations of the loop unless _n>=2.*/
  celt_assert(_n>=2);
  nc=1;
  i=_y[_n-1]<0;
  _u[0]=0;
  for(k=1;k<=_m+1;k++)_u[k]=(k<<1)-1;
  k=abs(_y[_n-1]);
  j=_n-2;
  nc+=_u[_m];
  i+=_u[k];
  k+=abs(_y[j]);
  if(_y[j]<0)i+=_u[k+1];
  while(j-->0){
    unext32(_u,_m+2,0);
    nc+=_u[_m];
    i+=_u[k];
    k+=abs(_y[j]);
    if(_y[j]<0)i+=_u[k+1];
  }
  /*If _m==0, nc should not be doubled.*/
  celt_assert(_m>0);
  *_nc=nc<<1;
  return i;
}

/*Returns the index of the given combination of _m elements chosen from a set
   of size _n with associated sign bits.
  _y:  The vector of pulses, whose sum of absolute values must be _m.
  _nc: Returns V(_n,_m).*/
celt_uint64_t icwrs64(int _n,int _m,celt_uint64_t *_nc,const int *_y,
 celt_uint64_t *_u){
  celt_uint64_t nc;
  celt_uint64_t i;
  int           j;
  int           k;
  /*We can't unroll the first two iterations of the loop unless _n>=2.*/
  celt_assert(_n>=2);
  nc=1;
  i=_y[_n-1]<0;
  _u[0]=0;
  for(k=1;k<=_m+1;k++)_u[k]=(k<<1)-1;
  k=abs(_y[_n-1]);
  j=_n-2;
  nc+=_u[_m];
  i+=_u[k];
  k+=abs(_y[j]);
  if(_y[j]<0)i+=_u[k+1];
  while(j-->0){
    unext64(_u,_m+2,0);
    nc+=_u[_m];
    i+=_u[k];
    k+=abs(_y[j]);
    if(_y[j]<0)i+=_u[k+1];
  }
  /*If _m==0, nc should not be doubled.*/
  celt_assert(_m>0);
  *_nc=nc<<1;
  return i;
}

static inline void encode_pulse32(int _n,int _m,const int *_y,ec_enc *_enc){
  VARDECL(celt_uint32_t,u);
  celt_uint32_t nc;
  celt_uint32_t i;
  SAVE_STACK;
  ALLOC(u,_m+2,celt_uint32_t);
  i=icwrs32(_n,_m,&nc,_y,u);
  ec_enc_uint(_enc,i,nc);
  RESTORE_STACK;
}

static inline void encode_pulse64(int _n,int _m,const int *_y,ec_enc *_enc){
  VARDECL(celt_uint64_t,u);
  celt_uint64_t nc;
  celt_uint64_t i;
  SAVE_STACK;
  ALLOC(u,_m+2,celt_uint64_t);
  i=icwrs64(_n,_m,&nc,_y,u);
  ec_enc_uint64(_enc,i,nc);
  RESTORE_STACK;
}

int get_required_bits(int N, int K, int frac)
{
   int nbits = 0;
   if(fits_in64(N,K))
   {
      VARDECL(celt_uint64_t,u);
      SAVE_STACK;
      ALLOC(u,N,celt_uint64_t);
      nbits = log2_frac64(ncwrs_u64(N,K,u), frac);
      RESTORE_STACK;
   } else {
      nbits = log2_frac64(N, frac);
      nbits += get_required_bits(N/2+1, (K+1)/2, frac);
      nbits += get_required_bits(N/2+1, K/2, frac);
   }
   return nbits;
}


void encode_pulses(int *_y, int N, int K, ec_enc *enc)
{
   if (K==0) {
   } else if (N==1)
   {
      ec_enc_bits(enc, _y[0]<0, 1);
   } else if(fits_in32(N,K))
   {
      encode_pulse32(N, K, _y, enc);
   } else if(fits_in64(N,K)) {
      encode_pulse64(N, K, _y, enc);
   } else {
     int i;
     int count=0;
     int split;
     split = (N+1)/2;
     for (i=0;i<split;i++)
        count += abs(_y[i]);
     ec_enc_uint(enc,count,K+1);
     encode_pulses(_y, split, count, enc);
     encode_pulses(_y+split, N-split, K-count, enc);
   }
}

static inline void decode_pulse32(int _n,int _m,int *_y,ec_dec *_dec){
  VARDECL(celt_uint32_t,u);
  celt_uint32_t nc;
  SAVE_STACK;
  ALLOC(u,_n,celt_uint32_t);
  nc=ncwrs_u32(_n,_m,u);
  cwrsi32(_n,_m,ec_dec_uint(_dec,nc),nc,_y,u);
  RESTORE_STACK;
}

static inline void decode_pulse64(int _n,int _m,int *_y,ec_dec *_dec){
  VARDECL(celt_uint64_t,u);
  celt_uint64_t nc;
  SAVE_STACK;
  ALLOC(u,_n,celt_uint64_t);
  nc=ncwrs_u64(_n,_m,u);
  cwrsi64(_n,_m,ec_dec_uint64(_dec,nc),nc,_y,u);
  RESTORE_STACK;
}

void decode_pulses(int *_y, int N, int K, ec_dec *dec)
{
   if (K==0) {
      int i;
      for (i=0;i<N;i++)
         _y[i] = 0;
   } else if (N==1)
   {
      int s = ec_dec_bits(dec, 1);
      if (s==0)
         _y[0] = K;
      else
         _y[0] = -K;
   } else if(fits_in32(N,K))
   {
      decode_pulse32(N, K, _y, dec);
   } else if(fits_in64(N,K)) {
      decode_pulse64(N, K, _y, dec);
   } else {
     int split;
     int count = ec_dec_uint(dec,K+1);
     split = (N+1)/2;
     decode_pulses(_y, split, count, dec);
     decode_pulses(_y+split, N-split, K-count, dec);
   }
}
