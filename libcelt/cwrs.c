/* (C) 2007 Timothy Terriberry
*/
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


/*#include <stdio.h>*/
#include <stdlib.h>

#include "cwrs.h"

/*Returns the numer of ways of choosing _m elements from a set of size _n with
   replacement when a sign bit is needed for each unique element.*/
#if 0
static unsigned ncwrs(int _n,int _m){
  static unsigned c[32][32];
  if(_n<0||_m<0)return 0;
  if(!c[_n][_m]){
    if(_m<=0)c[_n][_m]=1;
    else if(_n>0)c[_n][_m]=ncwrs(_n-1,_m)+ncwrs(_n,_m-1)+ncwrs(_n-1,_m-1);
  }
  return c[_n][_m];
}

#else

/*Returns the greatest common divisor of _a and _b.*/
static unsigned gcd(unsigned _a,unsigned _b){
  unsigned r;
  while(_b){
    r=_a%_b;
    _a=_b;
    _b=r;
  }
  return _a;
}

/*Returns _a*b/_d, under the assumption that the result is an integer, avoiding
   overflow.
  It is assumed, but not required, that _b is smaller than _a.*/
static unsigned umuldiv(unsigned _a,unsigned _b,unsigned _d){
  unsigned d;
  d=gcd(_b,_d);
  return (_a/(_d/d))*(_b/d);
}

unsigned ncwrs(int _n,int _m){
  unsigned ret;
  unsigned f;
  unsigned d;
  int      i;
  if(_n<0||_m<0)return 0;
  if(_m==0)return 1;
  if(_n==0)return 0;
  ret=0;
  f=_n;
  d=1;
  for(i=1;i<=_m;i++){
    ret+=f*d<<i;
#if 0
    f=umuldiv(f,_n-i,i+1);
    d=umuldiv(d,_m-i,i);
#else
    f=(f*(_n-i))/(i+1);
    d=(d*(_m-i))/i;
#endif
  }
  return ret;
}
#endif

/*Returns the _i'th combination of _m elements chosen from a set of size _n
   with associated sign bits.
  _x:      Returns the combination with elements sorted in ascending order.
  _s:      Returns the associated sign bits.*/
void cwrsi(int _n,int _m,unsigned _i,int *_x,int *_s){
  unsigned pn;
  int      j;
  int      k;
  pn=ncwrs(_n-1,_m);
  for(k=j=0;k<_m;k++){
    unsigned pp;
    unsigned p;
    unsigned t;
    pp=0;
    p=ncwrs(_n-j,_m-k)-pn;
    if(k>0){
      t=p>>1;
      if(t<=_i||_s[k-1])_i+=t;
    }
    pn=ncwrs(_n-j-1,_m-k-1);
    while(p<=_i){
      pp=p;
      j++;
      p+=pn;
      pn=ncwrs(_n-j-1,_m-k-1);
      p+=pn;
    }
    t=p-pp>>1;
    _s[k]=_i-pp>=t;
    _x[k]=j;
    _i-=pp;
    if(_s[k])_i-=t;
  }
}

/*Returns the index of the given combination of _m elements chosen from a set
   of size _n with associated sign bits.
  _x:      The combination with elements sorted in ascending order.
  _s:      The associated sign bits.*/
unsigned icwrs(int _n,int _m,const int *_x,const int *_s){
  unsigned pn;
  unsigned i;
  int      j;
  int      k;
  i=0;
  pn=ncwrs(_n-1,_m);
  for(k=j=0;k<_m;k++){
    unsigned pp;
    unsigned p;
    pp=0;
    p=ncwrs(_n-j,_m-k)-pn;
    if(k>0)p>>=1;
    pn=ncwrs(_n-j-1,_m-k-1);
    while(j<_x[k]){
      pp=p;
      j++;
      p+=pn;
      pn=ncwrs(_n-j-1,_m-k-1);
      p+=pn;
    }
    i+=pp;
    if((k==0||_x[k]!=_x[k-1])&&_s[k])i+=p-pp>>1;
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

/*
#define NMAX (10)
#define MMAX (9)

int main(int _argc,char **_argv){
  int n;
  for(n=0;n<=NMAX;n++){
    int m;
    for(m=0;m<=MMAX;m++){
      unsigned nc;
      unsigned i;
      nc=ncwrs(n,m);
      for(i=0;i<nc;i++){
        int x[MMAX];
        int s[MMAX];
        int x2[MMAX];
        int s2[MMAX];
        int y[NMAX];
        int j;
        int k;
        cwrsi(n,m,i,x,s);
        printf("%6u of %u:",i,nc);
        for(k=0;k<m;k++){
          printf(" %c%i",k>0&&x[k]==x[k-1]?' ':s[k]?'-':'+',x[k]);
        }
        printf(" ->");
        if(icwrs(n,m,x,s)!=i){
          fprintf(stderr,"Combination-index mismatch.\n");
        }
        comb2pulse(n,m,y,x,s);
        for(j=0;j<n;j++)printf(" %c%i",y[j]?y[j]<0?'-':'+':' ',abs(y[j]));
        printf("\n");
        pulse2comb(n,m,x2,s2,y);
        for(k=0;k<m;k++)if(x[k]!=x2[k]||s[k]!=s2[k]){
          fprintf(stderr,"Pulse-combination mismatch.\n");
          break;
        }
      }
      printf("\n");
    }
  }
  return 0;
}
*/
