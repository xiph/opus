#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include "cwrs.h"
#include <string.h>
#define NMAX (10)
#define MMAX (9)

int main(int _argc,char **_argv){
  int n;
  for(n=2;n<=NMAX;n++){
    int m;
    for(m=0;m<=MMAX;m++){
      celt_uint32_t uu[NMAX];
      celt_uint32_t inc;
      celt_uint32_t nc;
      celt_uint32_t i;
      nc=ncwrs_u32(n,m,uu);
      inc=nc/10000;
      if(inc<1)inc=1;
      for(i=0;i<nc;i+=inc){
        celt_uint32_t u[NMAX];
        int           x[MMAX];
        int           s[MMAX];
        int           x2[MMAX];
        int           s2[MMAX];
        int           y[NMAX];
        int           k;
        memcpy(u,uu,n*sizeof(*u));
        cwrsi32(n,m,i,x,s,u);
        /*printf("%6u of %u:",i,nc);*/
        /*for(k=0;k<m;k++){
          printf(" %c%i",k>0&&x[k]==x[k-1]?' ':s[k]?'-':'+',x[k]);
        }
        printf(" ->");*/
        memcpy(u,uu,n*sizeof(*u));
        if(icwrs32(n,m,x,s,u)!=i){
          fprintf(stderr,"Combination-index mismatch.\n");
          return 1;
        }
        comb2pulse(n,m,y,x,s);
        /*for(j=0;j<n;j++)printf(" %c%i",y[j]?y[j]<0?'-':'+':' ',abs(y[j]));
        printf("\n");*/
        pulse2comb(n,m,x2,s2,y);
        for(k=0;k<m;k++)if(x[k]!=x2[k]||s[k]!=s2[k]){
          fprintf(stderr,"Pulse-combination mismatch.\n");
          return 1;
        }
      }
      /*printf("\n");*/
    }
  }
  return 0;
}
