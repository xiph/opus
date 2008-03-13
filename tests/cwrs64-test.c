#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include "cwrs.h"
#define NMAX (32)
#define MMAX (16)

int main(int _argc,char **_argv){
  int n;
  for(n=0;n<=NMAX;n+=3){
    int m;
    for(m=0;m<=MMAX;m++){
      celt_uint64_t inc;
      celt_uint64_t nc;
      celt_uint64_t i;
      nc=ncwrs64(n,m);
      /* Testing all cases just wouldn't work! */
      inc = nc/1000;
      if (inc<1)
         inc = 1;
      /*printf("%d/%d: %llu",n,m, nc);*/
      for(i=0;i<nc;i+=inc){
        int x[MMAX];
        int s[MMAX];
        int x2[MMAX];
        int s2[MMAX];
        int y[NMAX];
        int k;
        cwrsi64(n,m,i,x,s);
        /*printf("%llu of %llu:",i,nc);
        for(k=0;k<m;k++){
          printf(" %c%i",k>0&&x[k]==x[k-1]?' ':s[k]?'-':'+',x[k]);
        }
        printf(" ->");*/
        if(icwrs64(n,m,x,s, NULL)!=i){
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
