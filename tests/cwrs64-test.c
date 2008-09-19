#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include "cwrs.h"
#include <string.h>

#define NMAX (32)
#define MMAX (16)

int main(int _argc,char **_argv){
  int n;
  for(n=2;n<=NMAX;n+=3){
    int m;
    for(m=1;m<=MMAX;m++){
      celt_uint64_t uu[NMAX];
      celt_uint64_t inc;
      celt_uint64_t nc;
      celt_uint64_t i;
      nc=ncwrs_u64(n,m,uu);
      /*Testing all cases just wouldn't work!*/
      inc=nc/1000;
      if(inc<1)inc=1;
      /*printf("%d/%d: %llu",n,m, nc);*/
      for(i=0;i<nc;i+=inc){
        celt_uint64_t u[NMAX>MMAX+2?NMAX:MMAX+2];
        int           y[NMAX];
        celt_uint64_t v;
        int           k;
        memcpy(u,uu,n*sizeof(*u));
        cwrsi64(n,m,i,nc,y,u);
        /*printf("%llu of %llu:",i,nc);
        for(k=0;k<n;k++)printf(" %+3i",y[k]);
        printf(" ->");*/
        if(icwrs64(n,m,&v,y,u)!=i){
          fprintf(stderr,"Combination-index mismatch.\n");
          return 1;
        }
        if(v!=nc){
          fprintf(stderr,"Combination count mismatch.\n");
          return 2;
        }
        /*printf(" %6llu\n",i);*/
      }
      /*printf("\n");*/
    }
  }
  return 0;
}
