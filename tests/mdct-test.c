#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include "mdct.h"


void check(float  * in,float  * out,int nfft,int isinverse)
{
    int bin,k;
    double errpow=0,sigpow=0;
    
    for (bin=0;bin<nfft/2;++bin) {
        double ansr = 0;
        double difr;

        for (k=0;k<nfft;++k) {
           double phase = 2*M_PI*(k+.5+.25*nfft)*(bin+.5)/nfft;
           double re = cos(phase);
            
           re /= nfft/4;

           ansr += in[k] * re;
        }
        /*printf ("%f %f\n", ansr, out[bin]);*/
        difr = ansr - out[bin];
        errpow += difr*difr;
        sigpow += ansr*ansr;
    }
    printf("nfft=%d inverse=%d,snr = %f\n",nfft,isinverse,10*log10(sigpow/errpow) );
}

void check_inv(float  * in,float  * out,int nfft,int isinverse)
{
   int bin,k;
   double errpow=0,sigpow=0;
    
   for (bin=0;bin<nfft;++bin) {
      double ansr = 0;
      double difr;

      for (k=0;k<nfft/2;++k) {
         double phase = 2*M_PI*(bin+.5+.25*nfft)*(k+.5)/nfft;
         double re = cos(phase);
            
         ansr += in[k] * re;
      }
      /*printf ("%f %f\n", ansr, out[bin]);*/
      difr = ansr - out[bin];
      errpow += difr*difr;
      sigpow += ansr*ansr;
   }
   printf("nfft=%d inverse=%d,snr = %f\n",nfft,isinverse,10*log10(sigpow/errpow) );
}


void test1d(int nfft,int isinverse)
{
    mdct_lookup cfg;
    size_t buflen = sizeof(float)*nfft;

    float  * in = (float*)malloc(buflen);
    float  * out= (float*)malloc(buflen);
    int k;

    mdct_init(&cfg, nfft);
    for (k=0;k<nfft;++k) {
        in[k] = (rand() % 65536) - 32768;
    }

#ifdef DOUBLE_PRECISION
    for (k=0;k<nfft;++k) {
       in[k].r *= 65536;
       in[k].i *= 65536;
    }
#endif
    
    if (isinverse)
    {
       for (k=0;k<nfft;++k) {
          in[k] /= nfft;
       }
    }
    
    /*for (k=0;k<nfft;++k) printf("%d %d ", in[k].r, in[k].i);printf("\n");*/
       
    if (isinverse)
    {
       mdct_backward(&cfg,in,out);
       check_inv(in,out,nfft,isinverse);
    } else {
       mdct_forward(&cfg,in,out);
       check(in,out,nfft,isinverse);
    }
    /*for (k=0;k<nfft;++k) printf("%d %d ", out[k].r, out[k].i);printf("\n");*/


    free(in);
    free(out);
    mdct_clear(&cfg);
}

int main(int argc,char ** argv)
{
    if (argc>1) {
        int k;
        for (k=1;k<argc;++k) {
            test1d(atoi(argv[k]),0);
            test1d(atoi(argv[k]),1);
        }
    }else{
        test1d(32,0);
        test1d(32,1);
        exit(0);
        test1d(36,0);
        test1d(36,1);
        test1d(50,0);
        test1d(50,1);
        test1d(120,0);
        test1d(120,1);
    }
    return 0;
}
