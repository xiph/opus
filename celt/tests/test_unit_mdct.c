#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define SKIP_CONFIG_H

#ifndef CUSTOM_MODES
#define CUSTOM_MODES
#endif

#include <stdio.h>

#define CELT_C
#include "mdct.h"
#include "stack_alloc.h"

#include "kiss_fft.c"
#include "mdct.c"
#include "mathops.c"
#include "entcode.c"

#ifndef M_PI
#define M_PI 3.141592653
#endif

int ret = 0;
void check(kiss_fft_scalar  * in,kiss_fft_scalar  * out,int nfft,int isinverse)
{
    int bin,k;
    double errpow=0,sigpow=0;
    double snr;
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
    snr = 10*log10(sigpow/errpow);
    printf("nfft=%d inverse=%d,snr = %f\n",nfft,isinverse,snr );
    if (snr<60) {
       printf( "** poor snr: %f **\n", snr);
       ret = 1;
    }
}

void check_inv(kiss_fft_scalar  * in,kiss_fft_scalar  * out,int nfft,int isinverse)
{
   int bin,k;
   double errpow=0,sigpow=0;
   double snr;
   for (bin=0;bin<nfft;++bin) {
      double ansr = 0;
      double difr;

      for (k=0;k<nfft/2;++k) {
         double phase = 2*M_PI*(bin+.5+.25*nfft)*(k+.5)/nfft;
         double re = cos(phase);

         /*re *= 2;*/

         ansr += in[k] * re;
      }
      /*printf ("%f %f\n", ansr, out[bin]);*/
      difr = ansr - out[bin];
      errpow += difr*difr;
      sigpow += ansr*ansr;
   }
   snr = 10*log10(sigpow/errpow);
   printf("nfft=%d inverse=%d,snr = %f\n",nfft,isinverse,snr );
   if (snr<60) {
      printf( "** poor snr: %f **\n", snr);
      ret = 1;
   }
}


void test1d(int nfft,int isinverse)
{
    mdct_lookup cfg;
    size_t buflen = sizeof(kiss_fft_scalar)*nfft;

    kiss_fft_scalar  * in = (kiss_fft_scalar*)malloc(buflen);
    kiss_fft_scalar  * in_copy = (kiss_fft_scalar*)malloc(buflen);
    kiss_fft_scalar  * out= (kiss_fft_scalar*)malloc(buflen);
    opus_val16  * window= (opus_val16*)malloc(sizeof(opus_val16)*nfft/2);
    int k;

    clt_mdct_init(&cfg, nfft, 0);
    for (k=0;k<nfft;++k) {
        in[k] = (rand() % 32768) - 16384;
    }

    for (k=0;k<nfft/2;++k) {
       window[k] = Q15ONE;
    }
    for (k=0;k<nfft;++k) {
       in[k] *= 32768;
    }

    if (isinverse)
    {
       for (k=0;k<nfft;++k) {
          in[k] /= nfft;
       }
    }

    for (k=0;k<nfft;++k)
       in_copy[k] = in[k];
    /*for (k=0;k<nfft;++k) printf("%d %d ", in[k].r, in[k].i);printf("\n");*/

    if (isinverse)
    {
       for (k=0;k<nfft;++k)
          out[k] = 0;
       clt_mdct_backward(&cfg,in,out, window, nfft/2, 0, 1);
       check_inv(in,out,nfft,isinverse);
    } else {
       clt_mdct_forward(&cfg,in,out,window, nfft/2, 0, 1);
       check(in_copy,out,nfft,isinverse);
    }
    /*for (k=0;k<nfft;++k) printf("%d %d ", out[k].r, out[k].i);printf("\n");*/


    free(in);
    free(out);
    clt_mdct_clear(&cfg);
}

int main(int argc,char ** argv)
{
    ALLOC_STACK;
    if (argc>1) {
        int k;
        for (k=1;k<argc;++k) {
            test1d(atoi(argv[k]),0);
            test1d(atoi(argv[k]),1);
        }
    }else{
        test1d(32,0);
        test1d(32,1);
        test1d(256,0);
        test1d(256,1);
        test1d(512,0);
        test1d(512,1);
#ifndef RADIX_TWO_ONLY
        test1d(40,0);
        test1d(40,1);
        test1d(120,0);
        test1d(120,1);
        test1d(240,0);
        test1d(240,1);
        test1d(480,0);
        test1d(480,1);
#endif
    }
    return ret;
}
