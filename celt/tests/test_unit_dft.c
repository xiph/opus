#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define SKIP_CONFIG_H

#ifndef CUSTOM_MODES
#define CUSTOM_MODES
#endif

#include <stdio.h>

#define CELT_C
#include "stack_alloc.h"
#include "kiss_fft.h"
#include "kiss_fft.c"
#include "mathops.c"
#include "entcode.c"


#ifndef M_PI
#define M_PI 3.141592653
#endif

int ret = 0;

void check(kiss_fft_cpx  * in,kiss_fft_cpx  * out,int nfft,int isinverse)
{
    int bin,k;
    double errpow=0,sigpow=0, snr;

    for (bin=0;bin<nfft;++bin) {
        double ansr = 0;
        double ansi = 0;
        double difr;
        double difi;

        for (k=0;k<nfft;++k) {
            double phase = -2*M_PI*bin*k/nfft;
            double re = cos(phase);
            double im = sin(phase);
            if (isinverse)
                im = -im;

            if (!isinverse)
            {
               re /= nfft;
               im /= nfft;
            }

            ansr += in[k].r * re - in[k].i * im;
            ansi += in[k].r * im + in[k].i * re;
        }
        /*printf ("%d %d ", (int)ansr, (int)ansi);*/
        difr = ansr - out[bin].r;
        difi = ansi - out[bin].i;
        errpow += difr*difr + difi*difi;
        sigpow += ansr*ansr+ansi*ansi;
    }
    snr = 10*log10(sigpow/errpow);
    printf("nfft=%d inverse=%d,snr = %f\n",nfft,isinverse,snr );
    if (snr<60) {
       printf( "** poor snr: %f ** \n", snr);
       ret = 1;
    }
}

void test1d(int nfft,int isinverse)
{
    size_t buflen = sizeof(kiss_fft_cpx)*nfft;

    kiss_fft_cpx  * in = (kiss_fft_cpx*)malloc(buflen);
    kiss_fft_cpx  * out= (kiss_fft_cpx*)malloc(buflen);
    kiss_fft_state *cfg = opus_fft_alloc(nfft,0,0);
    int k;

    for (k=0;k<nfft;++k) {
        in[k].r = (rand() % 32767) - 16384;
        in[k].i = (rand() % 32767) - 16384;
    }

    for (k=0;k<nfft;++k) {
       in[k].r *= 32768;
       in[k].i *= 32768;
    }

    if (isinverse)
    {
       for (k=0;k<nfft;++k) {
          in[k].r /= nfft;
          in[k].i /= nfft;
       }
    }

    /*for (k=0;k<nfft;++k) printf("%d %d ", in[k].r, in[k].i);printf("\n");*/

    if (isinverse)
       opus_ifft(cfg,in,out);
    else
       opus_fft(cfg,in,out);

    /*for (k=0;k<nfft;++k) printf("%d %d ", out[k].r, out[k].i);printf("\n");*/

    check(in,out,nfft,isinverse);

    free(in);
    free(out);
    free(cfg);
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
        test1d(128,0);
        test1d(128,1);
        test1d(256,0);
        test1d(256,1);
#ifndef RADIX_TWO_ONLY
        test1d(36,0);
        test1d(36,1);
        test1d(50,0);
        test1d(50,1);
        test1d(120,0);
        test1d(120,1);
#endif
    }
    return ret;
}
