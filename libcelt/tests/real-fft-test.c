#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "kiss_fftr.h"
#include "_kiss_fft_guts.h"
#include <stdio.h>
#include <string.h>

#define CELT_C 
#include "../libcelt/stack_alloc.h"
#include "../libcelt/kiss_fft.c"
#include "../libcelt/kiss_fftr.c"

#ifdef FIXED_DEBUG
long long celt_mips=0;
#endif
int ret=0;

static
kiss_fft_scalar rand_scalar(void) 
{
    return (rand()%32767)-16384;
}

static
double snr_compare( kiss_fft_cpx * vec1,kiss_fft_scalar * vec2, int n)
{
    int k;
    double sigpow=1e-10, noisepow=1e-10, err,snr;

    vec1[0].i = vec1[n].r;
    for (k=0;k<n;++k) {
        sigpow += (double)vec1[k].r * (double)vec1[k].r + 
                  (double)vec1[k].i * (double)vec1[k].i;
        err = (double)vec1[k].r - (double)vec2[2*k];
        /*printf ("%f %f\n", (double)vec1[k].r, (double)vec2[2*k]);*/
        noisepow += err * err;
        err = (double)vec1[k].i - (double)vec2[2*k+1];
        /*printf ("%f %f\n", (double)vec1[k].i, (double)vec2[2*k+1]);*/
        noisepow += err * err;

    }
    snr = 10*log10( sigpow / noisepow );
    if (snr<60) {
        printf( "** poor snr: %f **\n", snr);
        ret = 1;
    }
    return snr;
}

static
double snr_compare_scal( kiss_fft_scalar * vec1,kiss_fft_scalar * vec2, int n)
{
    int k;
    double sigpow=1e-10, noisepow=1e-10, err,snr;

    for (k=0;k<n;++k) {
        sigpow += (double)vec1[k] * (double)vec1[k];
        err = (double)vec1[k] - (double)vec2[k];
        noisepow += err * err;
    }
    snr = 10*log10( sigpow / noisepow );
    if (snr<60) {
        printf( "\npoor snr: %f\n", snr);
        ret = 1;
    }
    return snr;
}
#ifdef RADIX_TWO_ONLY
#define NFFT 1024
#else
#define NFFT 8*3*5
#endif

#ifndef NUMFFTS
#define NUMFFTS 10000
#endif


int main(void)
{
    int i;
    kiss_fft_cpx cin[NFFT];
    kiss_fft_cpx cout[NFFT];
    kiss_fft_scalar fin[NFFT];
    kiss_fft_scalar sout[NFFT];
    kiss_fft_cfg  kiss_fft_state;
    kiss_fftr_cfg  kiss_fftr_state;

    kiss_fft_scalar rin[NFFT+2];
    kiss_fft_scalar rout[NFFT+2];
    kiss_fft_scalar zero;
    ALLOC_STACK;
    memset(&zero,0,sizeof(zero) ); // ugly way of setting short,int,float,double, or __m128 to zero

    for (i=0;i<NFFT;++i) {
        rin[i] = rand_scalar();
#if defined(FIXED_POINT) && defined(DOUBLE_PRECISION)
        rin[i] *= 32768;
#endif
        cin[i].r = rin[i];
        cin[i].i = zero;
    }

    kiss_fft_state = kiss_fft_alloc(NFFT,0,0);
    kiss_fftr_state = kiss_fftr_alloc(NFFT,0,0);
    kiss_fft(kiss_fft_state,cin,cout);
    kiss_fftr(kiss_fftr_state,rin,sout);
    
    printf( "nfft=%d, inverse=%d, snr=%g\n",
            NFFT,0, snr_compare(cout,sout,(NFFT/2)) );

    memset(cin,0,sizeof(cin));
    cin[0].r = rand_scalar();
    cin[NFFT/2].r = rand_scalar();
    for (i=1;i< NFFT/2;++i) {
        //cin[i].r = (kiss_fft_scalar)(rand()-RAND_MAX/2);
        cin[i].r = rand_scalar();
        cin[i].i = rand_scalar();
    }

    // conjugate symmetry of real signal 
    for (i=1;i< NFFT/2;++i) {
        cin[NFFT-i].r = cin[i].r;
        cin[NFFT-i].i = - cin[i].i;
    }

    
#ifdef FIXED_POINT
#ifdef DOUBLE_PRECISION
    for (i=0;i< NFFT;++i) {
       cin[i].r *= 32768;
       cin[i].i *= 32768;
    }
#endif
    for (i=0;i< NFFT;++i) {
       cin[i].r /= NFFT;
       cin[i].i /= NFFT;
    }
#endif
    
    fin[0] = cin[0].r;
    fin[1] = cin[NFFT/2].r;
    for (i=1;i< NFFT/2;++i)
    {
       fin[2*i] = cin[i].r;
       fin[2*i+1] = cin[i].i;
    }
    
    kiss_ifft(kiss_fft_state,cin,cout);
    kiss_fftri(kiss_fftr_state,fin,rout);
    /*
    printf(" results from inverse kiss_fft : (%f,%f), (%f,%f), (%f,%f), (%f,%f), (%f,%f) ...\n "
            , (float)cout[0].r , (float)cout[0].i , (float)cout[1].r , (float)cout[1].i , (float)cout[2].r , (float)cout[2].i , (float)cout[3].r , (float)cout[3].i , (float)cout[4].r , (float)cout[4].i
            ); 

    printf(" results from inverse kiss_fftr: %f,%f,%f,%f,%f ... \n"
            ,(float)rout[0] ,(float)rout[1] ,(float)rout[2] ,(float)rout[3] ,(float)rout[4]);
*/
    for (i=0;i<NFFT;++i) {
        sout[i] = cout[i].r;
    }

    printf( "nfft=%d, inverse=%d, snr=%g\n",
            NFFT,1, snr_compare_scal(rout,sout,NFFT) );
    free(kiss_fft_state);
    free(kiss_fftr_state);

    return ret;
}
