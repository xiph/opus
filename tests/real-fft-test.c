#include "kiss_fftr.h"
#include "_kiss_fft_guts.h"
#include <sys/times.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

static double cputime(void)
{
    struct tms t;
    times(&t);
    return (double)(t.tms_utime + t.tms_stime)/  sysconf(_SC_CLK_TCK) ;
}

static
kiss_fft_scalar rand_scalar(void) 
{
#ifdef USE_SIMD
    return _mm_set1_ps(rand()-RAND_MAX/2);
#else
    kiss_fft_scalar s = (kiss_fft_scalar)(rand() -RAND_MAX/2);
    return s/2;
#endif
}

static
double snr_compare( kiss_fft_cpx * vec1,kiss_fft_scalar * vec2, int n)
{
    int k;
    double sigpow=1e-10, noisepow=1e-10, err,snr;

    for (k=1;k<n;++k) {
        sigpow += (double)vec1[k].r * (double)vec1[k].r + 
                  (double)vec1[k].i * (double)vec1[k].i;
        err = (double)vec1[k].r - (double)vec2[2*k];
        noisepow += err * err;
        err = (double)vec1[k].i - (double)vec2[2*k+1];
        noisepow += err * err;

    }
    snr = 10*log10( sigpow / noisepow );
    if (snr<10) {
        printf( "\npoor snr: %f\n", snr);
        exit(1);
    }
    return snr;
}

static
double snr_compare_scal( kiss_fft_scalar * vec1,kiss_fft_scalar * vec2, int n)
{
    int k;
    double sigpow=1e-10, noisepow=1e-10, err,snr;

    for (k=1;k<n;++k) {
        sigpow += (double)vec1[k] * (double)vec1[k];
        err = (double)vec1[k] - (double)vec2[k];
        noisepow += err * err;
    }
    snr = 10*log10( sigpow / noisepow );
    if (snr<10) {
        printf( "\npoor snr: %f\n", snr);
        exit(1);
    }
    return snr;
}
#define NFFT 8*3*5

#ifndef NUMFFTS
#define NUMFFTS 10000
#endif


int main(void)
{
    double ts,tfft,trfft;
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
    memset(&zero,0,sizeof(zero) ); // ugly way of setting short,int,float,double, or __m128 to zero

    srand(time(0));

    for (i=0;i<NFFT;++i) {
        rin[i] = rand_scalar();
        cin[i].r = rin[i];
        cin[i].i = zero;
    }

    kiss_fft_state = kiss_fft_alloc(NFFT,0,0,0);
    kiss_fftr_state = kiss_fftr_alloc(NFFT,0,0,0);
    kiss_fft(kiss_fft_state,cin,cout);
    kiss_fftr(kiss_fftr_state,rin,sout);
    
    printf( "nfft=%d, inverse=%d, snr=%g\n",
            NFFT,0, snr_compare(cout,sout,(NFFT/2)) );
    ts = cputime();
    for (i=0;i<NUMFFTS;++i) {
        kiss_fft(kiss_fft_state,cin,cout);
    }
    tfft = cputime() - ts;
    
    ts = cputime();
    for (i=0;i<NUMFFTS;++i) {
        kiss_fftr( kiss_fftr_state, rin, sout );
        /* kiss_fftri(kiss_fftr_state,cout,rin); */
    }
    trfft = cputime() - ts;

    printf("%d complex ffts took %gs, real took %gs\n",NUMFFTS,tfft,trfft);

    free(kiss_fft_state);
    free(kiss_fftr_state);

    kiss_fft_state = kiss_fft_alloc(NFFT,1,0,0);
    kiss_fftr_state = kiss_fftr_alloc(NFFT,1,0,0);

    memset(cin,0,sizeof(cin));
#if 1
    cin[0].r = rand_scalar();
    cin[NFFT/2].r = rand_scalar();
    for (i=1;i< NFFT/2;++i) {
        //cin[i].r = (kiss_fft_scalar)(rand()-RAND_MAX/2);
        cin[i].r = rand_scalar();
        cin[i].i = rand_scalar();
    }
#else
    cin[0].r = 12000;
    cin[3].r = 12000;
    cin[NFFT/2].r = 12000;
#endif

    // conjugate symmetry of real signal 
    for (i=1;i< NFFT/2;++i) {
        cin[NFFT-i].r = cin[i].r;
        cin[NFFT-i].i = - cin[i].i;
    }

    fin[0] = cin[0].r;
    fin[1] = cin[NFFT/2].r;
    for (i=1;i< NFFT/2;++i)
    {
       fin[2*i] = cin[i].r;
       fin[2*i+1] = cin[i].i;
    }
    
    kiss_fft(kiss_fft_state,cin,cout);
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

    return 0;
}
