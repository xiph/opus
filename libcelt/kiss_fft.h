#ifndef KISS_FFT_H
#define KISS_FFT_H

#include <stdlib.h>
#include <math.h>
#include "arch.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 ATTENTION!
 If you would like a :
 -- a utility that will handle the caching of fft objects
 -- real-only (no imaginary time component ) FFT
 -- a multi-dimensional FFT
 -- a command-line utility to perform ffts
 -- a command-line utility to perform fast-convolution filtering

 Then see kfc.h kiss_fftr.h kiss_fftnd.h fftutil.c kiss_fastfir.c
  in the tools/ directory.
*/

#ifdef USE_SIMD
# include <xmmintrin.h>
# define kiss_fft_scalar __m128
#define KISS_FFT_MALLOC(nbytes) memalign(16,nbytes)
#else	
#define KISS_FFT_MALLOC celt_alloc
#endif	


#ifdef FIXED_POINT
#include "arch.h"	
#ifdef DOUBLE_PRECISION
#  define kiss_fft_scalar celt_int32_t
#  define kiss_twiddle_scalar celt_int32_t
#  define KF_SUFFIX _celt_double
#else
#  define kiss_fft_scalar celt_int16_t
#  define kiss_twiddle_scalar celt_int16_t
#  define KF_SUFFIX _celt_single
#endif
#else
# ifndef kiss_fft_scalar
/*  default is float */
#   define kiss_fft_scalar float
#   define kiss_twiddle_scalar float
#   define KF_SUFFIX _celt_single
# endif
#endif


/* This adds a suffix to all the kiss_fft functions so we
   can easily link with more than one copy of the fft */
#define CAT_SUFFIX(a,b) a ## b
#define SUF(a,b) CAT_SUFFIX(a, b)

#define kiss_fft_alloc SUF(kiss_fft_alloc,KF_SUFFIX)
#define kf_work SUF(kf_work,KF_SUFFIX)
#define ki_work SUF(ki_work,KF_SUFFIX)
#define kiss_fft SUF(kiss_fft,KF_SUFFIX)
#define kiss_ifft SUF(kiss_ifft,KF_SUFFIX)
#define kiss_fft_stride SUF(kiss_fft_stride,KF_SUFFIX)
#define kiss_ifft_stride SUF(kiss_ifft_stride,KF_SUFFIX)


typedef struct {
    kiss_fft_scalar r;
    kiss_fft_scalar i;
}kiss_fft_cpx;

typedef struct {
   kiss_twiddle_scalar r;
   kiss_twiddle_scalar i;
}kiss_twiddle_cpx;

typedef struct kiss_fft_state* kiss_fft_cfg;

/** 
 *  kiss_fft_alloc
 *  
 *  Initialize a FFT (or IFFT) algorithm's cfg/state buffer.
 *
 *  typical usage:      kiss_fft_cfg mycfg=kiss_fft_alloc(1024,0,NULL,NULL);
 *
 *  The return value from fft_alloc is a cfg buffer used internally
 *  by the fft routine or NULL.
 *
 *  If lenmem is NULL, then kiss_fft_alloc will allocate a cfg buffer using malloc.
 *  The returned value should be free()d when done to avoid memory leaks.
 *  
 *  The state can be placed in a user supplied buffer 'mem':
 *  If lenmem is not NULL and mem is not NULL and *lenmem is large enough,
 *      then the function places the cfg in mem and the size used in *lenmem
 *      and returns mem.
 *  
 *  If lenmem is not NULL and ( mem is NULL or *lenmem is not large enough),
 *      then the function returns NULL and places the minimum cfg 
 *      buffer size in *lenmem.
 * */

kiss_fft_cfg kiss_fft_alloc(int nfft,void * mem,size_t * lenmem); 

void kf_work(kiss_fft_cpx * Fout,const kiss_fft_cpx * f,const size_t fstride,
             int in_stride,int * factors,const kiss_fft_cfg st,int N,int s2,int m2);

/** Internal function. Can be useful when you want to do the bit-reversing yourself */
void ki_work(kiss_fft_cpx * Fout, const kiss_fft_cpx * f, const size_t fstride,
             int in_stride,int * factors,const kiss_fft_cfg st,int N,int s2,int m2);

/**
 * kiss_fft(cfg,in_out_buf)
 *
 * Perform an FFT on a complex input buffer.
 * for a forward FFT,
 * fin should be  f[0] , f[1] , ... ,f[nfft-1]
 * fout will be   F[0] , F[1] , ... ,F[nfft-1]
 * Note that each element is complex and can be accessed like
    f[k].r and f[k].i
 * */
void kiss_fft(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout);
void kiss_ifft(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout);

/**
 A more generic version of the above function. It reads its input from every Nth sample.
 * */
void kiss_fft_stride(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout,int fin_stride);
void kiss_ifft_stride(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout,int fin_stride);

/** If kiss_fft_alloc allocated a buffer, it is one contiguous 
   buffer and can be simply free()d when no longer needed*/
#define kiss_fft_free celt_free


#ifdef __cplusplus
} 
#endif

#endif
