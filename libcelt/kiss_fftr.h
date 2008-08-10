#ifndef KISS_FTR_H
#define KISS_FTR_H

#include "kiss_fft.h"
#ifdef __cplusplus
extern "C" {
#endif

#define kiss_fftr_alloc SUF(kiss_fftr_alloc,KF_SUFFIX)
#define kiss_fftr_inplace SUF(kiss_fftr_inplace,KF_SUFFIX)
#define kiss_fftr_alloc SUF(kiss_fftr_alloc,KF_SUFFIX)
#define kiss_fftr_twiddles SUF(kiss_fftr_twiddles,KF_SUFFIX)
#define kiss_fftr SUF(kiss_fftr,KF_SUFFIX)
#define kiss_fftri SUF(kiss_fftri,KF_SUFFIX)

/* 
 
 Real optimized version can save about 45% cpu time vs. complex fft of a real seq.

 
 
 */

struct kiss_fftr_state{
      kiss_fft_cfg substate;
      kiss_twiddle_cpx * super_twiddles;
#ifdef USE_SIMD    
      long pad;
#endif    
   };

typedef struct kiss_fftr_state *kiss_fftr_cfg;


kiss_fftr_cfg kiss_fftr_alloc(int nfft,void * mem, size_t * lenmem);
/*
 nfft must be even

 If you don't care to allocate space, use mem = lenmem = NULL 
*/


/*
 input timedata has nfft scalar points
 output freqdata has nfft/2+1 complex points, packed into nfft scalar points
*/
void kiss_fftr_twiddles(kiss_fftr_cfg st,kiss_fft_scalar *freqdata);

void kiss_fftr(kiss_fftr_cfg st,const kiss_fft_scalar *timedata,kiss_fft_scalar *freqdata);
void kiss_fftr_inplace(kiss_fftr_cfg st, kiss_fft_scalar *X);

void kiss_fftri(kiss_fftr_cfg st,const kiss_fft_scalar *freqdata, kiss_fft_scalar *timedata);

/*
 input freqdata has  nfft/2+1 complex points, packed into nfft scalar points
 output timedata has nfft scalar points
*/

#define kiss_fftr_free speex_free

#ifdef __cplusplus
}
#endif
#endif
