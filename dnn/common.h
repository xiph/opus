

#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define RNN_INLINE inline
#define OPUS_INLINE inline

float lpc_from_cepstrum(float *lpc, const float *cepstrum);

static RNN_INLINE float ulaw2lin(float u)
{
    float s;
    float scale_1 = 32768.f/255.f;
    u = u - 128;
    s = u >= 0 ? 1 : -1;
    u = fabs(u);
    return s*scale_1*(exp(u/128.*log(256))-1);
}

static RNN_INLINE int lin2ulaw(float x)
{
    float u;
    float scale = 255.f/32768.f;
    int s = x >= 0 ? 1 : -1;
    x = fabs(x);
    u = (s*(128*log(1+scale*x)/log(256)));
    u = 128 + u;
    if (u < 0) u = 0;
    if (u > 255) u = 255;
    return (int)floor(.5 + u);
}


/** RNNoise wrapper for malloc(). To do your own dynamic allocation, all you need t
o do is replace this function and rnnoise_free */
#ifndef OVERRIDE_RNNOISE_ALLOC
static RNN_INLINE void *rnnoise_alloc (size_t size)
{
   return malloc(size);
}
#endif

/** RNNoise wrapper for free(). To do your own dynamic allocation, all you need to do is replace this function and rnnoise_alloc */
#ifndef OVERRIDE_RNNOISE_FREE
static RNN_INLINE void rnnoise_free (void *ptr)
{
   free(ptr);
}
#endif

/** Copy n elements from src to dst. The 0* term provides compile-time type checking  */
#ifndef OVERRIDE_RNN_COPY
#define RNN_COPY(dst, src, n) (memcpy((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Copy n elements from src to dst, allowing overlapping regions. The 0* term
    provides compile-time type checking */
#ifndef OVERRIDE_RNN_MOVE
#define RNN_MOVE(dst, src, n) (memmove((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Set n elements of dst to zero */
#ifndef OVERRIDE_RNN_CLEAR
#define RNN_CLEAR(dst, n) (memset((dst), 0, (n)*sizeof(*(dst))))
#endif



#endif
