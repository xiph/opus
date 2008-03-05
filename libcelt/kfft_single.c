#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef FIXED_POINT

#include "kfft_single.h"

#define SKIP_CONFIG_H
#include "kiss_fft.c"
#include "kiss_fftr.c"

#endif
