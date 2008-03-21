/* (C) 2008 Jean-Marc Valin, CSIRO
*/
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef KFFT_DOUBLE_H
#define KFFT_DOUBLE_H

#ifdef ENABLE_TI_DSPLIB

#include "dsplib.h"
#include "_kiss_fft_guts.h"

#define cpx32_fft_alloc(length) NULL
#define cpx32_fft_free(state)

#define cpx32_fft(state, X, Y, nx)\
    (\
      cfft32_SCALE(X,nx),\
      cbrev32(X,Y,nx)\
    )

#define cpx32_ifft(state, X, Y, nx) \
    (\
      cifft32_NOSCALE(X,nx),\
      cbrev32(X,Y,nx)\
    )


#else /* ENABLE_TI_DSPLIB */

#include "kiss_fft.h"
#include "_kiss_fft_guts.h"

#define cpx32_fft_alloc(length) kiss_fft_alloc(length, 0, 0);
#define cpx32_fft_free(state) kiss_fft_free(state)
#define cpx32_fft(state, X, Y, nx) kiss_fft(state,(const kiss_fft_cpx *)(X), (kiss_fft_cpx *)(Y))
#define cpx32_ifft(state, X, Y, nx) kiss_ifft(state,(const kiss_fft_cpx *)(X), (kiss_fft_cpx *)(Y))

#endif /* !ENABLE_TI_DSPLIB */

#endif /* KFFT_DOUBLE_H */
