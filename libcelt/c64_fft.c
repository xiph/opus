/* (c) Copyright 2008/2009 Xiph.Org Foundation */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
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

#include "c64_fft.h"

#include "dsp_fft16x16t.h"
#include "dsp_fft32x32s.h"
#include "dsp_ifft32x32.h"

#ifndef PI
# ifdef M_PI
#  define PI M_PI
# else
#  define PI 3.14159265358979323846
# endif
#endif


/* ======================================================================== */
/*  D2S -- Truncate a 'double' to a 'short', with clamping.                 */
/* ======================================================================== */
static short d2s(double d)
{
  if (d >=  32767.0) return  32767;
  if (d <= -32768.0) return -32768;
  return (short)d;
}


/* ======================================================================== */
/*  D2S -- Truncate a 'double' to a 'int',   with clamping.                 */
/* ======================================================================== */
static int d2i(double d)
{
  if (d >=  2147483647.0) return (int)0x7FFFFFFF;
  if (d <= -2147483648.0) return (int)0x80000000;
  return (int)d;
}


/* ======================================================================== */
/*  GEN_TWIDDLE -- Generate twiddle factors for TI's custom FFTs.           */
/*                                                                          */
/*  USAGE                                                                   */
/*      This routine is called as follows:                                  */
/*                                                                          */
/*          int gen_twiddle(short *w, int n, double scale)                  */
/*                                                                          */
/*          short  *w     Pointer to twiddle-factor array                   */
/*          int    n      Size of FFT                                       */
/*          double scale  Scale factor to apply to values.                  */
/*                                                                          */
/*      The routine will generate the twiddle-factors directly into the     */
/*      array you specify.  The array needs to be approximately 2*N         */
/*      elements long.  (The actual size, which is slightly smaller, is     */
/*      returned by the function.)                                          */
/* ======================================================================== */
int gen_twiddle16(short *w, int n, double scale)
{
  int i, j, k;
  
  for (j = 1, k = 0; j < n >> 2; j = j << 2)
    {
      for (i = 0; i < n >> 2; i += j << 1)
        {
	  w[k + 11] = d2s(scale * cos(6.0 * PI * (i + j) / n));
	  w[k + 10] = d2s(scale * sin(6.0 * PI * (i + j) / n));
	  w[k +  9] = d2s(scale * cos(6.0 * PI * (i    ) / n));
	  w[k +  8] = d2s(scale * sin(6.0 * PI * (i    ) / n));
	  
	  w[k +  7] = d2s(scale * cos(4.0 * PI * (i + j) / n));
	  w[k +  6] = d2s(scale * sin(4.0 * PI * (i + j) / n));
	  w[k +  5] = d2s(scale * cos(4.0 * PI * (i    ) / n));
	  w[k +  4] = d2s(scale * sin(4.0 * PI * (i    ) / n));
	  
	  w[k +  3] = d2s(scale * cos(2.0 * PI * (i + j) / n));
	  w[k +  2] = d2s(scale * sin(2.0 * PI * (i + j) / n));
	  w[k +  1] = d2s(scale * cos(2.0 * PI * (i    ) / n));
	  w[k +  0] = d2s(scale * sin(2.0 * PI * (i    ) / n));
	  
	  k += 12;
        }
    }
  
  return k;
}


/* ======================================================================== */
/*  GEN_TWIDDLE -- Generate twiddle factors for TI's custom FFTs.           */
/*                                                                          */
/*  USAGE                                                                   */
/*      This routine is called as follows:                                  */
/*                                                                          */
/*          int gen_twiddle(int *w, int n, double scale)                    */
/*                                                                          */
/*          int    *w     Pointer to twiddle-factor array                   */
/*          int    n      Size of FFT                                       */
/*          double scale  Scale factor to apply to values.                  */
/*                                                                          */
/*      The routine will generate the twiddle-factors directly into the     */
/*      array you specify.  The array needs to be approximately 2*N         */
/*      elements long.  (The actual size, which is slightly smaller, is     */
/*      returned by the function.)                                          */
/* ======================================================================== */
int gen_twiddle32(int *w, int n, double scale)
{
  int i, j, k, s=0, t;
  
  for (j = 1, k = 0; j < n >> 2; j = j << 2, s++)
    {
      for (i = t=0; i < n >> 2; i += j, t++)
        {
	  w[k +  5] = d2i(scale * cos(6.0 * PI * i / n));
	  w[k +  4] = d2i(scale * sin(6.0 * PI * i / n));
	  
	  w[k +  3] = d2i(scale * cos(4.0 * PI * i / n));
	  w[k +  2] = d2i(scale * sin(4.0 * PI * i / n));
	  
	  w[k +  1] = d2i(scale * cos(2.0 * PI * i / n));
	  w[k +  0] = d2i(scale * sin(2.0 * PI * i / n));
	  
	  k += 6;
        }
    }
  
  return k;
}

#define NBCACHE 3
static c64_fft_t *cache16[NBCACHE] = {NULL,};
static c64_fft_t *cache32[NBCACHE] = {NULL,};

c64_fft_t *c64_fft16_alloc(int length, int x, int y)
{
  c64_fft_t *state;
  celt_int16 *w, *iw;

  int i, c;

  for (c = 0; c < NBCACHE; c++) {
    if (cache16[c] == NULL)
      break;
    if (cache16[c]->nfft == length)
      return cache16[c];
  }

  state = (c64_fft_t *)celt_alloc(sizeof(c64_fft_t));
  state->shift = log(length)/log(2) - ceil(log(length)/log(4)-1);
  state->nfft = length;
  state->twiddle = celt_alloc(length*2*sizeof(celt_int16));
  state->itwiddle = celt_alloc(length*2*sizeof(celt_int16));

  gen_twiddle16((celt_int16 *)state->twiddle, length, 32767.0);

  w = (celt_int16 *)state->twiddle;
  iw = (celt_int16 *)state->itwiddle;

  for (i = 0; i < length; i++) {
    iw[2*i+0] = w[2*i+0];
    iw[2*i+1] = - w[2*i+1];
  }

  if (c < NBCACHE)
    cache16[c++] = state;
  if (c < NBCACHE)
    cache16[c] = NULL;

  return state;
}


c64_fft_t *c64_fft32_alloc(int length, int x, int y) 
{
  c64_fft_t *state;
  int i, c;

  for (c = 0; c < NBCACHE; c++) {
    if (cache32[c] == NULL)
      break;
    if (cache32[c]->nfft == length)
      return cache32[c];
  }

  state = (c64_fft_t *)celt_alloc(sizeof(c64_fft_t));
  state->shift = log(length)/log(2) - ceil(log(length)/log(4)-1);
  state->nfft = length;
  state->twiddle = celt_alloc(length*2*sizeof(celt_int32));
  state->itwiddle = celt_alloc(length*2*sizeof(celt_int32));

  // Generate the inverse twiddle first because it does not need scaling
  gen_twiddle32(state->itwiddle, length, 2147483647.000000000);

  for (i = 0; i < length; i++) {
    state->twiddle[2*i+0] = state->itwiddle[2*i+0] >> 1;
    state->twiddle[2*i+1] = state->itwiddle[2*i+1] >> 1;
  }

  if (c < NBCACHE)
    cache32[c++] = state;
  if (c < NBCACHE)
    cache32[c] = NULL;

  return state;
}


void c64_fft16_free(c64_fft_t *state) 
{
  c64_fft32_free(state);
}


void c64_fft32_free(c64_fft_t *state)
{
}


void c64_fft16_inplace(c64_fft_t * restrict state, celt_int16 *X)
{
  int i;
  VARDECL(celt_int16, cin);
  VARDECL(celt_int16, cout);
  SAVE_STACK;

  ALLOC(cin,  state->nfft*2, celt_int16);
  ALLOC(cout, state->nfft*2, celt_int16);

  for (i = 0; i < state->nfft; i++) {
    cin[2*i+0] = X[2*i+0];
    cin[2*i+1] = X[2*i+1];
  }

  DSP_fft16x16t((celt_int16 *)state->twiddle, state->nfft, cin, cout);

  for (i = 0; i < state->nfft; i++) {
    X[2*i+0] = cout[2*i+0];
    X[2*i+1] = cout[2*i+1];
  }
  
  RESTORE_STACK;
}



void c64_fft32(c64_fft_t * restrict state, const celt_int32 *X, celt_int32 *Y)
{
  int i;
  VARDECL(celt_int32, cin);
  SAVE_STACK;
  ALLOC(cin, state->nfft*2, celt_int32);

  for (i = 0; i < state->nfft; i++) {
    cin[2*i+0] = X[2*i+0] >> state->shift;
    cin[2*i+1] = X[2*i+1] >> state->shift;
  }

  DSP_fft32x32s(state->twiddle, state->nfft, cin, Y);

  RESTORE_STACK;
}


void c64_ifft16(c64_fft_t * restrict state, const celt_int16 *X, celt_int16 *Y)
{
  int i;
  VARDECL(celt_int16, cin);
  VARDECL(celt_int16, cout);
  SAVE_STACK;

  ALLOC(cin, state->nfft*2, celt_int16);
  if ((celt_int32)Y & 7) 
    ALLOC(cout, state->nfft*2, celt_int16);
  else
    cout = Y;

  for (i = 0; i < state->nfft; i++) {
    // No need to scale for this one but still need to save the input
    // because the fft is going to change it!
    cin[2*i+0] = X[2*i+0];
    cin[2*i+1] = X[2*i+1];
  }

  DSP_fft16x16t((celt_int16 *)state->itwiddle, state->nfft, cin, cout);

  if ((celt_int32)Y & 7)
    for (i = 0; i < state->nfft; i++) {
      Y[2*i+0] = cout[2*i+0];
      Y[2*i+1] = cout[2*i+1];
    }
  
  RESTORE_STACK;
}


void c64_ifft32(c64_fft_t * restrict state, const celt_int32 *X, celt_int32 *Y)
{
  int i;
  VARDECL(celt_int32, cin);
  SAVE_STACK;
  ALLOC(cin, state->nfft*2, celt_int32);

  celt_assert(Y & 7 == 0);

  for (i = 0; i < state->nfft; i++) {
    // No need to scale for this one but still need to save the input
    // because the fft is going to change it!
    cin[2*i+0] = X[2*i+0]; 
    cin[2*i+1] = X[2*i+1];
  }

  DSP_ifft32x32(state->itwiddle, state->nfft, cin, Y);

  RESTORE_STACK;
}


