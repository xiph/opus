/* Copyright (c) 2018 David Rowe
                 2018 Mozilla
                 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
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
/* NEON support for ARM machines */

#include <arm_neon.h>
#ifndef LPCNET_TEST
static OPUS_INLINE float32x4_t exp4_approx(float32x4_t x) {
  int32x4_t i;
  float32x4_t xf;

  x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(88.f)), vdupq_n_f32(-88.f));

  /* express exp(x) as exp2(x/log(2)), add 127 for the exponent later */
  x = vmlaq_f32(vdupq_n_f32(127.f), x, vdupq_n_f32(1.44269504f));

  /* split into integer and fractional parts */
  i = vcvtq_s32_f32(x);
  xf = vcvtq_f32_s32(i);
  x = vsubq_f32(x, xf);

  float32x4_t K0 = vdupq_n_f32(0.99992522f);
  float32x4_t K1 = vdupq_n_f32(0.69583354f);
  float32x4_t K2 = vdupq_n_f32(0.22606716f);
  float32x4_t K3 = vdupq_n_f32(0.078024523f);
  float32x4_t Y = vmlaq_f32(K0, x, vmlaq_f32(K1, x, vmlaq_f32(K2, K3, x)));

  /* compute 2^i */
  float32x4_t exponent = vreinterpretq_f32_s32(vshlq_n_s32(i, 23));

  Y = vmulq_f32(Y, exponent);
  return Y;
}

static OPUS_INLINE float celt_exp(float x)
{
   float out[4];
   float32x4_t X, Y;
   X = vdupq_n_f32(x);
   Y = exp4_approx(X);
   vst1q_f32(out, Y);
   return out[0];
}

static void softmax(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N-3;i+=4)
    {
        float32x4_t X, Y;
        X = vld1q_f32(&x[i]);
        Y = exp4_approx(X);
        vst1q_f32(&y[i], Y);
    }
    for (;i<N;i++)
        y[i] = celt_exp(x[i]);
}

static void vec_tanh(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N-3;i+=4)
    {
        const float32x4_t two = vdupq_n_f32(2.f);
        const float32x4_t one = vdupq_n_f32(1.f);
        float32x4_t X, Y;
        X = vld1q_f32(&x[i]);
        X = vmulq_f32(X, two);
        Y = exp4_approx(X);
        Y = vmulq_f32(vsubq_f32(Y, one),  vrecpeq_f32(vaddq_f32(Y, one)));
        vst1q_f32(&y[i], Y);
    }
    for (;i<N;i++)
    {
        float ex2;
        ex2 = celt_exp(2*x[i]);
        y[i] = (ex2-1)/(ex2+1);
    }
}

static void vec_sigmoid(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N-3;i+=4)
    {
        const float32x4_t one = vdupq_n_f32(1.f);
        float32x4_t X, Y;
        X = vld1q_f32(&x[i]);
        Y = exp4_approx(X);
        Y = vmulq_f32(Y,  vrecpeq_f32(vaddq_f32(Y, one)));
        vst1q_f32(&y[i], Y);
    }
    for (;i<N;i++)
    {
        float ex;
        ex = celt_exp(x[i]);
        y[i] = (ex)/(ex+1);
    }
}
#endif

static void sgemv_accum16(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
    int i, j;
    for (i=0;i<rows;i+=16)
    {
	float * restrict y = &out[i];
      
	/* keep y[0..15] in registers for duration of inner loop */
      
	float32x4_t y0_3 = vld1q_f32(&y[0]);
	float32x4_t y4_7 = vld1q_f32(&y[4]);
	float32x4_t y8_11 = vld1q_f32(&y[8]);
	float32x4_t y12_15 = vld1q_f32(&y[12]);
      
	for (j=0;j<cols;j++)
	{
	    const float * restrict w;
	    float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;
	    float32x4_t xj;

	    w = &weights[j*col_stride + i];
	    wvec0_3 = vld1q_f32(&w[0]);
	    wvec4_7 = vld1q_f32(&w[4]);
	    wvec8_11 = vld1q_f32(&w[8]);
	    wvec12_15 = vld1q_f32(&w[12]);
	    
	    xj = vld1q_dup_f32(&x[j]);
	 
	    y0_3 = vmlaq_f32(y0_3, wvec0_3, xj);
	    y4_7 = vmlaq_f32(y4_7, wvec4_7, xj);
	    y8_11 = vmlaq_f32(y8_11, wvec8_11, xj);
	    y12_15 = vmlaq_f32(y12_15, wvec12_15, xj);
	}

	/* save y[0..15] back to memory */
      
	vst1q_f32(&y[0], y0_3);
	vst1q_f32(&y[4], y4_7);
	vst1q_f32(&y[8], y8_11);
	vst1q_f32(&y[12], y12_15);
      
    }
}

static void sparse_sgemv_accum16(float *out, const float *w, int rows, const int *idx, const float *x)
{
    int i, j;
    for (i=0;i<rows;i+=16)
    {
	int cols;
	cols = *idx++;
	float * restrict y;
	y = &out[i];

	/* keep y[0..15] in registers for duration of inner loop */
      
	float32x4_t y0_3 = vld1q_f32(&y[0]);
	float32x4_t y4_7 = vld1q_f32(&y[4]);
	float32x4_t y8_11 = vld1q_f32(&y[8]);
	float32x4_t y12_15 = vld1q_f32(&y[12]);
      
	for (j=0;j<cols;j++)
	{
	    float32x4_t xj= vld1q_dup_f32(&x[*idx++]);
	    float32x4_t wvec;
	 
	    wvec = vld1q_f32(&w[0]); y0_3 = vmlaq_f32(y0_3, wvec, xj);
	    wvec = vld1q_f32(&w[4]); y4_7 = vmlaq_f32(y4_7, wvec, xj);
	    wvec = vld1q_f32(&w[8]); y8_11 = vmlaq_f32(y8_11, wvec, xj);
	    wvec = vld1q_f32(&w[12]); y12_15 = vmlaq_f32(y12_15, wvec, xj);
	 
	    w += 16;
	}

	/* save y[0..15] back to memory */
      
	vst1q_f32(&y[0], y0_3);
	vst1q_f32(&y[4], y4_7);
	vst1q_f32(&y[8], y8_11);
	vst1q_f32(&y[12], y12_15);
      
    }
}
