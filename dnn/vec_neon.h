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
static float celt_exp2(float x)
{
    int integer;
    float frac;
    union {
	float f;
	opus_uint32 i;
    } res;
    integer = floor(x);
    if (integer < -50)
	return 0;
    frac = x-integer;
    /* K0 = 1, K1 = log(2), K2 = 3-4*log(2), K3 = 3*log(2) - 2 */
    res.f = 0.99992522f + frac * (0.69583354f
				  + frac * (0.22606716f + 0.078024523f*frac));
    res.i = (res.i + (integer<<23)) & 0x7fffffff;
    return res.f;
}
#define celt_exp_neon(x) celt_exp2((x)*1.44269504f)

static float tansig_approx(float x)
{
    int i;
    float y, dy;
    float sign=1;
    /* Tests are reversed to catch NaNs */
    if (!(x<8))
        return 1;
    if (!(x>-8))
        return -1;
#ifndef FIXED_POINT
    /* Another check in case of -ffast-math */
    if (celt_isnan(x))
	return 0;
#endif
    if (x<0)
    {
	x=-x;
	sign=-1;
    }
    i = (int)floor(.5f+25*x);
    x -= .04f*i;
    y = tansig_table[i];
    dy = 1-y*y;
    y = y + x*dy*(1 - y*x);
    return sign*y;
}

static OPUS_INLINE float sigmoid_approx(float x)
{
    return .5f + .5f*tansig_approx(.5f*x);
}

static void softmax(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N;i++)
        y[i] = celt_exp_neon(x[i]);
}

static void vec_tanh(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N;i++)
    {
        y[i] = tansig_approx(x[i]);
    }
}

static void vec_sigmoid(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N;i++)
    {
        y[i] = sigmoid_approx(x[i]);
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
	    float xj= x[*idx++];
	    float32x4_t wvec;
	 
	    wvec = vld1q_f32(&w[0]); y0_3 = vmlaq_n_f32(y0_3, wvec, xj);
	    wvec = vld1q_f32(&w[4]); y4_7 = vmlaq_n_f32(y4_7, wvec, xj);
	    wvec = vld1q_f32(&w[8]); y8_11 = vmlaq_n_f32(y8_11, wvec, xj);
	    wvec = vld1q_f32(&w[12]); y12_15 = vmlaq_n_f32(y12_15, wvec, xj);
	 
	    w += 16;
	}

	/* save y[0..15] back to memory */
      
	vst1q_f32(&y[0], y0_3);
	vst1q_f32(&y[4], y4_7);
	vst1q_f32(&y[8], y8_11);
	vst1q_f32(&y[12], y12_15);
      
    }
}
