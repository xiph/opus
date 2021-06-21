/* Copyright (c) 2018 Mozilla
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
/*
  AVX implementation of vector operations, compile with -mavx
  AVX2/FMA implementation of vector operations, compile with -mavx2 -mfma
*/

#ifndef VEC_AVX_H
#define VEC_AVX_H

#include <immintrin.h>

#define DOT_PROD
#define USE_SU_BIAS

#ifdef __AVX2__
static inline __m256 exp8_approx(__m256 X)
{
   const __m256 K0 = _mm256_set1_ps(0.99992522f);
   const __m256 K1 = _mm256_set1_ps(0.69583354f);
   const __m256 K2 = _mm256_set1_ps(0.22606716f);
   const __m256 K3 = _mm256_set1_ps(0.078024523f);
   const __m256 log2_E = _mm256_set1_ps(1.44269504);
   const __m256 max_in = _mm256_set1_ps(50.f);
   const __m256 min_in = _mm256_set1_ps(-50.f);
   __m256 XF, Y;
   __m256i I;
   X = _mm256_mul_ps(X, log2_E);
   X = _mm256_max_ps(min_in, _mm256_min_ps(max_in, X));
   XF = _mm256_floor_ps(X);
   I = _mm256_cvtps_epi32(XF);
   X = _mm256_sub_ps(X, XF);
   Y = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(K3, X, K2), X, K1), X, K0);
   I = _mm256_slli_epi32(I, 23);
   Y = _mm256_castsi256_ps(_mm256_add_epi32(I, _mm256_castps_si256(Y)));
   return Y;
}
#else
#define _mm256_fmadd_ps(a,b,c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
#define _mm_fmadd_ps(a,b,c) _mm_add_ps(_mm_mul_ps(a, b), c)
static inline __m128 exp4_approx(__m128 X)
{
   const __m128 K0 = _mm_set1_ps(0.99992522f);
   const __m128 K1 = _mm_set1_ps(0.69583354f);
   const __m128 K2 = _mm_set1_ps(0.22606716f);
   const __m128 K3 = _mm_set1_ps(0.078024523f);
   const __m128 log2_E = _mm_set1_ps(1.44269504);
   const __m128 max_in = _mm_set1_ps(50.f);
   const __m128 min_in = _mm_set1_ps(-50.f);
   const __m128i mask = _mm_set1_epi32(0x7fffffff);
   __m128 XF, Y;
   __m128i I;
   X = _mm_mul_ps(X, log2_E);
   X = _mm_max_ps(min_in, _mm_min_ps(max_in, X));
   XF = _mm_floor_ps(X);
   I = _mm_cvtps_epi32(XF);
   X = _mm_sub_ps(X, XF);
   Y = _mm_fmadd_ps(_mm_fmadd_ps(_mm_fmadd_ps(K3, X, K2), X, K1), X, K0);
   I = _mm_slli_epi32(I, 23);
   Y = _mm_castsi128_ps(_mm_and_si128(mask, _mm_add_epi32(I, _mm_castps_si128(Y))));
   return Y;
}
static inline __m256 exp8_approx(__m256 X)
{
   __m256 Y;
   __m128 Xhi, Xlo, Yhi, Ylo;
   Xhi = _mm256_extractf128_ps(X, 1);
   Xlo = _mm256_extractf128_ps(X, 0);
   Yhi = exp4_approx(Xhi);
   Ylo = exp4_approx(Xlo);
   Y = _mm256_insertf128_ps(_mm256_setzero_ps(), Yhi, 1);
   Y = _mm256_insertf128_ps(Y, Ylo, 0);
   return Y;
}
#endif

static inline float celt_exp(float x)
{
   float out[8];
   __m256 X, Y;
   X = _mm256_set1_ps(x);
   Y = exp8_approx(X);
   _mm256_storeu_ps(out, Y);
   return out[0];
}

static inline void softmax(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N-7;i+=8)
    {
        __m256 X, Y;
        X = _mm256_loadu_ps(&x[i]);
        Y = exp8_approx(X);
        _mm256_storeu_ps(&y[i], Y);
    }
    for (;i<N;i++)
        y[i] = celt_exp(x[i]);
}

static inline void vec_tanh(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N-7;i+=8)
    {
        const __m256 two = _mm256_set1_ps(2.f);
        const __m256 one = _mm256_set1_ps(1.f);
        __m256 X, Y;
        X = _mm256_loadu_ps(&x[i]);
        X = _mm256_mul_ps(X, two);
        Y = exp8_approx(X);
        Y = _mm256_mul_ps(_mm256_sub_ps(Y, one),  _mm256_rcp_ps(_mm256_add_ps(Y, one)));
        _mm256_storeu_ps(&y[i], Y);
    }
    for (;i<N;i++)
    {
        float ex2;
        ex2 = celt_exp(2*x[i]);
        y[i] = (ex2-1)/(ex2+1);
    }
}

static inline void vec_sigmoid(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N-7;i+=8)
    {
        const __m256 one = _mm256_set1_ps(1.f);
        __m256 X, Y;
        X = _mm256_loadu_ps(&x[i]);
        Y = exp8_approx(X);
        /* Compute as 1-1/(1+e^x) to avoid >1 values caused by the reciprocal approximation. */
        Y = _mm256_sub_ps(one, _mm256_rcp_ps(_mm256_add_ps(Y, one)));
        _mm256_storeu_ps(&y[i], Y);
    }
    for (;i<N;i++)
    {
        float ex;
        ex = celt_exp(x[i]);
        y[i] = (ex)/(ex+1);
    }
}

static inline void sgemv_accum16(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   for (i=0;i<rows;i+=16)
   {
      float * restrict y;
      __m256 vy0, vy8;
      y = &out[i];
      vy0 = _mm256_loadu_ps(&y[0]);
      vy8 = _mm256_loadu_ps(&y[8]);
      for (j=0;j<cols;j++)
      {
         __m256 vxj;
         __m256 vw;
         vxj = _mm256_broadcast_ss(&x[j]);

         vw = _mm256_loadu_ps(&weights[j*col_stride + i]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         vw = _mm256_loadu_ps(&weights[j*col_stride + i + 8]);
         vy8 = _mm256_fmadd_ps(vw, vxj, vy8);
      }
      _mm256_storeu_ps (&y[0], vy0);
      _mm256_storeu_ps (&y[8], vy8);
   }
}
static inline void sparse_sgemv_accum16(float *out, const float *weights, int rows, const int *idx, const float *x)
{
   int i, j;
   for (i=0;i<rows;i+=16)
   {
      float * restrict y;
      int cols;
      __m256 vy0, vy8;
      y = &out[i];
      vy0 = _mm256_loadu_ps(&y[0]);
      vy8 = _mm256_loadu_ps(&y[8]);
      cols = *idx++;
      for (j=0;j<cols;j++)
      {
         int id;
         __m256 vxj;
         __m256 vw;
         id = *idx++;
         vxj = _mm256_broadcast_ss(&x[id]);

         vw = _mm256_loadu_ps(&weights[0]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         vw = _mm256_loadu_ps(&weights[8]);
         vy8 = _mm256_fmadd_ps(vw, vxj, vy8);
         weights += 16;
      }
      _mm256_storeu_ps (&y[0], vy0);
      _mm256_storeu_ps (&y[8], vy8);
   }
}

#ifdef DOT_PROD
#define USE_SU_BIAS

typedef signed char qweight;


#define MAX_INPUTS (2048)
#define MAX_OUTPUTS (8192)


#define SCALE (128.f*127.f)
#define SCALE_1 (1.f/128.f/127.f)

#if 1
static inline void sgemv_accum8x4(float *_out, const qweight *w, int rows, int cols, int col_stride, const float *_x)
{
   __m256i ones;
   int i, j;
   unsigned char x[MAX_INPUTS];
   int out[MAX_OUTPUTS];
   (void)col_stride;
   ones = _mm256_set1_epi16(1);
   for (i=0;i<rows;i++) out[i] = SCALE*_out[i];
   //for (i=0;i<cols;i++) x[i] = 127+floor(.5+127*_x[i]);
   __m256 const127 = _mm256_set1_ps(127.f);
   for (i=0;i<cols;i+=8) {
       __m256 xf;
       __m256i xi;
       xf = _mm256_loadu_ps(&_x[i]);
       //xf = _mm256_mul_ps(xf, const127);
       //xf = _mm256_add_ps(xf, const127);
       xf = _mm256_fmadd_ps(xf, const127, const127);
       xi = _mm256_cvtps_epi32(xf);
       xi = _mm256_packus_epi32(xi,  _mm256_setzero_si256());
       xi = _mm256_permute4x64_epi64(xi, 0xD8);
       xi = _mm256_packus_epi16(xi, _mm256_setzero_si256());
       xi = _mm256_permutevar8x32_epi32(xi, _mm256_setr_epi32(0,1, 0,0, 0,0, 0,0));
       //xi = _mm256_permute4x64_epi64(xi, 0x);
       _mm256_storeu_si256 ((__m256i *)&x[i], xi);
   }
   for (i=0;i<rows;i+=8)
   {
      int * restrict y;
      __m256i vy0;
      y = &out[i];
      vy0 = _mm256_loadu_si256((const __m256i *)&y[0]);
      for (j=0;j<cols;j+=4)
      {
         __m256i tmp;
         __m256i vxj;
         __m256i vw;
         vxj = _mm256_set1_epi32(*(int*)&x[j]);
         vw = _mm256_loadu_si256((const __m256i *)w); //_mm256_lddqu_si256?
         tmp = _mm256_maddubs_epi16(vxj, vw); //swap?
         tmp = _mm256_madd_epi16(tmp, ones);
         vy0 = _mm256_add_epi32(vy0, tmp);
         w += 32;
      }
      _mm256_storeu_si256 ((__m256i *)&y[0], vy0);
   }
   for (i=0;i<rows;i++) _out[i] = SCALE_1*out[i];
}
#else
static inline void sgemv_accum8x4(float *out, const qweight *w, int rows, int cols, int col_stride, const float *_x)
{
   int i, j;
   unsigned char x[MAX_INPUTS];
   (void)col_stride;
   for (i=0;i<rows;i++) out[i] *= SCALE;
   for (i=0;i<cols;i++) x[i] = 127+(int)floor(.5+127*_x[i]);
   for (i=0;i<rows;i+=8)
   {
      for (j=0;j<cols;j+=4)
      {
         float * restrict y;
         float xj0, xj1, xj2, xj3;
         xj0 = x[j+0];
         xj1 = x[j+1];
         xj2 = x[j+2];
         xj3 = x[j+3];
         y = &out[i];
         y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         y[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         y[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         y[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         y[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         y[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         w += 32;
      }
   }
   for (i=0;i<rows;i++) out[i] *= SCALE_1;
}
#endif

static inline void sparse_sgemv_accum8x4(float *_out, const qweight *w, int rows, int cols, const int *idx, const float *_x)
{
   __m256i ones;
   int i, j;
   unsigned char x[MAX_INPUTS];
   int out[MAX_OUTPUTS];
   ones = _mm256_set1_epi16(1);
   for (i=0;i<rows;i++) out[i] = SCALE*_out[i];
   //for (i=0;i<cols;i++) x[i] = 127+floor(.5+127*_x[i]);
   __m256 const127 = _mm256_set1_ps(127.f);
   for (i=0;i<cols;i+=8) {
       __m256 xf;
       __m256i xi;
       xf = _mm256_loadu_ps(&_x[i]);
       //xf = _mm256_mul_ps(xf, const127);
       //xf = _mm256_add_ps(xf, const127);
       xf = _mm256_fmadd_ps(xf, const127, const127);
       xi = _mm256_cvtps_epi32(xf);
       xi = _mm256_packus_epi32(xi,  _mm256_setzero_si256());
       xi = _mm256_permute4x64_epi64(xi, 0xD8);
       xi = _mm256_packus_epi16(xi, _mm256_setzero_si256());
       xi = _mm256_permutevar8x32_epi32(xi, _mm256_setr_epi32(0,1, 0,0, 0,0, 0,0));
       //xi = _mm256_permute4x64_epi64(xi, 0x);
       _mm256_storeu_si256 ((__m256i *)&x[i], xi);
   }
   for (i=0;i<rows;i+=8)
   {
      int * restrict y;
      int colblocks;
      __m256i vy0;
      colblocks = *idx++;
      y = &out[i];
      vy0 = _mm256_loadu_si256((const __m256i *)&y[0]);
      for (j=0;j<colblocks;j++)
      {
         __m256i tmp;
         __m256i vxj;
         __m256i vw;
         int pos;
         pos = 4 * (*idx++);
         vxj = _mm256_set1_epi32(*(int*)&x[pos]);
         vw = _mm256_loadu_si256((const __m256i *)w); //_mm256_lddqu_si256?
         tmp = _mm256_maddubs_epi16(vxj, vw); //swap?
         tmp = _mm256_madd_epi16(tmp, ones);
         vy0 = _mm256_add_epi32(vy0, tmp);
         w += 32;
      }
      _mm256_storeu_si256 ((__m256i *)&y[0], vy0);
   }
   for (i=0;i<rows;i++) _out[i] = SCALE_1*out[i];
}


#else /*DOT_PROD*/
typedef float qweight;
#define sgemv_accum8x4 sgemv_accum

static inline void sparse_sgemv_accum8x4(float *out, const qweight *weights, int rows, int ignore, const int *idx, const float *x)
{
   int i, j;
   (void)ignore;
   for (i=0;i<rows;i+=8)
   {
      float * restrict y;
      int cols;
      __m256 vy0;
      y = &out[i];
      vy0 = _mm256_loadu_ps(&y[0]);
      cols = *idx++;
      for (j=0;j<cols;j++)
      {
         int id;
         __m256 vxj;
         __m256 vw;
         id = *idx++;
         vxj = _mm256_broadcast_ss(&x[4*id]);
         vw = _mm256_loadu_ps(&weights[0]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         vxj = _mm256_broadcast_ss(&x[4*id+1]);
         vw = _mm256_loadu_ps(&weights[8]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         vxj = _mm256_broadcast_ss(&x[4*id+2]);
         vw = _mm256_loadu_ps(&weights[16]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         vxj = _mm256_broadcast_ss(&x[4*id+3]);
         vw = _mm256_loadu_ps(&weights[24]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         weights += 32;
      }
      _mm256_storeu_ps (&y[0], vy0);
   }
}
#endif /*DOT_PROD*/

#endif /*VEC_AVX_H*/
