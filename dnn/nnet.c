/* Copyright (c) 2018 Mozilla
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <math.h>
#include "opus_types.h"
#include "arch.h"
#include "common.h"
#include "tansig_table.h"
#include "nnet.h"
#include "nnet_data.h"

static OPUS_INLINE float tansig_approx(float x)
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

static OPUS_INLINE float relu(float x)
{
   return x < 0 ? 0 : x;
}

#ifdef __AVX2__
#include <immintrin.h>
static void gemm_accum16(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
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
static void sparse_gemm_accum16(float *out, const float *weights, int rows, const int *idx, const float *x)
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

#else
static void gemm_accum16(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   for (i=0;i<rows;i+=16)
   {
      for (j=0;j<cols;j++)
      {
         const float * restrict w;
         float * restrict y;
         float xj;
         w = &weights[j*col_stride + i];
         xj = x[j];
         y = &out[i];
         y[0] += w[0]*xj;
         y[1] += w[1]*xj;
         y[2] += w[2]*xj;
         y[3] += w[3]*xj;
         y[4] += w[4]*xj;
         y[5] += w[5]*xj;
         y[6] += w[6]*xj;
         y[7] += w[7]*xj;
         y[8] += w[8]*xj;
         y[9] += w[9]*xj;
         y[10] += w[10]*xj;
         y[11] += w[11]*xj;
         y[12] += w[12]*xj;
         y[13] += w[13]*xj;
         y[14] += w[14]*xj;
         y[15] += w[15]*xj;
      }
   }
}
#endif

static void gemm_accum(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   if (rows % 16 == 0 && cols % 16 == 0)
   {
      gemm_accum16(out, weights, rows, cols, col_stride, x);
   } else {
      for (i=0;i<rows;i++)
      {
         for (j=0;j<cols;j++)
            out[i] += weights[j*col_stride + i]*x[j];
      }
   }
}

void compute_activation(float *output, float *input, int N, int activation)
{
   int i;
   if (activation == ACTIVATION_SIGMOID) {
      for (i=0;i<N;i++)
         output[i] = sigmoid_approx(input[i]);
   } else if (activation == ACTIVATION_TANH) {
      for (i=0;i<N;i++)
         output[i] = tansig_approx(input[i]);
   } else if (activation == ACTIVATION_RELU) {
      for (i=0;i<N;i++)
         output[i] = relu(input[i]);
   } else if (activation == ACTIVATION_SOFTMAX) {
      float sum = 0;
      for (i=0;i<N;i++) {
         output[i] = exp(input[i]);
         sum += output[i];
      }
      sum = 1.f/(sum+1e-30);
      for (i=0;i<N;i++)
         output[i] = sum*output[i];
   } else {
      celt_assert(activation == ACTIVATION_LINEAR);
      for (i=0;i<N;i++)
         output[i] = input[i];
   }
}

void compute_dense(const DenseLayer *layer, float *output, const float *input)
{
   int i;
   int N, M;
   int stride;
   M = layer->nb_inputs;
   N = layer->nb_neurons;
   stride = N;
   celt_assert(input != output);
   for (i=0;i<N;i++)
      output[i] = layer->bias[i];
   gemm_accum(output, layer->input_weights, N, M, stride, input);
   compute_activation(output, output, N, layer->activation);
}

void compute_mdense(const MDenseLayer *layer, float *output, const float *input)
{
   int i, c;
   int N, M, C;
   int stride;
   float tmp[MAX_MDENSE_TMP];
   celt_assert(input != output);
   M = layer->nb_inputs;
   N = layer->nb_neurons;
   C = layer->nb_channels;
   celt_assert(N*C <= MAX_MDENSE_TMP);
   stride = N*C;
   for (i=0;i<N*C;i++)
      tmp[i] = layer->bias[i];
   gemm_accum(tmp, layer->input_weights, N*C, M, stride, input);
   compute_activation(tmp, tmp, N*C, ACTIVATION_TANH);
   for (i=0;i<N;i++)
      output[i] = 0;
   for (c=0;c<C;c++)
   {
      for (i=0;i<N;i++)
         output[i] += tmp[c*N + i]*layer->factor[c*N + i];
   }
   compute_activation(output, output, N, layer->activation);
}

void compute_gru(const GRULayer *gru, float *state, const float *input)
{
   int i;
   int N, M;
   int stride;
   float tmp[MAX_RNN_NEURONS];
   float z[MAX_RNN_NEURONS];
   float r[MAX_RNN_NEURONS];
   float h[MAX_RNN_NEURONS];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   celt_assert(input != state);
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   stride = 3*N;
   /* Compute update gate. */
   for (i=0;i<N;i++)
      z[i] = gru->bias[i];
   if (gru->reset_after)
   {
      for (i=0;i<N;i++)
         z[i] += gru->bias[3*N + i];
   }
   gemm_accum(z, gru->input_weights, N, M, stride, input);
   gemm_accum(z, gru->recurrent_weights, N, N, stride, state);
   compute_activation(z, z, N, ACTIVATION_SIGMOID);

   /* Compute reset gate. */
   for (i=0;i<N;i++)
      r[i] = gru->bias[N + i];
   if (gru->reset_after)
   {
      for (i=0;i<N;i++)
         r[i] += gru->bias[4*N + i];
   }
   gemm_accum(r, &gru->input_weights[N], N, M, stride, input);
   gemm_accum(r, &gru->recurrent_weights[N], N, N, stride, state);
   compute_activation(r, r, N, ACTIVATION_SIGMOID);

   /* Compute output. */
   for (i=0;i<N;i++)
      h[i] = gru->bias[2*N + i];
   if (gru->reset_after)
   {
      for (i=0;i<N;i++)
         tmp[i] = gru->bias[5*N + i];
      gemm_accum(tmp, &gru->recurrent_weights[2*N], N, N, stride, state);
      for (i=0;i<N;i++)
         h[i] += tmp[i] * r[i];
      gemm_accum(h, &gru->input_weights[2*N], N, M, stride, input);
   } else {
      for (i=0;i<N;i++)
         tmp[i] = state[i] * r[i];
      gemm_accum(h, &gru->input_weights[2*N], N, M, stride, input);
      gemm_accum(h, &gru->recurrent_weights[2*N], N, N, stride, tmp);
   }
   compute_activation(h, h, N, gru->activation);
   for (i=0;i<N;i++)
      h[i] = z[i]*state[i] + (1-z[i])*h[i];
   for (i=0;i<N;i++)
      state[i] = h[i];
}

void compute_gru2(const GRULayer *gru, float *state, const float *input)
{
   int i;
   int N, M;
   int stride;
   float zrh[3*MAX_RNN_NEURONS];
   float recur[3*MAX_RNN_NEURONS];
   float *z;
   float *r;
   float *h;
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   z = zrh;
   r = &zrh[N];
   h = &zrh[2*N];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   celt_assert(input != state);
   celt_assert(gru->reset_after);
   stride = 3*N;
   /* Compute update gate. */
   for (i=0;i<3*N;i++)
      zrh[i] = gru->bias[i];
   gemm_accum(zrh, gru->input_weights, 3*N, M, stride, input);
   for (i=0;i<3*N;i++)
      recur[i] = gru->bias[3*N + i];
   gemm_accum(recur, gru->recurrent_weights, 3*N, N, stride, state);
   for (i=0;i<2*N;i++)
      zrh[i] += recur[i];
   compute_activation(zrh, zrh, 2*N, ACTIVATION_SIGMOID);
   for (i=0;i<N;i++)
      h[i] += recur[2*N+i]*r[i];
   compute_activation(h, h, N, gru->activation);
   for (i=0;i<N;i++)
      h[i] = z[i]*state[i] + (1-z[i])*h[i];
   for (i=0;i<N;i++)
      state[i] = h[i];
}

void compute_gru3(const GRULayer *gru, float *state, const float *input)
{
   int i;
   int N;
   int stride;
   float zrh[3*MAX_RNN_NEURONS];
   float recur[3*MAX_RNN_NEURONS];
   float *z;
   float *r;
   float *h;
   N = gru->nb_neurons;
   z = zrh;
   r = &zrh[N];
   h = &zrh[2*N];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   celt_assert(input != state);
   celt_assert(gru->reset_after);
   stride = 3*N;
   RNN_COPY(zrh, input, 3*N);
   for (i=0;i<3*N;i++)
      recur[i] = gru->bias[3*N + i];
   gemm_accum(recur, gru->recurrent_weights, 3*N, N, stride, state);
   for (i=0;i<2*N;i++)
      zrh[i] += recur[i];
   compute_activation(zrh, zrh, 2*N, ACTIVATION_SIGMOID);
   for (i=0;i<N;i++)
      h[i] += recur[2*N+i]*r[i];
   compute_activation(h, h, N, gru->activation);
   for (i=0;i<N;i++)
      h[i] = z[i]*state[i] + (1-z[i])*h[i];
   for (i=0;i<N;i++)
      state[i] = h[i];
}

void compute_sparse_gru(const SparseGRULayer *gru, float *state, const float *input)
{
   int i, k;
   int N;
   float zrh[3*MAX_RNN_NEURONS];
   float recur[3*MAX_RNN_NEURONS];
   float *z;
   float *r;
   float *h;
   N = gru->nb_neurons;
   z = zrh;
   r = &zrh[N];
   h = &zrh[2*N];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   celt_assert(input != state);
   celt_assert(gru->reset_after);
   RNN_COPY(zrh, input, 3*N);
   for (i=0;i<3*N;i++)
      recur[i] = gru->bias[3*N + i];
   for (k=0;k<3;k++)
   {
      for (i=0;i<N;i++)
         recur[k*N + i] += gru->diag_weights[k*N + i]*state[i];
   }
   sparse_gemm_accum16(recur, gru->recurrent_weights, 3*N, gru->idx, state);
   for (i=0;i<2*N;i++)
      zrh[i] += recur[i];
   compute_activation(zrh, zrh, 2*N, ACTIVATION_SIGMOID);
   for (i=0;i<N;i++)
      h[i] += recur[2*N+i]*r[i];
   compute_activation(h, h, N, gru->activation);
   for (i=0;i<N;i++)
      h[i] = z[i]*state[i] + (1-z[i])*h[i];
   for (i=0;i<N;i++)
      state[i] = h[i];
}

void compute_conv1d(const Conv1DLayer *layer, float *output, float *mem, const float *input)
{
   int i;
   int N, M;
   int stride;
   float tmp[MAX_CONV_INPUTS];
   celt_assert(input != output);
   celt_assert(layer->nb_inputs*layer->kernel_size <= MAX_CONV_INPUTS);
   RNN_COPY(tmp, mem, layer->nb_inputs*(layer->kernel_size-1));
   RNN_COPY(&tmp[layer->nb_inputs*(layer->kernel_size-1)], input, layer->nb_inputs);
   M = layer->nb_inputs*layer->kernel_size;
   N = layer->nb_neurons;
   stride = N;
   for (i=0;i<N;i++)
      output[i] = layer->bias[i];
   gemm_accum(output, layer->input_weights, N, M, stride, tmp);
   compute_activation(output, output, N, layer->activation);
   RNN_COPY(mem, &tmp[layer->nb_inputs], layer->nb_inputs*(layer->kernel_size-1));
}

void compute_embedding(const EmbeddingLayer *layer, float *output, int input)
{
   int i;
   celt_assert(input >= 0);
   celt_assert(input < layer->nb_inputs);
   /*if (layer->dim == 64) printf("%d\n", input);*/
   for (i=0;i<layer->dim;i++)
   {
      output[i] = layer->embedding_weights[input*layer->dim + i];
   }    
}

void accum_embedding(const EmbeddingLayer *layer, float *output, int input)
{
   int i;
   celt_assert(input >= 0);
   celt_assert(input < layer->nb_inputs);
   /*if (layer->dim == 64) printf("%d\n", input);*/
   for (i=0;i<layer->dim;i++)
   {
      output[i] += layer->embedding_weights[input*layer->dim + i];
   }    
}

int sample_from_pdf(const float *pdf, int N, float exp_boost, float pdf_floor)
{
    int i;
    float sum, norm;
    float r;
    float tmp[DUAL_FC_OUT_SIZE];
    celt_assert(N <= DUAL_FC_OUT_SIZE);
    sum = 0;
    /* Decrease the temperature of the sampling. */
    for (i=0;i<N;i++)
    {
        tmp[i] = pow(pdf[i], 1.f+exp_boost);
        sum += tmp[i];
    }
    norm = 1.f/sum;
    /* Convert tmp to a CDF while subtracting the floor */
    tmp[0] = MAX16(0, norm*tmp[0] - pdf_floor);
    for (i=1;i<N;i++)
    {
        tmp[i] = tmp[i-1] + MAX16(0, norm*tmp[i] - pdf_floor);
    }
    /* Do the sampling (from the cdf). */
    r = tmp[N-1] * ((float)rand()/RAND_MAX);
    for (i=0;i<N-1;i++)
    {
        if (r < tmp[i]) return i;
    }
    return N-1;
}
