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
#include "dred_rdovae_constants.h"
#include "plc_data.h"

#ifdef NO_OPTIMIZATIONS
#if defined(_MSC_VER)
#pragma message ("Compiling without any vectorization. This code will be very slow")
#else
#warning Compiling without any vectorization. This code will be very slow
#endif
#endif


#define SOFTMAX_HACK

#define MAX_ACTIVATIONS (4096)

static OPUS_INLINE void vec_swish(float *y, const float *x, int N)
{
   int i;
   float tmp[MAX_ACTIVATIONS];
   celt_assert(N <= MAX_ACTIVATIONS);
   vec_sigmoid(tmp, x, N);
   for (i=0;i<N;i++)
      y[i] = x[i]*tmp[i];
}

static OPUS_INLINE float relu(float x)
{
   return x < 0 ? 0 : x;
}


static void sgemv_accum(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   if (rows % 16 == 0)
   {
      sgemv_accum16(out, weights, rows, cols, col_stride, x);
   } else {
      for (i=0;i<rows;i++)
      {
         for (j=0;j<cols;j++)
            out[i] += weights[j*col_stride + i]*x[j];
      }
   }
}

void compute_activation(float *output, const float *input, int N, int activation)
{
   int i;
   if (activation == ACTIVATION_SIGMOID) {
      vec_sigmoid(output, input, N);
   } else if (activation == ACTIVATION_TANH) {
      vec_tanh(output, input, N);
   } else if (activation == ACTIVATION_SWISH) {
      vec_swish(output, input, N);
   } else if (activation == ACTIVATION_RELU) {
      for (i=0;i<N;i++)
         output[i] = relu(input[i]);
   } else if (activation == ACTIVATION_SOFTMAX) {
#ifdef SOFTMAX_HACK
      RNN_COPY(output, input, N);
      /*for (i=0;i<N;i++)
         output[i] = input[i];*/
#else
      float sum = 0;
      softmax(output, input, N);
      for (i=0;i<N;i++) {
         sum += output[i];
      }
      sum = 1.f/(sum+1e-30);
      for (i=0;i<N;i++)
         output[i] = sum*output[i];
#endif
   } else {
      celt_assert(activation == ACTIVATION_LINEAR);
      for (i=0;i<N;i++)
         output[i] = input[i];
   }
}

void _lpcnet_compute_dense(const DenseLayer *layer, float *output, const float *input)
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
   sgemv_accum(output, layer->input_weights, N, M, stride, input);
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
   sgemv_accum(tmp, layer->input_weights, N*C, M, stride, input);
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

int sample_mdense(const MDenseLayer *layer, const float *input, const float *sampling_logit_table, kiss99_ctx *rng)
{
   int b, j, N, M, C, stride;
   int val=0;
   float thresholds[8];
   M = layer->nb_inputs;
   N = layer->nb_neurons;
   C = layer->nb_channels;
   celt_assert(N*C <= MAX_MDENSE_TMP);
   stride = M*C;
   
   celt_assert(N <= DUAL_FC_OUT_SIZE);

   /* Computing all the random thresholds in advance. These thresholds are directly
      based on the logit to avoid computing the sigmoid.*/
   for (b=0;b<8;b+=4) {
       uint32_t r = kiss99_rand(rng);
       thresholds[b] = sampling_logit_table[r&0xFF];
       thresholds[b+1] = sampling_logit_table[(r>>8)&0xFF];
       thresholds[b+2] = sampling_logit_table[(r>>16)&0xFF];
       thresholds[b+3] = sampling_logit_table[(r>>24)&0xFF];
   }

   for (b=0;b<8;b++)
   {
      int bit;
      int i;
      float sum1, sum2;
      
      i = (1<<b) | val;

      sum1 = layer->bias[i];
      sum2 = layer->bias[i + N];
      for (j=0;j<M;j++) {
         sum1 += layer->input_weights[i*stride + j]*input[j];
         sum2 += layer->input_weights[i*stride + j + M]*input[j];
      }
      sum1 = layer->factor[i]*tanh_approx(sum1);
      sum2 = layer->factor[N + i]*tanh_approx(sum2);
      sum1 += sum2;
      /*sum1 = 1.f/(1 + exp(-sum1));*/
#if 1 /* Sample the decision based on the logit. */
      bit = thresholds[b] < sum1;
#else
      sum1 = sigmoid_approx(sum1);
      bit = .025+.95*((rand()+.5f)/(RAND_MAX+1.f)) < sum1;
#endif
      val = (val << 1) | bit;
   }
   return val;

}


#if 0
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
   sgemv_accum(z, gru->input_weights, N, M, stride, input);
   sgemv_accum(z, gru->recurrent_weights, N, N, stride, state);
   compute_activation(z, z, N, ACTIVATION_SIGMOID);

   /* Compute reset gate. */
   for (i=0;i<N;i++)
      r[i] = gru->bias[N + i];
   if (gru->reset_after)
   {
      for (i=0;i<N;i++)
         r[i] += gru->bias[4*N + i];
   }
   sgemv_accum(r, &gru->input_weights[N], N, M, stride, input);
   sgemv_accum(r, &gru->recurrent_weights[N], N, N, stride, state);
   compute_activation(r, r, N, ACTIVATION_SIGMOID);

   /* Compute output. */
   for (i=0;i<N;i++)
      h[i] = gru->bias[2*N + i];
   if (gru->reset_after)
   {
      for (i=0;i<N;i++)
         tmp[i] = gru->bias[5*N + i];
      sgemv_accum(tmp, &gru->recurrent_weights[2*N], N, N, stride, state);
      for (i=0;i<N;i++)
         h[i] += tmp[i] * r[i];
      sgemv_accum(h, &gru->input_weights[2*N], N, M, stride, input);
   } else {
      for (i=0;i<N;i++)
         tmp[i] = state[i] * r[i];
      sgemv_accum(h, &gru->input_weights[2*N], N, M, stride, input);
      sgemv_accum(h, &gru->recurrent_weights[2*N], N, N, stride, tmp);
   }
   compute_activation(h, h, N, gru->activation);
   for (i=0;i<N;i++)
      h[i] = z[i]*state[i] + (1-z[i])*h[i];
   for (i=0;i<N;i++)
      state[i] = h[i];
}
#endif

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
#ifdef USE_SU_BIAS
   for (i=0;i<3*N;i++)
      zrh[i] = gru->subias[i];
#else
   for (i=0;i<3*N;i++)
      zrh[i] = gru->bias[i];
#endif
   sgemv_accum8x4(zrh, gru->input_weights, 3*N, M, stride, input);
   for (i=0;i<3*N;i++)
      recur[i] = gru->bias[3*N + i];
   sgemv_accum8x4(recur, gru->recurrent_weights, 3*N, N, stride, state);
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

#define MAX_RNN_NEURONS_ALL IMAX(IMAX(MAX_RNN_NEURONS, PLC_MAX_RNN_NEURONS), DRED_MAX_RNN_NEURONS)

void compute_gruB(const GRULayer *gru, const float* gru_b_condition, float *state, const float *input)
{
   int i;
   int N, M;
   int stride;
   float zrh[3*MAX_RNN_NEURONS_ALL];
   float recur[3*MAX_RNN_NEURONS_ALL];
   float *z;
   float *r;
   float *h;
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   z = zrh;
   r = &zrh[N];
   h = &zrh[2*N];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS_ALL);
   celt_assert(input != state);
   celt_assert(gru->reset_after);
   stride = 3*N;
   /* Compute update gate. */
#ifdef USE_SU_BIAS
   for (i=0;i<3*N;i++)
      zrh[i] = gru->subias[i] + gru_b_condition[i];
#else
   for (i=0;i<3*N;i++)
      zrh[i] = gru->bias[i] + gru_b_condition[i];
#endif
   sparse_sgemv_accum8x4(zrh, gru->input_weights, 3*N, M, gru->input_weights_idx, input);
#ifdef USE_SU_BIAS
   for (i=0;i<3*N;i++)
      recur[i] = gru->subias[3*N + i];
#else
   for (i=0;i<3*N;i++)
      recur[i] = gru->bias[3*N + i];
#endif
   sgemv_accum8x4(recur, gru->recurrent_weights, 3*N, N, stride, state);
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
   float zrh[3*MAX_RNN_NEURONS_ALL];
   float recur[3*MAX_RNN_NEURONS_ALL];
   float *z;
   float *r;
   float *h;
   N = gru->nb_neurons;
   z = zrh;
   r = &zrh[N];
   h = &zrh[2*N];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS_ALL);
   celt_assert(input != state);
   celt_assert(gru->reset_after);
   stride = 3*N;
   RNN_COPY(zrh, input, 3*N);
   for (i=0;i<3*N;i++)
      recur[i] = gru->bias[3*N + i];
   sgemv_accum8x4(recur, gru->recurrent_weights, 3*N, N, stride, state);
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

/* The input of this GRU is after the input matrix multiply. */
void compute_sparse_gru(const SparseGRULayer *gru, float *state, const float *input)
{
   int i, k;
   int N;
   float recur[3*MAX_RNN_NEURONS_ALL];
   float *z;
   float *r;
   float *h;
   const float *bias;
   N = gru->nb_neurons;
   z = recur;
   r = &recur[N];
   h = &recur[2*N];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS_ALL);
   celt_assert(input != state);
   celt_assert(gru->reset_after);
#ifdef USE_SU_BIAS
   bias = &gru->subias[3*N];
#else
   bias = &gru->bias[3*N];   
#endif
   for (k=0;k<2;k++)
   {
      for (i=0;i<N;i++)
         recur[k*N + i] = bias[k*N + i] + gru->diag_weights[k*N + i]*state[i] + input[k*N + i];
   }
   for (;k<3;k++)
   {
      for (i=0;i<N;i++)
         recur[k*N + i] = bias[k*N + i] + gru->diag_weights[k*N + i]*state[i];
   }
   sparse_sgemv_accum8x4(recur, gru->recurrent_weights, 3*N, N, gru->idx, state);
   compute_activation(recur, recur, 2*N, ACTIVATION_SIGMOID);
   for (i=0;i<N;i++)
      h[i] = h[i]*r[i] + input[2*N+i];
   compute_activation(h, h, N, gru->activation);
   for (i=0;i<N;i++)
      state[i] = z[i]*state[i] + (1-z[i])*h[i];
}

#define MAX_CONV_INPUTS_ALL IMAX(MAX_CONV_INPUTS, DRED_MAX_CONV_INPUTS)

void compute_conv1d(const Conv1DLayer *layer, float *output, float *mem, const float *input)
{
   int i;
   int N, M;
   int stride;
   float tmp[MAX_CONV_INPUTS_ALL];
   celt_assert(input != output);
   celt_assert(layer->nb_inputs*layer->kernel_size <= MAX_CONV_INPUTS_ALL);
   RNN_COPY(tmp, mem, layer->nb_inputs*(layer->kernel_size-1));
   RNN_COPY(&tmp[layer->nb_inputs*(layer->kernel_size-1)], input, layer->nb_inputs);
   M = layer->nb_inputs*layer->kernel_size;
   N = layer->nb_neurons;
   stride = N;
   for (i=0;i<N;i++)
      output[i] = layer->bias[i];
   sgemv_accum(output, layer->input_weights, N, M, stride, tmp);
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

void compute_gru_a_input(float *output, const float *input, int N, const EmbeddingLayer *layer1, int val1, const EmbeddingLayer *layer2, int val2, const EmbeddingLayer *layer3, int val3) {
   int i;
   for (i=0;i<3*N;i++) {
      output[i] = input[i] + layer1->embedding_weights[val1*layer1->dim + i]
                           + layer2->embedding_weights[val2*layer2->dim + i]
                           + layer3->embedding_weights[val3*layer3->dim + i];
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
