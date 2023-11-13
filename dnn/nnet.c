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
#include "nnet.h"
#include "dred_rdovae_constants.h"
#include "plc_data.h"
#include "fargan.h"
#include "os_support.h"
#include "vec.h"

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

void compute_generic_dense(const LinearLayer *layer, float *output, const float *input, int activation, int arch)
{
   compute_linear(layer, output, input, arch);
   compute_activation(output, output, layer->nb_outputs, activation);
}

#define MAX_RNN_NEURONS_ALL IMAX(IMAX(FARGAN_MAX_RNN_NEURONS, PLC_MAX_RNN_NEURONS), DRED_MAX_RNN_NEURONS)


void compute_generic_gru(const LinearLayer *input_weights, const LinearLayer *recurrent_weights, float *state, const float *in, int arch)
{
  int i;
  int N;
  float zrh[3*MAX_RNN_NEURONS_ALL];
  float recur[3*MAX_RNN_NEURONS_ALL];
  float *z;
  float *r;
  float *h;
  celt_assert(3*recurrent_weights->nb_inputs == recurrent_weights->nb_outputs);
  celt_assert(input_weights->nb_outputs == recurrent_weights->nb_outputs);
  N = recurrent_weights->nb_inputs;
  z = zrh;
  r = &zrh[N];
  h = &zrh[2*N];
  celt_assert(recurrent_weights->nb_outputs <= 3*MAX_RNN_NEURONS_ALL);
  celt_assert(in != state);
  compute_linear(input_weights, zrh, in, arch);
  compute_linear(recurrent_weights, recur, state, arch);
  for (i=0;i<2*N;i++)
     zrh[i] += recur[i];
  compute_activation(zrh, zrh, 2*N, ACTIVATION_SIGMOID);
  for (i=0;i<N;i++)
     h[i] += recur[2*N+i]*r[i];
  compute_activation(h, h, N, ACTIVATION_TANH);
  for (i=0;i<N;i++)
     h[i] = z[i]*state[i] + (1-z[i])*h[i];
  for (i=0;i<N;i++)
     state[i] = h[i];
}

void compute_glu(const LinearLayer *layer, float *output, const float *input, int arch)
{
   int i;
   float act2[MAX_INPUTS];
   celt_assert(layer->nb_inputs == layer->nb_outputs);
   compute_linear(layer, act2, input, arch);
   compute_activation(act2, act2, layer->nb_outputs, ACTIVATION_SIGMOID);
   if (input == output) {
     /* Give a vectorization hint to the compiler for the in-place case. */
     for (i=0;i<layer->nb_outputs;i++) output[i] = output[i]*act2[i];
   } else {
     for (i=0;i<layer->nb_outputs;i++) output[i] = input[i]*act2[i];
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
      OPUS_COPY(output, input, N);
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
      if (input != output) {
         for (i=0;i<N;i++)
            output[i] = input[i];
      }
   }
}

void _lpcnet_compute_dense(const DenseLayer *layer, float *output, const float *input, int arch)
{
   LinearLayer matrix;
   celt_assert(input != output);
   matrix.bias = layer->bias;
   matrix.subias = NULL;
   matrix.float_weights = layer->input_weights;
   matrix.weights = NULL;
   matrix.weights_idx = NULL;
   matrix.diag = NULL;
   matrix.nb_inputs = layer->nb_inputs;
   matrix.nb_outputs = layer->nb_neurons;
   matrix.scale = NULL;
   compute_linear(&matrix, output, input, arch);
   compute_activation(output, output, layer->nb_neurons, layer->activation);
}

#ifdef USE_SU_BIAS
#define bias_type subias
#else
#define bias_type bias
#endif
#define MAX_IDX_SIZE 8192

void compute_gruB(const GRULayer *gru, const float* gru_b_condition, float *state, const float *input, int arch)
{
  LinearLayer in_matrix, rec_matrix;
  int i, M, N;
  float bias[3*MAX_RNN_NEURONS_ALL];
  float scale[3*MAX_RNN_NEURONS_ALL];
  M = gru->nb_inputs;
  N = gru->nb_neurons;

  in_matrix.bias = bias;
  in_matrix.diag = NULL;
  in_matrix.nb_inputs = M;
  in_matrix.nb_outputs = 3*N;
  in_matrix.subias = bias;
#ifdef DISABLE_DOT_PROD
  for (i=0;i<3*N;i++) bias[i] = gru->bias[i] + gru_b_condition[i];
  in_matrix.scale = NULL;
  in_matrix.float_weights = gru->input_weights;
  in_matrix.weights = NULL;
#else
  for (i=0;i<3*N;i++) bias[i] = gru->bias_type[i] + gru_b_condition[i];
  for (i=0;i<3*N;i++) scale[i] = SCALE_1;
  in_matrix.scale = scale;
  in_matrix.weights = gru->input_weights;
  in_matrix.float_weights = NULL;
#endif
  in_matrix.weights_idx = gru->input_weights_idx;

  rec_matrix.bias = &gru->bias[3*N];
  rec_matrix.diag = NULL;
  rec_matrix.nb_inputs = N;
  rec_matrix.nb_outputs = 3*N;
  rec_matrix.scale = scale;
  rec_matrix.subias = &gru->subias[3*N];
#ifdef DISABLE_DOT_PROD
  rec_matrix.scale = NULL;
  rec_matrix.float_weights = gru->recurrent_weights;
  rec_matrix.weights = NULL;
#else
  rec_matrix.scale = scale;
  rec_matrix.weights = gru->recurrent_weights;
  rec_matrix.float_weights = NULL;
#endif
  rec_matrix.weights_idx = NULL;
  compute_generic_gru(&in_matrix, &rec_matrix, state, input, arch);
}


#define MAX_CONV_INPUTS_ALL DRED_MAX_CONV_INPUTS

void compute_generic_conv1d(const LinearLayer *layer, float *output, float *mem, const float *input, int input_size, int activation, int arch)
{
   float tmp[MAX_CONV_INPUTS_ALL];
   celt_assert(input != output);
   celt_assert(layer->nb_inputs <= MAX_CONV_INPUTS_ALL);
   OPUS_COPY(tmp, mem, layer->nb_inputs-input_size);
   OPUS_COPY(&tmp[layer->nb_inputs-input_size], input, input_size);
   compute_linear(layer, output, tmp, arch);
   compute_activation(output, output, layer->nb_outputs, activation);
   OPUS_COPY(mem, &tmp[input_size], layer->nb_inputs-input_size);
}

void compute_generic_conv1d_dilation(const LinearLayer *layer, float *output, float *mem, const float *input, int input_size, int dilation, int activation, int arch)
{
   float tmp[MAX_CONV_INPUTS_ALL];
   int ksize = layer->nb_inputs/input_size;
   int i;
   celt_assert(input != output);
   celt_assert(layer->nb_inputs <= MAX_CONV_INPUTS_ALL);
   if (dilation==1) OPUS_COPY(tmp, mem, layer->nb_inputs-input_size);
   else for (i=0;i<ksize-1;i++) OPUS_COPY(&tmp[i*input_size], &mem[i*input_size*dilation], input_size);
   OPUS_COPY(&tmp[layer->nb_inputs-input_size], input, input_size);
   compute_linear(layer, output, tmp, arch);
   compute_activation(output, output, layer->nb_outputs, activation);
   if (dilation==1) OPUS_COPY(mem, &tmp[input_size], layer->nb_inputs-input_size);
   else {
     OPUS_COPY(mem, &mem[input_size], input_size*dilation*(ksize-1)-input_size);
     OPUS_COPY(&mem[input_size*dilation*(ksize-1)-input_size], input, input_size);
   }
}


/* Computes non-padded convolution for input [ ksize1 x in_channels x (len2+ksize2) ],
   kernel [ out_channels x in_channels x ksize1 x ksize2 ],
   storing the output as [ out_channels x len2 ].
   We assume that the output dimension along the ksize1 axis is 1,
   i.e. processing one frame at a time. */
static void conv2d_float(float *out, const float *weights, int in_channels, int out_channels, int ktime, int kheight, const float *in, int height, int hstride)
{
   int i;
   int in_stride;
   in_stride = height+kheight-1;
   for (i=0;i<out_channels;i++) {
      int m;
      OPUS_CLEAR(&out[i*hstride], height);
      for (m=0;m<in_channels;m++) {
         int t;
         for (t=0;t<ktime;t++) {
            int h;
            for (h=0;h<kheight;h++) {
               int j;
               for (j=0;j<height;j++) {
                  out[i*hstride + j] += weights[i*in_channels*ktime*kheight + m*ktime*kheight + t*kheight + h] *
                                     in[t*in_channels*in_stride + m*in_stride + j + h];
               }
            }
         }
      }
   }
}

static void conv2d_3x3_float(float *out, const float *weights, int in_channels, int out_channels, const float *in, int height, int hstride)
{
   int i;
   int in_stride;
   int kheight, ktime;
   kheight = ktime = 3;
   in_stride = height+kheight-1;
   for (i=0;i<out_channels;i++) {
      int m;
      OPUS_CLEAR(&out[i*hstride], height);
      for (m=0;m<in_channels;m++) {
         int j;
         for (j=0;j<height;j++) {
            /* Unrolled version of previous function -- compiler will figure out the indexing simplifications. */
            out[i*hstride + j] += weights[i*in_channels*ktime*kheight + m*ktime*kheight + 0*kheight + 0]*in[0*in_channels*in_stride + m*in_stride + j + 0]
                                + weights[i*in_channels*ktime*kheight + m*ktime*kheight + 0*kheight + 1]*in[0*in_channels*in_stride + m*in_stride + j + 1]
                                + weights[i*in_channels*ktime*kheight + m*ktime*kheight + 0*kheight + 2]*in[0*in_channels*in_stride + m*in_stride + j + 2]
                                + weights[i*in_channels*ktime*kheight + m*ktime*kheight + 1*kheight + 0]*in[1*in_channels*in_stride + m*in_stride + j + 0]
                                + weights[i*in_channels*ktime*kheight + m*ktime*kheight + 1*kheight + 1]*in[1*in_channels*in_stride + m*in_stride + j + 1]
                                + weights[i*in_channels*ktime*kheight + m*ktime*kheight + 1*kheight + 2]*in[1*in_channels*in_stride + m*in_stride + j + 2]
                                + weights[i*in_channels*ktime*kheight + m*ktime*kheight + 2*kheight + 0]*in[2*in_channels*in_stride + m*in_stride + j + 0]
                                + weights[i*in_channels*ktime*kheight + m*ktime*kheight + 2*kheight + 1]*in[2*in_channels*in_stride + m*in_stride + j + 1]
                                + weights[i*in_channels*ktime*kheight + m*ktime*kheight + 2*kheight + 2]*in[2*in_channels*in_stride + m*in_stride + j + 2];
               }
      }
   }
}

#define MAX_CONV2D_INPUTS 8192

void compute_conv2d(const Conv2dLayer *conv, float *out, float *mem, const float *in, int height, int hstride, int activation)
{
   int i;
   const float *bias;
   float in_buf[MAX_CONV2D_INPUTS];
   int time_stride;
   celt_assert(in != out);
   time_stride = conv->in_channels*(height+conv->kheight-1);
   celt_assert(conv->ktime*time_stride <= MAX_CONV2D_INPUTS);
   OPUS_COPY(in_buf, mem, (conv->ktime-1)*time_stride);
   OPUS_COPY(&in_buf[(conv->ktime-1)*time_stride], in, time_stride);
   OPUS_COPY(mem, &in_buf[time_stride], (conv->ktime-1)*time_stride);
   bias = conv->bias;
   if (conv->kheight == 3 && conv->ktime == 3)
     conv2d_3x3_float(out, conv->float_weights, conv->in_channels, conv->out_channels, in_buf, height, hstride);
   else
     conv2d_float(out, conv->float_weights, conv->in_channels, conv->out_channels, conv->ktime, conv->kheight, in_buf, height, hstride);
   if (bias != NULL) {
     for (i=0;i<conv->out_channels;i++) {
       int j;
       for (j=0;j<height;j++) out[i*hstride+j] += bias[i];
     }
   }
   for (i=0;i<conv->out_channels;i++) {
     compute_activation(&out[i*hstride], &out[i*hstride], height, activation);
   }
}
