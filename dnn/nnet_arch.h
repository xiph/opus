/* Copyright (c) 2018-2019 Mozilla
                 2023 Amazon */
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

#ifndef NNET_ARCH_H
#define NNET_ARCH_H

#include "nnet.h"
#include "arch.h"
#include "os_support.h"
#include "vec.h"

#define CAT_SUFFIX2(a,b) a ## b
#define CAT_SUFFIX(a,b) CAT_SUFFIX2(a, b)

#define RTCD_SUF(name) CAT_SUFFIX(name, RTCD_ARCH)


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

void RTCD_SUF(compute_activation_)(float *output, const float *input, int N, int activation)
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


void RTCD_SUF(compute_linear_) (const LinearLayer *linear, float *out, const float *in)
{
   int i, M, N;
   const float *bias;
   celt_assert(in != out);
   bias = linear->bias;
   M = linear->nb_inputs;
   N = linear->nb_outputs;
   if (linear->float_weights != NULL) {
     if (linear->weights_idx != NULL) sparse_sgemv8x4(out, linear->float_weights, linear->weights_idx, N, in);
     else sgemv(out, linear->float_weights, N, M, N, in);
   } else if (linear->weights != NULL) {
     if (linear->weights_idx != NULL) sparse_cgemv8x4(out, linear->weights, linear->weights_idx, linear->scale, N, M, in);
     else cgemv8x4(out, linear->weights, linear->scale, N, M, in);
     /* Only use SU biases on for integer matrices on SU archs. */
#ifdef USE_SU_BIAS
     bias = linear->subias;
#endif
   }
   else OPUS_CLEAR(out, N);
   if (bias != NULL) {
      for (i=0;i<N;i++) out[i] += bias[i];
   }
   if (linear->diag) {
      /* Diag is only used for GRU recurrent weights. */
      celt_assert(3*M == N);
      for (i=0;i<M;i++) {
         out[i] += linear->diag[i]*in[i];
         out[i+M] += linear->diag[i+M]*in[i];
         out[i+2*M] += linear->diag[i+2*M]*in[i];
      }
   }
}


#endif
