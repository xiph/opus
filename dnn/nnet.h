/* Copyright (c) 2018 Mozilla
   Copyright (c) 2017 Jean-Marc Valin */
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

#ifndef _NNET_H_
#define _NNET_H_

#include <stddef.h>
#include "vec.h"
#include "kiss99.h"

#define ACTIVATION_LINEAR  0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH    2
#define ACTIVATION_RELU    3
#define ACTIVATION_SOFTMAX 4
#define ACTIVATION_SWISH   5

#define WEIGHT_BLOB_VERSION 0
#define WEIGHT_BLOCK_SIZE 64
typedef struct {
  const char *name;
  int type;
  int size;
  const void *data;
} WeightArray;

#define WEIGHT_TYPE_float 0
#define WEIGHT_TYPE_int 1
#define WEIGHT_TYPE_qweight 2
#define WEIGHT_TYPE_int8 3

typedef struct {
  char head[4];
  int version;
  int type;
  int size;
  int block_size;
  char name[44];
} WeightHead;

/* Generic sparse affine transformation. */
typedef struct {
  const float *bias;
  const float *subias;
  const opus_int8 *weights;
  const float *float_weights;
  const int *weights_idx;
  const float *diag;
  const float *scale;
  int nb_inputs;
  int nb_outputs;
} LinearLayer;

typedef struct {
  const float *bias;
  const float *input_weights;
  int nb_inputs;
  int nb_neurons;
  int activation;
} DenseLayer;

typedef struct {
  const float *bias;
  const float *input_weights;
  const float *factor;
  int nb_inputs;
  int nb_neurons;
  int nb_channels;
  int activation;
} MDenseLayer;

typedef struct {
  const float *bias;
  const float *subias;
  const qweight *input_weights;
  const int *input_weights_idx;
  const qweight *recurrent_weights;
  int nb_inputs;
  int nb_neurons;
  int activation;
  int reset_after;
} GRULayer;

typedef struct {
  const float *bias;
  const float *subias;
  const float *diag_weights;
  const qweight *recurrent_weights;
  const int *idx;
  int nb_neurons;
  int activation;
  int reset_after;
} SparseGRULayer;

typedef struct {
  const float *bias;
  const float *input_weights;
  int nb_inputs;
  int kernel_size;
  int nb_neurons;
  int activation;
} Conv1DLayer;

typedef struct {
  const float *embedding_weights;
  int nb_inputs;
  int dim;
} EmbeddingLayer;

void compute_linear(const LinearLayer *linear, float *out, const float *in);
void compute_generic_dense(const LinearLayer *layer, float *output, const float *input, int activation);
void compute_generic_gru(const LinearLayer *input_weights, const LinearLayer *recurrent_weights, float *state, const float *in);
void compute_generic_conv1d(const LinearLayer *layer, float *output, float *mem, const float *input, int input_size, int activation);
void compute_gated_activation(const LinearLayer *layer, float *output, const float *input, int activation);

void compute_activation(float *output, const float *input, int N, int activation);

void _lpcnet_compute_dense(const DenseLayer *layer, float *output, const float *input);

void compute_mdense(const MDenseLayer *layer, float *output, const float *input);

int sample_mdense(const MDenseLayer *layer,  const float *input, const float *sampling_logit_table, kiss99_ctx *rng);

void compute_gruB(const GRULayer *gru, const float* gru_b_condition, float *state, const float *input);

void compute_sparse_gru(const SparseGRULayer *gru, float *state, const float *input);

void compute_conv1d(const Conv1DLayer *layer, float *output, float *mem, const float *input);

void compute_embedding(const EmbeddingLayer *layer, float *output, int input);

void accum_embedding(const EmbeddingLayer *layer, float *output, int input);

void compute_gru_a_input(float *output, const float *input, int N, const EmbeddingLayer *layer1, int val1, const EmbeddingLayer *layer2, int val2, const EmbeddingLayer *layer3, int val3);

int sample_from_pdf(const float *pdf, int N, float exp_boost, float pdf_floor);


extern const WeightArray lpcnet_arrays[];
extern const WeightArray lpcnet_plc_arrays[];
extern const WeightArray rdovaeenc_arrays[];
extern const WeightArray rdovaedec_arrays[];
extern const WeightArray fwgan_arrays[];
extern const WeightArray nnpitch_arrays[];

int linear_init(LinearLayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *subias,
  const char *weights,
  const char *float_weights,
  const char *weights_idx,
  const char *diag,
  const char *scale,
  int nb_inputs,
  int nb_outputs);

int mdense_init(MDenseLayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *input_weights,
  const char *factor,
  int nb_inputs,
  int nb_neurons,
  int nb_channels,
  int activation);

int dense_init(DenseLayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *input_weights,
  int nb_inputs,
  int nb_neurons,
  int activation);

int gru_init(GRULayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *subias,
  const char *input_weights,
  const char *input_weights_idx,
  const char *recurrent_weights,
  int nb_inputs,
  int nb_neurons,
  int activation,
  int reset_after);

int sparse_gru_init(SparseGRULayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *subias,
  const char *diag_weights,
  const char *recurrent_weights,
  const char *idx,
  int nb_neurons,
  int activation,
  int reset_after);

int conv1d_init(Conv1DLayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *input_weights,
  int nb_inputs,
  int kernel_size,
  int nb_neurons,
  int activation);

int embedding_init(EmbeddingLayer *layer, const WeightArray *arrays,
  const char *embedding_weights,
  int nb_inputs,
  int dim);


#endif /* _MLP_H_ */
