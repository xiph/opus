/* Copyright (c) 2023 Amazon */
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

#include "nnet.h"

extern const WeightArray lpcnet_arrays[];

int parse_record(const unsigned char **data, int *len, WeightArray *array) {
  WeightHead *h = (WeightHead *)*data;
  if (*len < WEIGHT_BLOCK_SIZE) return -1;
  if (h->block_size < h->size) return -1;
  if (*len < h->block_size+WEIGHT_BLOCK_SIZE) return -1;
  if (h->name[sizeof(h->name)-1] != 0) return -1;
  if (h->size < 0) return -1;
  array->name = h->name;
  array->type = h->type;
  array->size = h->size;
  array->data = (*data)+WEIGHT_BLOCK_SIZE;
  
  *data += h->block_size+WEIGHT_BLOCK_SIZE;
  *len -= h->block_size+WEIGHT_BLOCK_SIZE;
  return array->size;
}

int parse_weights(WeightArray **list, const unsigned char *data, int len)
{
  int nb_arrays=0;
  int capacity=20;
  *list = malloc(capacity*sizeof(WeightArray));
  while (len > 0) {
    int ret;
    WeightArray array = {NULL, 0, 0, 0};
    ret = parse_record(&data, &len, &array);
    if (ret > 0) {
      if (nb_arrays+1 >= capacity) {
        /* Make sure there's room for the ending NULL element too. */
        capacity = capacity*3/2;
        *list = realloc(*list, capacity*sizeof(WeightArray));
      }
      (*list)[nb_arrays++] = array;
    }
  }
  (*list)[nb_arrays].name=NULL;
  return nb_arrays;
}

static const void *find_array(const WeightArray *arrays, const char *name) {
  while (arrays->name && strcmp(arrays->name, name) != 0) arrays++;
  return arrays->data;
}

int mdense_init(MDenseLayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *input_weights,
  const char *factor,
  int nb_inputs,
  int nb_neurons,
  int nb_channels,
  int activation)
{
  if ((layer->bias = find_array(arrays, bias)) == NULL) return 1;
  if ((layer->input_weights = find_array(arrays, input_weights)) == NULL) return 1;
  if ((layer->factor = find_array(arrays, factor)) == NULL) return 1;
  layer->nb_inputs = nb_inputs;
  layer->nb_neurons = nb_neurons;
  layer->nb_channels = nb_channels;
  layer->activation = activation;
  return 0;
}

int dense_init(DenseLayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *input_weights,
  int nb_inputs,
  int nb_neurons,
  int activation)
{
  if ((layer->bias = find_array(arrays, bias)) == NULL) return 1;
  if ((layer->input_weights = find_array(arrays, input_weights)) == NULL) return 1;
  layer->nb_inputs = nb_inputs;
  layer->nb_neurons = nb_neurons;
  layer->activation = activation;
  return 0;
}

int gru_init(GRULayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *subias,
  const char *input_weights,
  const char *input_weights_idx,
  const char *recurrent_weights,
  int nb_inputs,
  int nb_neurons,
  int activation,
  int reset_after)
{
  if ((layer->bias = find_array(arrays, bias)) == NULL) return 1;
  if ((layer->subias = find_array(arrays, subias)) == NULL) return 1;
  if ((layer->input_weights = find_array(arrays, input_weights)) == NULL) return 1;
  if ((layer->input_weights_idx = find_array(arrays, input_weights_idx)) == NULL) return 1;
  if ((layer->recurrent_weights = find_array(arrays, recurrent_weights)) == NULL) return 1;
  layer->nb_inputs = nb_inputs;
  layer->nb_neurons = nb_neurons;
  layer->activation = activation;
  layer->reset_after = reset_after;
  return 0;
}

int sparse_gru_init(SparseGRULayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *subias,
  const char *diag_weights,
  const char *recurrent_weights,
  const char *idx,
  int nb_neurons,
  int activation,
  int reset_after)
{
  if ((layer->bias = find_array(arrays, bias)) == NULL) return 1;
  if ((layer->subias = find_array(arrays, subias)) == NULL) return 1;
  if ((layer->diag_weights = find_array(arrays, diag_weights)) == NULL) return 1;
  if ((layer->recurrent_weights = find_array(arrays, recurrent_weights)) == NULL) return 1;
  if ((layer->idx = find_array(arrays, idx)) == NULL) return 1;
  layer->nb_neurons = nb_neurons;
  layer->activation = activation;
  layer->reset_after = reset_after;
  return 0;
}

int conv1d_init(Conv1DLayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *input_weights,
  int nb_inputs,
  int kernel_size,
  int nb_neurons,
  int activation)
{
  if ((layer->bias = find_array(arrays, bias)) == NULL) return 1;
  if ((layer->input_weights = find_array(arrays, input_weights)) == NULL) return 1;
  layer->nb_inputs = nb_inputs;
  layer->kernel_size = kernel_size;
  layer->nb_neurons = nb_neurons;
  layer->activation = activation;
  return 0;
}

int embedding_init(EmbeddingLayer *layer, const WeightArray *arrays,
  const char *embedding_weights,
  int nb_inputs,
  int dim)
{
  if ((layer->embedding_weights = find_array(arrays, embedding_weights)) == NULL) return 1;
  layer->nb_inputs = nb_inputs;
  layer->dim = dim;
  return 0;
}



#if 0
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>

int main()
{
  int fd;
  unsigned char *data;
  int len;
  int nb_arrays;
  int i;
  WeightArray *list;
  struct stat st;
  const char *filename = "weights_blob.bin";
  stat(filename, &st);
  len = st.st_size;
  fd = open(filename, O_RDONLY);
  data = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
  printf("size is %d\n", len);
  nb_arrays = parse_weights(&list, data, len);
  for (i=0;i<nb_arrays;i++) {
    printf("found %s: size %d\n", list[i].name, list[i].size);
  }
  printf("%p\n", list[i].name);
  free(list);
  munmap(data, len);
  close(fd);
  return 0;
}
#endif
