#ifndef MLP_H
#define MLP_H

#define MAX_NEURONS 20

typedef struct {
  const float *bias;
  const float *input_weights;
  int nb_inputs;
  int nb_neurons;
  int sigmoid;
} DenseLayer;

typedef struct {
  const float *bias;
  const float *input_weights;
  const float *recurrent_weights;
  int nb_inputs;
  int nb_neurons;
} GRULayer;

const DenseLayer layer0;
const GRULayer layer1;
const DenseLayer layer2;

void compute_dense(const DenseLayer *layer, float *output, const float *input);

void compute_gru(const GRULayer *gru, float *state, const float *input);

#endif
