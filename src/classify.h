#ifndef _CLASSIFY_H
#define _CLASSIFY_H
#include <stdint.h>
#include "weights.h"
void classifier_out(uint8_t *input , float *output);
float dot_product(float *input, float *weight, int length);
void neuron_layer(float *input, float * weight, float *bias, float *output, int input_per_neuron, int no_neurons);
void normalize_input(uint8_t *input, float *norm_out, int length);
#endif // _CLASSIFY_H
