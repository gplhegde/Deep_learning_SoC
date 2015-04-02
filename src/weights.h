#ifndef WEIGHTS_H
#define WEIGHTS_H
#include "parameter_config.h"
extern const float input_weights[NO_INPUT_NEURONS][NO_INPUTS];
extern const float hidden_weights[NO_HIDDEN_NEURONS][NO_INPUT_NEURONS];
extern const float output_weights[NO_OUTPUT_NEURONS][NO_HIDDEN_NEURONS];
extern const float input_bias[NO_INPUT_NEURONS];
extern const float hidden_bias[NO_HIDDEN_NEURONS];
extern const float output_bias[NO_OUTPUT_NEURONS];
#endif // WEIGHTS_H
