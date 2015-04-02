/*
Author: Gopalakrishna Hegde
Date: 19 Mar 2015
File: classify.c
*/
#include<stdio.h>
#include "classify.h"
#include "parameter_config.h"
#include <math.h>
#include <stdlib.h>

void classifier_out(uint8_t *input , float *output)
{
	float *y1, *y2, *norm_input;
	
	norm_input = (float *)malloc(NO_INPUTS * sizeof(float));
	y1 = (float *)malloc(NO_INPUT_NEURONS *sizeof(float));
	y2 = (float *)malloc(NO_HIDDEN_NEURONS * sizeof(float));
	
	// normalize input
	normalize_input(input, norm_input, NO_INPUTS);
#if 1
	// input layer
	neuron_layer(norm_input, input_weights[0], input_bias, y1, NO_INPUTS, NO_INPUT_NEURONS);

	// hidden layer
	neuron_layer(y1, hidden_weights[0], hidden_bias, y2, NO_INPUT_NEURONS, NO_HIDDEN_NEURONS);

	// output layer
	neuron_layer(y2, output_weights[0], output_bias, output, NO_HIDDEN_NEURONS,NO_OUTPUT_NEURONS);
#endif
	free(norm_input);
	free(y1);
	free(y2);

}
float dot_product(float *input, float *weight, int length)
{
	int i;
	float sum;

	for (i = 0; i< length; i++)
	{
		sum = sum + input[i] * weight[i];
	}
	
	return sum;
}

void neuron_layer(float *input, float * weight, float *bias, float *output, int input_per_neuron, int no_neurons)
{
	float sum;
	int i, neuron;
	
	for (neuron = 0; neuron < no_neurons; neuron++)	
	{
		sum = 0;
		// compute dot product
		for (i = 0; i < input_per_neuron; i++)
		{
			sum = sum + input[i] * (*(weight + neuron*input_per_neuron + i));
		}
		sum = sum + bias[neuron];
		// pass thru activation function
#ifdef BINARY_NEURON
		sum = (sum > 0)? 1: 0;
#elif defined (SIGMOID_NEURON)
		sum = 1 / (1 + exp((double)sum));
#elif defined (TANH_NEURON)
		sum = tanh((double)sum);
#else	// default case is linear activation function
		// nothing to do for linear neuron???
#endif 
		output[neuron] = sum;
	}
}

void normalize_input(uint8_t *input, float *norm_out, int length)
{
	uint8_t max;
	int i;

	// find the max input pixel
	max = input[0];
	for (i = 1; i < length; i++)
	{
		if(input[i] > max)
		{
			max = input[i];
		}
	}
	
	// normalize	
	for (i = 0; i < length; i++)
	{
		norm_out[i] = input[i]/max;
	}
}


