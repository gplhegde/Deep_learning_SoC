
/*
 * Author: Gopalakrishna Hegde
 * Date: 26 Feb 2015
 * filename: teradeep_main.c
 *
*/


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "vbx.h"
#include "vbx_types.h"
#include "vbx_test.h"
#include "vbx_port.h"
#include "deep_layer.h"
#include "connections.h"
#include "map_add.h"
#include "parameter_config.h"
#include "classify.h"

int8_t kernel_fix3[9] = {
13, 16, 0,
38, -16, 13,
-13, 127, -128
};
float kernel_float [9]= {
0.1, 0.123456, 0,
0.3, -0.125, 0.1,
-0.1, 0.99, -0.99
};



int8_t kernel_fix5[25] = {
13, 16, 0, -16, 13,
38, -16, 13, 16, 26,
-128, 127, -128, 82, 46,
16, 0, -16, 38, -13,
16, 13, -16, -26, 38
};
int8_t kernel_fix7[49] = {
13, 16, 0, -16, 13, -20, 25,
38, -16, 13, 16, 26, 12, -10,
-128, 127, -128, 82, 46, 32, 2,
16, 0, -16, 38, -13, 0, -10,
16, 13, -16, -26, 38, 15, 0,
38, -16, 13, 16, 26, 12, -10,
16, 0, -16, 38, -13, 0, -10
};
/*
float kernel_float [25]= {
0.1, 0.123456, 0, -0.123456, 0.1,
0.3, -0.125, 0.1, 0.125, -0.2
-1, 0.99, -0.99, 0.64, 0.36,
0.123456, 0, -0.123456, 0.3, -0.1,
0.125, 0.1, -0.125, -0.2, 0.3

};*/


int main (void)
{

	int row, col, map, mapping, con;
	vbx_timestamp_t start_time, stop_time;
	Mat in_img;
	uint8_t *l1_map[L1_MAX_MAPS], *l2_map[L2_MAX_MAPS], *l3_map[L3_MAX_MAPS];
	Mat l2_acc, l3_acc;

	int8_t  *l1_kernel[L1_MAX_MAPS], *l2_kernel[L2_MAX_MAPS], *l3_kernel[L3_MAX_MAPS];
	printf("--------------Deeplearn-Teradeep Application Start----------\n\n");
	printf("------Configuration Details--------- \n");
#ifdef MXP
	printf("MXP Enabled\n");
#else
	printf("MXP Disabled, APP is running entirely on ARM\n");
#endif

	printf("-------Parameter details---------");
	printf("L1 maps: %d, L2 maps: %d, L3 maps:%d\n", L1_MAX_MAPS, L2_MAX_MAPS, L3_MAX_MAPS);
	printf("L1 kernel size: %d,L3 kernel size: %d,L3 kernel size: %d\n", L1_K_ROWS, L2_K_ROWS, L3_K_ROWS);
	printf("Input neurons: %d, Hidden neurons: %d, Output neurons: %d\n", NO_INPUT_NEURONS, NO_HIDDEN_NEURONS, NO_OUTPUT_NEURONS);
//----------------------------------------------------------------------
	float * neuron_output = (float *) malloc(NO_OUTPUT_NEURONS*sizeof(float));
	in_img.img_data = (uint8_t *)malloc(MAX_ROWS*MAX_COLS*sizeof(uint8_t));
    in_img.no_rows = MAX_ROWS;
    in_img.no_cols = MAX_COLS;
//----------------------------------------------------------------------
	// allocate input and output buffer
	for (map = 0; map < L1_MAX_MAPS; map++)
	{
		l1_map[map] = (uint8_t *)malloc((in_img.no_rows/2)*(in_img.no_cols/2)*sizeof(uint8_t));
		memset(l1_map[map], 255, (in_img.no_rows/2)*(in_img.no_cols/2));
		l1_kernel[map]  = kernel_fix7;
	}

	for (map = 0; map < L2_MAX_MAPS; map++)
	{
		l2_map[map] = (uint8_t *)malloc((in_img.no_rows/4)*(in_img.no_cols/4)*sizeof(uint8_t));
		memset(l2_map[map], 255, (in_img.no_rows/4)*(in_img.no_cols/4));
		l2_kernel[map]  = kernel_fix5;
	}
	for (map = 0; map < L3_MAX_MAPS; map++)
	{
		l3_map[map] = (uint8_t *)malloc((in_img.no_rows/8)*(in_img.no_cols/8)*sizeof(uint8_t));
		memset(l3_map[map], 255, (in_img.no_rows/8)*(in_img.no_cols/8));
		l3_kernel[map]  = kernel_fix3;
	}
	
	l2_acc.img_data = (uint8_t *)malloc((in_img.no_rows/2)*(in_img.no_cols/2)*sizeof(uint8_t));
	l3_acc.img_data = (uint8_t *)malloc((in_img.no_rows/4)*(in_img.no_cols/4)*sizeof(uint8_t));
//----------------------------------------------------------------------
	vbx_test_init();
//----------------------------------------------------------------------
	// layer 1 computations
	printf("\nStart Layer1..................\n");
	vbx_timestamp_start();
	start_time = vbx_timestamp();
#ifdef MXP
	v_teradeep_layer1(in_img.img_data, l1_kernel, l1_map, in_img.no_rows, in_img.no_cols, L1_K_ROWS, L1_K_COLS, L1_MAX_MAPS);
#else
	s_teradeep_layer1(in_img.img_data, l1_kernel, l1_map, in_img.no_rows, in_img.no_cols, L1_K_ROWS, L1_K_COLS, L1_MAX_MAPS);
#endif // MXP
#if 0
	stop_time = vbx_timestamp();
	printf("Layer1 complete: max_rows = %d, max_cols = %d, kernel_size = %d, no_maps = %d\n", 
		in_img.no_rows, in_img.no_cols, L1_K_ROWS,L1_MAX_MAPS);
	printf("Runtime in sec = ");
	vbx_print_scalar_time(start_time, stop_time);

//----------------------------------------------------------------------
	// layer 2 computations
	printf("\nStart Layer2..................\n");
	start_time = vbx_timestamp();
#endif
	l2_acc.no_rows = in_img.no_rows/2;
	l2_acc.no_cols = in_img.no_cols/2;
	for(map = 0; map < L2_MAX_MAPS; map++)
	{
		// form input for layer 2 map by adding l1 maps
		memset(l2_acc.img_data, 0, (in_img.no_rows/2)*(in_img.no_cols/2)*sizeof(uint8_t));
		for(con = 0;con < L1_L2_CONNECTIONS; con++)
		{
			mapping = l1_l2_connections[map][con];
			img_add(l2_acc.img_data, l1_map[mapping], l2_acc.img_data, l2_acc.no_rows, l2_acc.no_cols);
		}
		// compute l2 map
#ifdef MXP
		v_teradeep_layer_generic(l2_acc.img_data, l2_kernel[map], l2_map[map], 
			in_img.no_rows/2, in_img.no_cols/2, L2_K_ROWS, L2_K_COLS);
#else
		s_teradeep_layer_generic(l2_acc.img_data, l2_kernel[map], l2_map[map], 
			in_img.no_rows/2, in_img.no_cols/2, L2_K_ROWS, L2_K_COLS);
#endif//MXP

	}
#if 0
	stop_time = vbx_timestamp();
	printf("Layer2 complete: max_rows = %d, max_cols = %d, kernel_size = %d, no_maps = %d\n", 
		in_img.no_rows/2, in_img.no_cols/2, L2_K_ROWS,L2_MAX_MAPS);
	printf("Runtime in sec = ");
	vbx_print_scalar_time(start_time, stop_time);
//----------------------------------------------------------------------
    // layer 3 computations
	printf("\nStart Layer3..................\n");
	start_time = vbx_timestamp();
#endif
    l3_acc.no_rows = in_img.no_rows/4;
    l3_acc.no_cols = in_img.no_cols/4;
    for(map = 0; map < L3_MAX_MAPS; map++)
    {
        // form input for layer 2 map by adding l1 maps
        memset(l3_acc.img_data, 0, (in_img.no_rows/4)*(in_img.no_cols/4)*sizeof(uint8_t));
        for(con = 0;con < L2_L3_CONNECTIONS; con++)
        {
            mapping = l2_l3_connections[map][con];
            img_add(l3_acc.img_data, l2_map[mapping], l3_acc.img_data, l3_acc.no_rows, l3_acc.no_cols);
        }
        // compute l2 map
#ifdef MXP
        v_teradeep_layer_generic(l3_acc.img_data, l3_kernel[map], l3_map[map], 
            in_img.no_rows/4, in_img.no_cols/4, L3_K_ROWS, L3_K_COLS);
#else
        s_teradeep_layer_generic(l3_acc.img_data, l3_kernel[map], l3_map[map], 
            in_img.no_rows/4, in_img.no_cols/4, L3_K_ROWS, L3_K_COLS);
#endif// MXP

    }
#if 0
	stop_time = vbx_timestamp();
	printf("Layer3 complete: max_rows = %d, max_cols = %d, kernel_size = %d, no_maps = %d\n", 
		in_img.no_rows/4, in_img.no_cols/4, L3_K_ROWS,L3_MAX_MAPS);
	printf("Runtime in sec = ");
	vbx_print_scalar_time(start_time, stop_time);
#endif

//----------------------------------------------------------------------
	// Classifier: lets give just 1 map as input to input neurons
	classifier_out(l3_map[0], neuron_output);
	stop_time = vbx_timestamp();
	printf("\nTotal Runtime = \n");
	vbx_print_scalar_time(start_time, stop_time);
	printf("\nEnd deep learning..................\n");
//----------------------------------------------------------------------
	// dealloate memory
	for ( map = 0; map < L1_MAX_MAPS; map++)
	{
		free(l1_map[map]);
	}

	for ( map = 0; map < L2_MAX_MAPS; map++)
	{
		free(l2_map[map]);
	}
	for ( map = 0; map < L3_MAX_MAPS; map++)
	{
		free(l3_map[map]);
	}
	free(l2_acc.img_data);
	free(l3_acc.img_data);
	free(neuron_output);
//----------------------------------------------------------------------
	// we are done with simulation...lets kill the simulator
	return 0;

}
