#ifndef _DEEP_LAYER_H_
#define _DEEP_LAYER_H_

#define MAX_ROWS 					(140)			// MAX input image size
#define MAX_COLS 					(140)
#define L1_K_ROWS					(7	)			// size of layer 1 kernel
#define L1_K_COLS					(7	)
#define L1_MAX_MAPS 				(64	)			// layer 1 max maps


#define L2_K_ROWS					(5	)			// size of layer 2 kernel
#define L2_K_COLS					(5	)
#define L2_MAX_MAPS 				(128)			// layer 2 max maps


#define L3_K_ROWS					(3	)			// size of layer 2 kernel
#define L3_K_COLS					(3	)
#define L3_MAX_MAPS 				(256)			// layer 2 max maps


#define L1_L2_CONNECTIONS 			(10	)			// input to layer2 is weighted sum of these many layer1 maps
#define L2_L3_CONNECTIONS 			(8	)			// input to layer3 is weighted sum of these many layer2 maps




int v_teradeep_layer1(uint8_t *in_img, int8_t **kernel, uint8_t **out_img, int max_row, int max_col, int k_rows, int k_cols, int max_maps);
int v_teradeep_layer_generic(uint8_t *in_img, int8_t *kernel, uint8_t *out_img, int max_row, int max_col, int k_rows, int k_cols);



void s_filter2D(uint8_t *image, int8_t *kernel, uint8_t *dest_image, int M, int N, int k_rows, int k_cols);
void s_max_pool(uint8_t *in_img, uint8_t *out_img, int max_rows, int max_cols);
int s_teradeep_layer_generic(uint8_t *in_img, int8_t *kernel, uint8_t *out_img, int max_row, int max_col, int k_rows, int k_cols);
int v_teradeep_layer1(uint8_t *in_img, int8_t **kernel, uint8_t **out_img, int max_row, int max_col, int k_rows, int k_cols, int max_maps);
int v_filter2D(uint8_t *in_img, int8_t *kernel, uint8_t *out_img, int max_row, int max_col, int k_rows, int k_cols);
#endif // _DEEP_LAYER_H_
