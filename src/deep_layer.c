/*
 * Author: Gopalakrishna Hegde
 * Date: 25 Feb 2015
 * filename: deep_layer.c
 *
*/


#include <stdio.h>
#include "vbx.h"
#include "vbx_types.h"
#include "vbx_port.h"
#include "deep_layer.h"
/* Takes L1_MAPS number of kernels, convolves the input image with all kernels and does max poooling and outputs L1_MAPS number of maps
   - kernel should be in Q1.7 fixed point 
	- boundary conditions are not handled
   process:	
		Input image--->2D filtering (convolution) ----------->Rectification---->max-pooling
			WxH									  64 maps(W-kcols+1)x(H-krows+1)		(W-kcols+1)/2x(H-krows+1)/2

	** 16 bit Saturation is not handled as input image has all +ve numbers. Thus there exists no case of -128x-128
NOTE: This API can be used to generate maps for other layers by sending max_maps = 1 and calling the same API repeatedly.
*/ 
int v_teradeep_layer1(uint8_t *in_img, int8_t **kernel, uint8_t **out_img, int max_row, int max_col, int k_rows, int k_cols, int max_maps)
{
	
	int MAX_K_COLS	= 9;									// max kernel size is 9x9
	int MAX_K_ROWS	= 9;
	int L1_MAPS		= 64;									// MAX no of maps at layer1
	int MAX_IMG_COLS	= 140;								// MAX no of columns in the image
	// Buffers for holding 3 rows of image
    uint8_t *in_rows[MAX_K_COLS],  *row_temp;				// pointers to input rows
	uint8_t *filt_out_even[L1_MAPS], *filt_out_odd[L1_MAPS];// buffers to hold 2 filtered rows which are inputs to max-pooling
	uint8_t *pool_out[L1_MAPS];								// final pooled row of a map.

    // Buffers for holding product of rows with kernel elements
    int16_t *par_prod[MAX_K_ROWS*MAX_K_COLS];

    // Buffers for holding filtered rows and final filtered row
    int32_t *row_acc[MAX_K_ROWS],  *final_sum, *temp_acc;
	int8_t *v_temp;
    int row, i, j, map;

	// Flush data cache
    vbx_dcache_flush_all();
//----------------------------------------------------------------------------------------------------------
	// input buffer and row accumulators allocation
	for (row = 0; row < k_rows; row++)
	{
		// allocate scratchpad for MAX_K_ROWS number of input rows
		if ( (NULL == (in_rows[row] = (uint8_t *)vbx_sp_malloc(MAX_IMG_COLS*sizeof(uint8_t)))) ||
		// allocate buffers for row accumulation
			 (NULL == (row_acc[row] = (int32_t *)vbx_sp_malloc(MAX_IMG_COLS*sizeof(int32_t)))))
		{
			printf("No scratchpad... exiting\n");
			return -1;
		}
	}
	// final result buffer for a row.
	if ((NULL == (final_sum = (int32_t *)vbx_sp_malloc(MAX_IMG_COLS*sizeof(int32_t)))) ||
		(NULL == (temp_acc = (int32_t *)vbx_sp_malloc(MAX_IMG_COLS*sizeof(int32_t)))) ||
		(NULL == (v_temp = (int8_t *)vbx_sp_malloc(MAX_IMG_COLS*sizeof(int8_t)))))
	{
		printf("No scratchpad... exiting\n");
		return -1;
	}

	// allcate buffers for partial products
	for (i = 0; i < k_rows*k_cols; i++)
	{
		if (NULL == (par_prod[i] = (int16_t*)vbx_sp_malloc(MAX_IMG_COLS*sizeof(int16_t))))
		{
			printf("No scrachpad available... exiting\n\n");
			return -1;
		}
	}
	// buffers for holding 2 filtered rows of all maps and also final pooled row of all maps
	for ( i = 0; i < max_maps; i++)
	{
		if ( (NULL == (filt_out_even[i] = (uint8_t *)vbx_sp_malloc(MAX_IMG_COLS*sizeof(uint8_t))))||
			 (NULL == (filt_out_odd[i] = (uint8_t *)vbx_sp_malloc(MAX_IMG_COLS*sizeof(uint8_t)))) ||
		     (NULL == (pool_out[i] = (uint8_t *)vbx_sp_malloc(MAX_IMG_COLS*sizeof(uint8_t))))
		) {
			printf("No scratchpad... exiting\n");
			return -1;
		}
	}

//----------------------------------------------------------------------------------------------------------
	vbx_set_vl(max_col);
    // transfer (k_rows -1) rows to scratchpad
	for (i = 0; i < k_rows - 1; i++)
	{
		vbx_dma_to_vector(in_rows[i], in_img + max_col * i, max_col * sizeof(uint8_t));
		vbx(SVBU, VSHR, in_rows[i], 1, in_rows[i] ); // convert from 8 bit unsigned to 8 bit signed
	}

    // Main processing loop
	for (row = 0; row < (max_row - k_rows + 1); row++)
    {
		// load new row
       	vbx_dma_to_vector(in_rows[k_rows-1], in_img+max_col*(row+k_rows-1), max_col*sizeof(uint8_t));
		vbx(SVBU, VSHR, in_rows[k_rows-1], 1, in_rows[k_rows-1] );

		// perform filtering with all kernels
		for (map = 0; map < max_maps; map++)
		{
			// reset accumulation buffers... any other better way to do this???
			for (i = 0; i < k_rows; i++)
			{
				vbx(SVW, VMUL, row_acc[i], 0, row_acc[i]);
			}
			// Multiplications required for filtering
			for (i = 0; i < k_rows; i++)
			{
				for (j = 0; j < k_cols; j++)
				{
					vbx(SVBH, VMUL, par_prod[i*k_cols+j], *(kernel[map] + i*k_rows + j), in_rows[i]);
					vbx(SVH, VSHL, par_prod[i*k_cols+j], 1, par_prod[i*k_cols+j]); // remove 1 sign bit
				}
			}
			// Add weighted rows with sliding
			for ( i = 0; i < k_rows; i++)
			{
				for (j = 0; j < k_cols-1; j+=2)
				{
					vbx(VVHW, VADD, temp_acc, (par_prod[i*k_cols+j]+j), (par_prod[i*k_cols+j+1]+j+1));
					vbx(VVW, VADD, row_acc[i], row_acc[i], temp_acc);
				}
				if ( j == k_cols-1) // odd number of cols in kernel
				{
					vbx(VVHW, VMOV, temp_acc, par_prod[i*k_cols+j] + j, 0);
					vbx(VVW, VADD, row_acc[i], row_acc[i], temp_acc);
				}
			}

			// reset final sum
			vbx(SVW, VMUL, final_sum, 0, final_sum);
			// add all accumulated rows to get final filtered row
			for ( i = 0; i < k_rows; i++)
			{
				vbx(VVW, VADD, final_sum, final_sum, row_acc[i]);
			}
			// Rectification
	        vbx(VVW, VMOV, temp_acc, final_sum, 0);
    	    vbx(SVW, VCMV_LTZ, temp_acc, 0, final_sum);		// now temp_acc contains only +ve numbers

			// Lets assume that the filter coefficients will add up to 1
			// Thus the sum will not overflow even after accumulation
			// convert to 8 bit . Here we consider 8 MSBs of 16 bit number excluding sign bit(here it is 0 since we did rectification)
			vbx(SVW, VSHR, temp_acc, 7, temp_acc);
			if ((row % 2) == 0) 
			{
				vbx(VVWB, VMOV, filt_out_even[map], temp_acc, 0);
			}
			else
			{
				vbx(VVWB, VMOV, filt_out_odd[map], temp_acc, 0);
			}
		}
		// max pool once we have 2 filtered rows
		if ((row % 2) != 0)
		{
			for (map = 0; map < max_maps; map++)
			{
				// max pool
		        // copy 1 row to max_row
        		vbx(VVB, VMOV, pool_out[map], filt_out_even[map], 0);
        		vbx(VVB, VSUB, v_temp, filt_out_even[map], filt_out_odd[map]);
        		vbx(VVB, VCMV_LTZ, pool_out[map], filt_out_odd[map], v_temp);

        		vbx(VVB, VSUB, v_temp, pool_out[map], pool_out[map] + 1);
        		vbx(VVB, VCMV_LTZ, pool_out[map], pool_out[map]+1, v_temp);

        		//  now throw away alternate elements
        		vbx(VVHB, VMOV, pool_out[map], (uint16_t*)pool_out[map], 0);
				vbx_dma_to_host(out_img[map]+((max_col/2) * (int)(row/2)), pool_out[map], ((max_col- k_cols + 1)/2) * sizeof(uint8_t));
			}
		}

        // Wrap the pointers
        row_temp = in_rows[0];
		for ( i = 0; i < k_rows-1; i++)
		{
			in_rows[i] = in_rows[i+1];
		}
		in_rows[k_rows-1] = row_temp;
	}
    vbx_sync();
    vbx_sp_free();
    return 0;
}


int v_teradeep_layer_generic(uint8_t *in_img, int8_t *kernel, uint8_t *out_img, int max_row, int max_col, int k_rows, int k_cols)
{
	
	int MAX_K_COLS	= 9;									// max kernel size is 9x9
	int MAX_K_ROWS	= 9;
	int MAX_IMG_COLS	= 140;								// MAX no of columns in the image
	// Buffers for holding 3 rows of image
    uint8_t *in_rows[MAX_K_COLS],  *row_temp;				// pointers to input rows
	uint8_t *filt_out_even, *filt_out_odd;// buffers to hold 2 filtered rows which are inputs to max-pooling
	uint8_t *pool_out;								// final pooled row of a map.

    // Buffers for holding product of rows with kernel elements
    int16_t *par_prod[MAX_K_ROWS*MAX_K_COLS];

    // Buffers for holding filtered rows and final filtered row
    int32_t *row_acc[MAX_K_ROWS],  *final_sum, *temp_acc;
	int8_t *v_temp;
    int row, i, j, map;

	// Flush data cache
    vbx_dcache_flush_all();
//----------------------------------------------------------------------------------------------------------
	// input buffer and row accumulators allocation
	for (row = 0; row < k_rows; row++)
	{
		// allocate scratchpad for MAX_K_ROWS number of input rows
		if ( (NULL == (in_rows[row] = (uint8_t *)vbx_sp_malloc(max_col*sizeof(uint8_t)))) ||
		// allocate buffers for row accumulation
			 (NULL == (row_acc[row] = (int32_t *)vbx_sp_malloc((max_col + k_cols)*sizeof(int32_t)))))
		{
			printf("No scratchpad... exiting\n");
			return -1;
		}
	}
	// final result buffer for a row.
	if ((NULL == (final_sum = (int32_t *)vbx_sp_malloc((max_col + k_cols)*sizeof(int32_t)))) ||
		(NULL == (temp_acc = (int32_t *)vbx_sp_malloc((max_col + k_cols)*sizeof(int32_t)))) ||
		(NULL == (v_temp = (int8_t *)vbx_sp_malloc((max_col + k_cols)*sizeof(int8_t)))))
	{
		printf("No scratchpad... exiting\n");
		return -1;
	}

	// allcate buffers for partial products
	for (i = 0; i < k_rows*k_cols; i++)
	{
		if (NULL == (par_prod[i] = (int16_t*)vbx_sp_malloc((max_col + k_cols)*sizeof(int16_t))))
		{
			printf("No scrachpad available... exiting\n\n");
			return -1;
		}
	}
	// buffers for holding 2 filtered rows of all maps and also final pooled row of all maps
	
	{
		if ( (NULL == (filt_out_even = (uint8_t *)vbx_sp_malloc((max_col + k_cols)*sizeof(uint8_t))))||
			 (NULL == (filt_out_odd = (uint8_t *)vbx_sp_malloc((max_col + k_cols)*sizeof(uint8_t)))) ||
		     (NULL == (pool_out = (uint8_t *)vbx_sp_malloc((max_col + k_cols)*sizeof(uint8_t))))
		) {
			printf("No scratchpad... exiting\n");
			return -1;
		}
	}

//----------------------------------------------------------------------------------------------------------
	vbx_set_vl(max_col);
    // transfer (k_rows -1) rows to scratchpad
	for (i = 0; i < k_rows - 1; i++)
	{
		vbx_dma_to_vector(in_rows[i], in_img + max_col * i, max_col * sizeof(uint8_t));
		vbx(SVBU, VSHR, in_rows[i], 1, in_rows[i] ); // convert from 8 bit unsigned to 8 bit signed
	}

    // Main processing loop
	for (row = 0; row < (max_row - k_rows + 1); row++)
    {
		// load new row
       	vbx_dma_to_vector(in_rows[k_rows-1], in_img+max_col*(row+k_rows-1), max_col*sizeof(uint8_t));
		vbx(SVBU, VSHR, in_rows[k_rows-1], 1, in_rows[k_rows-1] );

		// perform filtering with all kernels
		
		{
			// reset accumulation buffers... any other better way to do this???
			for (i = 0; i < k_rows; i++)
			{
				vbx(SVW, VMUL, row_acc[i], 0, row_acc[i]);
			}
			// Multiplications required for filtering
			for (i = 0; i < k_rows; i++)
			{
				for (j = 0; j < k_cols; j++)
				{
					vbx(SVBH, VMUL, par_prod[i*k_cols+j], *(kernel + i*k_rows + j), in_rows[i]);
					vbx(SVH, VSHL, par_prod[i*k_cols+j], 1, par_prod[i*k_cols+j]); // remove 1 sign bit
				}
			}
			// Add weighted rows with sliding
			for ( i = 0; i < k_rows; i++)
			{
				for (j = 0; j < k_cols-1; j+=2)
				{
					vbx(VVHW, VADD, temp_acc, (par_prod[i*k_cols+j]+j), (par_prod[i*k_cols+j+1]+j+1));
					vbx(VVW, VADD, row_acc[i], row_acc[i], temp_acc);
				}
				if ( j == k_cols-1) // odd number of cols in kernel
				{
					vbx(VVHW, VMOV, temp_acc, par_prod[i*k_cols+j] + j, 0);
					vbx(VVW, VADD, row_acc[i], row_acc[i], temp_acc);
				}
			}

			// reset final sum
			vbx(SVW, VMUL, final_sum, 0, final_sum);
			// add all accumulated rows to get final filtered row
			for ( i = 0; i < k_rows; i++)
			{
				vbx(VVW, VADD, final_sum, final_sum, row_acc[i]);
			}
			// Rectification
	        vbx(VVW, VMOV, temp_acc, final_sum, 0);
    	    vbx(SVW, VCMV_LTZ, temp_acc, 0, final_sum);		// now temp_acc contains only +ve numbers

			// Lets assume that the filter coefficients will add up to 1
			// Thus the sum will not overflow even after accumulation
			// convert to 8 bit . Here we consider 8 MSBs of 16 bit number excluding sign bit(here it is 0 since we did rectification)
			vbx(SVW, VSHR, temp_acc, 7, temp_acc);
			if ((row % 2) == 0) 
			{
				vbx(VVWB, VMOV, filt_out_even, temp_acc, 0);
			}
			else
			{
				vbx(VVWB, VMOV, filt_out_odd, temp_acc, 0);
			}
		}
		// max pool once we have 2 filtered rows
		if ((row % 2) != 0)
		{
			
			{
				// max pool
		        // copy 1 row to max_row
        		vbx(VVB, VMOV, pool_out, filt_out_even, 0);
        		vbx(VVB, VSUB, v_temp, filt_out_even, filt_out_odd);
        		vbx(VVB, VCMV_LTZ, pool_out, filt_out_odd, v_temp);

        		vbx(VVB, VSUB, v_temp, pool_out, pool_out + 1);
        		vbx(VVB, VCMV_LTZ, pool_out, pool_out+1, v_temp);

        		//  now throw away alternate elements
        		vbx(VVHB, VMOV, pool_out, (uint16_t*)pool_out, 0);
				vbx_dma_to_host(out_img+((max_col/2) * (int)(row/2)), pool_out, ((max_col- k_cols + 1)/2) * sizeof(uint8_t));
			}
		}

        // Wrap the pointers
        row_temp = in_rows[0];
		for ( i = 0; i < k_rows-1; i++)
		{
			in_rows[i] = in_rows[i+1];
		}
		in_rows[k_rows-1] = row_temp;
	}
    vbx_sync();
    vbx_sp_free();
    return 0;
}
// 2D filter API for ARM
void s_filter2D(uint8_t *image, int8_t *kernel, uint8_t *dest_image, int M, int N, int k_rows, int k_cols)
{
    int row, col, kernel_row, kernel_col;
    int sop;
	uint8_t pixel;
	int16_t prod;

    for (row = 0; row < M-k_rows+1; row++)
    {
        for (col = 0; col < N-k_cols+1; col++)
        {
            sop = 0;
            for ( kernel_row = 0; kernel_row < k_rows; kernel_row ++)
            {
                for (kernel_col = 0; kernel_col < k_cols; kernel_col++)
                {
					pixel = ((*(image + N*row + col + N*kernel_row + kernel_col)))>>1;
                    prod = ((*(kernel + k_cols*kernel_row + kernel_col)) * pixel)<<1;
                    sop += prod;
                }
            }
            if (sop < 0)
			{
				sop = 0;
			}

            *(dest_image + N*row + col) = (uint8_t)(sop >> 7);
        }
    }   
}       

void s_max_pool(uint8_t *in_img, uint8_t *out_img, int max_rows, int max_cols)
{
	int row, col;
	uint8_t max, temp_max;
	int dest_rows = max_rows/2;
	int dest_cols = max_cols/2;

	for (row = 0; row < dest_rows; row++)
	{
		for(col = 0; col< dest_cols; col++)
		{
			temp_max =( *(in_img + 2*max_cols*row + 2*col) > *(in_img + 2*max_cols*row + 2*col+1))? *(in_img + 2*max_cols*row + 2*col): *(in_img + 2*max_cols*row + 2*col+1);
			max = temp_max;
			temp_max = (*(in_img + max_cols*(2*row+1) + 2*col) > *(in_img + max_cols*(2*row+1) + 2*col+1))? *(in_img + max_cols*(2*row+1) + 2*col): *(in_img + max_cols*(2*row+1) + 2*col+1);
			max = max > temp_max? max: temp_max;

			*(out_img + dest_cols*row + col) = max;
		}
	}
}
int s_teradeep_layer_generic(uint8_t *in_img, int8_t *kernel, uint8_t *out_img, int max_row, int max_col, int k_rows, int k_cols)
{
	uint8_t *filt_img = (uint8_t*)malloc(max_row*max_col*sizeof(uint8_t));
	s_filter2D(in_img, kernel, filt_img, max_row, max_col, k_rows, k_cols);
	s_max_pool(filt_img, out_img, max_row, max_col);
	free(filt_img);
}

int s_teradeep_layer1(uint8_t *in_img, int8_t **kernel, uint8_t **out_img, int max_row, int max_col, int k_rows, int k_cols, int max_maps)
{
	int map;
	for (map = 0; map < max_maps; map++)
	{
		s_teradeep_layer_generic(in_img, kernel[map], out_img[map], max_row, max_col, k_rows, k_cols);
	}
}

int v_filter2D(uint8_t *in_img, int8_t *kernel, uint8_t *out_img, int max_row, int max_col, int k_rows, int k_cols)
{
	
	int MAX_K_COLS	= 9;									// max kernel size is 9x9
	int MAX_K_ROWS	= 9;
	int MAX_IMG_COLS	= 140;								// MAX no of columns in the image
	// Buffers for holding 3 rows of image
    uint8_t *in_rows[MAX_K_COLS],  *row_temp;				// pointers to input rows

    // Buffers for holding product of rows with kernel elements
    int16_t *par_prod[MAX_K_ROWS*MAX_K_COLS];

    // Buffers for holding filtered rows and final filtered row
    int32_t *row_acc[MAX_K_ROWS],  *final_sum, *temp_acc;
	uint8_t *filt_out, row_bot;
    int row, i, j, map;

	// Flush data cache
    vbx_dcache_flush_all();
//----------------------------------------------------------------------------------------------------------
	// input buffer and row accumulators allocation
	for (row = 0; row < k_rows; row++)
	{
		// allocate scratchpad for MAX_K_ROWS number of input rows
		if ( (NULL == (in_rows[row] = (uint8_t *)vbx_sp_malloc(max_col*sizeof(uint8_t)))) ||
		// allocate buffers for row accumulation
			 (NULL == (row_acc[row] = (int32_t *)vbx_sp_malloc((max_col )*sizeof(int32_t)))))
		{
			printf("No scratchpad... exiting\n");
			return -1;
		}
	}
	// final result buffer for a row.
	if ((NULL == (final_sum = (int32_t *)vbx_sp_malloc((max_col )*sizeof(int32_t)))) ||
		(NULL == (temp_acc = (int32_t *)vbx_sp_malloc((max_col )*sizeof(int32_t)))) ||
		(NULL == (row_bot = (uint8_t *)vbx_sp_malloc((max_col )*sizeof(uint8_t)))) ||
		(NULL == (filt_out = (uint8_t *)vbx_sp_malloc((max_col )*sizeof(uint8_t)))))
	{
		printf("No scratchpad... exiting\n");
		return -1;
	}

	// allcate buffers for partial products
	for (i = 0; i < k_rows*k_cols; i++)
	{
		if (NULL == (par_prod[i] = (int16_t*)vbx_sp_malloc((max_col)*sizeof(int16_t))))
		{
			printf("No scrachpad available... exiting\n\n");
			return -1;
		}
	}

//----------------------------------------------------------------------------------------------------------
	vbx_set_vl(max_col);
    // transfer (k_rows -1) rows to scratchpad
	for (i = 0; i < k_rows; i++)
	{
		vbx_dma_to_vector(in_rows[i], in_img + max_col * i, max_col * sizeof(uint8_t));
		vbx(SVBU, VSHR, in_rows[i], 1, in_rows[i] ); // convert from 8 bit unsigned to 8 bit signed
	}

    // Main processing loop
	for (row = 0; row < (max_row - k_rows + 1); row++)
    {
            if (row < max_row - k_rows)
            {
                vbx_dma_to_vector(row_bot, in_img+max_col*(row+k_rows), max_col*sizeof(uint8_t));
				vbx(SVBU, VSHR, row_bot, 1, row_bot );
            }
		// load new row
//       	vbx_dma_to_vector(in_rows[k_rows-1], in_img+max_col*(row+k_rows-1), max_col*sizeof(uint8_t));
		
			// reset accumulation buffers... any other better way to do this???
			for (i = 0; i < k_rows; i++)
			{
				vbx(SVW, VMUL, row_acc[i], 0, row_acc[i]);
			}
			// Multiplications required for filtering
			for (i = 0; i < k_rows; i++)
			{
				for (j = 0; j < k_cols; j++)
				{
					vbx(SVBH, VMUL, par_prod[i*k_cols+j], *(kernel + i*k_rows + j), in_rows[i]);
					vbx(SVH, VSHL, par_prod[i*k_cols+j], 1, par_prod[i*k_cols+j]); // remove 1 sign bit
				}
			}
			// Add weighted rows with sliding
			for ( i = 0; i < k_rows; i++)
			{
				for (j = 0; j < k_cols-1; j+=2)
				{
					vbx(VVHW, VADD, temp_acc, (par_prod[i*k_cols+j]+j), (par_prod[i*k_cols+j+1]+j+1));
					vbx(VVW, VADD, row_acc[i], row_acc[i], temp_acc);
				}
				if ( j == k_cols-1) // odd number of cols in kernel
				{
					vbx(VVHW, VMOV, temp_acc, par_prod[i*k_cols+j] + j, 0);
					vbx(VVW, VADD, row_acc[i], row_acc[i], temp_acc);
				}
			}

			// reset final sum
			vbx(SVW, VMUL, final_sum, 0, final_sum);
			// add all accumulated rows to get final filtered row
			for ( i = 0; i < k_rows; i++)
			{
				vbx(VVW, VADD, final_sum, final_sum, row_acc[i]);
			}
			vbx(SVWB, VSHR ,filt_out, 7, final_sum);
			vbx_dma_to_host((out_img + row*max_col), filt_out, (max_col-k_cols+1)*sizeof(uint8_t));
        // Wrap the pointers
        row_temp = in_rows[0];
		for ( i = 0; i < k_rows-1; i++)
		{
			in_rows[i] = in_rows[i+1];
		}
		in_rows[k_rows-1] = row_bot;
		row_bot = row_temp;
	}
    vbx_sync();
    vbx_sp_free();
    return 0;
}
