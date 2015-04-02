#ifndef _MAP_ADD_H_
#define _MAP_ADD_H_
#include "vbx.h"
#include "vbx_types.h"
#include "vbx_port.h"
#ifdef DEBUG
#define LOG(ARGS...) printf(ARGS)
#else
#define LOG(ARGS...) 
#endif
//#define MXP
// Structure for holding image as 2D matrix
typedef struct
{
    uint8_t *img_data;
    int no_rows;
    int no_cols;
} Mat;

void img_add(uint8_t *img_1, uint8_t  *img_2, uint8_t *img_out, int max_rows, int max_cols);
#endif //_MAP_ADD_H_
