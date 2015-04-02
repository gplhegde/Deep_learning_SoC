#include "map_add.h"
#include "vbx.h"
#include "vbx_types.h"

#define DOUBLE_BUFFERING
//#define MXP
//----------------------------------------------------------------------------
// Add two grayscale images and store the result in img_1
// input images must be of same size( = M*N ). Size checking has to be put as a wrapper to this API
static void v_img_add_weighted(uint8_t *img_1, uint8_t *img_2, uint8_t *res_img ,int M, int N, int scale)
{
    int row;
    uint8_t *img1_row_0 = vbx_sp_malloc(N);
    uint8_t *img2_row_0 = vbx_sp_malloc(N);
#ifdef DOUBLE_BUFFERING
    uint8_t *img1_row_1 = vbx_sp_malloc(N);
    uint8_t *img2_row_1 = vbx_sp_malloc(N);
#endif
    uint16_t *sum_0 = (uint16_t *)vbx_sp_malloc(N* sizeof(uint16_t));
    uint8_t *v_tmp;
    vbx_dcache_flush_all();
    vbx_set_vl(N);
#ifdef DOUBLE_BUFFERING
    vbx_dma_to_vector(img1_row_0, img_1, N);
    vbx_dma_to_vector(img2_row_0, img_2, N);
#endif
    for (row = 0; row < M*N; row+=N)
    {
//      vbx_dcache_flush_all();
        // transfer 1 row of both images to scratch pad 
#ifdef DOUBLE_BUFFERING
        if(row < M*N-N)
        {
            vbx_dma_to_vector(img1_row_1, img_1 + row+N, N);
            vbx_dma_to_vector(img2_row_1, img_2 + row+N, N);
        }
#else
        vbx_dma_to_vector(img1_row_0, img_1 + row, N);
        vbx_dma_to_vector(img2_row_0, img_2 + row, N);
#endif
        vbx(VVBHU, VADD, sum_0 , img1_row_0, img2_row_0);
        vbx(SVHBU, VSHR, img1_row_0, scale, sum_0);
        vbx_dma_to_host(res_img + row, img1_row_0, N);
#ifdef DOUBLE_BUFFERING
        v_tmp = img1_row_0; img1_row_0 = img1_row_1; img1_row_1  = v_tmp;
        v_tmp = img2_row_0; img2_row_0 = img2_row_1; img2_row_1  = v_tmp;
#endif

    }
//  vbx_sync();
    vbx_sp_free();
}

static void s_img_add_weighted(uint8_t *image1, uint8_t *image2, uint8_t *res_image, int M, int N, int scale)
{
        int row, col;
        for (row = 0; row < M; row++)
        {
                for(col = 0; col < N; col++)
                {
                        *(res_image + N*row +col) = (*(image1 +N* row+ col) + *(image2+row*N+col)) >> scale;
                }
        }

}

void img_add(uint8_t *img_1, uint8_t  *img_2, uint8_t *img_out, int max_rows, int max_cols)
{
    LOG("%s: Entry\n", __func__);
//#ifdef MXP
  //  v_img_add_weighted(img_1, img_2, img_out ,max_rows, max_cols, 1);
//#else

    s_img_add_weighted(img_1, img_2, img_out ,max_rows, max_cols, 1);
//#endif // MXP

    LOG("%s: Exit\n", __func__);
}
