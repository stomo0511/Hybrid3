/*
 * Hybrid_make_I.c
 *
 *  Created on: 2015/11/30
 *      Author: stomo
 */

#include "Hybrid_tile_QR.h"

#define M_p(MATRIX,i,j) ((MATRIX)+((i)+(j)*tile_row_num)*tile_size)
#define V_p(VECTOR,i,j) ((VECTOR)+((i)+(j)*tile_row_num)*tile_col)

#define M_Head(MATRIX,i,j) ((MATRIX)+(i)*tile_size+(j)*my_row*tile_col)
#define Comm_Buff_Y(num) (comm_tile)
#define Comm_Buff_T(num) (comm_tile+tile_size)
#define Comm_sendrecv(num) (comm_tile)
#define TAG(dest,source,number) ((dest)*2+(source)*3+(number)*5)

// 単位行列を作る
void Hybrid_make_I
(
	mat_stat_t * status
	,double * Id // to avoid conflict with imaginary unit in C99
	,int my_rank,int Proc_num
)
{
    double one = 1;
    double zero = 0;
    long int i;
    const unsigned int tile_row = mat_getTileM(status);
    const unsigned int tile_col = mat_getTileN(status);
    const unsigned long int my_row = mat_getMyM(status);
    const unsigned long int my_col = mat_getMyN(status);
    const unsigned long int tile_size = mat_TileSize(status);
    const unsigned long int tile_row_num = mat_getLTileM(status);
    const unsigned long int tile_col_num = mat_getLTileN(status);
    const unsigned long int grob_tile_col = mat_getGTileN(status);
    int thread_num;

    #pragma omp parallel
    {
        #pragma omp for
        for(i = 0; i < tile_col_num; i++)
        {
            cblas_dcopy(tile_size*tile_row_num,&zero,0,Id+tile_size*tile_row_num*i,1);
        }
        #pragma omp for
        for(i = 0; i < tile_col_num; i++){
            cblas_dcopy(tile_row,&one,0
                        ,Id+(i*tile_row_num+i*Proc_num+my_rank)*tile_row*tile_col,tile_col+1);
        }
    }
}

