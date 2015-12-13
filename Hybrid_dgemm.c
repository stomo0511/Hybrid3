/*
 * Hybrid_dgemm.c
 *
 *  Created on: 2015/11/30
 *      Author: stomo
 */

#include "Hybrid_tile_QR.h"
#include <time.h>
#include <cblas.h>

#define M_Head(MATRIX,i,j) ((MATRIX)+(i)*tile_size+(j)*my_row*tile_col)
#define Comm_Buff(num) (comm_tile+tile_size*num)
#define Comm_sendrecv(num) (comm_tile)

void Hybrid_dgemm(
		mat_stat_t * status
		,double * A,double * B,double * C
		,int my_rank,int Proc_num
)
{
	const int tile_row = mat_getTileM(status);
	const int tile_col = mat_getTileN(status);
	const unsigned long int my_row = mat_getMyM(status);
	const unsigned long int tile_row_num = mat_getLTileM(status);
	const unsigned long int tile_col_num = mat_getLTileN(status);
	const unsigned long int tile_size = mat_TileSize(status);
	double * comm_tile;
	double * thread_tile_A;
	double * thread_tile_C;
	char buff[20];
	const double one = 1;
	const double zero = 0;
	// MPI通信用タイルデータ
	MPI_Datatype MPI_TILE;
	// MPI_TILEの設定
	MPI_Type_contiguous(tile_size,MPI_DOUBLE,&MPI_TILE);
	MPI_Type_commit(&MPI_TILE);
	comm_tile = (double*)calloc(tile_size*tile_row_num,sizeof(double));
	#pragma omp parallel
	{
		long int i,j,k;
		// 全体のループ
		for(i = 0; i < tile_row_num; i++)
		{
			// 自分の持っている行列で計算するとき
			#pragma omp single
			{
				if(my_rank == i%Proc_num)
					cblas_dcopy(tile_size*tile_row_num,M_Head(A,0,i/Proc_num),1,Comm_Buff(0),1);
				MPI_Bcast(Comm_Buff(0),tile_row_num,MPI_TILE,i%Proc_num,MPI_COMM_WORLD);
			}
			#pragma omp for
			for(j = 0; j < tile_col_num; j++)
			{
				cblas_dcopy(tile_size,&zero,0,M_Head(C,i,j),1);
				for(k = 0; k < tile_row_num; k++)
				{
					cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans
						,tile_row,tile_col,tile_row,1.0
						,Comm_Buff(k),tile_row,M_Head(B,k,j),tile_row
						,1.0,M_Head(C,i,j),tile_row
					);
				}
			}
		}
	}

	free(comm_tile);
}

