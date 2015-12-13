/*
 * Hybrid_make_Q.c
 *
 *  Created on: 2015/11/30
 *      Author: stomo
 */

#include "Hybrid_tile_QR.h"

#  include <lapacke.h>
#  include <plasma.h>
#  include <core_blas.h>

static int crayj_dtsrft(
  int M, int N, int IB,
  double const *V, int LDV,
  double const *TAU,
  double *T, int LDT
)
{
  // borrowed from CORE_dtsqrt() in PLASMA 2.5.1
  static double const zzero = 0.0;
  int i, ii;

  for (ii = 0; ii < N; ii += IB) {
    int sb = ((N-ii < IB) ? (N-ii) : IB);
    for (i = 0; i < sb; ++i) {
      double alpha = -TAU[ii+i];
      cblas_dgemv(
        CblasColMajor, (CBLAS_TRANSPOSE)PlasmaTrans, M, i,
        (alpha), &V[LDV*ii], LDV,
        &V[LDV*(ii+i)], 1,
        (zzero), &T[LDT*(ii+i)], 1);
      cblas_dtrmv(
        CblasColMajor, (CBLAS_UPLO)PlasmaUpper,
        (CBLAS_TRANSPOSE)PlasmaNoTrans, (CBLAS_DIAG)PlasmaNonUnit, i,
        &T[LDT*ii], LDT,
        &T[LDT*(ii+i)], 1);
      T[LDT*(ii+i)+i] = TAU[ii+i];
    }
  }

  return PLASMA_SUCCESS;
}


#define M_p(MATRIX,i,j) ((MATRIX)+((i)+(j)*tile_row_num)*tile_size)
#define V_p(VECTOR,i,j) ((VECTOR)+((i)+(j)*tile_row_num)*tile_col)

#define M_Head(MATRIX,i,j) ((MATRIX)+(i)*tile_size+(j)*my_row*tile_col)
#define Comm_Buff_Y(num) (comm_tile)
#define Comm_Buff_T(num) (comm_tile+tile_size)
#define Comm_sendrecv(num) (comm_tile)
#define TAG(dest,source,number) ((dest)*2+(source)*3+(number)*5)

void Hybrid_make_Q(
	mat_stat_t * status
	,double * Q
	,double * RY,double * TAU
//	,double * T
	,int my_rank,int Proc_num
	,int ib
)
{
/*******************************************
	グローバル変数定義
*******************************************/
	double * comm_tile;
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
	double * Y_buff = calloc(tile_size,sizeof(double));
	double * T_buff = calloc(tile_size,sizeof(double));

	// MPI通信用タイルデータ
	MPI_Datatype MPI_TILE;
	comm_tile = calloc(tile_size*2,sizeof(double));
	// MPI_TILEの設定
	MPI_Type_contiguous(tile_size*2,MPI_DOUBLE,&MPI_TILE);
	MPI_Type_commit(&MPI_TILE);

	#pragma omp parallel for
	for(i = 0; i < tile_col_num; i++){
		cblas_dcopy(tile_row,&one,0
			,Q+(i*tile_row_num+i*Proc_num+my_rank)*tile_row*tile_col,tile_col+1);
	}

	#pragma omp parallel
	{
		//カーネルの作業領域
		double * work;
		long int i = 0,j,c;
		long int k,g_i,l_j;
		int ii,jj;
		g_i = 0;
		l_j = 0;
		work = (double*)calloc(tile_row*tile_col*3,sizeof(double));
		for(g_i = 0; g_i < grob_tile_col; g_i++)
		{
			i = g_i%Proc_num;
			// 自分が送信するとき
			if(my_rank == i)
			{
				#pragma omp single
				{
					// Yの作成
					for(ii = 0; ii < tile_col-1; ii++)
					{
						cblas_dcopy(tile_row-ii-1,M_Head(RY,g_i,l_j)+ii*tile_row+ii+1,1
							,Y_buff+ii*tile_row+ii+1,1);
						Y_buff[ii*tile_row+ii] = 1;
					}
					Y_buff[tile_row*tile_col-1] = 0;

					LAPACKE_dlarft_work(LAPACK_COL_MAJOR,
						lapack_const(PlasmaForward),
						lapack_const(PlasmaColumnwise),
						tile_row, tile_col,
						Y_buff, tile_row, V_p(TAU,g_i,l_j),
						T_buff, tile_row
					);


					// 送信する
					cblas_dcopy(tile_size,Y_buff,1,Comm_Buff_Y(g_i),1);
					cblas_dcopy(tile_size,T_buff,1,Comm_Buff_T(g_i),1);
					MPI_Bcast(Comm_sendrecv(g_i),1,MPI_TILE,my_rank,MPI_COMM_WORLD);
				}
				// 最初の行のlarfb
				#pragma omp for
				for(j = 0; j < tile_col_num;j++)
				{
					CORE_dormqr(PlasmaLeft, PlasmaTrans,
						tile_row, tile_col, tile_row, tile_col,
						Y_buff, tile_row, T_buff, tile_row,
						M_Head(Q,g_i,j), tile_row, work, tile_row
					);
				}
				#pragma omp for
				for(c = 0; c < tile_col; c++)
					cblas_dcopy(tile_row-c-1,&zero,0,M_Head(RY,g_i,l_j)+tile_row*c+c+1,1);

				k = g_i+1;
				// 下のSSRFBの計算
				for(k = g_i+1; k < tile_row_num; k++)
				{
					#pragma omp single
					{
						// Tを作成する
						crayj_dtsrft(
							tile_row, tile_col, tile_col,
							M_Head(RY,k,l_j), tile_row, V_p(TAU,k,l_j),
							T_buff, tile_row
						);
						// 送信する
						cblas_dcopy(tile_size,M_Head(RY,k,l_j),1,Comm_Buff_Y(g_i),1);
						cblas_dcopy(tile_size,T_buff,1,Comm_Buff_T(g_i),1);
						MPI_Bcast(Comm_sendrecv(g_i),1,MPI_TILE,my_rank,MPI_COMM_WORLD);
					}
					#pragma omp for
					for(j = 0; j < tile_col_num; j++)
					{
						CORE_dtsmqr(PlasmaLeft, PlasmaTrans,
							tile_row, tile_col, tile_row, tile_col, tile_col, tile_col,
							M_Head(Q,g_i,j), tile_row, M_Head(Q,k,j), tile_row,
							M_Head(RY,k,l_j), tile_row, T_buff, tile_row,
							work, tile_row
						);
					}
					#pragma omp single
					{
						cblas_dcopy(tile_size,&zero,0,M_Head(RY,k,l_j),1);
					}
				}
				// qrtカウンタを回す
				l_j++;
			}
			// 自分以外のYTを使う場合
			else
			{
				#pragma omp single
				{
					// 行列の受信
					MPI_Bcast(Comm_sendrecv(g_i),1,MPI_TILE,i,MPI_COMM_WORLD);
				}
				// 最初の行のlarfb
				#pragma omp for
				for(j = 0; j < tile_col_num;j++)
				{
					// larfbをやる
					LAPACKE_dlarfb_work(
						LAPACK_COL_MAJOR,
						lapack_const(PlasmaLeft),
						lapack_const(PlasmaTrans),
						lapack_const(PlasmaForward),
						lapack_const(PlasmaColumnwise),
						tile_row, tile_col, tile_col,
						Comm_Buff_Y(g_i), tile_row, Comm_Buff_T(g_i), tile_row,
						M_Head(Q,g_i,j), tile_row, work, tile_row
					);
				}
				for(k = g_i+1; k < tile_row_num; k++)
				{
					#pragma omp single
					{
						// 行列の受信
						MPI_Bcast(Comm_sendrecv(g_i),1,MPI_TILE,i,MPI_COMM_WORLD);
					}
					#pragma omp for
					for(j = 0; j < tile_col_num; j++)
					{
						// SSRFBの計算
						CORE_dtsmqr(PlasmaLeft, PlasmaTrans,
							tile_row, tile_col, tile_row, tile_col, tile_col, tile_col,
							M_Head(Q,g_i,j), tile_row, M_Head(Q,k,j), tile_row,
							Comm_Buff_Y(k), tile_row, Comm_Buff_T(k), tile_row,
							work, tile_row
						);
					}
				}
			}
		}
		free(work);
	}
	free(comm_tile);
	free(T_buff);
	free(Y_buff);
}

