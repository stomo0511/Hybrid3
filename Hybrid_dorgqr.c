/*
 * Hybrid_dorgqr.c
 *
 *  Created on: 2015/11/30
 *      Author: stomo
 */

#include "Hybrid_tile_QR.h"

#include <cblas.h>

#define M_p(MATRIX,i,j) ((MATRIX)+((i)+(j)*tile_row_num)*tile_size)
#define V_p(VECTOR,i,j) ((VECTOR)+((i)+(j)*tile_row_num)*tile_col)

#define M_Head(MATRIX,i,j) ((MATRIX)+(i)*tile_size+(j)*my_row*tile_col)
#define Comm_Buff_Y(num) (comm_tile)
#define Comm_Buff_T(num) (comm_tile+tile_size)
#define Comm_sendrecv(num) (comm_tile)
#define TAG(dest,source,number) ((dest)*2+(source)*3+(number)*5)

// 直交行列Qの生成
void Hybrid_dorgqr
(
	mat_stat_t * status
	,double * Q
	,double * RY,double * TAU
	,int my_rank,int Proc_num
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

	Hybrid_make_I(status,Q,my_rank,Proc_num);

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
					tile_dlarft(tile_row,tile_col,1,Y_buff,tile_row,V_p(TAU,g_i,l_j),T_buff,tile_row,work);

					// 送信する
					cblas_dcopy(tile_size,Y_buff,1,Comm_Buff_Y(g_i),1);
					cblas_dcopy(tile_size,T_buff,1,Comm_Buff_T(g_i),1);
					MPI_Bcast(Comm_sendrecv(g_i),1,MPI_TILE,my_rank,MPI_COMM_WORLD);
				}
				// 最初の行のlarfb
				#pragma omp for
				for(j = 0; j < tile_col_num;j++)
				{
					// larfbをやる
					#if DEBUG
						printf("[%d]dlarfb %d,%d,%d\n",my_rank,g_i,j,l_j);
					#endif
//					mat_print_tile(stdout,M_Head(Q,g_i,j),"Q",tile_row,tile_col);
//					tile_return_dlarfb
					tile_dlarfb
					(
						tile_row,tile_col,M_Head(Q,g_i,j),tile_row
						,Y_buff
						,T_buff
						,tile_row,work
					);
//					mat_print_tile(stdout,M_Head(Q,g_i,j),"Q'",tile_row,tile_col);
				}

				k = g_i+1;
				// 下のSSRFBの計算
				for(k = g_i+1; k < tile_row_num; k++)
				{
					#pragma omp single
					{
						// Tを作成する
						tile_dlarft(tile_row,tile_col,1,M_Head(RY,k,l_j),tile_row,V_p(TAU,k,l_j),T_buff,tile_row,work);
						// 送信する
						cblas_dcopy(tile_size,M_Head(RY,k,l_j),1,Comm_Buff_Y(g_i),1);
						cblas_dcopy(tile_size,T_buff,1,Comm_Buff_T(g_i),1);
						MPI_Bcast(Comm_sendrecv(g_i),1,MPI_TILE,my_rank,MPI_COMM_WORLD);
					}
					#pragma omp for
					for(j = 0; j < tile_col_num; j++)
					{
						tile_dssrfb
						(
							tile_row,tile_col
							,M_Head(Q,g_i,j)
							,M_Head(Q,k,j)
							,tile_row
							,M_Head(RY,k,l_j)
							,T_buff
							,tile_row
							,work
						);
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
					tile_dlarfb
					(
						tile_row,tile_col,M_Head(Q,g_i,j),tile_row
						,Comm_Buff_Y(g_i),Comm_Buff_T(g_i),tile_row,work
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

						tile_dssrfb
						(
							tile_row,tile_col
							,M_Head(Q,g_i,j)
							,M_Head(Q,k,j)
							,tile_row
							,Comm_Buff_Y(k)
							,Comm_Buff_T(k)
							,tile_row
							,work
						);
					}
				}
			}
		}
		free(work);
	}
	free(comm_tile);
}

