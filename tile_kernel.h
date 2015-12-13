/*
 * tile_kernel.h
 *
 *  Created on: 2015/11/30
 *      Author: stomo
 */

#ifndef TILE_KERNEL_H_
#define TILE_KERNEL_H_

#include<stdio.h>
#include<stdlib.h>
#include<cblas.h>
#include<math.h>

#define __KANEL_DEBUG__ 0

#define __DEBUG_OUT__ stdout

#define __DEBUG_HEAD__ "[DBG_KANEL]"

/*tile_init
	行列サイズを入力すると、作業用配列のメモリを確保する。
	char type		's' でfloat、'd'でdoubleのメモリを確保する。
	int tile_row		タイルの行数
	int tile_col		タイルの列数
*/
void* tile_init(char type,int tile_row,int tile_col);

// タイル専用のQR分解
void tile_dgeqrt
(
	const int row,
	const int col,
	const int block,
	double * A,
	const int lda,
	double * TAU,
	double * Y,
	double * T,
	const int ldt,
	double * work
);

// 上三角行列＋通常行列のQR分解
void tile_dtsqrt
(
	const int row,
	const int col,
	const int block,
	double * A1,
	const int lda1,
	double * A2,
	const int lda2,
	double * TAU,
	double * T,
	const int ldt,
	double * work
);
// tsqrtで出来たYとTを使って更新をする関数
void tile_dlarfb
(
	const int row,
	const int col,
	double * A,
	const int lda,
	const double * Y,
	const double * T,
	const int ldt,
	double * work
);
// tsqrtで出来たYとTを使って更新をする関数
void tile_dssrfb
(
	const int row,
	const int col,
	double * A1,
	double * A2,
	const int lda,
	const double * Y,
	const double * T,
	const int ldt,
	double * work
);

/********************************
	シーケンシャルな通常のQR分解
*********************************/
void tile_sub_dgeqr2
(
	const int row,
	const int col,
	double * A,
	const int lda,
	double * TAU,
	double * Y,
	double * work
);
/*********************
TAUを元にT行列を作る
*********************/
void tile_sub_dlarft
(
	const int row,
	const int col,
	const double * Y,
	const int lda,
	const double * TAU,
	double * T,
	const int ldt,
	double * work
);
/********************************
	dgeqr2による変換を適用する
********************************/
void tile_sub_dlarfb
(
	const int row,
	const int Y_col,
	const int A_col,
	const double * Y,
	const int ldy,
	const double * T,
	const int ldt,
	double * A,
	const int lda,
	double * work
);
/*********************************
	dtsqr2による変換を適用する
**********************************/
void tile_sub_dssrfb
(
	const int over_row,
	const int under_row,
	const int Y_col,
	const int A_col,
	const double * Y,
	const int ldy,
	const double * T,
	const int ldt,
	double * A1,
	const int lda1,
	double * A2,
	const int lda2,
	double * work
);
/**********************************
	TAUとT行列を使ってT行列を作る
**********************************/
void tile_dlarft
(
	const int row,
	const int col,
	const int block,
	const double * Y,
	const int lda,
	const double * TAU,
	double * T,
	const int ldt,
	double * work
);
void tile_dssrfb_head
(
	const int row,
	const int col,
	double * A1,
	double * A2,
	const int lda,
	const double * Y,
	const double * T,
	const int ldt,
	double * upper_mat
);
void tile_dssrfb_tail
(
	const int row,
	const int col,
	double * A2,
	const int lda,
	const double * Y,
	double * upper_mat
);

void tile_return_dlarfb
(
	const int row,
	const int col,
	double * A,
	const int lda,
	const double * Y,
	const double * T,
	const int ldt,
	double * work
);
void tile_return_dssrfb
(
	const int row,
	const int col,
	double * A1,
	double * A2,
	const int lda,
	const double * Y,
	const double * T,
	const int ldt,
	double * work
);
void tile_return_dlarft
(
	const int row,
	const int col,
	const double * Y,
	const int lda,
	const double * TAU,
	double * T,
	const int ldt,
	double * work
);

void tile_sub_dtsqr2
(
	const int row,
	const int col,
	double * A1,
	const int lda1,
	double * A2,
	const int lda2,
	double * TAU,
	double * work
);
#endif /* TILE_KERNEL_H_ */
