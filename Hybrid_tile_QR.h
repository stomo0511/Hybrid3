/*
 * Hybrid_tile_QR.h
 *
 *  Created on: 2015/11/30
 *      Author: stomo
 */

#ifndef HYBRID_TILE_QR_H_
#define HYBRID_TILE_QR_H_

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "mat_prod.h"
#include "tile_kernel.h"

void Hybrid_dgeqrf
(
		mat_stat_t *status,
		double *A, double *TAU,
		int my_rank, int Proc_num
);

void Hybrid_tile_QR(
		mat_stat_t *status,
		double *A, double *TAU,
		int my_rank, int Proc_num,int ib
);

void Hybrid_make_Q(
		mat_stat_t *status,
		double *Q,
		double *RY, double *TAU,
		int my_rank, int Proc_num,int ib
);

// 単位行列を作る
void Hybrid_make_I
(
		mat_stat_t *status,
		double *Id, // to avoid conflict with imaginary unit in C99
		int my_rank, int Proc_num
);

// Y要素を削除する
void Hybrid_remv_Y
(
		mat_stat_t *status,
		double *RY,
		int my_rank, int Proc_num
);

void Hybrid_dgemm(
		mat_stat_t *status,
		double *A, double *B, double *C,
		int my_rank, int Proc_num
);

// 直交行列Qの生成
void Hybrid_dorgqr
(
		mat_stat_t *status,
		double *Q,
		double *RY, double *TAU,
		int my_rank, int Proc_num
);

//　直交行列QをCにかける
void Hybrid_dormqr
(
		mat_stat_t *status,
		double *RY, double *TAU,
		double *C,
		int my_rank, int Proc_num
);

#endif /* HYBRID_TILE_QR_H_ */
