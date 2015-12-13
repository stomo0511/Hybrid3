/*
 * tile_kernel.c
 *
 *  Created on: 2015/11/30
 *      Author: stomo
 */

#include "tile_kernel.h"

// タイル専用のQR分解

/*tile_init
	行列サイズを入力すると、作業用配列のメモリを確保する。
	char type		's' でfloat、'd'でdoubleのメモリを確保する。
	int tile_row		タイルの行数
	int tile_col		タイルの列数
	現時点では、double型しか対応しない。
*/
void* tile_init(char type,int tile_row,int tile_col)
{
	void * wb;
	wb = calloc(tile_row*tile_col*3,sizeof(double));

	return wb;
}

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
)
{
	int i;

	for(i = 0; i < col-block; i+=block)
	{
		// QR分解
		tile_sub_dgeqr2(row-i,block,A+i*lda+i,lda,TAU+i,Y+i*lda+i,work);
		// TAUからT行列を作る
		tile_sub_dlarft(row-i,block,Y+i*lda+i,lda,TAU+i,T+i*ldt+i,ldt,work);
		// 全体にQR分解の変換を適用
		tile_sub_dlarfb(row-i,block,col-i-block,Y+i*lda+i,lda,T+i*ldt+i,ldt,A+(i+block)*lda+i,lda,work);
	}
	// 最後の部分のQR分解
	tile_sub_dgeqr2(row-i,col-i,A+i*lda+i,lda,TAU+i,Y+i*lda+i,work);

	tile_sub_dlarft(row-i,col-i,Y+i*lda+i,lda,TAU+i,T+i*ldt+i,ldt,work);
}

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
)
{
	int i;

	// ブロック毎ループ
	for(i = 0; i < col-block; i+=block)
	{
		tile_sub_dtsqr2(row,block,A1+i+i*lda1,lda1,A2+i*lda2,lda2,TAU+i,work);
		// TAUからT行列を作る
		tile_sub_dlarft(row,block,A2+i*lda2,lda2,TAU+i,T+i*ldt+i,ldt,work);
		// リフレクター更新
		tile_sub_dssrfb(
			block,row,block,col-i-block,A2+i*lda2,lda2,T+i+i*ldt,ldt
			,A1+i+(i+block)*lda1,lda1,A2+(i+block)*lda2,lda2
			,work
		);
	}
	if(col%block > 0)
	{
		i = block*((int)col/block);
		tile_sub_dtsqr2(row,col%block,A1+i+i*lda1,lda1,
			A2+i*lda2,lda2,TAU+i,work);
		tile_sub_dlarft(row,col%block,A2+i*lda2,lda2,TAU+i,T+i*ldt+i,ldt,work);
	}
	else
	{
		i = col-block;
		tile_sub_dtsqr2(row,block,A1+i+i*lda1,lda1,
			A2+i*lda2,lda2,TAU+i,work);
		tile_sub_dlarft(row,block,A2+i*lda2,lda2,TAU+i,T+i*ldt+i,ldt,work);
	}
}

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
)
{
	tile_sub_dlarfb(row,col,col,Y,lda,T,ldt,A,lda,work);
}

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
)
{

	tile_sub_dssrfb(row,row,col,col,Y,lda,T,ldt,A1,lda,A2,lda,work);
}

void tile_sub_dgeqr2
(
	const int row,
	const int col,
	double * A,
	const int lda,
	double * TAU,
	double * Y,
	double * work
)
{
	int max_loop;
	int i,j,k;
	double a_head,norm,norm_1,b,y_head,t;
	double * yA = work;

	for(i = 0; i < col-1;i++)
	{
		// yを作りながらyAを計算
		cblas_dgemm
		(
			CblasColMajor,CblasTrans,CblasNoTrans,
			1,col-i,row-i-1,1.0,
			A+i+lda*i+1,lda,A+i+lda*i+1,lda,0.0,yA,1
		);
		a_head = A[i+lda*i];
		norm_1 = yA[0];
		norm = norm_1+a_head*a_head;
		b = sqrt(norm);
		y_head = a_head-b;
		A[i+lda*i] = y_head;
		t = 2/(y_head*y_head+norm_1);
		// yAの最上行の計算
		cblas_daxpy
		(
			col-i-1,y_head,A+i+lda*(i+1),
			lda,yA+1,1
		);
		// A-tyyAの計算
		cblas_dgemm
		(
			CblasColMajor,CblasNoTrans,CblasNoTrans,
			row-i,col-i-1,1,-t,
			A+i+lda*i,lda,yA+1,1,1.0,A+i+lda*(i+1),lda
		);
		A[i+lda*i] = b;
		cblas_dscal(row-i-1,1/y_head,A+i+lda*i+1,1);
		cblas_dcopy(row-i-1,A+i+lda*i+1,1,Y+i+lda*i+1,1);
		Y[i+lda*i] = 1;
		TAU[i] = t*y_head*y_head;
	}
	// 最終
	if(row > col)
	{
		i = col-1;
		// yを作る
//		norm_1 = cblas_ddot(row-col,A+i+i*lda+1,1,A+i+i*lda+1,1);
		cblas_dgemm
		(
			CblasColMajor,CblasTrans,CblasNoTrans,
			1,col-i,row-i-1,1.0,
			A+i+lda*i+1,lda,A+i+lda*i+1,lda,0.0,yA,1
		);
		a_head = A[i+lda*i];
		norm_1 = yA[0];
		norm = norm_1+a_head*a_head;
		b = sqrt(norm);
		y_head = a_head-b;
		A[i+lda*i] = b;
		cblas_dscal(row-i-1,1/y_head,A+i+lda*i+1,1);
		cblas_dcopy(row-i-1,A+i+lda*i+1,1,Y+i+lda*i+1,1);
		Y[i+lda*i] = 1;
		t = 2/(y_head*y_head+norm_1);
//		TAU[i] = t;
		TAU[i] = t*y_head*y_head;
	}
	else if(row==col)
	{
		Y[i+lda*i] = 0;
	}
}

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
)
{
	int i;
//	int j;
	double * YtY = work;
	// YtYを作る
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,
		col,col,row,1.0,Y,lda,Y,lda,0.0,YtY,col);
/*	printf("YtY\n");
	for(i = 0; i < col; i++)
	{
		for(j = 0; j < col;j++)
		{
			printf("%lf\t",YtY[i+j*col]);
		}
	}
*/
	cblas_dcopy(col,TAU,1,T,ldt+1);
	// T1|YtY|T0
	for(i = 1; i < col; i++){
/*		printf("T\n%lf\nYtY\n",T[i+i*ldt]);
		for(j = 0;j < i; j++)
		{
			printf("%lf\t",YtY[i+j*col]);
		}
		printf("\n");
*/
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
			i,1,i,-T[i+i*ldt],T,ldt,YtY+i,col,
			0.0,T+i,ldt
		);
	}
}

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
)
{
	int i;
	double * dmat_1 = work;
	int ld1 = row;
//	double * TYA = work+Y_col*A_col;
//	double * YtA = work+Y_col*A_col*2;
	// YtA
	for(i = 0; i < A_col; i++)
		cblas_dcopy(row,A+i*lda,1,dmat_1+i*ld1,1);

	cblas_dtrmm(CblasColMajor,CblasLeft,CblasLower
		,CblasTrans,CblasNonUnit,row,A_col
		,1.0,Y,ldy,dmat_1,ld1);

	// TYA
	cblas_dtrmm(CblasColMajor,CblasLeft,CblasLower
		,CblasNoTrans,CblasNonUnit,Y_col,A_col
		,1.0,T,ldt,dmat_1,ld1);

	// A-YTYA
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans
		,row,A_col,Y_col,-1.0,Y,ldy,dmat_1,ld1,1.0,A,lda);
/*
	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans
		,Y_col,A_col,row,1.0,Y,ldy,A,lda,0.0,YtA,Y_col);

	// TYA
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans
		,Y_col,A_col,Y_col,1.0,T,ldt,YtA,Y_col,0.0,TYA,Y_col);

	// A-YTYA
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans
		,row,A_col,Y_col,-1.0,Y,ldy,TYA,Y_col,1.0,A,lda);
*/
}

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
)
{
	int i;
	double * dmat_1;
	double * dmat_2;
	int ld1;
	int out_i,out_j;
	dmat_1 = work;
	ld1 = over_row;

	// A1+YtA2の計算
	for(i = 0; i < A_col;i++)
		cblas_dcopy(over_row,A1+lda1*i,1,dmat_1+ld1*i,1);

	cblas_dgemm
	(
		CblasColMajor,CblasTrans,CblasNoTrans,
		Y_col,A_col,under_row,1.0,Y,ldy,A2,lda2,1.0,dmat_1,ld1
	);

	// T(A1+YtA2)の計算
	cblas_dtrmm(CblasColMajor,CblasLeft,CblasLower
		,CblasNoTrans,CblasNonUnit,Y_col,A_col
		,1.0,T,ldt,dmat_1,ld1);

	// A1-T(A1+YtA2)の計算
	for(i = 0; i < A_col; i++)
		cblas_daxpy(over_row,-1.0,dmat_1+ld1*i,1,A1+lda1*i,1);

	// A2-YT(A1+YtA2)の計算
	cblas_dgemm
	(
		CblasColMajor,CblasNoTrans,CblasNoTrans,
		under_row,A_col,Y_col,-1.0,Y,ldy,dmat_1,ld1,1.0,A2,lda2
	);
}

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
)
{
	int i,j;
	int q,l;
	int ldyty = (col/block)*block + block;
	double * YtY = work;
	double * T1YtY = work+row*col;

//	cblas_dcopy(col,TAU,1,T,ldt+1);
		// YtYを作る
/*	for(i = 0; i < col; i+=block)
	{
		for(j = i+block; j < row; j+=block)
		{
			cblas_dgemm(
				CblasRowMajor,CblasNoTrans,CblasTrans,
				block,block,row,1.0,Y+i*lda,lda,Y+j*lda,lda,
				0.0,YtY+j+i*row,row
			);
		}
	}
*/
	if(block < 2)
	{
		tile_sub_dlarft(row,col,Y,lda,	TAU,T,ldt,work);
	}
	else if( block < col )
	{
//		printf("0\n");
		cblas_dgemm(
			CblasRowMajor,CblasNoTrans,CblasTrans,
			col,col,row,1.0,Y,lda,Y,lda,0.0,YtY,ldyty
		);
//		printf("1\n");

			// T1|YtY|T2
		for(i = block;i < col-block;i+=block)
		{
//			printf("2\n");
			// T1 YtY
			cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
				i,block,i,1.0,T,ldt,YtY+i,ldyty,0.0,T1YtY,block);
//			printf("3\n");
			// T1YtY T2
			cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
				i,block,block,-1.0,T1YtY,block,T+i+i*ldt,ldt
				,0.0,T+i,ldt);
		}
		// T1|YtY|T0
		for(i = (col/block)*block;i < col; i++)
		{
//			printf("4\n");
			cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
				i,1,i,-T[i+i*ldt],T,ldt,YtY+i,ldyty,
				0.0,T+i,ldt
			);
		}
	}
}

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
)
{
	int i;
	double * dmat_1;
	int ld1;
	int out_i,out_j;
	dmat_1 = upper_mat;
	ld1 = row;
	// A1+YtA2の計算
	for(i = 0; i < col;i++)
		cblas_dcopy(row,A1+lda*i,1,dmat_1+ld1*i,1);

	cblas_dgemm
	(
		CblasColMajor,CblasTrans,CblasNoTrans,
		col,col,row,1.0,Y,lda,A2,lda,1.0,dmat_1,ld1
	);

	// T(A1+YtA2)の計算
	cblas_dtrmm(CblasColMajor,CblasLeft,CblasLower
		,CblasNoTrans,CblasNonUnit,col,col
		,1.0,T,ldt,dmat_1,ld1);

	// A1-T(A1+YtA2)の計算
	for(i = 0; i < col; i++)
		cblas_daxpy(row,-1.0,dmat_1+ld1*i,1,A1+lda*i,1);
}

void tile_dssrfb_tail
(
	const int row,
	const int col,
	double * A2,
	const int lda,
	const double * Y,
	double * upper_mat
)
{
	int i;
	double * dmat_1;
	int ld1;
	dmat_1 = upper_mat;
	ld1 = row;
	// A2-YT(A1+YtA2)の計算
	cblas_dgemm
	(
		CblasColMajor,CblasNoTrans,CblasNoTrans,
		row,col,col,-1.0,Y,lda,dmat_1,ld1,1.0,A2,lda
	);
}

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
)
{
	int i;
	int j;
	double * dmat_1 = work;
	int ld1 = row;
	double zero = 0;
	cblas_dcopy(row*col,&zero,0,work,1);
	// YtA
	for(i = 0; i < col; i++)
		cblas_dcopy(row,A+i*lda,1,dmat_1+i*ld1,1);
/*
	printf("A\n");
	for(i = 0; i < col; i++)
	{
		for(j = 0; j < row;j++)
		{
			printf("%lf\t",A[j*col+i]);
		}
		printf("\n");
	}
	printf("\n");
	printf("Y\n");
	for(i = 0; i < col; i++)
	{
		for(j = 0; j < row;j++)
		{
			printf("%lf\t",Y[j*col+i]);
		}
		printf("\n");
	}
	printf("\n");
*/
	cblas_dtrmm(CblasColMajor,CblasLeft,CblasLower
		,CblasTrans,CblasNonUnit,row,col
		,1.0,Y,lda,dmat_1,ld1);
/*
	printf("YtA\n");
	for(i = 0; i < col; i++)
	{
		for(j = 0; j < row;j++)
		{
			printf("%lf\t",dmat_1[j*col+i]);
		}
		printf("\n");
	}
	printf("\n");

	printf("T\n");
	for(i = 0; i < col; i++)
	{
		for(j = 0; j < row;j++)
		{
			printf("%lf\t",T[j*col+i]);
		}
		printf("\n");
	}
	printf("\n");
*/
	// TYA
	cblas_dtrmm(CblasRowMajor,CblasRight,CblasUpper
		,CblasTrans,CblasNonUnit,col,col
		,1.0,T,ldt,dmat_1,ld1);
/*
	printf("TYA\n");
	for(i = 0; i < col; i++)
	{
		for(j = 0; j < row;j++)
		{
			printf("%lf\t",dmat_1[j*col+i]);
		}
		printf("\n");
	}
	printf("\n");
*/
	// A-YTYA
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans
		,row,col,col,-1.0,Y,lda,dmat_1,ld1,1.0,A,lda);
/*
	printf("A-YTYA\n");
	for(i = 0; i < col; i++)
	{
		for(j = 0; j < row;j++)
		{
			printf("%lf\t",A[j*col+i]);
		}
		printf("\n");
	}
	printf("\n");
*/
}

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
)
{
	int i;
	double * dmat_1;
	double * dmat_2;
	int ld1;
	int out_i,out_j;
	dmat_1 = work;
	ld1 = row;

	// A1+YtA2の計算
	for(i = 0; i < col;i++)
		cblas_dcopy(row,A1+lda*i,1,dmat_1+ld1*i,1);

	cblas_dgemm
	(
		CblasColMajor,CblasTrans,CblasNoTrans,
		col,col,row,1.0,Y,lda,A2,lda,1.0,dmat_1,ld1
	);

	// T(A1+YtA2)の計算
	cblas_dtrmm(CblasRowMajor,CblasRight,CblasUpper
		,CblasTrans,CblasNonUnit,col,col
		,1.0,T,ldt,dmat_1,ld1);

	// A1-T(A1+YtA2)の計算
	for(i = 0; i < col; i++)
		cblas_daxpy(row,-1.0,dmat_1+ld1*i,1,A1+lda*i,1);

	// A2-YT(A1+YtA2)の計算
	cblas_dgemm
	(
		CblasColMajor,CblasNoTrans,CblasNoTrans,
		row,col,col,-1.0,Y,lda,dmat_1,ld1,1.0,A2,lda
	);

}

// dtsqr2
// シーケンシャルな上三角行列＋通常行列のQR分解
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
)
{
	int i;
	double * dvec_1 = work;
	double t,norm_1,norm,a_head,y_head,b;
	// dtsqr2
	for(i = 0;i < col-1; i++)
	{
		cblas_dgemm
		(
			CblasColMajor,CblasTrans,CblasNoTrans,
			1,col-i,row,1.0
			,A2+i*lda2,lda2
			,A2+i*lda2,lda2,0.0,dvec_1,1
		);
		a_head = A1[i+i*lda1];
		norm_1 = dvec_1[0];
		norm = norm_1+a_head*a_head;
		b = sqrt(norm);
		y_head = a_head-b;
		t = 2/(y_head*y_head+norm_1);
		cblas_daxpy(col-i-1,y_head,A1+i+(i+1)*lda1,lda1,dvec_1+1,1);
		// A-tyyAの計算
		// A1-tyyAの計算
		cblas_daxpy(col-i-1,-t*y_head,dvec_1+1,1
			,A1+i+(i+1)*lda1,lda1);
		// A2-tyyAの計算
		cblas_dger(CblasColMajor,row,col-i-1,-t,A2+i*lda2,1,dvec_1+1,1,A2+(i+1)*lda2,lda2);
//		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans
//			,1,col-i-1,1,-t,A2+i*lda2,lda2
//			,dvec_1+1,1,1.0,A2+(i+1)*lda2,lda2);

		A1[i+i*lda1] = b;
		cblas_dscal(row,1/y_head,A2+i*lda2,1);
		TAU[i] = t*y_head*y_head;
	}
	cblas_dgemm
	(
		CblasColMajor,CblasTrans,CblasNoTrans,
		1,col-i,row,1.0
		,A2+i*lda2,lda2
		,A2+i*lda2,lda2,0.0,dvec_1,1
	);
	a_head = A1[i+i*lda1];
	norm_1 = dvec_1[0];
	norm = norm_1+a_head*a_head;
	b = sqrt(norm);
	y_head = a_head-b;
	t = 2/(y_head*y_head+norm_1);
	A1[i+i*lda1] = b;
	cblas_dscal(row,1/y_head,A2+i*lda2,1);
	TAU[i] = t*y_head*y_head;
}
