/*
 * main.c
 *
 *  Created on: 2015/11/30
 *      Author: stomo
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <cblas.h>

#include "Hybrid_tile_QR.h"

#define FLOD (4.0/3.0)

int main(int argc, char * argv[])
{
	long int matrix_size;
	long int tile_size;
	long int size;
	double error;
	double t1,t2,times;
	double flops;
	int i,j,k;
	int lock;
	int my_rank;
	int Proc_num;
	unsigned int seed;

	FILE *file_out;
	char filename[100];
	char findf[100];

	// 行列の定義
	double *A;
	double *TAU;
	double *O;
	double *Q;
	double *QR;

	// 変数の定義
	double zero = 0;
	mat_stat_t mat_stat;
	int ib;
	double elapsed_qr;

	// MPIの初期データを取得
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &Proc_num);

	// 行列のデータを取得
	if(argc < 3)
	{
		if(my_rank == 0)
		{
			fprintf(stderr,"This program need two arguments.\n");
		}
		MPI_Finalize();
		exit(EXIT_SUCCESS);
		return 0;
	}

	matrix_size = atoi(argv[1]);
	tile_size = atoi(argv[2]);

#ifdef DEBUG
	if(my_rank == 0)
		printf("matrix_size %d\ttile_size %d\n",matrix_size,tile_size);
#endif

	if (argc >= 4)
	{
		ib = atoi(argv[3]);
		if (ib <= 0)
		{
			fprintf(stderr, "CRAYJ:[%5d] ERROR: ib is invalid: ib == %d\n",
					my_rank, ib);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	else
	{
		ib = tile_size;
	}

#ifdef DEBUG
	printf("N=%d, NTILE=%d, IB=%d\n", matrix_size, tile_size,ib);
#endif

	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);

	size = matrix_size*matrix_size;

#ifdef DEBUG
	if(my_rank == 0)
		printf("total matrix size=%ld\n",size);
#endif
	size = size/Proc_num;
#ifdef DEBUG
	if(my_rank == 0)
		printf("process's matrix size=%ld\n",size);
#endif

	// mat_prodの初期化
	mat_init(&mat_stat,matrix_size,matrix_size,tile_size,tile_size,my_rank,Proc_num);

	// 行列の確保
	A = calloc(mat_mysize(&mat_stat),sizeof(double));
	TAU = calloc(mat_myTAU(&mat_stat),sizeof(double));
	QR = calloc(mat_mysize(&mat_stat),sizeof(double));
	O = calloc(mat_mysize(&mat_stat),sizeof(double));
	Q = calloc(mat_mysize(&mat_stat),sizeof(double));

	if(A == NULL)
		fprintf(stderr,"error matrix A\n");
	if(TAU == NULL)
		fprintf(stderr,"error matrix T\n");
	if(Q == NULL)
		fprintf(stderr,"error matrix Q\n");
	if(O == NULL)
		fprintf(stderr,"error matrix O\n");
	if(QR == NULL)
		fprintf(stderr,"error matrix R\n");

	// 行列の作成ランダム
#ifdef DEBUG
	if(my_rank == 0)
		printf("make matrix\n");
#endif

	// 行列のシードを決める
	seed = 20151130;
	mat_make(&mat_stat,A,seed,my_rank);

#ifdef DEBUG
	if(my_rank == 0)
		printf("run qr\n");
#endif

	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();

	Hybrid_tile_QR(&mat_stat,A,TAU,my_rank,Proc_num,ib);

	MPI_Barrier(MPI_COMM_WORLD);
	t2 = MPI_Wtime();

#ifdef DEBUG
	if(my_rank == 0)
		printf("end qr\n");
#endif

	elapsed_qr = t2 - t1;
	flops = (FLOD)*((double)matrix_size*(double)matrix_size*(double)matrix_size)/elapsed_qr;
	if(my_rank == 0)
	{
		printf("node= %d\tN= %d\tNB= %d\tIB= %d\ttime= %-.6f s\tflops= %lf Gflops\n",
				Proc_num,matrix_size,tile_size,ib,elapsed_qr,flops*1e-9);
		fflush(stdout);
	}

#ifdef DEBUG
	if(my_rank == 0)
		printf("run make q\n");
#endif

	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();

	Hybrid_make_Q(&mat_stat,Q,A,TAU,my_rank,Proc_num,ib);

	MPI_Barrier(MPI_COMM_WORLD);

	t2 = MPI_Wtime();

#ifdef DEBUG
	if(my_rank == 0)
		printf("end make q\n");
#endif

	if (my_rank == 0)
	{
		printf("CRAYJ: Elapsed time for Hybrid_make_Q() == %.6f sec\n", t2-t1);
		fflush(stdout);
	}
	t1 = MPI_Wtime();

	error = mat_comp3(&mat_stat,Q,my_rank,Proc_num);

	if (my_rank == 0)
	{
		printf("||Q||oo=\t%e\n", error);
		fflush(stdout);
	}

	cblas_dcopy(mat_mysize(&mat_stat),Q,1,O,1);

#ifdef DEBUG
	if(my_rank == 0)
		printf("run q'q\n");
#endif

	MPI_Barrier(MPI_COMM_WORLD);
	Hybrid_dgemm(&mat_stat,Q,O,QR,my_rank,Proc_num);
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef DEBUG
	if(my_rank == 0)
		printf("end q'q\n");
#endif

#ifdef DEBUG
	if(my_rank == 0)
		printf("run ||I-Q'Q||oo\n");
#endif

	Hybrid_make_I(&mat_stat,O,my_rank,Proc_num);

	cblas_daxpy(mat_mysize(&mat_stat),-1.0,O,1,QR,1);

	mat_dprint(&mat_stat,O,QR);
	error = mat_comp3(&mat_stat,O,my_rank,Proc_num);
	if(my_rank == 0)
	{
		printf("||I-Q'Q||oo=\t%e\n",error);
	}

#ifdef DEBUG
	if(my_rank == 0)
	{
		printf("end ||I-q'q||oo\n");
		printf("run make QR\n");
	}
#endif

	MPI_Barrier(MPI_COMM_WORLD);
	Hybrid_dgemm(&mat_stat,Q,A,QR,my_rank,Proc_num);
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef DEBUG
	if(my_rank == 0)
	{
		printf("end make QR\n");
		printf("run ||A-QR||oo\n");
	}
#endif

	mat_remv(&mat_stat,QR,seed,my_rank);
	mat_dprint(&mat_stat,O,QR);
	error = mat_comp3(&mat_stat,O,my_rank,Proc_num);
	if(my_rank == 0)
	{
		printf("||A-QR||oo=\t%e\n",error);
		fprintf(stderr,"end ||A-QR||oo\n");
	}

	t2 = MPI_Wtime();
	if (0 == my_rank)
	{
		printf("CRAYJ: Elapsed time for calculation norms = %f sec.\n", t2-t1);
	}

	free(Q);
	free(O);
	free(QR);
	free(TAU);
	free(A);

	MPI_Finalize();
	exit(EXIT_SUCCESS);
	return 0;
}
