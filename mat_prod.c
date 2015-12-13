/*
 * mat_prod.c
 *
 *  Created on: 2015/11/30
 *      Author: stomo
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "mat_prod.h"

// 行列の初期化を行う
void mat_init(mat_stat_t * status
              ,long int mat_m,long int mat_n
              ,int tile_m,int tile_n
              ,int my_rank,int proc_num)
{
    long int i;
    long int j;
    status->mat_M = mat_m;
    status->mat_N = mat_n;
    i = mat_m/tile_m;
    status->my_mat_M = mat_m;
    status->my_tile_M = i;
    status->glob_tile_M = i;
    i = mat_n/tile_n;
    status->glob_tile_N = i;
    j = i/proc_num;
    if(i%proc_num > my_rank)
        j++;
    status->my_mat_N = tile_n*j;
    status->my_tile_N = j;
    status->tile_M = tile_m;
    status->tile_N = tile_n;
}

// 行列の誤差を出力する関数
// 行列のmaxノルム

double mat_comp2(mat_stat_t * status,double * matA,double * matB,int my_rank,int proc_num)
{
    double e = 0,s;
    int i,j,ii,jj;
    double * ebf = calloc(proc_num,sizeof(double));
    double n;

    for(j = 0; j < mat_getLTileN(status);j++)
    {
        for(jj = 0; jj < mat_getTileN(status); jj++)
        {
            for(i = 0; i < mat_getLTileM(status);i++)
            {
                for(ii = 0; ii < mat_getTileM(status);ii++)
                {
                    s += fabs(fabs(matA[mat_getPosID(status,i,j,ii,jj)])
                              -fabs(matB[mat_getPosID(status,i,j,ii,jj)]));
                }
            }
            s = fabs(s);
            if(s > e)
                e = s;
        }
    }
    MPI_Gather(&e,1,MPI_DOUBLE,ebf,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    e = 0;
    if(my_rank == 0){
        for(i = 0; i < proc_num;i++){
            if(ebf[i] > e)
                e = ebf[i];
        }
    }
    return e;
}

double mat_comp(mat_stat_t * status,double * matA,unsigned int seed,int my_rank,int proc_num)
{
    double e = 0,s;
    int i,j,ii,jj;
    double * ebf = calloc(proc_num,sizeof(double));
    double n;
    srand(seed+my_rank);
    for(j = 0; j < mat_getLTileN(status);j++)
    {
        for(jj = 0; jj < mat_getTileN(status); jj++)
        {
            for(i = 0; i < mat_getLTileM(status);i++)
            {
                for(ii = 0; ii < mat_getTileM(status);ii++)
                {
                    #if __CHECK_MODE__
                    n = (double)((double)(rand()%10));
                    #else
                    n = (double)((double)rand()/(double)RAND_MAX);
                    #endif
                    s += fabs(fabs(matA[mat_getPosID(status,i,j,ii,jj)])-fabs(n));
                }
            }
            s = fabs(s);
            if(s > e)
                e = s;
        }
    }
    MPI_Gather(&e,1,MPI_DOUBLE,ebf,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    e = 0;
    if(my_rank == 0){
        for(i = 0; i < proc_num;i++){
            if(ebf[i] > e)
                e = ebf[i];
        }
    }
    return e;
}
double mat_comp3(mat_stat_t * status,double * matE,int my_rank,int proc_num)
{
    double e = 0,s;
    int i,j;
    double * ebf = calloc(proc_num,sizeof(double));
    double n;

    for(i = 0; i< mat_getMyN(status);i++)
    {
        s = 0;
        for(j = 0; j < mat_getMyM(status);j++){
            s += fabs(matE[i*mat_getMyM(status)+j]);
        }
        if(s > e)
            e = s;

        if (isnan(s)) {
            e = s;
            break;
        }

    }

    MPI_Gather(&e,1,MPI_DOUBLE,ebf,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    e = 0;
    if(my_rank == 0){
        for(i = 0; i < proc_num;i++){
            if(ebf[i] > e)
                e = ebf[i];

            if (isnan(ebf[i])) {
                e = ebf[i];
                break;
            }
        }
    }
    free(ebf);

    return e;
}

// 行列を作成する関数
void mat_make(mat_stat_t * status,double * matA,unsigned int seed,int my_rank)
{
    int i,j,ii,jj;
    double n;

    srand(seed+my_rank);

    for(j = 0; j < mat_getLTileN(status);j++)
    {
        for(jj = 0; jj < mat_getTileN(status); jj++)
        {
            for(i = 0; i < mat_getLTileM(status);i++)
            {
                for(ii = 0; ii < mat_getTileM(status);ii++)
                {
                    #if __CHECK_MODE__
                    n = (double)((double)(rand()%10));
                    #else
                    n = (double)((double)rand()/(double)RAND_MAX);
                    #endif
                    matA[mat_getPosID(status,i,j,ii,jj)] = n;
                }
            }
        }
    }
}

void mat_remv(mat_stat_t * status,double * matA,unsigned int seed,int my_rank)
{
    int i,j,ii,jj;
    double n;

    srand(seed+my_rank);
    for(j = 0; j < mat_getLTileN(status);j++)
    {
        for(jj = 0; jj < mat_getTileN(status); jj++)
        {
            for(i = 0; i < mat_getLTileM(status);i++)
            {
                for(ii = 0; ii < mat_getTileM(status);ii++)
                {
                    #if __CHECK_MODE__
                    n = (double)((double)(rand()%10));
                    #else
                    n = (double)((double)rand()/(double)RAND_MAX);
                    #endif
                    matA[mat_getPosID(status,i,j,ii,jj)] -= n;
                }
            }
        }
    }
}

// タイル配列の行列を出力する関数
void mat_print(mat_stat_t * status, FILE *out,double * mat,const char * head_string)
{
    char buff[10000];
    int i,j,k,l;
    int tile_row = mat_getTileM(status);
    int tile_col = mat_getTileN(status);
    int tile_row_num = mat_getLTileM(status);
    int tile_col_num = mat_getLTileN(status);
//crayj>>>
    int np, me, tag, dummy;
    MPI_Status stat;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    tag = 100;
    dummy = 0;
    if (0 != me) {
        MPI_Recv(&dummy, 1, MPI_INT, me-1, tag, MPI_COMM_WORLD, &stat);
    }
//crayj<<<

//	sprintf(buff,"%s\n",head_string);
//crayj	fprintf(out,"%s\n",head_string);
//crayj>>>
    fprintf(out,"[%5d] %s\n", me, head_string);
//crayj<<<
    for(i = 0; i < tile_row_num; i++){
        for(k = 0; k < tile_row; k++){
            for(j = 0; j < tile_col_num; j++){
                for(l = 0; l < tile_col; l++)
                {
/*					sprintf(buff,"%s%lf\t",buff
                                        ,(mat+(i+tile_row_num*j)*tile_row*tile_col)[k+tile_row*l]
					);
*/					fprintf(out,"%lf\t",(mat+(i+tile_row_num*j)*tile_row*tile_col)[k+tile_row*l]);
                }
            }
//			sprintf(buff,"%s\n",buff);
            fprintf(out,"\n");
        }
    }
    fprintf(out,"\n");
//crayj>>>
    fflush(out);
    if (np-1 != me) {
        MPI_Send(&dummy, 1, MPI_INT, me+1, tag, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
//crayj<<<
}
void mat_dprint(mat_stat_t * status,double * out,double * mat)
{
    int i,j,k,l,c;
    int tile_row = mat_getTileM(status);
    int tile_col = mat_getTileN(status);
    int tile_row_num = mat_getLTileM(status);
    int tile_col_num = mat_getLTileN(status);

    c = 0;
    for(j = 0; j < tile_col_num; j++)
    {
        for(l = 0; l < tile_col; l++)
            for(i = 0; i < tile_row_num; i++)
            {
                for(k = 0; k < tile_row; k++)
                {
                    out[c] = (mat+(i+tile_row_num*j)*tile_row*tile_col)[k+tile_row*l];
                    c++;
                }
            }
    }
}

// タイルを出力する関数
void mat_print_tile(FILE *out,double * tile,const char * head_string
                    ,const int tile_row,const int tile_col )
{
    char buff[10000];
    int i,j;
    sprintf(buff,"%s\n",head_string);
    for(i = 0; i < tile_row; i++){
        for(j = 0; j < tile_col; j++)
        {
            sprintf(buff,"%s%lf\t",buff,tile[i+tile_row*j]);
        }
        sprintf(buff,"%s\n",buff);
    }
    fprintf(out,"%s\n",buff);
}

