/*
 * mat_prod.h
 *
 *  Created on: 2015/11/30
 *      Author: stomo
 */

#ifndef MAT_PROD_H_
#define MAT_PROD_H_

// 行列のステータス
typedef struct mstatus
{

	unsigned long int mat_M;			// 行列のMサイズ
	unsigned long int mat_N;			// 行列のNサイズ
	unsigned long int my_mat_M;		// 自分のランクの持つMサイズ
	unsigned long int my_mat_N;		// 自分のランクの持つNサイズ
	unsigned int tile_M;				// タイルのMサイズ
	unsigned int tile_N;				// タイルのNサイズ
	unsigned int glob_tile_M;		// タイルのM方向の数
	unsigned int glob_tile_N;		// タイルのN方向の数
	unsigned int my_tile_M;			// 自分の持つタイルのM方向の数
	unsigned int my_tile_N;			// 自分の持つタイルのN方向の数
} mat_stat_t;

/****************************************
 行列の初期化を行う
 mat_m 行列全体の行数
 mat_n 行列全体の列数
 tile_m タイルの行数
 tile_n タイルの列数
 my_rank MPIランク
 proc_num MPIプロセス数
*****************************************/
void mat_init( mat_stat_t *status,
		long int mat_m, long int mat_n,
		int tile_m, int tile_n,
		int my_rank, int proc_num);

// 簡単な関数はdefine関数を用いる
#define mat_mysize(st) (unsigned long int)(((st)->my_mat_M)*((st)->my_mat_N))
#define mat_myTAU(st) (unsigned long int)(((st)->my_mat_M/2)*((st)->my_mat_N/2))
#define mat_TileSize(st) (unsigned int)(((st)->tile_M)*((st)->tile_N))
#define mat_getTileM(st) ((st)->tile_M)
#define mat_getTileN(st) ((st)->tile_N)
#define mat_getM(st) ((st)->mat_M)
#define mat_getN(st) ((st)->mat_N)
#define mat_getMyM(st) ((st)->my_mat_M)
#define mat_getMyN(st) ((st)->my_mat_N)
#define mat_getGTileM(st) ((st)->glob_tile_M)
#define mat_getGTileN(st) ((st)->glob_tile_N)
#define mat_getLTileM(st) ((st)->my_tile_M)
#define mat_getLTileN(st) ((st)->my_tile_N)

// タイル状行列の位置番号をタイル位置とタイル内の番号から返す関数
#define mat_getPosID(st,ti,tj,si,sj) \
(((ti)+(tj)*((st)->my_tile_M))\
*((st)->tile_M)*((st)->tile_N)+(si)+(sj)*((st)->tile_M))

// タイル状行列のタイルの位置を返す
#define mat_getMatPos(st,i,j) (((i)+(j)*((st)->my_tile_M))\
*((unsigned int)(((st)->tile_M)*((st)->tile_N))))

// タイル状ベクトルのタイルの位置を返す
#define mat_getVecPos(st,i,j) (((i)+(j)*((st)->my_tile_M))\
*((unsigned int)((st)->tile_N)))

// 行列の誤差を出力する関数
double mat_comp(mat_stat_t * status,double * matA,unsigned int seed,int my_rank,int proc_num);
double mat_comp2(mat_stat_t * status,double * matA,double * matB,int my_rank,int proc_num);
double mat_comp3(mat_stat_t * status,double * matE,int my_rank,int proc_num);

// 行列を作成する関数
void mat_make(mat_stat_t * status,double * matA,unsigned int seed,int my_rank);
void mat_remv(mat_stat_t * status,double * matA,unsigned int seed,int my_rank);

// タイル配列の行列を出力する関数
void mat_print(mat_stat_t * status,FILE * out,double * mat,const char * head_string);
void mat_dprint(mat_stat_t * status,double * out,double * mat);
// タイルを出力する関数
void mat_print_tile(FILE * out,double * tile,const char * head_string
	,const int tile_row,const int tile_col
	);



#endif /* MAT_PROD_H_ */
