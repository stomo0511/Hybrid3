/*
 * TileSch.h
 *
 *  Created on: 2015/11/30
 *      Author: stomo
 */

#ifndef TILESCH_H_
#define TILESCH_H_

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// スケジュール用構造体
typedef struct list_data
{
    long int i;
    long int j;
    long int k;
    unsigned int priority;
    unsigned int data;
} TilSch_t;

typedef struct list_data_pry
{
    TilSch_t pos;
    unsigned int priority;
} TilSch_sch_t;

typedef struct list_data_table
{
    TilSch_t pos;
    unsigned int depend;
} TilSch_table_t;

typedef struct list_data_buff
{
    TilSch_t pos;
    double * pointer;
    long int life;
} TilSch_buff_t;

typedef struct sche_list
{
    TilSch_t * list;
    long int head;
    long int tail;
    long int max;
    long int max_size;
} TilSch_List_t;

typedef struct sche_table
{
    TilSch_table_t * list;
    long int head;
    long int tail;
    long int max;
    long int use;
} TilSch_Table_t;

typedef struct sche_buff
{
    TilSch_buff_t * list;
    long int head;
    long int tail;
    long int max;
    long int use;
    int buff_size;
} TilSch_Buff_t;

// 初期化
void TilSch_init(TilSch_List_t * list,long int max);

// スケジュール取得
int TilSch_pop(TilSch_List_t * list,TilSch_t * data);

// スケジュール挿入
int TilSch_push(TilSch_List_t * list,TilSch_t data);

// 空かどうか
int TilSch_empty(TilSch_List_t * list);

// 一杯かどうか
int TilSch_fully(TilSch_List_t * list);

// 終了処理
void TilSch_end(TilSch_List_t * list);

// ソート

// 特定のデータの依存書き換えを行う
int TilSch_depend(TilSch_List_t * list,TilSch_t data,unsigned int add_depend);

// 特定のデータを削除する
int TilSch_delete(TilSch_List_t * list,TilSch_t data);

// データ用定数
#define TSNONE		(0x0000)
#define TSFULL		(0xffff)
// 1バイトフィルタ
#define TSFilter1	(0x000f)
#define TSFilter2	(0x00f0)
#define TSFilter3	(0x0f00)
#define TSFilter4	(0xf000)

// 依存定数
#define TSDep_MEM	(0x0001)	//(.00000001)2
#define TSDep_LIN	(0x0002)	//(.00000010)2
#define TSDep_YT	(0x0004)	//(.00000100)2
#define TSDep_END	(0x0007)	//(.00000111)2

// 項目定数
#define TSKa_GEQ	(0x0010)	// geqrt
#define TSKa_TSQ	(0x0020)	// tsqrt
#define TSKa_LAR	(0x0030)	// larfb
#define TSKa_SSR	(0x0040)	// ssrfb
#define TSKa_SEC	(0x00a0)	// send
#define TSKa_REC	(0x00b0)	// recv

// 特定スレッド定数
#define TSTh_COM	=0x0100;	// comunication

#endif /* TILESCH_H_ */
