/*
 * TileSch.c
 *
 *  Created on: 2015/11/30
 *      Author: stomo
 */

#include "TileSch.h"

void TilSch_init(TilSch_List_t * list,long int max)
{
    list->head = 0;
    list->tail = 0;
    list->max = max;
    list->max_size = 0;
    list->list = calloc(max,sizeof(TilSch_t));
}

int TilSch_pop(TilSch_List_t * list,TilSch_t * data)
{
    int flag;

    #pragma omp critical (tilsch)
    {
        if( (list->tail<=list->head))
        {
            flag = 0;
        }
        else
        {
            (*data) = list->list[(list->head)%list->max];
            list->head++;
            flag = 1;
        }
    }

    return flag;
}

int TilSch_push(TilSch_List_t * list,TilSch_t data)
{
    int flag;

    #pragma omp critical (tilsch)
    {
        if(list->max <= (list->tail-list->head))
        {
            flag = 0;
        }
        else
        {
            list->list[(list->tail)%(list->max)] = data;
            list->tail++;
            if(list->tail-list->head > list->max_size)
                list->max_size = list->tail-list->head;
            flag = 1;
        }
    }

    return flag;
}

// 空かどうか
int TilSch_empty(TilSch_List_t * list)
{
    int flag;

    #pragma omp critical (tilsch)
    {
        flag = (list->tail<=list->head);
    }

    return flag;
}
// 一杯かどうか
int TilSch_fully(TilSch_List_t * list)
{
    int flag;

    #pragma omp critical (tilsch)
    {
        flag = (list->max <= (list->tail-list->head));
    }

    return flag;
}

void TilSch_end(TilSch_List_t * list)
{
    // デバッグ用
    printf("Q_size[%ld,%ld,%ld,%ld]\n",list->head,list->tail,list->max,list->max_size);

}

// 特定のデータの依存書き換えを行う
int TilSch_depend(TilSch_List_t * list,TilSch_t data,unsigned int add_depend)
{
    long int l;
    int flag = 0;
    #pragma omp critical (tilsch)
    {
        if(add_depend == TSDep_END)
        {
            flag = 1;
        }
        else{
            for(l = list->head; l < list->tail; l++)
            {
                if((list->list[l].i == data.i) &&
                   (list->list[l].j == data.j) &&
                   (list->list[l].k == data.k))
                {
                    flag = 1;
                    break;
                }
            }
            if(flag)
            {
                list->list[l].data |= add_depend;
                if((list->list[l].data&TSFilter1) == TSDep_END)
                {
                    flag = 1;
                }
                else
                {
                    flag = 0;
                }
            }
            else
            {
                list->list[list->tail] = data;
                list->list[list->tail].data = add_depend;
                list->tail++;
                if(list->tail-list->head > list->max_size)
                    list->max_size = list->tail-list->head;
            }
        }
    }

    return flag;
}

// 特定のデータを削除する
int TilSch_delete(TilSch_List_t * list,TilSch_t data)
{
    long int l;
    int flag = 0;

    #pragma omp critical (tilsch)
    {
        for(l = list->head; l < list->tail; l++)
        {
            if((list->list[l].i == data.i) &&
               (list->list[l].j == data.j) &&
               (list->list[l].k == data.k) &&
               ((list->list[l].data&TSFilter3) == (data.i&TSFilter3)))
            {
                flag = 1;
                break;
            }
        }
        if(flag)
        {
            for(l++;l < list->tail;l++)
            {
                list->list[l-1] = list->list[l];
            }
            list->tail--;
        }
    }

    return flag;
}


