//crayj>>>
#if defined(CRAYJ_USE_COREBLAS)
#  include "plasma.h"
#  include "core_blas.h"
#endif
#if defined(CRAYJ_TIMING_FILE_PER_RANK)
#  if ! defined(CRAYJ_TIMING)
#    define CRAYJ_TIMING 1
#  endif
#endif
#if defined(CRAYJ_TIMELINE) || defined(CRAYJ_DYNAMIC_COMM_SCHED)
#  include <stdbool.h>
#endif

//crayj<<<
#include "Hybrid_tile_QR.h"
#include "TileSch.h"
#include "tile_kernel.h"
#include <signal.h>

#define __TILE_BLOCK__ 10

#define M_p(MATRIX,i,j) ((MATRIX)+(mat_getMatPos(status,(i),(j))))
#define V_p(VECTOR,i,j) ((VECTOR)+(mat_getVecPos(status,(i),(j))))

#define PT_p(i,j,k) ((prog_table)+((i)*mat_getGTileM(status)*mat_getGTileN(status)+(j)*mat_getGTileM(status)+(k)))

#define BP_p(i,j) ((buff_pos)+((i)+mat_getGTileM(status)*(j)))

#define BF(id) ((glob_YT)+((id)*mat_TileSize(status)*2))
#define BF_ID(i,j) ((i)+mat_getGTileM(status)*(j))

#define __DEBUGOUT__(str)
#define __DBGKANEL__(str,pi,pj,pk)
#define __DBGQUEUE__(str,gi,gj,gk,li,lj,lk)
#define __DBGDEPEND__(str,pi,pj,pk,dp)
#define __DBGBUFF__(str,gi,gj,gk,id)

#define DEP_MEM	(0x0001)	//(00000001)2
#define DEP_LIN	(0x0002)	//(00000010)2
#define DEP_YTM	(0x0004)	//(00000100)2
#define DEP_END	(0x0007)	//(00000111)2
#define DEP_NON	(0x0000)	//(00000000)2

void sigcatch(int sig)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    fprintf(stderr,"ID[%d,%d] is Segmentation fault (signal 11)\n",my_rank,omp_get_thread_num());
    exit(-1);
}

//crayj>>>
static MPI_Datatype MPI_TILE;
//crayj<<<
#if defined(CRAYJ_TIMING)
//crayj>>>
#  define CRAYJ_MAX_THREADS 32
static double crayj_elapsed[CRAYJ_MAX_THREADS][8];
static int    crayj_ncalls [CRAYJ_MAX_THREADS][8];

static double ela_send, ela_recv;
static int n_send, n_recv;
#  pragma omp threadprivate(ela_send,ela_recv,n_send,n_recv)
//crayj<<<
#endif
#if defined(CRAYJ_TIMELINE)
#  define CRAYJ_FILENAME_MAX 128
#endif

#if defined(CRAYJ_TIMELINE)
//crayj>>>
static double timebase;
static FILE * timeline;
#  pragma omp threadprivate(timeline)
//crayj<<<
#endif
#if defined(CRAYJ_DYNAMIC_COMM_SCHED)
//crayj>>>
static unsigned long *send_gis;
//crayj<<<
#endif
//crayj>>>

// building blocks for communication threads
static
double *
sender_get_next_buffer(
    mat_stat_t *status, long int const *buff_pos, double *glob_YT,
    unsigned long g_i, unsigned long g_k
    )
{
    double *buff = NULL;
    long int buff_pos_id = -1;

#pragma omp critical (buff_pos)
    {
        if (*BP_p(g_k,g_i) >= 0) {
            buff_pos_id = *BP_p(g_k, g_i);
            __DBGBUFF__("send get", g_i, g_i, g_k, *BP_p(g_k, g_i));
        } 
    }
    if (buff_pos_id >= 0) {
        buff = BF(buff_pos_id);
    }
    return buff;
}

static
void
sender_send(
    mat_stat_t *status, double *buff, unsigned long g_i, unsigned long g_k
    )
{
    double ttmp;
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

//    __DBGKANEL__("send",g_i,g_i,g_k);

#if defined(CRAYJ_TIMELINE)
    if (NULL != timeline) {
        fprintf(timeline, "%f, send start %d %d %d\n", MPI_Wtime() - timebase,
                g_i, g_i, g_k);
        fflush(timeline);
    }
#endif
#if defined(CRAYJ_TIMING)
    ttmp = MPI_Wtime();
#endif
#  if defined(CRAYJ_DYNAMIC_COMM_SCHED)
    MPI_Bcast(&g_k, 1, MPI_UNSIGNED_LONG, myrank, MPI_COMM_WORLD);
#  endif
    MPI_Bcast(buff, 1, MPI_TILE, myrank, MPI_COMM_WORLD);
#if defined(CRAYJ_TIMING)
    ela_send += MPI_Wtime() - ttmp;
    ++n_send;
#endif
#if defined(CRAYJ_TIMELINE)
    if (NULL != timeline) {
        fprintf(timeline, "%f, send end   %d %d %d\n", MPI_Wtime() - timebase,
                g_i, g_i, g_k);
        fflush(timeline);
    }
#endif
}

static double* recver_get_new_buffer(
    mat_stat_t *status, double *glob_YT, 
    long buff_table_size, long buff_head, long *buff_tail,
    long *buff_pos_id
    )
{
    double *buff = NULL;
#define CRAYJ_NWAIT_MAX 100000
    int nwait = 0;
    while (buff == NULL)
    {
#pragma omp critical (buff_pos)
        {
#if defined(CRAYJ_TIMELINE)
            fprintf(timeline, "%f, buff_head == %d, buff_tail == %d, buff_table_size == %d\n",
                    MPI_Wtime() - timebase, buff_head, *buff_tail, buff_table_size);
            fflush(timeline);
#endif

            if(*buff_tail-buff_head < buff_table_size)
            {
                *buff_pos_id = *buff_tail;
                (*buff_tail)++;
                buff = BF((*buff_pos_id)%buff_table_size);
                // __DBGBUFF__("recv set",g_i,g_i,g_k,buff_pos_id%buff_table_size);
            }
        }

        ++nwait;
        if (CRAYJ_NWAIT_MAX < nwait) {
            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            printf("CRAYJ:[%5d] ERROR: %s:%d: "
                   "couldn't get a new buffer\n",
                   myrank, __FILE__, __LINE__);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return buff;
}

static void recver_recv(
    mat_stat_t *status, double *recvbuff, unsigned long g_i, unsigned long *g_k
    )
{
#if defined(CRAYJ_TIMING)
    double ttmp;
#endif
    int src;
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    src = g_i % nprocs;

//    __DBGKANEL__("recv", g_i, g_i, *g_k);
#if defined(CRAYJ_TIMELINE)
    if (NULL != timeline) {
        fprintf(timeline, "%f, recv start %d %d %d\n", MPI_Wtime() - timebase,
                g_i, g_i, *g_k);
        fflush(timeline);
    }
#endif
#if defined(CRAYJ_TIMING)
    ttmp = MPI_Wtime();
#endif
#  if defined(CRAYJ_DYNAMIC_COMM_SCHED)
    MPI_Bcast(g_k, 1, MPI_UNSIGNED_LONG, src, MPI_COMM_WORLD);
#  endif
    MPI_Bcast(recvbuff, 1, MPI_TILE, src, MPI_COMM_WORLD);
#if defined(CRAYJ_TIMING)
    ela_recv += MPI_Wtime() - ttmp;
    ++n_recv;
#endif
#if defined(CRAYJ_TIMELINE)
    if (NULL != timeline) {
        fprintf(timeline, "%f, recv end   %d %d %d\n", MPI_Wtime() - timebase,
                g_i, g_i, *g_k);
        fflush(timeline);
    }
#endif
}

static
void
recver_postprocess(
    mat_stat_t *status, unsigned long g_i, unsigned long g_k,
    long *buff_pos, long buff_pos_id, 
    long *buff_table, long buff_table_size, long *buff_ticket,
    unsigned char *prog_table, TilSch_List_t *sche_list
    )
{
    TilSch_t data;
    long ti, ticket = 0;
    int myrank, nprocs;
    unsigned long g_j;

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    for (ti = g_i+1; ti < mat_getGTileN(status); ti++)
    {
        if (myrank == ti%nprocs)
        {
            ticket++;
        }
    }

    // バッファに登録
#pragma omp critical (buff_pos)
    {
        buff_table[buff_pos_id%buff_table_size] = BF_ID(g_k,g_i);
        *BP_p(g_k,g_i) = buff_pos_id%buff_table_size;
        buff_ticket[buff_pos_id%buff_table_size] = ticket;
    }
    // カーネル登録
    for (g_j = g_i+1; g_j < mat_getGTileN(status); g_j++)
    {
//      __DBGKANEL__("find",g_i,g_j,g_k);
        if (myrank == g_j%nprocs)
        {
            int input = 0;
//          __DBGKANEL__("find ok",g_i,g_j,g_k);
#pragma omp critical (prog_table)
            {
                *PT_p(g_i,g_j,g_k) |= DEP_YTM;
                __DBGDEPEND__("depend",g_i,g_j,g_k,*PT_p(g_i,g_j,g_k));
                if(*PT_p(g_i,g_j,g_k) == DEP_END)
                {
                    input = 1;
                }
            }
            data.i = g_i;
            data.j = g_j;
            data.k = g_k;
            // l_j = g_j/Proc_num;
            // __DBGQUEUE__("push",g_i,g_j,g_k,l_i,l_j,l_k);
            while(input&&!TilSch_push(sche_list,data)){}
#if defined(CRAYJ_TIMELINE)
            if (input && NULL != timeline) {
                fprintf(timeline, "%f, push %d %d %d\n",
                        MPI_Wtime() - timebase, g_i, g_j, g_k);
                fflush(timeline);
            }
#endif
        }
    }
}

static int compare(const void *px, const void *py)
{
    unsigned long const x = *(unsigned long *)px;
    unsigned long const y = *(unsigned long *)py;
    if (x < y)       return -1;
    else if (x == y) return  0;
    else             return  1;
}

//crayj<<<

void Hybrid_tile_QR
(
    mat_stat_t * status
    ,double * A,double * TAU
//	,double * T
    ,int my_rank,int Proc_num
#if defined(CRAYJ_USE_COREBLAS)
//crayj>>>
    ,int ib
//crayj<<<
#endif
    )
{
    /***************************************************************************
		グローバル変数定義
    ***************************************************************************/
    // 共有データ
#if ! defined(CRAYJ_USE_VOLATILE)
    TilSch_List_t sche_list;
#  if defined(CRAYJ_TWO_LEVEL_SCHED)
//crayj>>>
    TilSch_List_t priority_list;
//crayj<<<
#  endif
    unsigned char * prog_table;
    int run_flag = 1;
#else
    volatile TilSch_List_t sche_list;
#  if defined(CRAYJ_TWO_LEVEL_SCHED)
//crayj>>>
    volatile TilSch_List_t priority_list;
//crayj<<<
#  endif
    volatile unsigned char * prog_table;
    volatile int run_flag = 1;
#endif
	
#if ! defined(CRAYJ_USE_VOLATILE)
    long int * buff_pos;
    long int buff_head = 0;
    long int buff_tail = 0;
#else
//crayj>>>
    volatile long int * buff_pos;
    volatile long int buff_head = 0;
    volatile long int buff_tail = 0;
//crayj<<<
#endif
	
    double * glob_YT;
    long int * buff_table;
    long int * buff_ticket;
    long int buff_table_size;
    double * T;

    // 通信ステータス
//crayj	MPI_Datatype MPI_TILE;
#if defined(CRAYJ_TIMING)
//crayj>>>
    int dummy, tag, tid;
//crayj<<<
#endif

#if defined(CRAYJ_VERBOSE)
//crayj>>>
    if (0 == my_rank) {
        printf("CRAYJ: The following optimization flags have been enabled:\n");
        printf("CRAYJ:   CRAYJ_VERBOSE\n");
#  if defined(CRAYJ_TIMING)
        printf("CRAYJ:   CRAYJ_TIMING\n");
#  endif
#  if defined(CRAYJ_TIMELINE)
        printf("CRAYJ:   CRAYJ_TIMELINE\n");
#  endif
#  if defined(CRAYJ_USE_OMP_FLUSH)
        printf("CRAYJ:   CRAYJ_USE_OMP_FLUSH\n");
#  endif
#  if defined(CRAYJ_USE_VOLATILE)
        printf("CRAYJ:   CRAYJ_USE_VOLATILE\n");
#  endif
#  if defined(CRAYJ_USE_COREBLAS)
        printf("CRAYJ:   CRAYJ_USE_COREBLAS\n");
#  endif
#  if defined(CRAYJ_TWO_LEVEL_SCHED)
        printf("CRAYJ:   CRAYJ_TWO_LEVEL_SCHED\n");
#  endif
#  if defined(CRAYJ_DYNAMIC_COMM_SCHED)
        printf("CRAYJ:   CRAYJ_DYNAMIC_COMM_SCHED\n");
#  endif
    }
//crayj<<<
#endif

    TilSch_init(&sche_list,mat_getLTileM(status)*mat_getLTileN(status)*mat_getLTileN(status)/20+100);
#if defined(CRAYJ_TWO_LEVEL_SCHED)
//crayj>>>
    TilSch_init(&priority_list,mat_getLTileM(status)*mat_getLTileN(status)*mat_getLTileN(status)/20+100);
//crayj<<<
#endif
    prog_table = calloc(mat_getGTileM(status)*mat_getGTileN(status)*mat_getGTileN(status),1);
	
    // バッファのポジションを記録
    buff_pos = calloc(mat_getGTileM(status)*mat_getGTileN(status),sizeof(long int));

    // バッファの最大数
/*	if(Proc_num < mat_getLTileN(status))
	{
        buff_table_size = mat_getLTileM(status)* mat_getLTileN(status);
	}
	else
	{
        buff_table_size = mat_getLTileM(status)*Proc_num;
	}
*/
//crayj	buff_table_size = mat_getLTileM(status)*2;
//crayj>>>
    buff_table_size = (mat_getLTileM(status)+1)*(mat_getGTileN(status)+1)/2;
    if (0 == buff_table_size) {
        ++buff_table_size;
    }
    // This could be a little bit larger than actually required,
    // but it helps avoid deadlock caused by the shortage of the buffer.
//crayj<<<

    // バッファのIDを記録
    buff_table = calloc(buff_table_size,sizeof(long int));	
    buff_ticket = calloc(buff_table_size,sizeof(long int));
    glob_YT = calloc(buff_table_size*mat_TileSize(status)*2,sizeof(double));
    T = calloc(mat_mysize(status),sizeof(double));

    // MPI_TILEの設定
//	MPI_Type_contiguous(mat_TileSize(status)+mat_getTileN(status),MPI_DOUBLE,&MPI_TILE);
    MPI_Type_contiguous(mat_TileSize(status)*2,MPI_DOUBLE,&MPI_TILE);
    MPI_Type_commit(&MPI_TILE);
//crayj	signal(SIGSEGV,sigcatch);

#if (! defined(CRAYJ_TWO_LEVEL_SCHED)) && (! defined(CRAYJ_TIMELINE))
#pragma omp parallel shared(sche_list,run_flag) firstprivate(prog_table,buff_pos,glob_YT,buff_table,buff_ticket,buff_table_size)
#endif
//crayj>>>
#if defined(CRAYJ_TWO_LEVEL_SCHED) && defined(CRAYJ_TIMELINE)
#pragma omp parallel default(shared) shared(sche_list,run_flag)         \
    shared(priority_list,timebase)                                      \
    firstprivate(prog_table,buff_pos,glob_YT,buff_table,buff_ticket,buff_table_size) 
#elif defined(CRAYJ_TWO_LEVEL_SCHED) 
#pragma omp parallel default(shared) shared(sche_list,run_flag)         \
    shared(priority_list)                                               \
    firstprivate(prog_table,buff_pos,glob_YT,buff_table,buff_ticket,buff_table_size) 
#elif defined(CRAYJ_TIMELINE)
#pragma omp parallel default(shared) shared(sche_list,run_flag)         \
    shared(timebase)                                                    \
    firstprivate(prog_table,buff_pos,glob_YT,buff_table,buff_ticket,buff_table_size) 
#endif
//crayj<<<
    {
        unsigned long int g_i,g_j,g_k,l_i,l_j,l_k;
        long int ti;
        long int ticket;
        int ii,jj,kk;
        int my_flag = 1;
        int my_turn = 1;
        int comm_flag = 1;
        long int buff_pos_id;
        int input = 0;
        long int my_count;
        double * work;
        const double dzero = 0.0;
        TilSch_t data;
        double * buff;
        double * buff_Y;
        double * buff_T;
        double * geY;
        double * prT;
        double * comm_buff;
#if defined(CRAYJ_USE_COREBLAS)
//crayj>>>
        int retval;
//crayj<<<
#endif
#if defined(CRAYJ_TIMELINE)
//crayj>>>
        char filename[CRAYJ_FILENAME_MAX];
        bool flag_notmyturn = false;
//crayj<<<
#endif
#if defined(CRAYJ_TIMING) || defined(CRAYJ_TIMELINE)
//crayj>>>
        int tid = omp_get_thread_num();
//crayj<<<
#endif
#if defined(CRAYJ_TIMING)
//crayj>>>
        double ela_geqrt, ela_tsqrt, ela_larfb, ela_ssrfb, ela_taskwait;
        int n_geqrt, n_tsqrt, n_larfb, n_ssrfb, n_taskwait, n_notmyturn;
        double ela_total;
//		double ela_send, ela_recv;
//		int n_send, n_recv;
        double ttmp;

        ela_geqrt = 0.0;
        ela_tsqrt = 0.0;
        ela_larfb = 0.0;
        ela_ssrfb = 0.0;
        ela_taskwait = 0.0;
        ela_send = 0.0;
        ela_recv = 0.0;
        n_geqrt = 0;
        n_tsqrt = 0;
        n_larfb = 0;
        n_ssrfb = 0;
        n_taskwait = 0;
        n_notmyturn = 0;
        n_send = 0;
        n_recv = 0;
        ela_total = MPI_Wtime();
//crayj<<<
#endif
#if defined(CRAYJ_TIMELINE)
//crayj>>>
        snprintf(filename, CRAYJ_FILENAME_MAX, 
                 "crayj_timeline_%05d_%02d.txt", my_rank, tid);
        timeline = fopen(filename, "w");
        if (NULL == timeline) {
            fprintf(stderr, "CRAYJ[%d]: "
                    "WARNING: file open failed: filename == %s\n", 
                    my_rank, filename);
            fprintf(stderr, "CRAYJ[%d]: "
                    "WARNING: timeline output disabled.\n",
                    my_rank);
        }
#pragma omp barrier
#pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            timebase = MPI_Wtime();
        }
#pragma omp barrier
        if (NULL != timeline) {
            fprintf(timeline, "%f, init start\n", MPI_Wtime() - timebase);
            fflush(timeline);
        }
//crayj<<<
#endif
        __DEBUGOUT__("init");
        work = (double *)tile_init('d',mat_getTileM(status),mat_getTileN(status));
        geY = calloc(mat_TileSize(status),sizeof(double));
        prT = calloc(mat_TileSize(status),sizeof(double));
        {
#pragma omp sections
            {
                // バッファテーブルポジションを初期化
#pragma omp section
                {
                    for(g_j = 0; g_j < mat_getGTileM(status)*mat_getGTileN(status); g_j++)
                    {
                        buff_pos[g_j] = -1;
                    }
#if defined(CRAYJ_TIMELINE)
//crayj>>>
                    if (NULL != timeline) {
                        fprintf(timeline, "%f, init buff_pos done.\n",
                                MPI_Wtime() - timebase);
                    }
//crayj<<<
#endif
                }
#pragma omp section
                {
                    __DEBUGOUT__("prog_init");
                    // プログレステーブルの初期化
                    for(g_j = 0; g_j < mat_getGTileN(status); g_j++)
                    {
                        for(g_k = 0; g_k < mat_getGTileM(status); g_k++)
                        {
                            if(g_j == 0)
                            {
                                *PT_p(0,g_j,g_k) = DEP_MEM + DEP_YTM;
                            }
                            else if(g_k == 0)
                            {
                                *PT_p(0,g_j,g_k) = DEP_MEM + DEP_LIN;
                            }
                            else
                            {
                                *PT_p(0,g_j,g_k) = DEP_MEM;
                            }
                        }
                    }
#if defined(CRAYJ_TIMELINE)
//crayj>>>
                    if (NULL != timeline) {
                        fprintf(timeline, "%f, init prog_table (1) done.\n",
                                MPI_Wtime() - timebase);
                    }
//crayj<<<
#endif
                }
#pragma omp section
                {
                    for(g_i = 1; g_i < mat_getGTileN(status); g_i++)
                    {
                        for(g_j = g_i; g_j < mat_getGTileN(status); g_j++)
                        {
                            for(g_k = g_i; g_k < mat_getGTileM(status); g_k++)
                            {
                                if(g_i==g_j&&g_j==g_k)
                                {
                                    *PT_p(g_i,g_j,g_k) = DEP_YTM+DEP_LIN;
                                }
                                else if(g_j == g_i)
                                {
                                    *PT_p(g_i,g_j,g_k) = DEP_YTM;
                                }
                                else if(g_k == g_i)
                                {
                                    *PT_p(g_i,g_j,g_k) = DEP_LIN;
                                }
                                else
                                {
                                    *PT_p(g_i,g_j,g_k) = DEP_NON;
                                }					
                            }
                        }
                    }
#if defined(CRAYJ_TIMELINE)
//crayj>>>
                    if (NULL != timeline) {
                        fprintf(timeline, "%f, init prog_table (2) done.\n",
                                MPI_Wtime() - timebase);
                    }
//crayj<<<
#endif
                }
#pragma omp section
                {
                    // 最初の計算の登録
                    if(my_rank == 0)
                    {
                        data.i = 0;
                        data.j = 0;
                        data.k = 0;
#if ! defined(CRAYJ_TWO_LEVEL_SCHED)
                        TilSch_push(&sche_list,data);
#  if defined(CRAYJ_TIMELINE)
//crayj>>>
                        fprintf(timeline, "%f, push %d %d %d\n", 
                                MPI_Wtime() - timebase, data.i, data.j, data.k);
//crayj<<<
#  endif
#else
//crayj>>>
                        TilSch_push(&priority_list,data);
#  if defined(CRAYJ_TIMELINE)
                        fprintf(timeline, "%f, push priority (1) %d %d %d\n", 
                                MPI_Wtime() - timebase, data.i, data.j, data.k);
#  endif
//crayj<<<
#endif
                        __DBGQUEUE__("first",(long int)0,(long int)0,(long int)0,(long int)0,(long int)0,(long int)0);
                    }
                }
            } // end omp sections
#pragma omp barrier
#if defined(CRAYJ_TIMELINE)
//crayj>>>
            if (NULL != timeline) {
                fprintf(timeline, "%f, init done.\n",
                        MPI_Wtime() - timebase);
            }
//crayj<<<
#endif
            ///////////////////////////////////////////////////////////////////////////
            // 通信用スレッド Start
            // MPIで分散した場合、通信用のスレッドを立ち上げる
            ///////////////////////////////////////////////////////////////////////////
            if(Proc_num > 1 && omp_get_thread_num() == 0)
            {
#if defined(CRAYJ_DYNAMIC_COMM_SCHED)
//crayj>>>
                bool finished = false; 
#  if defined(CRAYJ_TIMELINE)
                bool nosend = false;
#  endif

                send_gis = (unsigned long *)malloc(Proc_num*sizeof(unsigned long));
                if (NULL == send_gis)
                {
                    fprintf(stderr, "CRAYJ:[%5d] ERROR: %s:%d: malloc failed.\n", 
                            my_rank, __FILE__, __LINE__);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
//crayj<<<
#endif
//crayj>>>
//                comm_buff = shmalloc(mat_TileSize(status)*2*sizeof(double));
                comm_buff = malloc(mat_TileSize(status)*2*sizeof(double));
                if (NULL == comm_buff)
                {
                    fprintf(stderr, "CRAYJ:[%5d] ERROR: %s:%d: shmalloc failed.\n", 
                            my_rank, __FILE__, __LINE__);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
//crayj<<<
                __DEBUGOUT__("start_comm");
#if defined(CRAYJ_TIMELINE)
//crayj>>>
                if (NULL != timeline)
                {
                    fprintf(timeline, "%f, start communication.\n", MPI_Wtime() - timebase);
                    fflush(timeline);
                }
//crayj<<<
#endif
#if ! defined(CRAYJ_DYNAMIC_COMM_SCHED)
                ///////////////////////////////////////////////////////////////////////////
                // Static 通信スケジューリング
                for(g_i = 0; g_i < mat_getGTileN(status);g_i++)
                {
                    // いらなくなったバッファの消去
#pragma omp critical (buff_pos)
                    {
                        for(ti = buff_head;ti < buff_tail; ti++)
                        {
                            if(buff_ticket[ti%buff_table_size] == 0)
                            {
                                buff_head = ti;
                            }
                            else
                            {
                                break;
                            }
                        }
                    }

                    g_j = g_i;
                    // 自分が送り元の時
                    if(my_rank == g_i%Proc_num)
                    {
                        for(g_k = g_i ; g_k < mat_getGTileM(status); g_k++)
                        {
#  if defined(CRAYJ_TIMELINE)
//crayj>>>
                            if (NULL != timeline)
                            {
                                fprintf(timeline, "%f, send waiting for %d %d %d\n",
                                        MPI_Wtime() - timebase, g_i, g_j, g_k);
                                fflush(timeline);
                            }
//crayj<<<
#  endif
                            buff = NULL;
                            // 通信するバッファの取得
                            while(buff == NULL)
                            {
//crayj>>>
                                buff = sender_get_next_buffer(status, buff_pos, glob_YT, g_i, g_k);
//crayj<<<
                            }
//crayj>>>
                            sender_send(status, buff, g_i, g_k);
//crayj<<<
                        }
                    }
                    else
                    {
                        for(g_k = g_i; g_k < mat_getGTileM(status); g_k++)
                        {
#  if defined(CRAYJ_TIMELINE)
//crayj>>>
                            if (NULL != timeline)
                            {
                                fprintf(timeline, "%f, recv waiting for %d %d %d\n",
                                        MPI_Wtime() - timebase, g_i, g_i, g_k);
                                fflush(timeline);
                            }
//crayj<<<
#  endif
                            // 新たなバッファを取得する
//crayj>>>
                            buff = recver_get_new_buffer(
                                status, glob_YT, buff_table_size,
                                buff_head, &buff_tail, &buff_pos_id
                                );
//crayj<<<
                            // 受信開始
//crayj>>>
                            recver_recv(
                                status, comm_buff, g_i, &g_k
                                );
//crayj<<<
                            cblas_dcopy(mat_TileSize(status)*2,comm_buff,1,buff,1);
//crayj>>>
                            recver_postprocess (
                                status, g_i, g_k,
                                buff_pos, buff_pos_id,
                                buff_table, buff_table_size, buff_ticket,
                                prog_table, &sche_list
                                );
//crayj<<<
                        } // end for g_k
                    } // end of receiver
#  if defined(CRAYJ_TIMELINE)
//crayj>>>
                    if (NULL != timeline) fflush(timeline);
//crayj<<<
#  endif
                } // end for g_i

#else // CRAYJ_DYNAMIC_COMM_SCHED
                ///////////////////////////////////////////////////////////////////////////
                // Dynamic 通信スケジューリング
                g_i = my_rank;
                g_k = g_i;
                finished = false;
#  if defined(CRAYJ_TIMELINE)
                nosend = false;
#  endif
                while (! finished)
                {
                    unsigned long send_g_i, recv_g_i;
                    unsigned long recv_g_k;
                    unsigned long const g_nrows = mat_getGTileM(status);
                    unsigned long const g_ncols = mat_getGTileN(status);

                    // いらなくなったバッファの消去
#pragma omp critical (buff_pos)
                    {
                        for(ti = buff_head;ti < buff_tail; ti++)
                        {
                            if(buff_ticket[ti%buff_table_size] == 0)
                            {
                                buff_head = ti;
                            }
                            else
                            {
                                break;
                            }
                        }
                    }

                    if (g_i >= g_ncols)
                    {
                        send_g_i = g_ncols;
                    }
                    else
                    {
                        buff = sender_get_next_buffer(status, buff_pos, glob_YT, g_i, g_k);
                        if (NULL != buff)
                        {
                            send_g_i = g_i;
                        }
                        else
                        {
                            send_g_i = g_ncols;	
                        }
                    }

                    ///////////////////////////////////////////////////////////////////////////
                    MPI_Allgather(&send_g_i, 1, MPI_UNSIGNED_LONG,
                                  send_gis, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

                    qsort(send_gis, Proc_num, sizeof(unsigned long), compare);
#  if defined(CRAYJ_TIMELINE)
                    if (send_gis[0] >= g_ncols && ! nosend) {
                        if (NULL != timeline) {
                            fprintf(timeline, "%f, no one wants to send.\n", 
                                    MPI_Wtime() - timebase);
                            fflush(timeline);
                        }
                        nosend = true;
                    }
#  endif
                    for (int i = 0; i < Proc_num; ++i) {

                        if (send_gis[i] >= g_ncols) {
                            break;
                        } else if (send_gis[i] == g_i) {
                            // I'm the sender this time
                            sender_send(status, buff, g_i, g_k
                                );

                            ++g_k;
                            if (g_k >= g_nrows) {
                                g_i += Proc_num;
                                g_k = g_i;
                            }
#  if defined(CRAYJ_TIMELINE)
                            nosend = false;
#  endif
                        }
                        else
                        {
                            // I'm a receiver this time
                            double *recvbuff = recver_get_new_buffer(
                                status, 
                                glob_YT, 
                                buff_table_size, 
                                buff_head, 
                                &buff_tail, 
                                &buff_pos_id
                                );

                            recv_g_i = send_gis[i];
                            recv_g_k = 0UL;

                            recver_recv(
                                status,
                                comm_buff, 
                                recv_g_i, 
                                &recv_g_k
                                );

                            cblas_dcopy(mat_TileSize(status)*2, comm_buff, 1, recvbuff, 1);

                            recver_postprocess(
                                status, 
                                recv_g_i, 
                                recv_g_k,
                                buff_pos, 
                                buff_pos_id, 
                                buff_table, 
                                buff_table_size, 
                                buff_ticket,
                                prog_table, 
                                &sche_list
                                );
#  if defined(CRAYJ_TIMELINE)
                            nosend = false;
#  endif
                        }
                        if (send_gis[i] == g_ncols - 1)
                        {
                            finished = true;
                        }
                    }
                } // end while (! finished)
#endif // CRAYJ_DYNAMIC_COMM_SCHED

#pragma omp critical (endflag)
                run_flag = 0;
#if defined(CRAYJ_USE_OMP_FLUSH)
//crayj<<<
#pragma omp flush
//crayj>>>
#endif
#if defined(CRAYJ_TIMELINE)
//crayj>>>
                if (NULL != timeline)
                {
                    fprintf(timeline, "%f, reached end of main loop\n", 
                            MPI_Wtime() - timebase);
                    fflush(timeline);
                }
//crayj<<<
#endif
#if defined(CRAYJ_DYNAMIC_COMM_SCHED)
//crayj>>>
                free(send_gis);
//crayj<<<
#endif

                free(comm_buff);

            }
            ///////////////////////////////////////////////////////////////////////////
            // 通信用スレッド End
            ///////////////////////////////////////////////////////////////////////////
            // 計算用スレッド Start
            ///////////////////////////////////////////////////////////////////////////
            else
            {
                __DEBUGOUT__("start");
                while(my_flag)
                {
                    // 計算の取得
#if defined(CRAYJ_TIMING)
//crayj>>>
                    ttmp = MPI_Wtime();
//crayj<<<
#endif
#if ! defined(CRAYJ_TWO_LEVEL_SCHED)
#pragma omp critical (sche_list)
                    my_turn = TilSch_pop(&sche_list,&data);
#  if defined(CRAYJ_TIMELINE)
//crayj>>>
                    if (my_turn) {
                        if (NULL != timeline)
                            fprintf(timeline, "%f, pop %d %d %d\n",
                                    MPI_Wtime() - timebase, data.i, data.j, data.k);
                        flag_notmyturn = false;
                    }
//crayj<<<
#  endif
#else
//crayj>>>
#pragma omp critical (priority_list)
                    my_turn = TilSch_pop(&priority_list,&data);
#  if defined(CRAYJ_TIMELINE)
                    if (my_turn) {
                        if (NULL != timeline)
                            fprintf(timeline, "%f, pop priority %d %d %d\n",
                                    MPI_Wtime() - timebase, data.i, data.j, data.k);
                        flag_notmyturn = false;
                    }
#  endif
                    if (! my_turn) {
#pragma omp critical (sche_list)
                        my_turn = TilSch_pop(&sche_list,&data);
#  if defined(CRAYJ_TIMELINE)
                        if (my_turn) {
                            if (NULL != timeline)
                                fprintf(timeline, "%f, pop %d %d %d\n",
                                        MPI_Wtime() - timebase, data.i, data.j, data.k);
                            flag_notmyturn = false;
                        }
#  endif
                    }
//crayj<<<
#endif
#if defined(CRAYJ_TIMING)
//crayj>>>
                    ela_taskwait += MPI_Wtime() - ttmp;
                    ++n_taskwait;
                    if (! my_turn) ++n_notmyturn;
//crayj<<<
#endif
#if defined(CRAYJ_TIMELINE)
//crayj>>>
                    if ((! my_turn) && (false == flag_notmyturn)) {
                        if (NULL != timeline) 
                            fprintf(timeline, "%f, not my turn start\n", 
                                    MPI_Wtime() - timebase);
                        flag_notmyturn = true;
                    }
//crayj<<<
#  endif
                    if(my_turn)
                    {
                        g_i = data.i;
                        g_j = data.j;
                        g_k = data.k;
                        l_i = g_i;
                        l_j = g_j/Proc_num;
                        l_k = g_k;
                        __DBGQUEUE__("pop",g_i,g_j,g_k,l_i,l_j,l_k);
						
                        ////////////////////////////////////////////////
                        // GEQRT
                        ////////////////////////////////////////////////
                        if( g_i==g_j && g_j==g_k )
                        {
                            // いらなくなったバッファの消去
#pragma omp critical (buff_pos)
                            {
                                for(ti = buff_head;ti < buff_tail; ti++)
                                {
                                    if(buff_ticket[ti%buff_table_size] == 0)
                                    {
                                        buff_head = ti;
                                    }
                                    else
                                    {
                                        break;
                                    }
                                }
                            }

//                            __DBGKANEL__("dgeqrt",g_i,g_j,g_k);
#if defined(CRAYJ_TIMING)
//crayj>>>
                            ttmp = MPI_Wtime();
//crayj<<<
#endif
#if ! defined(CRAYJ_USE_COREBLAS)
                            tile_dgeqrt(mat_getTileM(status),mat_getTileN(status),__TILE_BLOCK__
                                        ,M_p(A,l_i,l_j),mat_getTileM(status)
                                        ,V_p(TAU,l_i,l_j)
                                        ,geY
                                        ,prT
                                        ,mat_getTileM(status)
                                        ,work
                                );
#else
//crayj>>>
                            retval = CORE_dgeqrt(mat_getTileM(status), mat_getTileN(status), ib,
                                                 M_p(A,l_i,l_j), mat_getTileM(status),
                                                 prT, mat_getTileM(status),
                                                 V_p(TAU,l_i,l_j),
                                                 work
                                );

                            if (0 != retval)
                            {
                                fprintf(stderr, "CRAYJ:[%5d] ERROR: CORE_dgeqrt failed: retval == %d\n",
                                        my_rank, retval);
                                fprintf(stderr, "CRAYJ:[%5d] ERROR: mat_getTileM(status) == %d, mat_getTileN(status) == %d, ib == %d\n", 
                                        my_rank, mat_getTileM(status), mat_getTileN(status), ib);
                                MPI_Abort(MPI_COMM_WORLD, 1);
                            }
//crayj<<<
#endif
#if defined(CRAYJ_TIMING)
//crayj>>>
                            ela_geqrt += MPI_Wtime() - ttmp;
                            ++n_geqrt;
//crayj<<<
#endif
                            // 次のtsqrtのタスク作成
                            if(g_k+1 < mat_getLTileM(status))
                            {
                                input = 0;
                                // プログレステーブルの更新
#pragma omp critical (prog_table)
                                {
                                    *PT_p(g_i,g_j,g_k+1) |= DEP_LIN;
                                    __DBGDEPEND__("depend",g_i,g_j,g_k+1,*PT_p(g_i,g_j,g_k+1));
                                    if(*PT_p(g_i,g_j,g_k+1) == DEP_END)
                                    {
                                        input = 1;
                                    }
                                }
		
                                data.i = g_i;
                                data.j = g_j;
                                data.k = g_k+1;
                                __DBGQUEUE__("push",g_i,g_j,g_k+1,l_i,l_j,l_k+1);
#if ! defined(CRAYJ_TWO_LEVEL_SCHED)
                                while(input&&!TilSch_push(&sche_list,data)){}
#  if defined(CRAYJ_TIMELINE)
//crayj>>>
                                if (input && NULL != timeline)
                                    fprintf(timeline, "%f, push %d %d %d\n", 
                                            MPI_Wtime() - timebase, g_i, g_j, g_k+1);
//crayj<<<
#  endif
#else
//crayj>>>
                                while(input&&!TilSch_push(&priority_list,data)){}
#  if defined(CRAYJ_TIMELINE)
                                if (input && NULL != timeline)
                                    fprintf(timeline, "%f, push priority (2) %d %d %d\n", 
                                            MPI_Wtime() - timebase, g_i, g_j, g_k+1);
#  endif
//crayj<<<
#endif
                            }
                            else
                            {
#pragma omp critical (endflag)
                                run_flag = 0;
#if defined(CRAYJ_USE_OMP_FLUSH)
//crayj>>>
#pragma omp flush
//crayj<<<
#endif
#if defined(CRAYJ_TIMELINE)
//crayj>>>
                                if (NULL != timeline)
                                    fprintf(timeline, "%f, reached end of main loop\n",
                                            MPI_Wtime() - timebase);
//crayj<<<
#endif
                            }

                            // 新たなバッファを取得する
                            buff = NULL;
                            while(buff == NULL)
                            {
#pragma omp critical (buff_pos)
                                {
                                    if(buff_tail-buff_head < buff_table_size)
                                    {
                                        buff_pos_id = buff_tail%buff_table_size;
                                        buff_tail++;
                                        buff = BF(buff_pos_id);
                                        __DBGBUFF__("geqrt set",g_i,g_j,g_i,buff_pos_id);
                                    }
                                }
                            }
                            buff_Y = buff;
                            buff_T = buff+mat_TileSize(status);

#if ! defined(CRAYJ_USE_COREBLAS)
                            cblas_dcopy(mat_TileSize(status),geY,1,buff_Y,1);
#else
                            cblas_dcopy(mat_TileSize(status),M_p(A,l_i,l_j),1,buff_Y,1);
#endif
                            cblas_dcopy(mat_TileSize(status),prT,1,buff_T,1);
                            ticket = 0;
                            for(ti = g_i+1; ti < mat_getGTileN(status); ti++)
                            {
                                if(my_rank == ti%Proc_num)
                                {
                                    ticket++;
                                }
                            }
                            // バッファに登録
#pragma omp critical (buff_pos)
                            {
                                buff_table[buff_pos_id] = BF_ID(g_i,g_j);
                                *BP_p(g_i,g_j) = buff_pos_id;
                                buff_ticket[buff_pos_id] = ticket;
                            }

//							mat_print_tile(stdout,buff_Y,"dgeqrt R",mat_getTileM(status),mat_getTileN(status));
//							mat_print_tile(stdout,buff_T,"dgeqrt T",mat_getTileM(status),mat_getTileN(status));

                            // 次のlarfbのタスク作成
                            for(g_j=g_i+1;g_j < mat_getGTileN(status);g_j++)
                            {
                                if(my_rank == g_j%Proc_num)
                                {
                                    input = 0;
#pragma omp critical (prog_table)
                                    {
                                        *PT_p(g_i,g_j,g_k) |= DEP_YTM;
                                        __DBGDEPEND__("depend",g_i,g_j,g_k,*PT_p(g_i,g_j,g_k));
                                        if(*PT_p(g_i,g_j,g_k) == DEP_END)
                                        {
                                            input = 1;
                                        }
                                    }
                                    data.i = g_i;
                                    data.j = g_j;
                                    data.k = g_k;
                                    l_j = g_j/Proc_num;
                                    __DBGQUEUE__("push",g_i,g_j,g_k,l_i,l_j,l_k);
                                    while(input&&!TilSch_push(&sche_list,data)){}
#if defined(CRAYJ_TIMELINE)
//crayj>>>
                                    if (input && NULL != timeline)
                                        fprintf(timeline, "%f, push %d %d %d\n", 
                                                MPI_Wtime() - timebase, g_i, g_j, g_k);
//crayj<<<
#endif
                                }
                            }
                        }
                        // tsqrt
                        else if(g_i==g_j)
                        {
                            // カーネルの計算
//                            __DBGKANEL__("dtsqrt",g_i,g_j,g_k);
                            //					mat_print_tile(stdout,M_p(A,i,j),"dtsqrt A1",mat_getTileM(status),mat_getTileN(status));
                            //					mat_print_tile(stdout,M_p(A,k,j),"dtsqrt A2",mat_getTileM(status),mat_getTileN(status));
#if defined(CRAYJ_TIMING)
//crayj>>>
                            ttmp = MPI_Wtime();
//crayj<<<
#endif
#if ! defined(CRAYJ_USE_COREBLAS)
                            tile_dtsqrt(mat_getTileM(status),mat_getTileN(status),__TILE_BLOCK__
                                        ,M_p(A,l_i,l_j),mat_getTileM(status)
                                        ,M_p(A,l_k,l_j),mat_getTileM(status)
                                        ,V_p(TAU,l_k,l_j)
                                        ,prT
                                        ,mat_getTileM(status)
                                        ,work
                                );
#else
//crayj>>>
                            retval = CORE_dtsqrt(mat_getTileM(status), mat_getTileN(status), ib,
                                                 M_p(A,l_i,l_j), mat_getTileM(status),
                                                 M_p(A,l_k,l_j), mat_getTileM(status),
                                                 prT, mat_getTileM(status),
                                                 V_p(TAU,l_k,l_j),
                                                 work
                                );
                            if (0 != retval) {
                                fprintf(stderr, "CRAYJ[%5d]: ERROR: CORE_dtsqrt failed: retval == %d\n",
                                        my_rank, retval);
                                MPI_Abort(MPI_COMM_WORLD, 1);
                            }
//crayj<<<
#endif
#if defined(CRAYJ_TIMING)
//crayj>>>
                            ela_tsqrt += MPI_Wtime() - ttmp;
                            ++n_tsqrt;
//crayj<<<
#endif
		
                            // 次のtsqrtの登録
                            if(g_k+1 < mat_getLTileM(status))
                            {
                                input = 0;
                                // プログレステーブルの更新
#pragma omp critical (prog_table)
                                {
                                    *PT_p(g_i,g_j,g_k+1) |= DEP_LIN;
                                    __DBGDEPEND__("depend",g_i,g_j,g_k+1,*PT_p(g_i,g_j,g_k+1));
                                    if(*PT_p(g_i,g_j,g_k+1) == DEP_END)
                                    {
                                        input = 1;
                                    }
                                }
                                data.i = g_i;
                                data.j = g_j;
                                data.k = g_k+1;
                                __DBGQUEUE__("push",g_i,g_j,g_k+1,l_i,l_j,l_k+1);
#if ! defined(CRAYJ_TWO_LEVEL_SCHED)
                                while(input&&!TilSch_push(&sche_list,data)){}
#  if defined(CRAYJ_TIMELINE)
//crayj>>>
                                if (input && NULL != timeline)
                                    fprintf(timeline, "%f, push %d %d %d\n", 
                                            MPI_Wtime() - timebase, g_i, g_j, g_k+1);
//crayj<<<
#  endif
#else
//crayj>>>
                                while(input&&!TilSch_push(&priority_list,data)){}
#  if defined(CRAYJ_TIMELINE)
                                if (input && NULL != timeline)
                                    fprintf(timeline, "%f, push priority (3) %d %d %d\n", 
                                            MPI_Wtime() - timebase, g_i, g_j, g_k+1);
#  endif
//crayj<<<
#endif
                            }
                            // 新たなバッファを取得する
                            buff = NULL;
                            while(buff == NULL)
                            {
#pragma omp critical (buff_pos)
                                {
                                    if(buff_tail-buff_head < buff_table_size)
                                    {
                                        buff_pos_id = buff_tail%buff_table_size;
                                        buff_tail++;
                                        buff = BF(buff_pos_id);
                                        __DBGBUFF__("tsqrt set",g_i,g_j,g_k,buff_pos_id);
                                    }
                                }
                            }
                            buff_Y = buff;
                            buff_T = buff+mat_TileSize(status);
		
                            cblas_dcopy(mat_TileSize(status),M_p(A,l_k,l_j),1,buff_Y,1);
                            cblas_dcopy(mat_TileSize(status),prT,1,buff_T,1);
//							mat_print_tile(stdout,buff_Y,"dtsqrt Y",mat_getTileM(status),mat_getTileN(status));
//							mat_print_tile(stdout,buff_T,"dtsqrt T",mat_getTileM(status),mat_getTileN(status));
                            ticket = 0;
                            for(ti = g_i+1; ti < mat_getGTileN(status); ti++)
                            {
                                if(my_rank == ti%Proc_num)
                                {
                                    ticket++;
                                }
                            }

                            // バッファに登録
#pragma omp critical (buff_pos)
                            {
                                buff_table[buff_pos_id] = BF_ID(g_k,g_j);
                                *BP_p(g_k,g_j) = buff_pos_id;
                                buff_ticket[buff_pos_id] = ticket;
                            }
                            // 次のssrfbのタスク作成
                            for(g_j=g_i+1;g_j < mat_getGTileN(status);g_j++)
                            {
                                if(my_rank == g_j%Proc_num)
                                {
                                    input = 0;
#pragma omp critical (prog_table)
                                    {
                                        *PT_p(g_i,g_j,g_k) |= DEP_YTM;
                                        __DBGDEPEND__("depend",g_i,g_j,g_k,*PT_p(g_i,g_j,g_k));
                                        if(*PT_p(g_i,g_j,g_k) == DEP_END)
                                        {
                                            input = 1;
                                        }
                                    }
                                    data.i = g_i;
                                    data.j = g_j;
                                    data.k = g_k;
                                    l_j = g_j/Proc_num;
                                    __DBGQUEUE__("push",g_i,g_j,g_k,l_i,l_j,l_k);
                                    while(input&&!TilSch_push(&sche_list,data)){}
#if defined(CRAYJ_TIMELINE)
//crayj>>>
                                    if (input && NULL != timeline)
                                        fprintf(timeline, "%f, push %d %d %d\n", 
                                                MPI_Wtime() - timebase, g_i, g_j, g_k);
//crayj<<<
#endif
                                }
                            }
                        }
                        // larfb
                        else if(g_i==g_k)
                        {
                            // バッファの取得
#pragma omp critical (buff_pos)
                            {
//								printf("[%d,%d]BP[%ld]\n",my_rank,omp_get_thread_num(),*BP_p(g_i,g_i));
                                buff = BF(*BP_p(g_i,g_i));
                                __DBGBUFF__("larfb get",g_i,g_j,g_k,*BP_p(g_i,g_i));
                            }
                            buff_Y = buff;
                            buff_T = buff+mat_TileSize(status);
//                            __DBGKANEL__("dlarfb",g_i,g_j,g_k);
//							mat_print_tile(stdout,buff_Y,"dlarfb Y",mat_getTileM(status),mat_getTileN(status));
//							mat_print_tile(stdout,buff_T,"dlarfb T",mat_getTileM(status),mat_getTileN(status));
                            //					mat_print_tile(stdout,M_p(A,i,j),"dlarfb A",mat_getTileM(status),mat_getTileN(status));
						
#if defined(CRAYJ_TIMING)
//crayj>>>
                            ttmp = MPI_Wtime();
//crayj<<<
#endif
#if ! defined(CRAYJ_USE_COREBLAS)
                            tile_dlarfb(mat_getTileM(status),mat_getTileN(status)
                                        ,M_p(A,l_i,l_j),mat_getTileM(status)
                                        ,buff_Y
                                        ,buff_T,mat_getTileM(status)
                                        ,work
                                );
#else
//crayj>>>
                            retval = CORE_dormqr(PlasmaLeft, PlasmaTrans,
                                                 mat_getTileM(status), mat_getTileN(status),
                                                 mat_getTileM(status), ib,
                                                 buff_Y, mat_getTileM(status),
                                                 buff_T, mat_getTileM(status),
                                                 M_p(A,l_i,l_j), mat_getTileM(status),
                                                 work, mat_getTileN(status)
                                );
                            if (0 != retval) {
                                fprintf(stderr, "CRAYJ:[%5d] ERROR: CORE_dormqr failed: retval == %d\n",
                                        my_rank, retval);
                                MPI_Abort(MPI_COMM_WORLD, 1);
                            }
//crayj<<<
#endif
#if defined(CRAYJ_TIMING)
//crayj>>>
                            ela_larfb += MPI_Wtime() - ttmp;
                            ++n_larfb;
//crayj<<<
#endif
                            //					mat_print_tile(stdout,M_p(A,i,j),"dlarfb R",mat_getTileM(status),mat_getTileN(status));
#pragma omp critical (buff_pos)
                            {
                                if(buff_ticket[*BP_p(g_i,g_i)] > 0)
                                    buff_ticket[*BP_p(g_i,g_i)]--;
                                else
                                    printf("buff_ticket error\n");
                            }
                            if(g_k+1 < mat_getLTileM(status))
                            {
                                input = 0;
                                // プログレステーブルの更新
#pragma omp critical (prog_table)
                                {
                                    *PT_p(g_i,g_j,g_k+1) |= DEP_LIN;
                                    __DBGDEPEND__("depend",g_i,g_j,g_k+1,*PT_p(g_i,g_j,g_k+1));
                                    if(*PT_p(g_i,g_j,g_k+1) == DEP_END)
                                    {
                                        input = 1;
                                    }
                                }
                                data.i = g_i;
                                data.j = g_j;
                                data.k = g_k+1;
                                __DBGQUEUE__("push",g_i,g_j,g_k+1,l_i,l_j,l_k+1);
                                while(input&&!TilSch_push(&sche_list,data)){}
#if defined(CRAYJ_TIMELINE)
//crayj>>>
                                if (input && NULL != timeline)
                                    fprintf(timeline, "%f, push %d %d %d\n", 
                                            MPI_Wtime() - timebase, g_i, g_j, g_k+1);
//crayj<<<
#endif
                            }
                        }
                        // ssrfb
                        else
                        {
#pragma omp critical (buff_pos)
                            {
//								printf("[%d,%d]BP[%ld]\n",my_rank,omp_get_thread_num(),*BP_p(g_k,g_i));
                                buff = BF(*BP_p(g_k,g_i));
                                __DBGBUFF__("ssrfb get",g_i,g_j,g_k,*BP_p(g_k,g_i));
                            }
                            buff_Y = buff;
                            buff_T = buff+mat_TileSize(status);
//                            __DBGKANEL__("dssrfb",g_i,g_j,g_k);
//							mat_print_tile(stdout,buff_Y,"dssrfb Y",mat_getTileM(status),mat_getTileN(status));
//							mat_print_tile(stdout,buff_T,"dssrfb T",mat_getTileM(status),mat_getTileN(status));

                            // カーネルの計算
                            //					mat_print_tile(stdout,M_p(A,i,j),"dssrfb A1",mat_getTileM(status),mat_getTileN(status));
                            //					mat_print_tile(stdout,M_p(A,k,j),"dssrfb A2",mat_getTileM(status),mat_getTileN(status));
		
#if defined(CRAYJ_TIMING)
//crayj>>>
                            ttmp = MPI_Wtime();
//crayj<<<
#endif
#if ! defined(CRAYJ_USE_COREBLAS)
/*							tile_dssrfb(mat_getTileM(status),mat_getTileN(status)
                                                        ,M_p(A,l_i,l_j),M_p(A,l_k,l_j),mat_getTileM(status)
                                                        ,buff_Y,buff_T,mat_getTileM(status)
                                                        ,work
							);
*/
                            tile_dssrfb_head(mat_getTileM(status),mat_getTileN(status)
                                             ,M_p(A,l_i,l_j),M_p(A,l_k,l_j),mat_getTileM(status)
                                             ,buff_Y,buff_T,mat_getTileM(status)
                                             ,work
                                );
#else
//crayj>>>
                            retval = CORE_dtsmqr(PlasmaLeft, PlasmaTrans,
                                                 mat_getTileM(status), mat_getTileN(status), mat_getTileM(status),
                                                 mat_getTileN(status), mat_getTileN(status), ib,
                                                 M_p(A,l_i,l_j), mat_getTileM(status),
                                                 M_p(A,l_k,l_j), mat_getTileM(status),
                                                 buff_Y, mat_getTileM(status), buff_T, mat_getTileM(status),
                                                 work, ib
                                );
                            if (0 != retval) {
                                fprintf(stderr, "CRAYJ:[%5d] ERROR: CORE_dtsmqr failed: retval == %d\n",
                                        my_rank, retval);
                                MPI_Abort(MPI_COMM_WORLD, 1);
                            }
//crayj<<<
#endif
#if defined(CRAYJ_TIMING)
//crayj>>>
                            ela_ssrfb += MPI_Wtime() - ttmp;
                            ++n_ssrfb;
//crayj<<<
#endif

                            //					mat_print_tile(stdout,M_p(A,i,j),"dssrfb R1",mat_getTileM(status),mat_getTileN(status));
                            //					mat_print_tile(stdout,M_p(A,k,j),"dssrfb R2",mat_getTileM(status),mat_getTileN(status));

                            // 次のssrfbのタスク作成
                            if(g_k+1 < mat_getLTileM(status))
                            {
                                input = 0;
                                // プログレステーブルの更新
#pragma omp critical (prog_table)
                                {
                                    *PT_p(g_i,g_j,g_k+1) |= DEP_LIN;
                                    __DBGDEPEND__("depend",g_i,g_j,g_k+1,*PT_p(g_i,g_j,g_k+1));
                                    if(*PT_p(g_i,g_j,g_k+1) == DEP_END)
                                    {
                                        input = 1;
                                    }
                                }

                                data.i = g_i;
                                data.j = g_j;
                                data.k = g_k+1;
                                __DBGQUEUE__("push",g_i,g_j,g_k+1,l_i,l_j,l_k+1);
                                while(input&&!TilSch_push(&sche_list,data)){}
#if defined(CRAYJ_TIMELINE)
//crayj>>>
                                if (input && NULL != timeline)
                                    fprintf(timeline, "%f, push %d %d %d\n", 
                                            MPI_Wtime() - timebase, g_i, g_j, g_k+1);
//crayj<<<
#endif

                            }
							
#if ! defined(CRAYJ_USE_COREBLAS)
#  if defined(CRAYJ_TIMING)
//crayj>>>
                            ttmp = MPI_Wtime();
//crayj<<<
#  endif
                            tile_dssrfb_tail(mat_getTileM(status),mat_getTileN(status)
                                             ,M_p(A,l_k,l_j),mat_getTileM(status),buff_Y,work
                                );
#  if defined(CRAYJ_TIMING)
//crayj>>>
                            ela_ssrfb += MPI_Wtime() - ttmp;
//crayj<<<
#  endif
#endif
							
#pragma omp critical (buff_pos)
                            {
                                if(buff_ticket[*BP_p(g_k,g_i)] > 0)
                                    buff_ticket[*BP_p(g_k,g_i)]--;
                                else
                                    printf("buff_ticket error\n");
                            }
                            // メモリ依存解決
                            input = 0;
                            // プログレステーブルの更新
#pragma omp critical (prog_table)
                            {
                                *PT_p(g_i+1,g_j,g_k) |= DEP_MEM;
                                if(*PT_p(g_i+1,g_j,g_k) == DEP_END)
                                {
                                    input = 1;
                                }
                            }
                            data.i = g_i+1;
                            data.j = g_j;
                            data.k = g_k;
                            __DBGQUEUE__("push",g_i+1,g_j,g_k,l_i,l_j,l_k);
#if ! defined(CRAYJ_TWO_LEVEL_SCHED)
                            while(input&&!TilSch_push(&sche_list,data)){}
#  if defined(CRAYJ_TIMELINE)
//crayj>>>
                            if (input && NULL != timeline)
                                fprintf(timeline, "%f, push %d %d %d\n", 
                                        MPI_Wtime() - timebase, g_i+1, g_j, g_k);
//crayj<<<
#  endif
#else
//crayj>>>
                            if (data.i == data.j) {
                                while(input&&!TilSch_push(&priority_list,data)){}
#  if defined(CRAYJ_TIMELINE)
                                if (input && NULL != timeline)
                                    fprintf(timeline, "%f, push priority (4) %d %d %d\n", 
                                            MPI_Wtime() - timebase, g_i+1, g_j, g_k);
#  endif
                            } else {
                                while(input&&!TilSch_push(&sche_list,data)){}
#  if defined(CRAYJ_TIMELINE)
                                if (input && NULL != timeline)
                                    fprintf(timeline, "%f, push normal (4) %d %d %d\n", 
                                            MPI_Wtime() - timebase, g_i+1, g_j, g_k);
#  endif
                            }
//crayj<<<
#endif
                        }
                    }
                    // 終了判定
#pragma omp critical (endflag)
                    {
#if defined(CRAYJ_USE_OMP_FLUSH)
//crayj>>>
#pragma omp flush
//crayj<<<
#endif
                        my_flag = run_flag;
                    }
#if defined(CRAYJ_TIMELINE)
//crayj>>>
                    if (0 == my_flag) {
                        if (NULL != timeline) 
                            fprintf(timeline, "%f, found end flag\n",
                                    MPI_Wtime() - timebase);
                    }

                    if (NULL != timeline) fflush(timeline);
//crayj<<<
#endif
                }
            }
//			TilSch_end(&sche_list);
            __DEBUGOUT__("end");
        }
#if defined(CRAYJ_TIMING)
//crayj>>>
        ela_total = MPI_Wtime() - ela_total;
        crayj_elapsed[tid][0] = ela_geqrt;
        crayj_elapsed[tid][1] = ela_tsqrt;
        crayj_elapsed[tid][2] = ela_larfb;
        crayj_elapsed[tid][3] = ela_ssrfb;
        crayj_elapsed[tid][4] = ela_taskwait;
        crayj_elapsed[tid][5] = ela_total;
        crayj_elapsed[tid][6] = ela_send;
        crayj_elapsed[tid][7] = ela_recv;
        crayj_ncalls[tid][0] = n_geqrt;
        crayj_ncalls[tid][1] = n_tsqrt;
        crayj_ncalls[tid][2] = n_larfb;
        crayj_ncalls[tid][3] = n_ssrfb;
        crayj_ncalls[tid][4] = n_taskwait;
        crayj_ncalls[tid][5] = n_notmyturn;
        crayj_ncalls[tid][6] = n_send;
        crayj_ncalls[tid][7] = n_recv;
//crayj<<<
#endif
#if defined(CRAYJ_TIMELINE)
//crayj>>>
        if (NULL != timeline) fclose(timeline);
//crayj<<<
#endif

        free(prT);
        free(geY);

    }

    free(T);
    free(glob_YT);
    free(buff_ticket);
    free(buff_table);

    free(buff_pos);
    free(prog_table);

#if defined(CRAYJ_TIMING)
//crayj>>>
    {   FILE *fp;
	if (my_rank == 0) {
            fprintf(stderr, "CRAYJ[%d]: REMARK: "
                    "timing info output to stdout.\n",
                    my_rank);
	}
	fp = stdout;

	if (NULL != fp) {
            tag = 100;

            if (my_rank == 0) {

                fprintf(fp, "CRAYJ:[%5d] Kernel Timing [sec.]\n", my_rank);
                fprintf(fp, "CRAYJ:, %6s, %3s, %10s, %10s, %10s, %10s, %10s, %10s\n",
                        "rank", "tid", "GEQRT", "TSQRT", "LARFB", "SSRFB", "TASKWAIT", "TOTAL");

            }

            if (my_rank != 0) {
                MPI_Recv(&dummy, 1, MPI_INT, my_rank-1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (tid = 0; tid < omp_get_max_threads(); ++tid) {
                fprintf(fp, "CRAYJ:, %6d, %3d, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f\n",
                        my_rank, tid,
                        crayj_elapsed[tid][0], crayj_elapsed[tid][1], crayj_elapsed[tid][2],
                        crayj_elapsed[tid][3], crayj_elapsed[tid][4], crayj_elapsed[tid][5]
                    );
            }
            fflush(fp);
            if (my_rank != Proc_num-1) {
                MPI_Send(&dummy, 1, MPI_INT, my_rank+1, tag, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);

            if (my_rank == 0) {
                fputc('\n', fp);
                fprintf(fp, "CRAYJ:[%5d] The number of calls\n", my_rank);
                fprintf(fp, "CRAYJ:, %6s, %3s, %10s, %10s, %10s, %10s, %10s, %10s\n", "rank", "tid", 
                        "GEQRT", "TSQRT", "LARFB", "SSRFB", "TASKWAIT", "NOTMYTURN");
            }

            if (my_rank != 0) {
                MPI_Recv(&dummy, 1, MPI_INT, my_rank-1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (tid = 0; tid < omp_get_max_threads(); ++tid) {
                fprintf(fp, "CRAYJ:, %6d, %3d, %10d, %10d, %10d, %10d, %10d, %10d\n",
                        my_rank, tid,
                        crayj_ncalls[tid][0], crayj_ncalls[tid][1], crayj_ncalls[tid][2],
                        crayj_ncalls[tid][3], crayj_ncalls[tid][4], crayj_ncalls[tid][5]
                    );
            }
            fflush(fp);
            if (my_rank != Proc_num-1) {
                MPI_Send(&dummy, 1, MPI_INT, my_rank+1, tag, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);

            if (my_rank == 0) {
                fputc('\n', fp);
                fprintf(fp, "CRAYJ:[%5d] MPI_Bcast Timing [sec.]\n", my_rank);
                fprintf(fp, "CRAYJ:, %6s, %10s, %10s, %10s, %10s\n", 
                        "rank", "send time", "send count", "recv time", "recv count");
            }

            if (my_rank != 0) {
                MPI_Recv(&dummy, 1, MPI_INT, my_rank-1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            fprintf(fp, "CRAYJ:, %6d, %10.6f, %10d, %10.6f, %10d\n", my_rank, 
                    crayj_elapsed[0][6], crayj_ncalls[0][6], 
                    crayj_elapsed[0][7], crayj_ncalls[0][7]);
            fflush(fp);
            if (my_rank != Proc_num-1) {
                MPI_Send(&dummy, 1, MPI_INT, my_rank+1, tag, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);

	} // end if (NULL != fp)

    } // end FILE *fp;
//crayj<<<
#endif
}

