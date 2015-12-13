BLAS_ROOT = /opt/OpenBLAS
BLAS_INC_DIR = $(BLAS_ROOT)/include
BLAS_LIB_DIR = $(BLAS_ROOT)/lib
BLAS_LIBS = -lopenblas_seq
#
PLASMA_ROOT = /opt/PLASMA
PLASMA_INC_DIR = $(PLASMA_ROOT)/include
PLASMA_LIB_DIR = $(PLASMA_ROOT)/lib
PLASMA_LIBS = -lplasma -lcoreblas -lquark
#
VT_INC = /usr/include/openmpi-x86_64/vampirtrace
#
# for Debug
CC = mpicc -fopenmp
CFLAGS = -I$(BLAS_INC_DIR) -I$(PLASMA_INC_DIR) 

# for Performance Evaluation
#CFLAGS = -O3 -I$(BLAS_INC_DIR) -I$(PLASMA_INC_DIR)

# for Trace
#CC = vtcc -vt:cc mpicc -vt:inst manual -fopenmp
#CFLAGS = -DVTRACE -I$(BLAS_INC_DIR) -I$(PLASMA_INC_DIR) -I$(VT_INC)

#CFLAGS += -DCRAYJ_VERBOSE -DCRAYJ_DYNAMIC_COMM_SCHED -DCRAYJ_FREE_HEAP
CFLAGS += -DCRAYJ_DYNAMIC_COMM_SCHED -DCRAYJ_USE_COREBLAS -DCRAYJ_REMOVE_REDUNDANT_DLARFT -DCRAYJ_FREE_HEAP -DDEBUG -DLOG

OBJS = main.o Hybrid_tile_QR.o Hybrid_make_Q.o Hybrid_make_I.o Hybrid_dgemm.o Hybrid_dorgqr.o Hybrid_dormqr.o TileSch.o mat_prod.o tile_kernel.o

all: Hybrid3

Hybrid3 : $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) -L$(PLASMA_LIB_DIR) $(PLASMA_LIBS) -L$(BLAS_LIB_DIR) $(BLAS_LIBS)  

.c.o :
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f Makefile~ *.c~ *.h~ *.o
