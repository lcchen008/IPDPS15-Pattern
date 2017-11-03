OPT=
MPI_HOME=/usr/local/mvapich2/1.7-intel
CFLAGS=$(OPT) -O3 -I . -I ./lib -std=c++0x
NVFLAGS=$(OPT) -O3 -arch=sm_20 -lpthread -I . -I$(MPI_HOME)/include/  -L$(MPI_HOME)/lib
LDFLAGS=$(OPT) -lpthread -I . -I ./lib
#CXX=/usr/mpi/gccmpicxx
#NVCC=/usr/local/cuda/bin/nvcc 
NVCC=/usr/local/cuda/4.2.9/bin/nvcc
#NVCC=/usr/local/cuda/5.0.35/bin/nvcc
#MPICC=mpicc
CXX=mpicxx
CC=gcc

LIBS = -lm

OBJS=ljs.o \
     input.o\
     atom.o\
     neighbor.o\
     thermo.o\
     force.o\
     setup.o\
     misc.o\
     timer.o\
     include/runtime.o\
     lib/buffer.o \
     lib/buffer_cuda.o\
     lib/grid_cpu.o\
     lib/grid_view_cpu.o\
     lib/grid_cuda.o\
     lib/grid_view_cuda.o\
     lib/data_util.o\
     lib/grid.o\
     lib/grid_mpi.o\
     lib/grid_view.o\
     lib/stencil_runtime.o\
     lib/cu_util.o\
     lib/time_util.o\
     lib/compute_cpu.o\
     irregular/partition_view.o\
     irregular/irregular_runtime.o\
     irregular/partition_cuda.o\
     irregular/cpu_kernel.o\
     irregular/cpu_kernel_tools.o\
     irregular/partition_util.o\
     regular/data_partition_mpi.o\
     regular/splitter.o\
     regular/cpu_kernel.o\
     regular/cpu_kernel_tools.o\
     regular/regular_runtime.o
	
.PHONY: main 

main: $(OBJS)
	$(NVCC) $(NVFLAGS) -o main $(OBJS) -lmpich -lmpl
%.o : %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $*.o
%.o : %.cpp
	$(CXX) $(CFLAGS) -c $< -o $*.o
%.o : %.c
	$(CC) $(CFLAGS) -DBLOCK_ADPATIVE -c $< -o $*.o $(LIBS) 

 

clean:
	-rm main $(OBJS)
