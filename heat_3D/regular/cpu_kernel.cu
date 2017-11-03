#ifndef CPU_KERNEL
#define CPU_KERNEL
#endif

//#include <iostream>
//using namespace std;
#include "parameters.h"
#include "regular_runtime.h"
#include "args.h"
#include <math.h>
#include <stdio.h>
#include "roc.cu"
#include "data_type.h"
#include "args.h"
#include "../regular.h"

void *compute_cpu_reg(void *arg)
{
    struct cpu_args_reg *args = (struct cpu_args_reg *)arg;
    RegularRuntime *runtime = args->runtime; 
    int tid = args->tid;

    size_t offset_number = runtime->dp_mpi_->num_offsets();
	
    //if(tid == 0)
    //	printf("=====================>offset number: %d\n", offset_number);

    pthread_mutex_t *mutex = runtime->mutex;
    int task_index = 0;
    int size = 0;
    int remain = 0;
    size_t *device_offset = runtime->task_offset_; //used for competing

    //void *input = runtime->input_pin_;
    //Offset *data_offset = runtime->offsets_pin_;
    void *input = runtime->input_pin_;
    Offset *data_offset = runtime->offsets_pin_;

    CO *objects = runtime->roc_;

    int map_idx = runtime->map_idx_;
    int reduce_idx = runtime->reduce_idx_;

    FFCPU::map_fp map_ptr = FFCPU::map_function_table[map_idx];
    FFCPU::reduce_fp reduce_ptr = FFCPU::reduce_function_table[reduce_idx];

    while(1)
    {
        pthread_mutex_lock(mutex);

        task_index = *device_offset;

        *device_offset +=  R_CPU_BLOCK_SIZE;

	remain = offset_number - task_index;
	
	size = *device_offset < offset_number? R_CPU_BLOCK_SIZE:remain;

	//printf("=================> CPU task idx: %d, size: %d, offset_number: %d\n", task_index, size, offset_number);

        pthread_mutex_unlock(mutex);

        if(task_index >= offset_number)
        {
            break; 
        }

        for(int i = 0; i < size; i++)
        {
            //FFCPU::map(&objects[tid], input, data_offset[task_index + i].offset, runtime->parameter_, 0);  
            map_ptr(&objects[tid], input, data_offset[task_index + i].offset, runtime->parameter_, 0, reduce_ptr);  
        }
    }

    //printf("OOOO+++++OOOOO++++\n");

    if(tid != 0)
        objects[0].merge(&objects[tid], reduce_ptr);

    return (void *)0;
}
