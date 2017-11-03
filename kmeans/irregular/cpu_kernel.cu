#include <iostream>
using namespace std;
#define CPU_KERNEL
#include "parameters.h"
#include "irregular_runtime.h"
#include "../irregular.h"
#include "cpu_args.h"
#include <math.h>
#include <stdio.h>
#include "roc.cu"
#include "data_type.h"

void *compute_cpu(void *arg)
{
    CpuArgs *args = (CpuArgs *)arg;
    IrregularRuntime *runtime = args->runtime; 
    int tid = args->tid;

    partition_cpu *p_cpu = runtime->p_cpu_;

    int number_parts = p_cpu->my_num_parts();
    int number_edges = p_cpu->my_num_edges();
    //pthread_mutex_t *mutex = args->mutex;
    int task_index = 0;
    Part *parts = p_cpu->my_parts();

    int *part_index = runtime->part_index;

    void *point_data = p_cpu->my_node_data();
    void *edge_data = p_cpu->my_edges();

    Cobject *object = &(runtime->roc_)[tid];
    pthread_mutex_t *mutex = args->mutex;

    void *parameter = runtime->parameter_;

    struct part new_part;

    int map_idx = runtime->map_idx_;
    int reduce_idx = runtime->reduce_idx_;

    FCPU::irmap_fp map_ptr = FCPU::map_function_table[map_idx];
    FCPU::irreduce_fp reduce_ptr = FCPU::reduce_function_table[reduce_idx];

    while(1)
    {
    	pthread_mutex_lock(mutex);
    	task_index = *(runtime->cpu_edge_offset_);
	*(runtime->cpu_edge_offset_) += CPU_BLOCK_SIZE;
    	pthread_mutex_unlock(mutex); 

    	if(task_index >= number_edges)
    	{
    	    break;
    	}

    	int start = task_index;
    	int end = ((task_index + CPU_BLOCK_SIZE - 1) > (number_edges - 1))?(number_edges-1):(start + CPU_BLOCK_SIZE-1);

	//printf("start: %d end: %d\n", start, end);

    	for(int j = start; /*j < (start+CPU_BLOCK_SIZE) && j < number_edges*/j<=end; j++)
    	{
            //FCPU::map(object, point_data, part_index, edge_data, parameter, j, 0);  
            map_ptr(object, point_data, part_index, edge_data, parameter, j, 0, reduce_ptr);  
    	}
    }

    return (void *)0;
}

void *init_roc(void *arg)
{
	struct init_args *args = (struct init_args *)arg;
	(args->roc)->init(args->tid);	
	return (void *)0;
}

void *mergetc(void *arg)
{
	struct merge_args *args = (struct merge_args *)arg;
	(args->roc1)->mergec(args->roc2, args->tid);
	return (void *)0;
}
