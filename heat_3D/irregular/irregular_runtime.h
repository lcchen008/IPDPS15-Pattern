#ifndef IRREGULAR_IRREGULAR_RUNTIME_H_
#define IRREGULAR_IRREGULAR_RUNTIME_H_
#include "partition_view.h"
#include "partition_mpi.h"
#include "../lib/common.h"
#include "partition_cpu.h"
#include "partition_cuda.h"
#include "rog.h"
#include "roc.h"
#include "reorder.h"

class IrregularRuntime
{
public:
	partition_view *pv_;
	partition_mpi *p_mpi_;
	partition_cpu *p_cpu_;
	partition_cuda **p_cuda_;

	IRIndex global_num_edges_;
	IRIndex  global_num_nodes_;
	EDGE *global_edges_;
	void *global_edge_data_;
	void *global_node_data_;
	int edge_data_elm_size_;
	int node_data_elm_size_;
	int reduction_elm_size_;

	int *device_node_start_;
	int *device_node_sum_;

	int node_num_dims_;
	void *node_coordinates_;
	int coordinate_size_;

	//contains the reduction results from each device
	void *reduction_result_;

	int map_idx_;
	int reduce_idx_;

	//reduction objects for reductions
        Cobject *roc_;	//cpu reduction objects
        Gobject **rog_; //gpu reduction objects

	int **task_offset_d_;

	int my_rank_;
	int num_procs_;

	int current_iter_;
	int num_iters_;

	int num_gpus_;
	int num_devices_;
	double *speeds_;

	int *cpu_edge_offset_;

	pthread_t tids_[CPU_THREADS];

	int *part_index;
	int *part_index_d;	

	int parameter_size_;
	void *parameter_;
	void **parameter_d_;

	Reorder *partitioner_;

	IrregularRuntime(IRIndex num_edges, 
			IRIndex num_nodes, 
			EDGE *edges,
			void *edge_data, 
			void *node_data,
			int edge_data_elm_size,
			int node_data_elm_size,
			int reduction_elm_size,
			int node_num_dims,
			void *node_coordinates,
			int coordinate_size,
			void *parameter,
			int parameter_size,
			int num_procs,
			int num_iters);

	void IrregularInit();

	void split();

	static void *launch(void *arg);

	void IrregularStart(); 

	void IrregularBarrier();

	void mergecg();

	void merge_cobjects();

	void get_reduction_result(void *buffer);

	void reset_node_data(void *node_data);

	void re_init();

	~IrregularRuntime();

	void set_map_idx(int idx);

	void set_reduce_idx(int idx);
};

#endif
