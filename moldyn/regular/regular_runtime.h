#ifndef REGULAR_RUNTIME_H_
#define REGULAR_RUNTIME_H_

#include <pthread.h>
#include "data_type.h"
#include "data_partition_mpi.h"
#include "splitter.h"
#include "roc.h"
#include "rog.h"

class RegularRuntime
{
public:
	int num_devices_;
	int num_gpus_;
	int num_procs_;
	size_t num_offsets_;
	int my_rank_;

	size_t input_size_;
	int parameter_size_;

	int map_idx_;
	int reduce_idx_;

	//indicates if the reduction should be reduced to one value
	bool scaler;

	void *input_;
	Offset *offsets_;
	void *parameter_;	
	//device parameters for each device
	void **parameter_d_;

	//my_input in pinned memory
	void *input_pin_;
	Offset *offsets_pin_;

	splitter *splitter_;
	data_partition_mpi * dp_mpi_;

	CO *roc_;
	GO **rog_;

	//gpu0's buffer used to store rogs from peer gpus
	GO *rog_peer_;

	//task offset for devices
	int **device_offsets_h;
	int **device_offsets_d;

	//streams for devices
	cudaStream_t **streams_;

	//global offset for task scheduling
	size_t *task_offset_;

	//buffers for input
	void **input_buffer_d_;	
	//buffers for data offsets
	Offset **offset_buffer_d_;

	//global mutex for scheduling
	pthread_mutex_t *mutex;

	unsigned int total_key_num_;
        unsigned int total_key_size_;
        unsigned int total_value_num_;
        unsigned int total_value_size_;

	struct output reduction_output;

	void split();

	static void *start_gpu(void *arg);

	static void *start_cpu(void *arg);

	RegularRuntime(
			int num_procs,
			void *input,
			size_t input_size,
			Offset *offsets,
			size_t offset_number,
			void *parameters,
			int parameter_size
			);

	void RegularInit();

	void RegularStart();

	struct output RegularGetOutput();

	//merges reduction objects within each device
	void merge_device();

	//merges reduction objects from different nodes
	void merge_nodes();

	//merges two device mem reduction objects
	void merge_two_gpu_objects(GO *object1, GO *object2);

	void set_map_idx(int idx);
	void set_reduce_idx(int idx);
};

#endif
