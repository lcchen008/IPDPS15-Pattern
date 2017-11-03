#include "regular_runtime.h"
#include "../lib/cu_util.h"
#include "args.h"
#include "cpu_kernel.h"
#include "gpu_kernel.cu"
#include "data_type.h"
#include <mpi.h>
#include "../lib/macro.h"
#include <stdio.h> 
#include "../lib/time_util.h"

RegularRuntime::RegularRuntime(
	int num_procs,
	void *input,
	size_t input_size,
	Offset *offsets,
	size_t num_offsets,
	void *parameters,
	int parameter_size
):num_procs_(num_procs),
input_(input), 
input_size_(input_size), 
offsets_(offsets), 
num_offsets_(num_offsets),
parameter_(parameters),
parameter_size_(parameter_size)
{
		
}

void RegularRuntime::RegularInit()
{
	//MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);	

	this->num_gpus_ = GetGPUNumber();

	this->num_devices_ = num_gpus_ + 1;

	splitter_ = new splitter(num_procs_, my_rank_, input_, input_size_, offsets_, num_offsets_, parameter_, parameter_size_);

	dp_mpi_ = splitter_->gen_partition();

	input_pin_ = dp_mpi_->input_pin();

	offsets_pin_ = dp_mpi_->offset_pin();

	//create GPU reduction objects
	rog_ = (GO **)malloc(sizeof(GO *)*num_gpus_);

	//create CPU reduction objects
	roc_ = (CO *)malloc(sizeof(CO) * CPU_THREADS);
	memset(roc_, 0, sizeof(CO)*CPU_THREADS);

	for(int i = 0; i < CPU_THREADS; i++)
		roc_[i].num_buckets = R_NUM_BUCKETS_G;
	
	//init offsets
	task_offset_ = (size_t *)malloc(sizeof(size_t));	
	*task_offset_ = 0;

	mutex = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));

	pthread_mutex_init(mutex, NULL);

	//default map and reduce functions indices are 0
	map_idx_ = 0;
	reduce_idx_ = 0;

	//create device related things parameters
	parameter_d_ = (void **)malloc(sizeof(void *) * num_gpus_);
	device_offsets_h = (int **)malloc(sizeof(int *)*num_gpus_);
	device_offsets_d = (int **)malloc(sizeof(int *)*num_gpus_);
	streams_ = (cudaStream_t **)malloc(sizeof(cudaStream_t *)*num_gpus_);
	//input_buffer_d_ = (void **)malloc(sizeof(void *)*num_gpus_);
	//offset_buffer_d_ = (Offset **)malloc(sizeof(Offset *)*num_gpus_);


	GO *rogh = (GO *)malloc(sizeof(GO));
	memset(rogh, 0, sizeof(GO));
	rogh->num_buckets = R_NUM_BUCKETS_G;

	//init offsets
	for(int i = 0; i < GPU_THREADS * GPU_BLOCKS/WARP_SIZE; i++)
	{
		rogh->offsets[i] = GLOBAL_POOL_SIZE * i / (GPU_THREADS * GPU_BLOCKS / WARP_SIZE);	
	}

	for(int i = 0; i < num_gpus_; i++)
	{
		CUDA_SAFE_CALL(cudaSetDevice(i));
		CUDA_SAFE_CALL(cudaMalloc(&parameter_d_[i], parameter_size_));
		CUDA_SAFE_CALL(cudaMemcpy(parameter_d_[i], parameter_, parameter_size_, cudaMemcpyHostToDevice));

		streams_[i] = (cudaStream_t *)malloc(sizeof(cudaStream_t) * 2);

		cudaStreamCreate(&streams_[i][0]);
		cudaStreamCreate(&streams_[i][1]);
	
		cudaHostAlloc(&device_offsets_h[i], sizeof(int)*2, cudaHostAllocPortable|cudaHostAllocMapped);
	
		device_offsets_h[i][0] = 0;
		device_offsets_h[i][1] = 0;

		cudaMalloc(&device_offsets_d[i], sizeof(int)*2);
		
		//CUDA_SAFE_CALL(cudaMalloc(&input_buffer_d_[i], input_size_/16*2));	

		//CUDA_SAFE_CALL(cudaMalloc(&offset_buffer_d_[i], sizeof(Offset)*num_offsets_/16*2));	
		//CUDA_SAFE_CALL(cudaMalloc(&rog_[i], sizeof(GO)*2));
		
		CUDA_SAFE_CALL(cudaMalloc((void **)&rog_[i], sizeof(GO)));
		CUDA_SAFE_CALL(cudaMemcpy(rog_[i], rogh, sizeof(GO), cudaMemcpyHostToDevice));

		//allocate peer device rog buffers
		if(i==0)
		{
			CUDA_SAFE_CALL(cudaMalloc((void **)&rog_peer_, sizeof(GO)));
			CUDA_SAFE_CALL(cudaMemcpy(rog_peer_, rogh, sizeof(GO), cudaMemcpyHostToDevice));
		}
		//CUDA_SAFE_CALL(cudaMemcpy(rog_[i] + 1, rogh, sizeof(GO), cudaMemcpyHostToDevice));
	}

	free(rogh);
}

void *RegularRuntime::start_cpu(void *arg)
{
	RegularRuntime *runtime = (RegularRuntime *)arg;	
	struct cpu_args_reg args[CPU_THREADS];	
	pthread_t tid[CPU_THREADS];

	double before_cpu = rtclock();
	for(int i = 0; i < CPU_THREADS; i++)
	{
		args[i].tid = i;	
		args[i].runtime = runtime;

		pthread_create(&tid[i], NULL, compute_cpu_reg, &args[i]);
	}

	for(int j = 0; j < CPU_THREADS; j++)
	{
		pthread_join(tid[j], NULL);
	}

	double after_cpu = rtclock();

	printf("%d CPU time: %f\n", runtime->my_rank_,  after_cpu - before_cpu);

	return (void *)0;
}

void *RegularRuntime::start_gpu(void *arg)
{
	return (void *)0;
}

void RegularRuntime::RegularStart()
{
	pthread_t tid_cpu;	

	//create CPU thread
	pthread_create(&tid_cpu, NULL, start_cpu, this);

	size_t total_offsets = dp_mpi_->num_offsets();

	dim3 grid(GPU_BLOCKS, 1, 1);		
	dim3 block(GPU_THREADS, 1, 1);

	int count = 0;

	printf("O+O+O+O+O+total offsets: %d %d\n", my_rank_, total_offsets);

	double before_gpu = rtclock();
	while(*task_offset_ < total_offsets)
	{
		for(int i = 0; i < num_gpus_; i++)
		{
			CUDA_SAFE_CALL(cudaSetDevice(i));

			int size;
			size_t start;

			//int starts[2]; //starting number for each stream
			//int sizes[2]; //size for each stream
			
			//int input_start[2];
			//int input_end[2];
			//int input_size[2];

			pthread_mutex_lock(mutex);	

			start = *task_offset_;	

			if(start >= total_offsets)
			{
				pthread_mutex_unlock(mutex);
				break;
			}

			int remain = total_offsets - start;

			*task_offset_ += R_GPU_PREALLOC_SIZE; 

			size = *task_offset_ < total_offsets ? R_GPU_PREALLOC_SIZE : remain;

			pthread_mutex_unlock(mutex);

			int tmp = 0;
			CUDA_SAFE_CALL(cudaMemcpy(device_offsets_d[i], &tmp, sizeof(int), cudaMemcpyHostToDevice));

			//TODO: launch kernel for stream 0
			compute_gpu<<<grid, block>>>
			(
				input_pin_,		

				offsets_pin_,

				device_offsets_d[i],

				size,

				start,

				rog_[i],

				parameter_d_[i],

				map_idx_,

				reduce_idx_	
			);
		}

		count++;

		if(count%4==0)
		{
			for(int i = 0; i < num_gpus_; i++)
			{
				CUDA_SAFE_CALL(cudaSetDevice(i));
				CUDA_SAFE_CALL(cudaDeviceSynchronize());	
			}
		}
	}

	if(count%4!=0)
	{
		for(int i = 0; i < num_gpus_; i++)
		{
			CUDA_SAFE_CALL(cudaSetDevice(i));
			CUDA_SAFE_CALL(cudaDeviceSynchronize());	
		}
	}

	double after_gpu = rtclock();

	printf("%d gpu time: %f\n", my_rank_, after_gpu - before_gpu);

	//start GPUs
	
	//join GPU threads
	//for(int i = 0; i < num_gpus_; i++)
	//{
	//	pthread_join(tid_gpu[i], NULL);		
	//}

	//join CPU thread
	pthread_join(tid_cpu, NULL);
	//merge_device();
}

struct output RegularRuntime::RegularGetOutput()
{
	
}

void RegularRuntime::merge_device()
{
	//first, merge gpu objects. Send objects from gpus to gpu0	
	for(int i = 1; i < num_gpus_; i++)
	{
		CUDA_SAFE_CALL(cudaSetDevice(0));
		CUDA_SAFE_CALL(cudaMemcpyPeer(rog_peer_, 0, rog_[i], i, sizeof(GO)));
		merge_two_gpu_objects(rog_[0], rog_peer_);
	}

	//next, merge cpu object to here.
	CUDA_SAFE_CALL(cudaMemcpy(rog_peer_, roc_, sizeof(GO), cudaMemcpyHostToDevice));
	merge_two_gpu_objects(rog_[0], rog_peer_);
}

void RegularRuntime::merge_two_gpu_objects(GO *object1, GO *object2)
{
	dim3 grid(GPU_BLOCKS, 1, 1);		
	dim3 block(GPU_THREADS, 1, 1);

	merged<<<grid, block>>>
	(
		object1,
		object2,
		reduce_idx_
	);

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void RegularRuntime::merge_nodes()
{
	for(int i = 2; i <= num_procs_; i*=2)	
	{
		int step = i/2;
		//receiver
		if(my_rank_ % i == 0)		
		{
			int sender = my_rank_ + step;	
		}
		//sender
		if((my_rank_ - step) > 0 && ((my_rank_ - step)%i == 0))
		{
			int receiver = my_rank_ - step;	
		}
	}
}

void RegularRuntime::set_map_idx(int idx)
{
	this->map_idx_ = idx;
}

void RegularRuntime::set_reduce_idx(int idx)
{
	this->reduce_idx_ = idx;
}
