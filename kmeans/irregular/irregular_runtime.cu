#include "partitioner.cpp"
#include "irregular_runtime.h"
#include "../lib/cu_util.h"
#include "partition_cuda.h"
#include "partition_cpu.h"
#include "cpu_args.h"
#include "parameters.h"
#include "gpu_kernel.cu"
#include <mpi.h>
#include "../lib/macro.h"
#include "cpu_kernel.h"
#include <stdio.h>
#include "reorder.cpp"
#include "../lib/time_util.h"
#include <math.h>

IrregularRuntime::IrregularRuntime(IRIndex num_edges, 
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
			int num_iters):
			global_num_edges_(num_edges),
			global_num_nodes_(num_nodes),
			global_edges_(edges),
			global_edge_data_(edge_data),
			global_node_data_(node_data),
			edge_data_elm_size_(edge_data_elm_size),
			node_data_elm_size_(node_data_elm_size),
			reduction_elm_size_(reduction_elm_size),
			node_num_dims_(node_num_dims),
			node_coordinates_(node_coordinates),
			coordinate_size_(coordinate_size),
			num_procs_(num_procs),
			num_iters_(num_iters),
			parameter_size_(parameter_size),
			current_iter_(0),
			parameter_(parameter)
{}

void IrregularRuntime::IrregularInit()
{
	//=====reorder input data to reduce crossing edges
	int *partitions = (int *)malloc(sizeof(int)*node_num_dims_);
	memset(partitions, 0, sizeof(int)*node_num_dims_);
	partitions[0] = num_procs_;

	partitioner_ = new Reorder(global_num_nodes_,
	global_num_edges_,
	node_num_dims_, 
	partitions);

	if(coordinate_size_==4)
		partitioner_->partition<float>((float *)node_coordinates_);
	else if(coordinate_size_==8)
		partitioner_->partition<double>((double *)node_coordinates_);

	partitioner_->reorder_edges(global_edges_);

	partitioner_->reorder_satellite_data(global_node_data_, node_data_elm_size_);

	//=========

	//MPI_Init(&argc, &argv);	

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);

	printf("my rank is: %d\n", my_rank_);

	MPI_Barrier(MPI_COMM_WORLD);

	this->num_gpus_ = GetGPUNumber();

	this->num_devices_ = num_gpus_ + 1;

	pv_ = new partition_view(num_procs_, 
	my_rank_, 
	global_num_nodes_, 
	global_num_edges_, 
	global_edges_, 
	global_edge_data_, 
	global_node_data_, 
	edge_data_elm_size_, 
	node_data_elm_size_,
	reduction_elm_size_);

	//printf("pv_ created...\n");

	//create per node partition
	p_mpi_ = pv_->CreatePartition();	

	MPI_Barrier(MPI_COMM_WORLD);

	//printf("p_mpi created...\n");

	cudaHostAlloc((void **)&part_index, sizeof(int) * p_mpi_->my_num_nodes(), cudaHostAllocPortable|cudaHostAllocMapped);

	cudaHostGetDevicePointer((void **)&part_index_d, part_index, 0);

	//init part_index

	//allocate reduction result space
	reduction_result_ = malloc(reduction_elm_size_ * p_mpi_->my_num_nodes()); 

	p_cpu_ = NULL;

	//allocate gpu partiiton pointers
	p_cuda_ = (partition_cuda**)malloc(num_gpus_ * sizeof(partition_cuda*));
	memset(p_cuda_, 0, num_gpus_ * sizeof(partition_cuda *));
	
	speeds_ = (double *)malloc(sizeof(double) * num_devices_);

	device_node_start_ = (int *)malloc(sizeof(int)*num_devices_);
	device_node_sum_ = (int *)malloc(sizeof(int)*num_devices_);

	//initial speeds are equal
	for(int i = 0; i < num_devices_; i++)
	{
		speeds_[i] = 1;	
	}

	//initialize gpu reduction objects
	rog_ = (Gobject **)malloc(sizeof(Gobject *)*num_gpus_);
	task_offset_d_ = (int **)malloc(sizeof(int *)*num_gpus_);
	parameter_d_ = (void **)malloc(sizeof(void *)*num_gpus_);

	map_idx_ = 0;
	reduce_idx_ = 0;

	Gobject *rogh = (Gobject *)malloc(sizeof(Gobject));

	for(int i = 0; i < NUM_BUCKETS_G; i++)
	{
		(rogh->keys)[i] = EMPTY_BUCKET_VALUE;
		//(rogh->values)[i] = EMPTY_BUCKET_VALUE;
		(rogh->locks)[i] = 0;	
		(rogh->pairs_per_bucket)[i] = 0;
	}
	
	//printf("here........\n");

	for(int i = 0; i < num_gpus_; i++)
	{
		//printf("gpu id: %d+++++++++\n", i);

		CUDA_SAFE_CALL(cudaSetDevice(i));
		cudaMalloc((void **)&rog_[i], sizeof(Gobject));
		cudaMemcpy(rog_[i], rogh, sizeof(Gobject), cudaMemcpyHostToDevice);	

		int init_val = 0;
		cudaMalloc((void **)&task_offset_d_[i], sizeof(int));
		cudaMemcpy(task_offset_d_[i], &init_val, sizeof(int), cudaMemcpyHostToDevice);

		//printf("copy task offset done++++++++++\n");

		//copy parameters
		cudaMalloc((void **)&parameter_d_[i], parameter_size_);
		cudaMemcpy(parameter_d_[i], parameter_, parameter_size_, cudaMemcpyHostToDevice);

		//printf("copy parameter done++++++++++\n");
	}

	free(rogh);

	//initialize reduction objects for CPU cores 
	cudaHostAlloc((void **)&roc_, sizeof(Cobject) * CPU_THREADS, cudaHostAllocMapped); 

	for(int i = 0; i < NUM_BUCKETS_G; i++)
	{
		(roc_[0].keys)[i] = EMPTY_BUCKET_VALUE;
		//(roc->values)[i] = EMPTY_BUCKET_VALUE;
		(roc_[0].locks)[i] = 0;	
		(roc_[0].pairs_per_bucket)[i] = 0;
	}

	for(int i = 1; i<CPU_THREADS; i++)
	{
		memcpy(&roc_[i], &roc_[0], sizeof(Cobject));
	}

	cpu_edge_offset_ = (int *)malloc(sizeof(int));
	*cpu_edge_offset_ = 0;

}

void IrregularRuntime::split()
{
	//split the reduction space based on the speed of each device	
	//split into cpu partition and gpu partitions

	//Partitioner partitioner(part_index, p_mpi_->my_num_nodes(), p_mpi_->my_num_edges(), num_devices_, speeds_, reduction_elm_size_, node_num_dims_, my_rank_); 	
	Partitioner partitioner(part_index, p_mpi_->my_own_num_nodes(), p_mpi_->my_num_edges(), num_devices_, speeds_, reduction_elm_size_, node_num_dims_, my_rank_); 	

	printf("my num nodes: %d\n", p_mpi_->my_num_nodes());

	//printf("coordinate size: %d****************\n", coordinate_size_);

	if(coordinate_size_ == 4)
		partitioner.partition_device_nodes<float>((float *)node_coordinates_ + pv_->my_node_start() * node_num_dims_);

	else if(coordinate_size_ == 8)
		partitioner.partition_device_nodes<double>((double *)node_coordinates_ + pv_->my_node_start() * node_num_dims_);

	//printf("before edges generatedoooooooooooooooooooooooooooo...................\n");

	partitioner.generate_device_edges(p_mpi_->my_edges());
	//partitioner.reorder_satellite_data(p_mpi_->my_node_data(), node_data_elm_size_);

	//printf("edges generatedoooooooooooooooooooooooooooo...................\n");

	if(p_cpu_)
	{
		delete p_cpu_;
		printf("CPU deleted....\n");
	}

	p_cpu_ = new partition_cpu
	(	
		p_mpi_->my_num_nodes(), 
		p_mpi_->my_node_data(), 
		partitioner.get_cpu_num_edges(), 
		partitioner.get_cpu_edges(), 
		p_mpi_->my_edge_data(), 
		p_mpi_->node_data_elm_size(), 
		p_mpi_->edge_data_elm_size(), 
		p_mpi_->reduction_elm_size(), 
		partitioner.get_cpu_num_parts(), 
		partitioner.get_cpu_parts()
	);

	for(int i = 0; i < num_devices_; i++)
	{
		device_node_start_[i] = partitioner.get_node_start(i); 	
		device_node_sum_[i] = partitioner.get_node_sum(i); 	
	}

	//gpu partitions
	for(int i = 0; i < num_gpus_; i++)
	{
		CUDA_SAFE_CALL(cudaSetDevice(i));

		if(p_cuda_[i])
		{
			delete p_cuda_[i];
			printf("GPU %d deleted....\n", i);
		}

		p_cuda_[i] = new partition_cuda
		(p_mpi_->my_num_nodes(), 
		p_mpi_->my_node_data(), 
		p_mpi_->my_node_data_d(),
		partitioner.get_gpu_num_edges()[i], 
		partitioner.get_gpu_edges()[i], 
		p_mpi_->my_edge_data(), 
		p_mpi_->node_data_elm_size(), 
		p_mpi_->edge_data_elm_size(), 
		p_mpi_->reduction_elm_size(), 
		partitioner.get_gpu_num_parts()[i], 
		partitioner.get_gpu_parts()[i]);	

		p_cuda_[i]->Allocate();
	}
}

void *IrregularRuntime::launch(void *arg)
{
	IrregularRuntime *runtime = (IrregularRuntime *)arg;
	
        pthread_mutex_t mutex;
        pthread_mutex_init(&mutex, NULL);

        //create CPU_THREAD threads
        //pthread_t tid[CPU_THREADS];
        CpuArgs args[CPU_THREADS];
	pthread_t tid[CPU_THREADS];

	double before_launch = rtclock();
	for(int i = 0; i < CPU_THREADS; i++)
	{
		args[i].tid = i;
		args[i].runtime = runtime;
		args[i].mutex = &mutex;

		pthread_create(&tid[i], NULL, compute_cpu, &args[i]);	
	}

	for(int j = 0; j < CPU_THREADS; j++)
	{
		pthread_join(tid[j], NULL);	
	}

	double after_launch = rtclock();

	printf("CPU time: %f\n", after_launch - before_launch);

	//runtime->speeds_[0] = sqrt(1/(after_launch - before_launch));
	runtime->speeds_[0] = (1/(after_launch - before_launch))/3;

	return (void *)0;
}

void IrregularRuntime::IrregularStart()
{
	if(current_iter_!=0)
	{
		re_init();	
	}

	if(current_iter_==0||current_iter_==1)
	{
		split();
		printf("%d split done....\n", my_rank_);
	}

	if(current_iter_==0||current_iter_==1)
	{
		//printf("oooooooooooo exchanging halo size ooooooooooooo\n");
		pv_ -> exchange_halo_size_info();

		//printf("oooooooooooo exchanging halo size info done...oooooooooooo\n");
		pv_ -> exchange_halo_node_info();
	}

	printf("oooooooooooo exchanging halo node data ooooooooooooo\n");
	double before_halo = rtclock();
	pv_ -> exchange_halo_node_data(p_mpi_);
	double after_halo = rtclock();

	printf("=============>exchange halo time: %f\n", after_halo - before_halo);

	//printf("+O+O+O+O+O+O+O exchange done +O+O+O+O+O+O\n");

	//launch CPU
	pthread_t tid;	
	pthread_create(&tid, NULL, launch, this);

	//launch GPU
	double before_gpu = rtclock();
	for(int i = 0; i < num_gpus_; i++)
	{
		CUDA_SAFE_CALL(cudaSetDevice(i));
		dim3 grid(GPU_BLOCKS, 1, 1);		
		dim3 block(GPU_THREADS, 1, 1);

		compute_gpu<<<grid, block, 0>>>
		(
			p_cuda_[i]->my_parts_d(),
			p_cuda_[i]->my_node_data_device(),
			part_index_d,
			p_cuda_[i]->my_edges_d(),
			parameter_d_[i],
			//TASK OFFSET
			task_offset_d_[i],
			p_cuda_[i]->my_num_parts(),
			rog_[i],
			map_idx_,
			reduce_idx_	
		);
	}

	for(int i = 0; i < num_gpus_; i++)
	{
		CUDA_SAFE_CALL(cudaSetDevice(i));
		cudaDeviceSynchronize();	
	}

	checkCUDAError("~~~internal error checking...");

	double after_gpu = rtclock();

	printf("gpu time: %f\n", after_gpu - before_gpu);

        for(int i = 0; i < num_gpus_; i++)
        {
                //speeds_[i + 1] = sqrt(1/(after_gpu - before_gpu));
                speeds_[i + 1] = (1/(after_gpu - before_gpu));
        }

	pthread_join(tid, NULL);

	current_iter_++;
}

void IrregularRuntime::get_reduction_result(void *buffer)
{
	//first, combine cpu reduction objects
	merge_cobjects();
	//then, copy cpu result		
	memcpy(reduction_result_, &(roc_[0].values), sizeof(VALUE)*device_node_sum_[0]);
	//then, copy gpu results

	for(int i = 0; i < num_gpus_; i++)
	{
		cudaSetDevice(i);	
		cudaMemcpy((char *)reduction_result_ + device_node_start_[i+1] * reduction_elm_size_, 
		&(rog_[i]->values), 
		device_node_sum_[i+1]*reduction_elm_size_, 
		cudaMemcpyDeviceToHost);
	}

	memcpy((char *)buffer + pv_->my_node_start()*reduction_elm_size_, reduction_result_, reduction_elm_size_ * pv_->my_num_nodes());		
}

void IrregularRuntime::merge_cobjects()
{
	pthread_t tid[CPU_THREADS];
	struct merge_args merge_args[CPU_THREADS]; 

	for(int j = 1; j < 8; j++)
	{
		for(int i = 0; i < CPU_THREADS; i++)
		{
			merge_args[i].tid = i;
			merge_args[i].roc1 = &roc_[0];
			merge_args[i].roc1 = &roc_[j];
			pthread_create(&tid[i], NULL, mergetc, &merge_args[i]);
		}

		//join the threads
        	for(int j = 0; j < CPU_THREADS; j++)
        	{
        	    pthread_join(tid[j], NULL); 
        	}
	}
}

void IrregularRuntime::reset_node_data(void *node_data)
{
	memcpy(p_mpi_->my_node_data(), (char *)node_data + pv_->my_node_start()*node_data_elm_size_, pv_->my_num_nodes() * reduction_elm_size_);		
}

void IrregularRuntime::IrregularBarrier()
{
	MPI_Barrier(MPI_COMM_WORLD);
}

IrregularRuntime::~IrregularRuntime()
{
	delete pv_;
	delete p_mpi_;
	delete p_cpu_;
	delete [] p_cuda_;

	delete [] rog_;
	for(int i = 0; i < num_gpus_; i++)
	{
		cudaFree(rog_[i]);	
	}
}

void IrregularRuntime::re_init()
{
	int init_val = 0;
	for(int i = 0; i < num_gpus_; i++)
	{
		cudaSetDevice(i);
		cudaMemcpy(task_offset_d_[i], &init_val, sizeof(int), cudaMemcpyHostToDevice);
		
		dim3 grid(GPU_BLOCKS, 1, 1);
		dim3 block(GPU_THREADS, 1, 1);
		init_rog<<<grid, block>>>(rog_[i]);
		cudaThreadSynchronize();
	}

	//init cpu related 
	*cpu_edge_offset_ = 0;

        pthread_t tid[CPU_THREADS]; 
	struct init_args init_args[CPU_THREADS];
        for(int i = 0; i < CPU_THREADS; i++)
	{
		init_args[i].tid = i;
		init_args[i].roc = &roc_[i];
		pthread_create(&tid[i], NULL, init_roc, &init_args[i]);	
	}

	//join the threads
        for(int j = 0; j < CPU_THREADS; j++)
        {
            pthread_join(tid[j], NULL); 
        }
}

void IrregularRuntime::set_map_idx(int idx)
{
	this->map_idx_ = idx;
}

void IrregularRuntime::set_reduce_idx(int idx)
{
	this->reduce_idx_ = idx;
}
