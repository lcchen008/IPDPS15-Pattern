#include "stencil_runtime.h"
#include "cu_util.h"
#include "common.h"
#include "cpu_util.h"
#include "macro.h"
#include "compute_cpu.h"
#include <vector>
#include "data_util.h"
#include <mpi.h>
#include <iostream>
#include "time_util.h"
#include "array.h"
#include "buffer.h"
#include "compute_cuda.cu"
#include "CU_DS.h"
#include <stdio.h>
//#include "mpi_util.h"

using namespace std;

StencilRuntime::StencilRuntime(int num_dims, int unit_size,
			const IndexArray &global_size, 
			int proc_num_dims,
			IntArray proc_size, 
			int stencil_width, int num_iters, 
			void *parameter, size_t parameter_size):num_dims_(num_dims),
			unit_size_(unit_size), global_size_(global_size), 
			proc_num_dims_(proc_num_dims), proc_size_(proc_size), 
			stencil_width_(stencil_width), num_iters_(num_iters), 
			current_iter_(0), parameter_(parameter), parameter_size_(parameter_size)
{
	
}

void StencilRuntime::StencilInit()
{
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);

	this->num_gpus_ = GetGPUNumber();

	this->num_devices_ = num_gpus_ + 1;

	gv_ = new Grid_view(this->num_dims_, this->global_size_, this->proc_num_dims_, this->proc_size_, this->my_rank_);

	grid_ = gv_->CreateGrid(unit_size_, num_dims_, global_size_, stencil_width_);

	//allocate gpu grids pointers
	gv_cuda_ = (Grid_view_cuda **)malloc(sizeof(Grid_view_cuda *) * num_gpus_);
	grid_cuda_ = (GridCuda **)malloc(sizeof(GridCuda *) * num_gpus_);
	parameter_cuda_ = (void **)malloc(sizeof(void *) * num_gpus_);

	internal_tiles = (vector <struct Tile> *)malloc(num_devices_ * sizeof(vector <struct Tile>));

	border_tiles = (vector <struct Tile> *)malloc(num_devices_ * sizeof(vector <struct Tile>));

	requests = (vector <MPI_Request> *)malloc(num_dims_ * sizeof(vector <MPI_Request>));

	speeds_ = (double *)malloc(sizeof(double) * num_devices_);

	//initial speed are equal
	for(int i = 0; i < num_devices_; i++)
	{
		speeds_[i] = 1;	
	}

	stencil_idx_ = 0;

	cout<<"Rank: "<<my_rank_<<" Init done..."<<endl;
}

void StencilRuntime::along_dim()
{
	int along_which = num_dims_ - 1;	

	//IndexArray size = grid_->my_size();

	//for(int i = num_dims_ - 2; i >= 0; i--)
	//{
	//	if(size[i] > size[along_which])	
	//	{
	//		along_which = i;	
	//	}
	//}

	this->along_ = along_which;
}

//split the per-node grid with initial parameters(evenly)
void StencilRuntime::split()
{
	IndexArray size = grid_->my_size();
	int along_size = size[along_];

	starts_ = (int *)malloc(num_devices_ * sizeof(int));
	along_partitions_ = (int *)malloc(num_devices_ * sizeof(int));	
	int accumulated = 0;
	
	double total_speed = 0;
	for(int i = 0; i < num_devices_; i++)
	{
		total_speed += speeds_[i];	
	}

	for(int i = 0; i <= num_devices_ - 2; i++)
	{
		along_partitions_[i] = along_size*speeds_[i]/total_speed;	
		cout<<"#########along "<<i<<": "<<along_partitions_[i]<<endl;
		starts_[i] = accumulated;
		accumulated  += along_partitions_[i];
	}

	along_partitions_[num_devices_ - 1] = along_size - accumulated;
	cout<<"#########along "<<(num_devices_ - 1)<<": "<<along_partitions_[num_devices_ - 1]<<endl;
	starts_[num_devices_ - 1] = accumulated;
}

//allocate grid views and grids
//also copies data into sub-grids
void StencilRuntime::create_grids()
{
	IndexArray offset(stencil_width_, stencil_width_, stencil_width_);

	IndexArray size = grid_->my_size();
	size[along_] = along_partitions_[0];

	//allocate cpu grids
	gv_cpu_ = new Grid_view_cpu();	
	grid_cpu_ = gv_cpu_->CreateGrid(unit_size_, num_dims_, offset, size, stencil_width_, num_devices_, 0);

	IndexArray my_offset(stencil_width_, stencil_width_, stencil_width_);
	grid_cpu_->copy_host_to_host(my_offset, offset, grid_->buffer(), size);

	cout<<"################copy to cpu done..."<<endl;

	
	for(int i = 0; i <= num_gpus_ - 1; i++)
	{
		cout<<"copying GPU: "<<i<<endl;
		//don't forget to set device
		CUDA_SAFE_CALL(cudaSetDevice(i));

		if(current_iter_ == 0)
		{
    			CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
			CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		}

		offset[along_] += along_partitions_[i];
		size[along_] = along_partitions_[i+1];

		//cout<<"gpu "<<i<<" size: "<<size<<endl;

		gv_cuda_[i] = new Grid_view_cuda();	
		grid_cuda_[i] = gv_cuda_[i]->CreateGrid(unit_size_, num_dims_, offset, size, stencil_width_, num_devices_, i + 1);
		grid_cuda_[i]->copy_host_to_device(my_offset, offset, size, grid_->buffer());
		parameter_cuda_[i] = NULL;

		if(parameter_size_ > 0)
		{
			cudaMalloc(&parameter_cuda_[i], parameter_size_);
			cudaMemcpy(parameter_cuda_[i], parameter_, parameter_size_, cudaMemcpyHostToDevice);
		}
	}
}

void *StencilRuntime::launch(void *arg)
{
	int *compute_type = (int *)((void **)arg)[0];

	int *ptr_type = (int *)((void **)arg)[1];	
	
	void *object = ((void **)arg)[2];

	int device_type = *ptr_type;
	
	StencilRuntime *runtime = (StencilRuntime *)object;

	if(device_type == CPU)
	{
		pthread_t tid[CPU_THREADS];
		Cargs args[CPU_THREADS];	

		double before_launch = rtclock();

		for(int i = 0; i < CPU_THREADS; i++)
		{
			args[i].tid = i;	
			args[i].runtime = runtime;

			if(*compute_type == INTERNAL)
			{
				args[i].tiles = &(runtime->internal_tiles)[0];
			}
			else
			{
				args[i].tiles = &(runtime->border_tiles)[0];
			}

			pthread_create(&tid[i], NULL, compute_tiles, &args[i]);
		}

		for(int j = 0; j < CPU_THREADS; j++)
		{
			pthread_join(tid[j], NULL);	
		}

		double after_launch = rtclock();

		printf("exec time: %f\n", after_launch - before_launch);

		if(*compute_type == INTERNAL)
			runtime->speeds_[0] = 1/(after_launch - before_launch);
	}
	
	return (void *)0;
}

void StencilRuntime::tile_grids()
{
	//tile CPU internal	
	IndexArray internal_start;   	
	IndexArray internal_size;

	IndexArray my_size;
	IndexArray my_real_size;

	for(int i = 0; i < num_dims_; i++)
	{
		internal_start[i] += 2*stencil_width_;  
		internal_size[i] = grid_cpu_->my_size()[i] - 2*stencil_width_; 
	}

	internal_tiles[0].clear();

	tiling(num_dims_, internal_size, internal_start, CPU_TILE_SIZE, internal_tiles[0]);

	cout<<"############internal size: "<<internal_size<<endl;
	//cout<<"tiling done........"<<endl;

	//for(int i = 0; i < internal_tiles[0].size(); i++)
	//{
	//	cout<<"offset: "<<(internal_tiles[0])[i].offset<<"size: "<<(internal_tiles[0])[i].size<<endl;	
	//}

	//tile CPU border
	my_size = grid_cpu_->my_size();
	my_real_size = grid_cpu_->my_real_size();

	border_tiles[0].clear();
	tiling_border(num_dims_, my_size, my_real_size, grid_cpu_->halo(), CPU_TILE_SIZE, border_tiles[0]);

	//for(int i = 0; i < border_tiles[0].size(); i++)
	//{
	//	cout<<"offset: "<<( border_tiles[0])[i].offset<<"size: "<<( border_tiles[0])[i].size<<endl;	
	//}

	printf("^^^^^^^CPU internal tiles: %d, CPU border tiles: %d\n", internal_tiles[0].size(), border_tiles[0].size());

	//tile GPUs internal
	for(int j = 0; j < num_gpus_; j++)
	{
		IndexArray internal_start;   	
		IndexArray internal_size;
		for(int i = 0; i <num_dims_; i++)
		{
			internal_start[i] += 2*stencil_width_;  
			internal_size[i] = grid_cuda(j)->my_size()[i] - 2*stencil_width(); 
		}
		
		internal_tiles[j + 1].clear();
		tiling(num_dims(), internal_size, internal_start, GPU_TILE_SIZE, internal_tiles[j + 1]);

		my_size = grid_cuda_[j]->my_size();
		my_real_size = grid_cuda_[j]->my_real_size();

		//tile gpu border
		border_tiles[j + 1].clear();
		tiling_border(num_dims_, my_size, my_real_size, grid_cuda_[j]->halo(), GPU_TILE_SIZE, border_tiles[j + 1]);
	}

	//for(int i = 0; i < internal_tiles[2].size(); i++)
	//{
	//	cout<<"my size: "<<grid_cuda_[0]->my_size()<<"offset: "<<( internal_tiles[1])[i].offset<<"size: "<<( internal_tiles[1])[i].size<<endl;	
	//}
}

void StencilRuntime::interdevice_exchange()
{
	//inter device exchange
	for(int i = 0; i < num_gpus_; i++)
	{
		int cur_device;
		cudaGetDevice(&cur_device);

		if(i!=cur_device)
		CUDA_SAFE_CALL(cudaSetDevice(i));
		//the highest dimension
		grid_cuda_[i]->send_to_neighbors(along_, stencil_width_, grid_, grid_cpu_, grid_cuda_);	

		for(int dim = num_dims_ - 2; dim >= 0; dim--)
		{
			if(proc_size_[dim]>1)
			{
				//forward
				grid_cuda_[i]->copy_device_to_map(dim, stencil_width_, true);

				//backward
				grid_cuda_[i]->copy_device_to_map(dim, stencil_width_, false);
			}
		}
	}

	grid_cpu_->send_to_neighbors(along_, stencil_width_, grid_, grid_cuda_);	

	//next, synchronize the previous operations
	for(int i = 0; i < num_gpus_; i++)	
	{
		CUDA_SAFE_CALL(cudaSetDevice(i));
		cudaDeviceSynchronize();	
	}

	//conduct synchronous operations
	for(int i = 0; i < num_gpus_; i++)
	{
		grid_cuda_[i]->copy_map_to_neighbor(along_, stencil_width_, grid_, grid_cpu_);	

		for(int dim = num_dims_ - 2; dim >= 0; dim--)
		{
			//copy side halos to global grid
			if(proc_size_[dim]>1)
			{
				//forward
				grid_cuda_[i]->copy_map_to_global_grid(grid_, dim, starts_[i+1], along_partitions_[i+1], stencil_width_, true);

				//backward
				grid_cuda_[i]->copy_map_to_global_grid(grid_, dim, starts_[i+1], along_partitions_[i+1], stencil_width_, false);
			}
		}
	}

	//copy side to global
	for(int dim = num_dims_ - 2; dim >= 0; dim--)
	{
		if(proc_size_[dim] > 1)		
		{	
			//forward
			grid_cpu_->copy_to_global_grid(grid_, dim, starts_[0], along_partitions_[0], stencil_width_, true);	
			//backward
			grid_cpu_->copy_to_global_grid(grid_, dim, starts_[0], along_partitions_[0], stencil_width_, false);	
		}
	}

	//exchange among nodes
	//double before_exchange = rtclock();
	//UnsignedArray halo(stencil_width_, stencil_width_, stencil_width_);
	//Width2 w = {halo, halo};
	//gv_->ExchangeBoundaries(grid_, w, false, false);
	

	MPI_Barrier(MPI_COMM_WORLD);

	double before_internal = rtclock();
	process_internal();
	double after_internal = rtclock();

	printf("PURE INTERNAL TIME: %f\n", after_internal - before_internal);

	double before_ex = rtclock();

	for(int i = grid_->num_dims() - 1; i >= 0; i--)
	{
		requests[i].clear();		

		gv_->ExchangeBoundariesAsync(grid_, i, stencil_width_, stencil_width_, false, false, requests[i]);
	}
	
	double copy_time = 0; 

	for(int i = grid_->num_dims() - 1; i >= 0; i--)
	{
		for (int j = 0; j < requests[i].size(); j++)
		{
			MPI_Request *req = &(requests[i][j]);	
			CHECK_MPI(MPI_Wait(req, MPI_STATUS_IGNORE));

			double before_copy = rtclock();
			grid_->CopyinHalo(i, stencil_width_, false);
			grid_->CopyinHalo(i, stencil_width_, true);
			double after_copy = rtclock();

			copy_time += (after_copy - before_copy);
		}

		printf("DIM %d copy time: %f\n", i, copy_time);
	}

	double after_ex = rtclock();

	printf("RANK: %d PURE EX TIME: %f, PURE COPY TIME: %f\n", my_rank_, after_ex - before_ex, copy_time);

	MPI_Barrier(MPI_COMM_WORLD);

	
	//double after_exchange = rtclock();

	//printf("exchange time: %f\n", after_exchange - before_exchange);

	//next step is to copy self halo buffer to sub devices 

	//synchronous operations first
	//copy from global grid receive buffer to cpu local halo
	for(int dim = num_dims_ - 2; dim >= 0; dim--)
	{
		if(proc_size_[dim] > 1)		
		{	
			//forward
			grid_cpu_->copy_from_global_grid(grid_, dim, starts_[0], along_partitions_[0], stencil_width_, true);	
			//backward
			grid_cpu_->copy_from_global_grid(grid_, dim, starts_[0], along_partitions_[0], stencil_width_, false);	
		}
	}

	for(int i = 0; i < num_gpus_; i++)
	{

		for(int dim = num_dims_ - 2; dim >= 0; dim--)
		{
			if(proc_size_[dim]>1)
			{
				//copy from global receive buffer to local mapped buffer
				//forward
				grid_cuda_[i]->copy_global_grid_to_map(grid_, dim, starts_[i+1], along_partitions_[i+1], stencil_width_, true);
				//backward
				grid_cuda_[i]->copy_global_grid_to_map(grid_, dim, starts_[i+1], along_partitions_[i+1], stencil_width_, false);

			}
		}
	}

	//asynchrounous operations next
	for(int i = 0; i < num_gpus_; i++)
	{
		CUDA_SAFE_CALL(cudaSetDevice(i));

		for(int dim = num_dims_ - 2; dim >= 0; dim--)
		{
			if(proc_size_[dim]>1)
			{
				//copy from map buffer to gpu device halo
				//forward
				grid_cuda_[i]->copy_map_to_device(dim, stencil_width_, true);
				//backward
				grid_cuda_[i]->copy_map_to_device(dim, stencil_width_, false);
			}
		}
	}

	//next, synchronize the previous operations
	for(int i = 0; i < num_gpus_; i++)	
	{
		CUDA_SAFE_CALL(cudaSetDevice(i));
		cudaDeviceSynchronize();	
	}

	//copy top from global to gpu
	grid_cuda_[num_gpus_ - 1]->copy_from_top(along_, stencil_width_, grid_);

	//copy botton from global to cpu
	grid_cpu_->copy_from_bottom(along_, stencil_width_, grid_);
}

void StencilRuntime::process_internal()
{
	pthread_t tid[2];			
	int device_types[2];
	device_types[0] = CPU;
	device_types[1] = GPU;
	int compute_type = INTERNAL;
	void *arg1[3] = {&compute_type, &device_types[0], this};

	//launch CPU
	pthread_create(&tid[0], NULL, launch, arg1);
	cout<<"CPU thread launched..."<<endl;
	
	double before_gpu = rtclock();
	for(int j = 0; j < num_gpus_; j++)
	{
		Ltile *tiles_d;
		int *dims_d;

		CUDA_SAFE_CALL(cudaSetDevice(j));

		cudaMalloc((void **)&tiles_d, sizeof(Ltile) * internal_tiles[j + 1].size());
		cudaMalloc((void **)&dims_d, sizeof(int) * 3);

		cudaMemcpy(tiles_d, (Ltile *)&(internal_tiles[j + 1])[0], sizeof(Ltile) * internal_tiles[j + 1].size(), cudaMemcpyHostToDevice);
		cudaMemcpy(dims_d, &(grid_cuda(j)->my_real_size())[0], sizeof(int)*3, cudaMemcpyHostToDevice);

	 	dim3 grid(GPU_BLOCKS, 1, 1);		
		dim3 block(GPU_THREADS, 1, 1);
		
		compute_cuda_internal<<<grid, block, 0>>>
		(
			internal_tiles[j + 1].size(),		
			tiles_d,
			grid_cuda(j)->data_in(),
			grid_cuda(j)->data_out(),
			num_dims(),
			unit_size(),
			dims_d,	
			stencil_idx_,
			parameter_cuda_[j],
			parameter_size_
		);
	}

    	cudaThreadSynchronize();
	checkCUDAError("~~~internal error checking...");
	double after_gpu = rtclock();

	for(int i = 0; i < num_gpus_; i++)
	{
		speeds_[i + 1] = 1/(after_gpu - before_gpu);	
	}

	printf("gpu time: %f\n", after_gpu - before_gpu);

	pthread_join(tid[0], NULL);
}

void StencilRuntime::process_border()
{

	printf("processing border...\n");

	pthread_t tid[2];			
	int device_types[2];
	device_types[0] = CPU;
	device_types[1] = GPU;
	int compute_type = BORDER;
	void *arg1[3] = {&compute_type , &device_types[0], this};

	//launch CPU
	pthread_create(&tid[0], NULL, launch, arg1);
	cout<<"CPU thread launched..."<<endl;
	
	double before_gpu = rtclock();
	for(int j = 0; j < num_gpus_; j++)
	{
		Ltile *tiles_d;
		int *dims_d;

		CUDA_SAFE_CALL(cudaSetDevice(j));

		cudaMalloc((void **)&tiles_d, sizeof(Ltile) * border_tiles[j + 1].size());
		cudaMalloc((void **)&dims_d, sizeof(int) * 3);

		cudaMemcpy(tiles_d, (Ltile *)&(border_tiles[j + 1])[0], sizeof(Ltile) * border_tiles[j + 1].size(), cudaMemcpyHostToDevice);
		cudaMemcpy(dims_d, &(grid_cuda(j)->my_real_size())[0], sizeof(int)*3, cudaMemcpyHostToDevice);

		//printf("SIZE OF INDEXARRAY: %d\n", sizeof(IndexArray));

	 	dim3 grid(GPU_BLOCKS, 1, 1);		
		dim3 block(GPU_THREADS, 1, 1);
		
		compute_cuda_internal<<<grid, block, 0>>>
		(
			border_tiles[j + 1].size(),		
			tiles_d,
			grid_cuda(j)->data_in(),
			grid_cuda(j)->data_out(),
			num_dims(),
			unit_size(),
			dims_d,	
			stencil_idx_,
			parameter_cuda_[j],
			parameter_size_
		);
	}

    	cudaThreadSynchronize();
	checkCUDAError("~~~border error checking...");
	double after_gpu = rtclock();
	printf("gpu time: %f\n", after_gpu - before_gpu);

	pthread_join(tid[0], NULL);	
}

void StencilRuntime::StencilBegin()
{
	if(num_iters_>1)
	{
		profile_iter();	
		clean_grids();
		current_iter_ = 1;
	}
	
	along_dim();
	
	split();

	create_grids();

	tile_grids();

	interdevice_exchange();

	process_border();

	current_iter_++;

	double before = rtclock();
	for(; current_iter_ <= num_iters_ - 1; current_iter_++)
	{
		printf("%d ====>iter: %d\n", my_rank_, current_iter_);
		double before_internal = rtclock();	
			interdevice_exchange();	
		double after_internal = rtclock();	
		printf("internal time: %f\n", after_internal - before_internal);
			process_border();
		double after_border = rtclock();
		printf("border time: %f\n", after_border - after_internal);
	}
	double after = rtclock();

	printf("#########RANK %d EXECUTION TIME: %f\n", my_rank_, after - before);
}

void StencilRuntime::profile_iter()
{
	along_dim();	

	split();

	create_grids();

	tile_grids();

	interdevice_exchange();

	process_border();
}

void StencilRuntime::clean_grids()
{
	delete gv_cpu_;
	gv_cpu_ = NULL;
	delete grid_cpu_;
	grid_cpu_ = NULL;
	for(int i = 0; i < num_gpus_; i++)
	{
		delete grid_cuda_[i];		
		grid_cuda_[i] = NULL;
		delete gv_cuda_[i];
		gv_cuda_[i] = NULL;
	}
}

void StencilRuntime::StencilFinalize()
{
	delete gv_;	
	delete grid_;

	delete grid_cpu_;

	delete gv_cpu_;

	for(int i = 0; i < num_gpus_; i++)
	{
		delete grid_cuda_[i];		
		delete gv_cuda_[i];
	}

	free(grid_cuda_);
	free(gv_cuda_);

	free(internal_tiles);
	free(border_tiles);
	free(requests);

	//MPI_Finalize();
}
