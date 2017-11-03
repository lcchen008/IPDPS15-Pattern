#include "partition_cuda.h"
#include "../lib/macro.h"
#include <stdio.h> 
#include <stdlib.h> 

void partition_cuda::Allocate()
{
	//node data come from zero-copy	

	//allocate edges
	cudaMalloc((void **)&my_edges_d_, sizeof(EDGE)*my_num_edges_);
	//copy in edge_data
	cudaMemcpy(my_edges_d_, my_edges_, sizeof(EDGE)*my_num_edges_, cudaMemcpyHostToDevice);

	CUDA_SAFE_CALL(cudaMalloc((void **)&my_node_data_device_, my_num_nodes_ * node_data_elm_size_));				
	CUDA_SAFE_CALL(cudaMemcpy(my_node_data_device_, my_node_data_d_, my_num_nodes_ * node_data_elm_size_, cudaMemcpyHostToDevice));

	//allocate edge data
	my_edge_data_ = NULL;
	my_edge_data_d_ = NULL;

	if(edge_data_elm_size_!=0)
	{
		cudaMalloc((void **)&my_edge_data_d_, edge_data_elm_size_*my_num_edges_);

		cudaMemcpy(my_edge_data_d_, my_edge_data_, edge_data_elm_size_*my_num_edges_, cudaMemcpyHostToDevice);
	}

	//allocate parts
	cudaMalloc((void **)&my_parts_d_, sizeof(Part) * my_num_parts_);
	//copy in parts
	cudaMemcpy(my_parts_d_, my_parts_, sizeof(Part) * my_num_parts_, cudaMemcpyHostToDevice);
}

partition_cuda::~partition_cuda()
{
	//cudaFreeHost(my_node_data_d_);	

	if(my_edge_data_d_)
		cudaFree(my_edge_data_d_);

	cudaFree(my_edges_d_);
	free(my_parts_);
	cudaFree(my_parts_d_);
}
