#include "data_partition_mpi.h"
#include <stdio.h>

data_partition_mpi::data_partition_mpi(void *input,
			size_t input_size,
			Offset *offsets,
			size_t offset_number
):input_(input),input_size_(input_size),offsets_(offsets),offset_number_(offset_number)
{
	//allocate pinned memory copy
	cudaHostAlloc((void **)&input_pin_, input_size_,  cudaHostAllocPortable|cudaHostAllocMapped);

	//copy data into it
	memcpy(input_pin_, input_, input_size_);

	printf("=================> offset number: %ld input size: %ld\n", offset_number_, input_size_);

	//allocate pinned offset copy
	cudaHostAlloc((void **)&offsets_pin_, offset_number_ * sizeof(Offset),   cudaHostAllocPortable|cudaHostAllocMapped);

	memcpy(offsets_pin_, offsets_, offset_number_ * sizeof(Offset));
}
