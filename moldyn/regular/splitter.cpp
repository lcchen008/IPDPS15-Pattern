#include "splitter.h"
#include "data_partition_mpi.h"

splitter::splitter(
			int num_procs,
			int my_rank,
			void *input,
			int input_size,
			Offset *offsets,
			int num_offsets,
			void *parameters,
			int parameter_size
			):num_procs_(num_procs),
		my_rank_(my_rank),	
		input_(input),
		input_size_(input_size),
		offsets_(offsets),
		num_offsets_(num_offsets),
		parameters_(parameters),
		parameter_size_(parameter_size)
{}

data_partition_mpi *splitter::gen_partition()
{
	int average = num_offsets_/num_procs_;
	my_offset_start_ = average * my_rank_;	
	my_num_offsets_ = my_rank_==num_procs_ - 1 ? num_offsets_ - (num_procs_ - 1)*average : average; 
	int my_input_start = offsets_[my_offset_start_].offset; 
	int last_offset = my_offset_start_ + my_num_offsets_ - 1;
	int my_input_end = offsets_[last_offset].offset + offsets_[last_offset].size - 1;
	int my_input_size = my_input_end - my_input_start + 1;

	void *my_input = (char *)input_ + my_input_start;
	Offset *my_offset = offsets_ + my_offset_start_; 

	//shift the offsets	
	for(int i = 0; i < my_num_offsets_; i++)
	{
		my_offset[i].offset -= my_input_start;	
	}

	data_partition_mpi *d = new data_partition_mpi(my_input, my_input_size, my_offset, my_num_offsets_);

	return d; 
}
