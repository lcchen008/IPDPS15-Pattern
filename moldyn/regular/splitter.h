//splits data cross nodes
#ifndef REGULAR_SPLITTER_H_
#define REGULAR_SPLITTER_H_

#include "data_type.h"
#include "data_partition_mpi.h"

class splitter
{
	protected:
		int num_procs_;		
		int my_rank_;
		void *input_;
		int input_size_;
		Offset *offsets_;
		int num_offsets_;
		void *parameters_;
		int parameter_size_;

		int my_offset_start_;
		int my_num_offsets_;

	public:
		splitter(
			int num_procs,
			int my_rank,
			void *input,
			int input_size,
			Offset *offsets,
			int num_offsets,
			void *parameters,
			int parameter_size
			);

		data_partition_mpi *gen_partition();
};

#endif
