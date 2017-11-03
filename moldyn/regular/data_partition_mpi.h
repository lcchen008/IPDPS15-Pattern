#ifndef REGULAR_DATA_MPI_H_
#define REGULAR_DATA_MPI_H_

#include "data_type.h"
#include <stdlib.h>

class data_partition_mpi
{
protected:
	void *input_;	
	size_t input_size_;
	Offset *offsets_;
	size_t offset_number_;

	void *input_pin_;
	Offset *offsets_pin_;
public:
	data_partition_mpi(void *input,
			size_t input_size,
			Offset *offsets,
			size_t offset_number
			);

	void *input(){return input_;}
	size_t input_size(){return input_size_;}
	Offset *offset(){return offsets_;}
	size_t num_offsets(){return offset_number_;}

	void *input_pin(){return input_pin_;}
	Offset *offset_pin(){return offsets_pin_;}

};

#endif
