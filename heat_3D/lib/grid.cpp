#include "grid.h"
#include "stdio.h"

Grid::Grid(int unit_size, int num_dims, const IndexArray &size):
	unit_size_(unit_size), num_dims_(num_dims), size_(size) 
{
	num_eles_ = size.accumulate(num_dims_);	
	data_buffer_[0] = NULL;
	data_buffer_[1] = NULL;
}

void Grid::InitBuffer()
{
	data_buffer_[0] = new BufferHost();
	data_buffer_[0]->Allocate(num_dims_, unit_size_, size_);
	data_buffer_[1] = new BufferHost();
	data_buffer_[1]->Allocate(num_dims_, unit_size_, size_);

  	data_[0] = (char*)data_buffer_[0]->Get();
  	data_[1] = (char*)data_buffer_[1]->Get();

	//printf("data_[0]: %ld, data_[1]: %ld", data_[0], data_[1]);
}

void Grid::DeleteBuffers()
{
	if (data_[0]) 
	{
    		delete data_buffer_[0];
    		data_[0] = NULL;
  	}
  	if (data_[1]) 
	{
    		delete data_buffer_[1];
    		data_[1] = NULL;
  	}
}

Grid::~Grid()
{
	DeleteBuffers();
}
