#ifndef LIB_GRID_H
#define LIB_GRID_H
#include "buffer.h"
#include "array.h"
#include <stdio.h>

class Grid
{
public:
	//one for input, one for output
	Grid(int unit_size, int num_dims, const IndexArray &size);
	static Grid* Create(int unit_size, int num_dims, const IndexArray &size)
	{
		Grid *g = new Grid(unit_size, num_dims, size);	
		g->InitBuffer();
		return g;
	}

	Grid(){}

	int unit_size() const
	{
		return unit_size_;
	}

	int num_dims() const
	{
		return num_dims_;
	}

	const IndexArray &size() const
	{
		return size_;
	}

	BufferHost *buffer() const
	{
		return data_buffer_[0];	
	}

	char *data() const
	{
		return data_[0];
	}

	char *data_in()
	{
		return data_[0]; 	
	}

	char *data_out()
	{
		return data_[1]; 	
	}
	
	virtual ~Grid();

protected:
	virtual void InitBuffer();
  	virtual void DeleteBuffers();
	void Swap()
	{
		std::swap(data_[0], data_[1]);
  		std::swap(data_buffer_[0], data_buffer_[1]);
	}

	BufferHost *data_buffer_[2];
	char *data_[2];
	int unit_size_;
	int num_dims_;
	size_t num_eles_;
	IndexArray size_;
};


#endif
