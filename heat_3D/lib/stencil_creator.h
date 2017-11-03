#ifndef STENCIL_CREATOR_H_
#define STENCIL_CREATOR_H_

#include "stencil_runtime.h"
#include "array.h"

template <class T>
class StencilCreator
{
protected:
	int num_dims_;
	IndexArray global_size_;
	int proc_num_dims_;
	IntArray proc_size_;
	int stencil_width_;
	int num_iters_;
	void *parameter_;
	size_t parameter_size_;
	

public:		
	StencilCreator(int num_dims, const IndexArray &global_size, int proc_num_dims, IntArray &proc_size, int stencil_width, int num_iters, bool periodic, bool diagonal, void *parameter, int parameter_size):num_dims_(num_dims),global_size_(global_size),proc_num_dims_(proc_num_dims),proc_size_(proc_size),stencil_width_(stencil_width),num_iters_(num_iters), parameter_(parameter), parameter_size_(parameter_size)
	{
				
	}

	StencilRuntime *CreateRuntime()
	{
		return new StencilRuntime(num_dims_, sizeof(T), global_size_, proc_num_dims_, proc_size_, stencil_width_, num_iters_, parameter_, parameter_size_); 	
	}
};

#endif
