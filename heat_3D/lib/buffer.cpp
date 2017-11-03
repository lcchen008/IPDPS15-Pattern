#include "buffer.h"
#include <assert.h>
#include <string.h>
#include "data_util.h"
#include <stdio.h>
#include <stdio.h>

//constructor
Buffer::Buffer(BufferDeleter deleter):size_(0), buf_(NULL), deleter_(deleter)
{

}

Buffer::~Buffer()
{
	Free();
}

void Buffer::Allocate(int num_dims, int elm_size, IndexArray size)
{	
	num_dims_ = num_dims;
	elm_size_ = elm_size;
	size_ = size;
	assert(buf_ == NULL);	
	buf_ = GetChunk(size);
}

void Buffer::Free()
{
	if(deleter_)
		deleter_(buf_);
	buf_ = NULL;
	size_.Set(0);
}

//the deleter used by host buffer is the default "free"
BufferHost::BufferHost(): Buffer(free)
{

}

BufferHost::~BufferHost(){Free();}

//allocate from host memory
void *BufferHost::GetChunk(IndexArray &size)
{
	size_t s = size.accumulate(num_dims_);
	
	if(s == 0) return NULL;

	//printf("&&&&&&&CPU linear size: %ld\n", elm_size_*s);

	void *p = calloc(elm_size_, s);

  	assert(p);
  	return p;
}

void BufferHost::copy_in_or_out(BufferHost *src, 
			IndexArray &my_offset,
			IndexArray &src_offset,
			IndexArray &size, bool in)
{
	//printf("in buffer copy in or out\n");		
	int elm_size = elm_size_;		
	char *dst_addr = (char *)Get();
	char *src_addr = (char *)src->Get();

	int num_dims = num_dims_;

	IndexArray offset_tmp(0);
	IndexArray src_start;
	IndexArray dst_start;

	if(num_dims==3)
	{
		for(int i = 0; i < size[2]; i++)	
		{
			offset_tmp[2] = i;
			for(int j = 0; j < size[1]; j++)
			{
				offset_tmp[1] = j;
				src_start = src_offset + offset_tmp;
				dst_start = my_offset + offset_tmp;
				intptr_t linear_offset_src = GridCalcOffset3D(src_start, src->size()) * elm_size_;
				intptr_t linear_offset_dst = GridCalcOffset3D(dst_start, size_)* elm_size_;
				if(in)	
				{
					//printf("size 0: %d\n", size[0]);
					memcpy((char *)Get()+linear_offset_dst, 
					(char *)src->Get() + linear_offset_src, size[0]*elm_size_);
				}

				else
				{
					memcpy((char *)src->Get() + 
					linear_offset_src, (char *)Get() +
					linear_offset_dst, size[0]*elm_size_);
				}
			}
		}
	}

	else if(num_dims==2)
	{
		src_offset[2] = 0;
		my_offset[2] = 0;
		for(int i = 0; i < size[1]; i++)	
		{
			offset_tmp[1] = i;	
			src_start = src_offset + offset_tmp; 
			dst_start = my_offset + offset_tmp;

			intptr_t linear_offset_src = GridCalcOffset3D(src_start, src->size()) * elm_size_;
			intptr_t linear_offset_dst = GridCalcOffset3D(dst_start, size_) * elm_size_;
				
			if(in)
			{
				memcpy((char *)Get()+linear_offset_dst, 
				(char *)src->Get() + linear_offset_src, size[0]*elm_size_);
			}

			else
			{
				memcpy((char *)src->Get() + 
				linear_offset_src, (char *)Get() +
				linear_offset_dst,size[0]*elm_size_);
			}
		}
	}
}

void BufferHost::copy_in(BufferHost *src, 
			IndexArray &my_offset, 
			IndexArray &src_offset,
			IndexArray &size)
{
	copy_in_or_out(src, my_offset, src_offset, size, true);
}

void BufferHost::copy_out(BufferHost *src, 
			IndexArray &my_offset, 
			IndexArray &src_offset,
			IndexArray &size)
{
	copy_in_or_out(src, my_offset, src_offset, size, false);
}

