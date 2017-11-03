#ifndef LIB_BUFFER_H_
#define LIB_BUFFER_H_

#include <cstdlib>
#include "array.h"


class Buffer
{

protected:
	//define a function type
	typedef void (*BufferDeleter) (void *); 	
	//create a buffer
	explicit Buffer(BufferDeleter deleter);
public:
	virtual ~Buffer();
	void Allocate(int num_dims, int elm_size, IndexArray size);
	const void *Get() const {return buf_;}
	//return the pointer to data memory chunk
	void *Get() {return buf_;}
	//return size of buffer
	IndexArray &size() {return size_;}
	int num_dims() const{return num_dims_;}
	int elm_size() const{return elm_size_;}
	//virtual void copy_in(Buffer &src, IndexArray &my_offset, IndexArray &src_offset, IndexArray &size);
	//free the buffer
	void Free();
	void *buf_;

			
protected:
	//pure virtual func, used to allocate a space from mem/device mem
	virtual void *GetChunk(IndexArray &size) = 0;

	//size of buffer
	IndexArray size_;
	//memory chunk for the real data
	int num_dims_;
	int elm_size_;

	//function to delete buffer
	void (*deleter_)(void *ptr);
};

class BufferHost:public Buffer
{
public:
	BufferHost();
	virtual ~BufferHost();

	void copy_in_or_out(BufferHost *src, 
			IndexArray &my_offset,
			IndexArray &src_offset,
			IndexArray &size, bool in);
	virtual void copy_in(BufferHost *src, 
			IndexArray &my_offset, 
			IndexArray &src_offset,
			IndexArray &size);
	void copy_out(BufferHost *src, 
			IndexArray &my_offset, 
			IndexArray &src_offset,
			IndexArray &size);

protected:
	virtual void *GetChunk(IndexArray &size);
};

#endif
