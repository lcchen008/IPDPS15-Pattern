#ifndef LIB_BUFFER_CUDA_H_
#define LIB_BUFFER_CUDA_H_

#include "array.h"
#include "buffer.h"

class BufferCUDAHost;

//buffer on device memory
class BufferCUDADev:public Buffer 
{
public:
  	BufferCUDADev();
  	virtual ~BufferCUDADev();

	void copy_in_or_out(BufferHost *src, 
			IndexArray &my_offset,
			IndexArray &src_offset,
			IndexArray &size, bool in);

	void copy_in_from_map(BufferCUDAHost *src, 
			IndexArray &src_offset,
			IndexArray &my_offset, 
			IndexArray &size, cudaStream_t stream);

	void copy_out_to_map(BufferCUDAHost *dst, 
			IndexArray &dst_offset,
			IndexArray &my_offset, 
			IndexArray &size, cudaStream_t stream);

	void copy_in_from_host(BufferHost *src, 
			IndexArray &my_offset, 
			IndexArray &src_offset,
			IndexArray &size);

	void copy_out_to_host(BufferHost *src, 
			IndexArray &my_offset, 
			IndexArray &src_offset,
			IndexArray &size);
	
protected:
  	virtual void *GetChunk(IndexArray &size);
private:
  	static void DeleteChunk(void *ptr);
};

// Pinned buffer on *host* memory
class BufferCUDAHost: public Buffer 
{
public:
  	BufferCUDAHost();
  	virtual ~BufferCUDAHost();
	cudaStream_t strm(){return strm_;}
	void *DeviceBuf(){return buf_d_;};
  
protected:
	cudaStream_t strm_;
	void *buf_d_;
  	virtual void *GetChunk(IndexArray &size);
private:
  	static void DeleteChunk(void *ptr);  
};

#endif
