#include "grid_mpi.h"
#include "data_util.h"
#include "macro.h"
//#include "buffer_cuda.h"
#include <string.h>
#include <iostream>
#include <stdio.h>
using namespace std;

size_t GridMPI::CalcHaloSize(int dim, unsigned width) 
{
  IndexArray halo_size = my_real_size_;
  halo_size[dim] = width;
  return halo_size.accumulate(num_dims_);
}

GridMPI::GridMPI(int unit_size, int num_dims,
          const IndexArray &size,
          const IndexArray &my_offset,
          const IndexArray &my_size, 
          const Width2 &halo):
	Grid(unit_size, num_dims, size),
	my_offset_(my_offset),
	my_size_(my_size),
	halo_(halo),
	halo_self_fw_(NULL),
	halo_self_bw_(NULL),
	halo_peer_fw_(NULL),
	halo_peer_bw_(NULL)
{
	my_real_size_ = my_size_;	
	my_real_offset_ = my_offset_;
	for(int i = 0; i < num_dims; i++)
	{
		my_real_size_[i] += halo.fw[i] + halo.bw[i];
    		my_real_offset_[i] -= halo.bw[i];
	}

}

GridMPI *GridMPI::Create(
    	int unit_size,
    	int num_dims, const IndexArray &size,
    	const IndexArray &local_offset,
    	const IndexArray &local_size,
    	const Width2 &halo)
{
  GridMPI *g = new GridMPI(
      	unit_size,
      	num_dims, size,
      	local_offset,
      	local_size,
      	halo);

  	g->InitBuffer();

  	return g;
}

//we are using double buffering
void GridMPI::InitBuffer() 
{
  	data_buffer_[0] = new BufferHost();
  	data_buffer_[0]->Allocate(num_dims_, unit_size_, my_real_size_);
  	data_[0] = (char*)data_buffer_[0]->Get();
  	data_buffer_[1] = new BufferHost();
	data_buffer_[1]->Allocate(num_dims_, unit_size_, my_real_size_);
  	data_[1] = (char*)data_buffer_[1]->Get();

	
  	InitHaloBuffers();
}

void GridMPI::InitHaloBuffers()
{
	halo_self_fw_ = new BufferHost*[num_dims_];
  	halo_self_bw_ = new BufferHost*[num_dims_];
  	halo_peer_fw_ = new BufferHost*[num_dims_];
  	halo_peer_bw_ = new BufferHost*[num_dims_];

  
  	for (int i = 0; i < num_dims_; ++i) 
	{
    		halo_self_fw_[i] = halo_self_bw_[i] = NULL;
    		halo_peer_fw_[i] = halo_peer_bw_[i] = NULL;

    		if (halo_.fw[i]) 
		{
      			halo_self_fw_[i] = new BufferHost();
      			halo_peer_fw_[i] = new BufferHost();

			IndexArray size = my_real_size_;
			size[i] = halo_.fw[i];
			
			halo_self_fw_[i]->Allocate(num_dims_, unit_size_, size);
			halo_peer_fw_[i]->Allocate(num_dims_, unit_size_, size);
			//printf("halo: #######%ld\n", halo_peer_fw_[i]);	
    		} 

    		if (halo_.bw[i]) 
		{
      			IndexArray size = my_real_size_; 
			size[i] = halo_.bw[i]; 

			halo_self_bw_[i]= new BufferHost();	
			halo_peer_bw_[i]= new BufferHost();	
			halo_self_bw_[i]->Allocate(num_dims_, unit_size_, size);
			halo_peer_bw_[i]->Allocate(num_dims_, unit_size_, size);
    		} 
  	}

	printf("Initting.......................\n");

	//IndexArray copy_size = my_real_size_; 
	//copy_size[num_dims_-1] = halo_.bw[num_dims_-1]; 
	//cout<<"&&&&&&&&&&&&my real size: "<<copy_size<<endl;
	//BufferCUDAHost *copy_buffer_ = new BufferCUDAHost(); 

	//copy_buffer_ -> Allocate(num_dims_, unit_size_, copy_size);

	//delete copy_buffer_;

	
	//IndexArray copy_size = my_real_size_; 
	//copy_size[num_dims_-1] = halo_.bw[num_dims_-1]; 
	//cout<<"&&&&&&&&&&&&my real size: "<<copy_size<<endl;
	//BufferCUDAHost *copy_buffer_ = new BufferCUDAHost();

	//size_t s = copy_size.accumulate(num_dims_); 

  	//if (s == 0) exit(-1);

	//size_t linear_size = GetLinearSize(num_dims_, unit_size_, copy_size);
	////printf("&&&&&&linear size: %ld, elm_size_: %ld&&&&&&\n", linear_size, elm_size_);
	////cout<<"multi dim size: "<<size<<endl;

  	//void *ptr = NULL;  

	//CUDA_SAFE_CALL(cudaHostAlloc(&ptr, 
	//linear_size, 
	//cudaHostAllocPortable|cudaHostAllocMapped));

	//printf("linear size: %ld ptr: %ld\n", linear_size, ptr);

	//copy_buffer_->buf_ = ptr;

	/////copy_buffer_ -> Allocate(num_dims_, unit_size_, copy_size);
	//delete copy_buffer_;

	//printf("^^^^^^^^^^^^^^^^^^buffer pointer: %ld\n", copy_buffer_->Get());
	//printf("^^^^^^^^^^^^^^^^^^Device buffer pointer: %ld\n", copy_buffer_->DeviceBuf());
}

//copy data from Halo to grid 
void GridMPI::CopyinHalo(int dim, unsigned width, bool fw)
{
	//highest dim does not need copying
	if(dim == num_dims_ -1)
		return;

	IndexArray halo_offset(0);

	if(fw) 
	{
    		halo_offset[dim] = my_real_size_[dim] - halo_.fw[dim];
  	} 

	else 
	{
    		halo_offset[dim] = halo_.bw[dim] - width;
  	}

  	char *halo_buf = (char *)((fw ? halo_peer_fw_[dim] : halo_peer_bw_[dim])->Get());

  	IndexArray halo_size = my_real_size_;

  	halo_size[dim] = width;

  	CopyinSubgrid(unit_size_, num_dims_, data_[0], my_real_size_,
                halo_buf, halo_offset, halo_size);
}

//copy halo from the grid into the (send) buffer
//note that the memory area to be sent is within the private grid
//, other than the halo area
void GridMPI::CopyoutHalo(int dim, unsigned width, bool fw)
{
  	IndexArray halo_offset(0);

  	if(fw) 
	{
    		halo_offset[dim] = halo_.bw[dim];
  	} 
	
	else 
	{
    		halo_offset[dim] = my_real_size_[dim] - halo_.fw[dim] - width;
  	}

	//std::cout << "halo offset: "
        //      << halo_offset << "\n";

	//char *tmp;
  	//char **halo_buf = &tmp; 
	
	//char *halo_buf = fw ? (char *)(halo_self_fw_[dim]->buf_) : (char *)(halo_self_bw_[dim]->buf_);

  	// The slowest changing dimension does not need actual copying
  	// because its halo region is physically continuous.
  	if (dim == (num_dims_ - 1)) 
	{
    		char *p = data_[0] + GridCalcOffset3D(halo_offset, my_real_size_) * unit_size_;
    		//*halo_buf = p;
		if(fw)
		{
			halo_self_fw_[dim]->buf_ = p;
		}
		else
		{
			halo_self_bw_[dim]->buf_ = p;	
		}

    		return;
  	} 

	else 
	{
    		//IndexArray halo_size = my_real_size_;
    		//halo_size[dim] = width;
    		//CopyoutSubgrid(unit_size_, num_dims_, data_[0], my_real_size_, halo_buf, halo_offset, halo_size);

    		return;
  	}
}

void GridMPI::DeleteBuffers()
{
	DeleteHaloBuffers();
	Grid::DeleteBuffers();
}

GridMPI::~GridMPI()
{
	DeleteBuffers();
}

void GridMPI::DeleteHaloBuffers()
{
	for (int i = 0; i < num_dims_ - 1; ++i) 
	{
    		if (halo_self_fw_) delete (halo_self_fw_[i]);
    		if (halo_self_bw_) delete (halo_self_bw_[i]);
    		if (halo_peer_fw_) delete (halo_peer_fw_[i]);
    		if (halo_peer_bw_) delete (halo_peer_bw_[i]);
  	}

	if(halo_self_fw_)
  		PS_XDELETEA(halo_self_fw_);
	if(halo_self_bw_)
  		PS_XDELETEA(halo_self_bw_);
	if(halo_peer_fw_)
  		PS_XDELETEA(halo_peer_fw_);
	if(halo_peer_bw_)
  		PS_XDELETEA(halo_peer_bw_);
	//printf("Deleting...................................\n");
	//delete copy_buffer_;
}

void GridMPI::Copyin(const void *src)
{
  	void *dst = buffer()->Get();
  	if (HasHalo()) 
	{
    		CopyinSubgrid(unit_size(), num_dims_,
                dst, my_real_size(),
                src, halo_.bw, my_size());
  	} 

	else 
	{
    		memcpy(dst, src, GetLocalBufferSize());
  	}

  	return;
}

void GridMPI::Copyout(void *dst)
{
  	const void *src = buffer()->Get();
  	if (HasHalo()) 
	{
    		IndexArray offset(halo_.bw);
    		CopyoutSubgrid(unit_size(), num_dims_,
                src, my_real_size(),
                dst, offset, my_size());
  	} 
	
	else 
	{
    		memcpy(dst, src, GetLocalBufferSize());
  	}

  	return;
}
