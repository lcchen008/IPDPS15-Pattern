#ifndef LIB_GRID_MPI_H_
#define LIB_GRID_MPI_H_

#include "grid.h"
#include "array.h"
#include "DS.h"
#include "buffer_cuda.h"

class Grid_view;

class GridMPI: public Grid
{

friend class Grid_view;

protected:
	GridMPI(int unit_size, int num_dims,
        const IndexArray &size, //global offset
        const IndexArray &my_offset,
        const IndexArray &my_size, 
        const Width2 &halo);

	GridMPI(){}

	//////////////device info///////////////
	int num_devices_;

	/////////////logical data//////////////
	//offset related to the global grid
	IndexArray my_offset_;	
	//size of the subgrid
	IndexArray my_size_;

	////////////real data including halo/////////////
	Width2 halo_;
	IndexArray my_real_offset_; 
	IndexArray my_real_size_;

	BufferHost **halo_self_fw_;
	BufferHost **halo_self_bw_;

	BufferHost **halo_peer_fw_;
	BufferHost **halo_peer_bw_;


	size_t CalcHaloSize(int dim, unsigned width);    
  
  	//! Allocates buffers, including halo buffers.
  	virtual void InitBuffer();
  	//! Allocates buffers for halo communications.
  	virtual void InitHaloBuffers();
  	//! Deletes buffers, including halo buffers.
  	virtual void DeleteBuffers();
  	//! Deletes halo buffers.
  	virtual void DeleteHaloBuffers();

	//returns buffer for remote halo
  	char *GetHaloPeerBuf(int dim, bool fw, unsigned width);

	//copy grid data into this grid
	virtual void Copyin(const void *src);
	//copy grid data out from this grid
	virtual void Copyout(void *dst);	

	
public:
	static GridMPI *Create(
      		int unit_size,
      		int num_dims, const IndexArray &size,
      		const IndexArray &my_offset,
      		const IndexArray &my_size,
      		const Width2 &halo);

  	const IndexArray& my_size() const { return my_size_; }  
  	const IndexArray& my_offset() const { return my_offset_; }
  	const IndexArray& my_real_size() const { return my_real_size_; }

  	Width2 &halo() { return halo_; }
  	bool HasHalo() const { return ! (halo_.fw == 0 && halo_.bw == 0); }  

	//Buffer for sending forward
	BufferHost **halo_self_fw(){return halo_self_fw_;}
	//Buffer for sending backward
	BufferHost **halo_self_bw(){return halo_self_bw_;}

	//Buffer for receiving forward
	BufferHost **halo_peer_fw(){return halo_peer_fw_;}
	//Buffer for receiving backward
	BufferHost **halo_peer_bw(){return halo_peer_bw_;}

	//BufferCUDAHost *copy_buffer(){return copy_buffer_;}

  	size_t GetLocalBufferRealSize() const 
	{
    		return my_real_size_.accumulate(num_dims_) * unit_size_;
  	}

	size_t GetLocalBufferSize()
	{
		return my_size_.accumulate(num_dims_) * unit_size_;	
	}

	//copy halo data into this grid
	virtual void CopyinHalo(int dim, unsigned width, bool fw);
	//copy halo from the grid into the (send) buffer
  	virtual void CopyoutHalo(int dim, unsigned width, bool fw);

	void copy_to_copy_buffer();

	void copy_from_copy_buffer();

	virtual ~GridMPI();
};

#endif
