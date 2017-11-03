#ifndef LIB_GRID_CUDA_H_
#define LIB_GRID_CUDA_H_

#include "buffer.h"
#include "buffer_cuda.h"
#include "array.h"
#include "grid_mpi.h"

class GridCPU;

class GridCuda: public GridMPI
{
protected:
	int num_devices_;
	int device_id_;

	GridCuda(int unit_size, 
			int num_dims, 
			const IndexArray &size, 
			const IndexArray &my_offset, 
			const IndexArray &my_size, 
			const Width2 &halo,
			int num_devices,
			int device_id);

	BufferCUDADev *data_buffer_[2];
	char *data_[2];

	//halo data is put in zero-copy
	BufferCUDAHost **halo_self_fw_;
	BufferCUDAHost **halo_self_bw_;

	BufferCUDAHost **halo_peer_fw_;
	BufferCUDAHost **halo_peer_bw_;

	BufferCUDAHost *buffer_global_; //used for copying to global grid
	BufferCUDAHost *buffer_cpu_send_;   //used for copying to CPU grid
	BufferCUDAHost *buffer_cpu_recv_;   //used for copying from CPU grid

	cudaStream_t compute_stream_;
	cudaStream_t copy_out_stream_;
	cudaStream_t copy_in_stream_;

public:

	BufferCUDADev *buffer()
	{
		return data_buffer_[0];	
	}

	static GridCuda *Create(int unit_size,
      		int num_dims, const IndexArray &size,
      		const IndexArray &my_offset,
      		const IndexArray &my_size,
      		const Width2 &halo, int num_devices, int device_id);

	//copy a portion of data into GPU devie memory
	void copy_host_to_device(IndexArray &my_offset, 
			IndexArray &src_offset, 
			IndexArray &size, 
			BufferHost *src);

	//used to copy out border from device memory to host(mapped) memory
	void send_to_neighbors(int dim, int stencil_width, GridMPI *grid_global, GridCPU *grid_cpu, GridCuda **grid_cuda);

	//copies data from the top of global grid to cuda grid
	void copy_from_top(int dim, int stencil_width, GridMPI *grid_global);

	void copy_device_to_map(int dim, unsigned width, bool fw);
	
	void copy_map_to_device(int dim, unsigned width, bool fw);

	void copy_map_to_global_grid(GridMPI *grid_global, int dim, int along_start, int along_length, unsigned width, bool fw);

	void copy_global_grid_to_map(GridMPI *grid_global, int dim, int along_start, int along_length, unsigned width, bool fw);

	void copy_map_to_neighbor(int dim, int stencil_width, GridMPI *grid_global, GridCPU *grid_cpu);

	void copy_from_neighbor_map(int dim, int stencil_width, GridMPI *grid_global);

	//Buffer for sending forward
	BufferCUDAHost **halo_self_fw(){return halo_self_fw_;}
	//Buffer for sending backward
	BufferCUDAHost **halo_self_bw(){return halo_self_bw_;}

	//Buffer for receiving forward
	BufferCUDAHost **halo_peer_fw(){return halo_peer_fw_;}
	//Buffer for receiving backward
	BufferCUDAHost **halo_peer_bw(){return halo_peer_bw_;}

	BufferCUDAHost *buffer_global(){return buffer_global_;}

	BufferCUDAHost *buffer_cpu_send(){return buffer_cpu_send_;}

	BufferCUDAHost *buffer_cpu_recv(){return buffer_cpu_recv_;}

	char *data_in()
	{
		return data_[0]; 	
	}

	char *data_out()
	{
		return data_[1]; 	
	}

	void set_device_id(int device_id);
	void set_num_devices(int num_devices);
	int device_id(){return device_id_;} 
	int num_devices(){return num_devices_;}

	virtual void InitBuffer();
	virtual void InitHaloBuffers();
	virtual void DeleteBuffers();
	virtual void DeleteHaloBuffers();
};

#endif
