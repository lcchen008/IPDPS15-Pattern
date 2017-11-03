#ifndef LIB_GRID_CPU_H_
#define LIB_GRID_CPU_H_

#include "buffer.h"
#include "grid.h"
#include "array.h"
#include "grid_mpi.h"

class GridCuda;

class GridCPU:public GridMPI
{
public:
	int num_devices_;
	int device_id_;

	void copy_host_to_host(IndexArray &my_offset, IndexArray &src_offset, BufferHost *src, IndexArray &size);

	GridCPU(int unit_size, int num_dims, 
	const IndexArray &size,
        const IndexArray &my_offset,
        const IndexArray &my_size, 
        const Width2 &halo, int num_devices, int device_id);

	//GridCPU(const GridMPI &grid);

	static GridCPU *Create(int unit_size, int num_dims, 
			const IndexArray &size,
			const IndexArray &local_offset, 
			const IndexArray &localsize, 
			const Width2 &halo, 
			int num_devices, int device_id);	

	void set_num_devices(int num_devices);
	int num_devices(){return num_devices_;}
	//BufferHost *buffer() const
	//{
	//	return data_buffer_[0];	
	//}
	void send_to_neighbors(int dim, int stencil_width, GridMPI *grid_global, GridCuda **grid_cuda);

	//copy from the bottom of the global grid into the local cpu buffer halo area
	void copy_from_bottom(int dim, int stencil_width, GridMPI *grid_global);

	//copy from halo buffer to global send buffer
	void copy_to_global_grid(GridMPI *grid_global, int dim, int along_start, int along_length, unsigned width, bool fw);

	//copy from global receive buffer to my halo buffer
	void copy_from_global_grid(GridMPI *grid_global, int dim, int along_start, int along_length, unsigned width, bool fw);

	virtual void InitBuffer();
	virtual void DeleteBuffers();
};


#endif
