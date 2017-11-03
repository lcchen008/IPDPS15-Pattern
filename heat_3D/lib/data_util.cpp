#include "data_util.h"
#include "util.h"
#include "macro.h"
#include <string.h>
#include <list>
#include "DS.h"
#include <stdio.h>
#include <iostream>

using namespace std;


static size_t get1DOffset(const IndexArray &md_offset,
		          const IndexArray &size,
			  int num_dims) 
{
	size_t offset_1d = 0;
	size_t ref_offset = 1;

	for (int i = 0; i < num_dims; ++i) 
	{
		offset_1d += (md_offset[i] * ref_offset);
		ref_offset *= size[i];
	}

        return offset_1d;
}

// Copy a continuous sub grid.
/*
 * This is a special case of CopySubgrid.
 *
 * \param buf Destination buffer.
 * \param src Source buffer.
 * \param elm_size Element size in bytes.
 * \param num_dims Number of dimensions.
 * \param grid_size Number of elements of each dimension in the grid.
 * \param subgrid_offset Offset to copy from or into.
 * \param subgrid_size Size of sub grid.
 * \param is_copyin Flag to indicate copyin or copyout.
 */
static void CopyContinuousSubgrid(
    	void *buf,const void *src,
    	size_t elm_size, int num_dims,
    	const IndexArray &grid_size,
    	const IndexArray &subgrid_offset,
    	const IndexArray &subgrid_size,
    	bool is_copyin) 
{
  	intptr_t linear_offset = GridCalcOffset3D(subgrid_offset, grid_size) * elm_size;
  	size_t size = subgrid_size.accumulate(num_dims) * elm_size;
  	if(is_copyin) 
	{
    		memcpy((void*)(((intptr_t)buf) + linear_offset), src, size);
  	} 

	else 
	{
    		memcpy(buf, (void*)(((intptr_t)src) + linear_offset), size);
  	}
}

//buf is always the dst, it may be the grid, if is_copyin is true, otherwise, it is the halo buffer
static void CopySubgrid(void *buf, const void *src,
                        size_t elm_size, int num_dims,
                        const IndexArray &grid_size,
                        const IndexArray &subgrid_offset,
                        const IndexArray &subgrid_size,
                        bool is_copyin) 
{
	bool continuous = true;
  	for (int i = 0; i < num_dims - 1; ++i) 
	{
    		if (subgrid_offset[i] != 0 ||
        	subgrid_size[i] != grid_size[i]) 
		{
      			continuous = false;
      			break;
    		}
  	}

  	if (continuous) 
	{
    		CopyContinuousSubgrid(buf, src, elm_size,
                          num_dims, grid_size,
                          subgrid_offset, subgrid_size,
                          is_copyin);
    		return;
  	}

  	intptr_t subgrid_original = (intptr_t) (is_copyin ? src : buf);
  	std::list<IndexArray> *offsets = new std::list<IndexArray>;
  	std::list<IndexArray> *offsets_new = new std::list<IndexArray>;

	LOG_DEBUG() << ": CopySubgrid"
              << "subgrid offset: " << subgrid_offset
              << "subgrid size: " << subgrid_size
              << "grid size: " << grid_size
              << "\n";

  	// Collect 1-D continuous regions
  	offsets->push_back(subgrid_offset);

  	for (int i = num_dims - 1; i >= 1; --i) 
	{
    		FOREACH (oit, offsets->begin(), offsets->end()) 
		{
      			const IndexArray &cur_offset = *oit;
      			for (PSIndex j = 0; j < subgrid_size[i]; ++j) 
			{
        			IndexArray new_offset = cur_offset;
        			new_offset[i] += j;

        			// support periodic access
        			new_offset[i] = (new_offset[i] + grid_size[i]) %  grid_size[i];
        			offsets_new->push_back(new_offset);
      			}
    		}

    		std::swap(offsets, offsets_new);
    		offsets_new->clear();
  	}

	// Copy the collected 1-D continuous regions. The region is allowed
	// to periodically access off grid boundaries
	
	// Copy the collected 1-D continuous regions. The region is allowed
	// to periodically access off grid boundaries
  	FOREACH (oit, offsets->begin(), offsets->end()) 
	{
    		IndexArray &offset = *oit;
    		PSIndex next_offset;
    		PSIndex initial_size;
    		IndexArray ss = subgrid_size;
    		bool done = false;
    		while (!done) 
		{
      			initial_size = subgrid_size[0];
      			if (offset[0] < 0) 
			{
        			// Copy sub region with backward periodic access
        			offset[0] = (offset[0] + grid_size[0]) % grid_size[0];
        			ss[0] = grid_size[0] - offset[0];
        			next_offset = 0;
      			} 
			
			else if (offset[0] + ss[0] > grid_size[0]) 
			{
        			// Copy data only within the continuous region and leave the
        			// remaining, which then results in periodic access
        			ss[0] = grid_size[0] - offset[0];
        			next_offset = 0;
      			} 

			else 
			{
        			done = true;
      			}
      			
			size_t grid_1d_offset =
          			get1DOffset(offset, grid_size, num_dims) * elm_size;

      			const void *src_ptr;
      			void *dst_ptr;

      			if (is_copyin) 
			{
      			  	src_ptr = src;
      			  	dst_ptr = (void *)((intptr_t)buf + grid_1d_offset);
      			} 

			else 
			{
      			  	src_ptr = (const void *)((intptr_t)src + grid_1d_offset);
      			  	dst_ptr = buf;
      			}

      			size_t line_size = ss[0] * elm_size;
      			memcpy(dst_ptr, src_ptr, line_size);

      			if (is_copyin) 
			{
      			  	src = (const void*)((intptr_t)src + line_size);
      			} 
			
			else 
			{
      			  	buf = (void*)((intptr_t)buf + line_size);
      			}

      			offset[0] = next_offset;
      			ss[0] = initial_size - ss[0];
    		}
  	}

  	delete offsets;
  	delete offsets_new;
}

void CopyoutSubgrid(size_t elm_size, int num_dims,
                    const void *grid, const IndexArray &grid_size,
                    void *subgrid,
                    const IndexArray &subgrid_offset,
                    const IndexArray &subgrid_size) 
{
  	CopySubgrid(subgrid, grid, elm_size, num_dims, grid_size,
              	subgrid_offset, subgrid_size, false);
  	return;
}

void CopyinSubgrid(size_t elm_size, int num_dims,
                   void *grid, const IndexArray &grid_size,
                   const void *subgrid,
                   const IndexArray &subgrid_offset,
                   const IndexArray &subgrid_size) 
{
  	CopySubgrid(grid, subgrid, elm_size, num_dims,
              grid_size, subgrid_offset, subgrid_size,
              true);

  	return;
}

void tiling(int num_dims, IndexArray &size, IndexArray &start_offset, int tile_size, std::vector<struct Tile> &offset_vector)
{
	IndexArray offset_new;
	IndexArray part_size(tile_size);

	if(num_dims == 2)
	{
		int dim1_tiles = ceil((double)size[1]/tile_size);
		int dim0_tiles = ceil((double)size[0]/tile_size);

		for(int i = 0; i < dim1_tiles; i++)	
		{
			offset_new[1] = start_offset[1] + tile_size * i;
			part_size[1] = tile_size;

			if((size[1]%tile_size != 0) && (i == dim1_tiles - 1))
			{
				part_size[1] = size[1] - i * tile_size;	
			}

			for(int j = 0; j < dim0_tiles; j++)
			{
				offset_new[0] = start_offset[0] + tile_size * j;	
				part_size[0] = tile_size;

				if((size[0]%tile_size != 0) && (j == dim0_tiles - 1))
				{
					part_size[0] = size[0] - j * tile_size;	
				}

				struct Tile tile_tmp = {offset_new, part_size};				
				offset_vector.push_back(tile_tmp);
			}
		}
	}

	else if(num_dims == 3)
	{
		int dim2_tiles = ceil((double)size[2]/tile_size);
		int dim1_tiles = ceil((double)size[1]/tile_size);
		int dim0_tiles = ceil((double)size[0]/tile_size);

		//printf("tiles: %d %d %d\n", dim0_tiles, dim1_tiles, dim2_tiles);		

		for(int i = 0; i < dim2_tiles; i++)	
		{
			offset_new[2] = start_offset[2] + tile_size * i;
			part_size[2] = tile_size;

			if((size[2]%tile_size!=0) && (i == dim2_tiles - 1))
			{
				part_size[2] = size[2] - i * tile_size;	
			}

			for(int j = 0; j < dim1_tiles; j++)
			{
				offset_new[1] = start_offset[1] + tile_size * j;	
				part_size[1] = tile_size;

				if((size[1]%tile_size!=0) && (j == dim1_tiles - 1))
				{
					part_size[1] = size[1] - j * tile_size;	
				}

				for(int k = 0; k < dim0_tiles; k++)
				{
					offset_new[0] = start_offset[0] + tile_size * k;
					part_size[0] = tile_size;
					
					if((size[0]%tile_size != 0)&&(k == dim0_tiles - 1))
					{
						part_size[0] = size[0] - k * tile_size;	
					}
	
					//if(part_size==0&&offset_new==0)
					//printf("current tile: %d, %d, %d, size: %d, %d, %d\n", offset_new[0], offset_new[1], offset_new[2], part_size[0], part_size[1],  part_size[2]);

					struct Tile tile_tmp = {offset_new, part_size};				
					offset_vector.push_back(tile_tmp);
				}
			}
		}
	}
}

//size is the logical size of the grid, real_size is the size representing the actual space it takes
void tiling_border(int num_dims, IndexArray &size, IndexArray &real_size, Width2 halo, int tile_size, std::vector<struct Tile> &offset_vector)
{
	IndexArray start = halo.bw;		
	IndexArray size_to_tile = size;

	for(int i = 0; i < num_dims; i++)
	{
		//first tile bw	 border
		if(halo.bw[i])
		{
			start = halo.bw;		
			size_to_tile = size;
			size_to_tile[i] = halo.bw[i];
			tiling(num_dims, size_to_tile, start, tile_size, offset_vector);
		}

		//then tile fw border
		if(halo.fw[i])
		{
			start = halo.bw;
			size_to_tile = size;
			start[i] = real_size[i] - 2 * halo.fw[i];
			size_to_tile[i] = halo.fw[i];
			tiling(num_dims, size_to_tile, start, tile_size, offset_vector);
		}
	}
}

size_t GetLinearSize(int num_dims, size_t elm_size, const IndexArray &s)
{
	return s.accumulate(num_dims) * elm_size;
}
