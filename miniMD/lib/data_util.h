#ifndef LIB_DATA_UTIL_H_
#define LIB_DATA_UTIL_H_

#include "common.h"
#include "array.h"
#include <vector>
#include "DS.h"

#define _OFFSET_TYPE PSIndex

inline _OFFSET_TYPE GridCalcOffset3D(PSIndex x, PSIndex y, PSIndex z,
                                     PSIndex xsize, PSIndex ysize) 
{
  	return ((_OFFSET_TYPE)x) + ((_OFFSET_TYPE)y) * ((_OFFSET_TYPE)xsize)
      	+ ((_OFFSET_TYPE)z) * ((_OFFSET_TYPE)xsize) * ((_OFFSET_TYPE)ysize);
}

inline _OFFSET_TYPE GridCalcOffset3D(PSIndex x, PSIndex y, PSIndex z,
                                     const IndexArray &size) 
{
  	return GridCalcOffset3D(x, y, z, size[0], size[1]);
}

inline intptr_t GridCalcOffset3D(const IndexArray &index,
                                 const IndexArray &size) 
{
  	return GridCalcOffset3D(index[0], index[1], index[2], size[0], size[1]);
}

void CopyoutSubgrid(size_t elm_size, int num_dims,
                    const void *grid, const IndexArray &grid_size,
                    void *subgrid,
                    const IndexArray &subgrid_offset,
                    const IndexArray &subgrid_size);

void CopyinSubgrid(size_t elm_size, int num_dims,
                   void *grid, const IndexArray &grid_size,
                   const void *subgrid,
                   const IndexArray &subgrid_offset,
                   const IndexArray &subgrid_size); 


void tiling(int num_dims, IndexArray &size, 
		IndexArray &start_offset, int tile_size, 
		std::vector<struct Tile> &offset_vector);

void tiling_border(int num_dims, IndexArray &size, IndexArray &real_size, Width2 halo_, int tile_size, std::vector<struct Tile> &offset_vector);

size_t GetLinearSize(int num_dims, size_t elm_size, const IndexArray &s);
#endif
