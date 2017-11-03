#ifndef CPU_KERNEL
#define CPU_KERNEL

#endif

#include "compute_cpu.h"
#include "grid_view.h"
#include "grid_mpi.h"
#include "DS.h"
#include "cpu_util.h"
#include <math.h>
#include "../stencil.h"
#include <stdio.h>
#include <iostream>
#include "array.h"
using namespace FCPU;

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


void process_a_tile(int num_dims, int unit_size, void *input, void *output, struct Tile &tile, IndexArray &size, stencil_fp stencil_ptr)
{
	IndexArray offset_tmp;
	//std::cout<<"tile is: "<<tile<<std::endl;

	if(num_dims == 2)	
	{
		for(int i = 0; i < tile.size[1]; i++)	
			for(int j = 0; j < tile.size[0]; j++)
			{
				offset_tmp[0] = j;
				offset_tmp[1] = i;
				offset_tmp += tile.offset;
				
				//FCPU::stencil(input, output, (int *)&offset_tmp[0], (int *)&size[0]);
				stencil_ptr(input, output, (int *)&offset_tmp[0], (int *)&size[0]);
			}
	}

	else if(num_dims == 3)
	{
		for(int i = 0; i < tile.size[2]; i++)	
			for(int j = 0; j < tile.size[1]; j++)
			{
				for(int k = 0; k < tile.size[0]; k++)
				{
					offset_tmp[0] = k;
					offset_tmp[1] = j;
					offset_tmp[2] = i;

					offset_tmp += tile.offset;

					//FCPU::stencil(input, output, (int *)&offset_tmp[0], (int *)&size[0]);
					stencil_ptr(input, output, (int *)&offset_tmp[0], (int *)&size[0]);
				}
			}
	}
}

void *compute_tiles(void *arg)
{
	Cargs *args = (Cargs *)arg;		
	int tid = args->tid;
	StencilRuntime *runtime = args->runtime;

	int width = runtime->stencil_width();	
	int num_dims = runtime->num_dims();
	int unit_size = runtime->unit_size();
	GridMPI *grid = runtime->grid_cpu();

	char *input = grid->data_in();

	char *output = grid->data_out();

	int stencil_idx = runtime->get_stencil_idx();

	stencil_fp stencil_ptr = stencil_function_table[stencil_idx];

	IndexArray total_size = grid->my_real_size();

	//std::cout<<"^^^total_size^^^^^: "<<total_size<<std::endl;

	std::vector<struct Tile> tiles = *(args->tiles);

	int num_tiles = tiles.size();

	if(tid == 0)
		printf("##########num_tiles: %d\n", num_tiles);

	int tiles_per_proc = floor((double)num_tiles/CPU_THREADS); 

	int rem = num_tiles - tiles_per_proc * CPU_THREADS;

	int start = tid * tiles_per_proc;

	int end = (tid + 1) * tiles_per_proc;

	//printf("thread %d is computing... start is: %d and end is: %d\n", tid, start, end);
	//printf("input: %ld, output: %ld\n", input, output);

	for(int i = start; i < end; i++)
	{
		//printf("processing a tile**************\n");
		struct Tile tile = tiles[i];	
		//printf("current tile: %d, %d, %d, size: %d, %d, %d\n", tile.offset[0], tile.offset[1], tile.offset[2], tile.size[0], tile.size[1], tile.size[2]);
		process_a_tile(num_dims, unit_size, input, output, tile, total_size, stencil_ptr);
	}

	//process remaining tiles
	if(tid < rem)
	{
		struct Tile tile = tiles[tiles_per_proc * CPU_THREADS + tid];	
		process_a_tile(num_dims, unit_size, input, output, tile, total_size, stencil_ptr);
	}

	return (void *)0;
}
