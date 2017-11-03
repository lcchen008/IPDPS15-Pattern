#include "grid_view.h"
#include "grid_mpi.h"
#include "array.h"
#include "macro.h"
#include <mpi.h> 
#include <iostream>
#include "data_util.h"
#include "util.h"

using namespace std;

static void partition(int num_dims, int num_procs, const IndexArray &size, 
		const IndexArray &num_partitions, int **partitions, 
		int **offsets, std::vector<IndexArray> &proc_indices, 
		IndexArray &min_partition)
{
	for(int i = 0; i < num_procs; i++)
	{
		int t = i;
		IndexArray idx;	
		for(int j = 0; j < num_dims; j++)
		{
			idx[j] = t % num_partitions[j];	
			t /= num_partitions[j]; 	
		}

		proc_indices.push_back(idx);
	}

	min_partition.Set(INT_MAX);

	for(int i = 0; i < num_dims; i++)
	{
		partitions[i] = new int[num_partitions[i]];	
		offsets[i] = new int[num_partitions[i]];	
		int offset = 0;

		for(int j = 0; j < num_partitions[i]; j++)
		{
			int rem = size[i]%num_partitions[i];
			partitions[i][j] = size[i]/num_partitions[i];	
			if(num_partitions[i] - j <= rem)
			{
				partitions[i][j]++;
			}
			offsets[i][j] = offset;
			offset += partitions[i][j];
			min_partition[i] = std::min(min_partition[i], partitions[i][j]);
		}
	}
}

int Grid_view::GetProcessRank(const IndexArray &proc_index) const 
{
	int rank = 0;
	int offset = 1;
	for (int i = 0; i < num_dims; ++i) 
	{
		rank += proc_index[i] * offset;
		offset *= proc_size[i];
        }

        return rank;
}

//global_size here is the global size
Grid_view::Grid_view(int num_dims, IndexArray &global_size, int proc_num_dims, 
		IndexArray &proc_size, int my_rank):num_dims(num_dims),
		global_size(global_size),proc_num_dims(proc_num_dims),
		proc_size(proc_size),my_rank(my_rank)
{
	partitions = new int* [num_dims];
	offsets = new int* [num_dims];
	num_procs = proc_size.accumulate(proc_num_dims);

	partition(num_dims, num_procs, global_size, proc_size, partitions, offsets, proc_indices, min_partition);	
	my_idx = proc_indices[my_rank];

	for (int i = 0; i < num_dims; ++i) 
	{
		my_offset[i] = offsets[i][my_idx[i]]; // For example {0,0,31}
	        my_size[i] = partitions[i][my_idx[i]]; // For example {0,0,11}
	}

	//comm = MPI_COMM_WORLD;	

 	for (int i = 0; i < num_dims; ++i) 
	{
    		IndexArray neighbor = my_idx; // Usually {0, 0, my_rank_}
    		neighbor[i] += 1;
    		// wrap around for periodic boundary access
    		neighbor[i] %= proc_size[i];
    		fw_neighbors[i] = GetProcessRank(neighbor);
    		neighbor[i] = my_idx[i] - 1;
    		// wrap around for periodic boundary access
    		if (neighbor[i] < 0) 
		{
      			neighbor[i] = (neighbor[i] + proc_size[i]) % proc_size[i];
    		}

    		bw_neighbors[i] = GetProcessRank(neighbor);
  	}

	if(my_rank == 0)
	{
		cout<<"my rank: "<<my_rank<<endl;
		cout<<"my backward neighbors: "<<bw_neighbors<<endl;
		cout<<"my forward neighbors: "<<fw_neighbors<<endl;
	}
}

//size here indicates the local size in my process
GridMPI *Grid_view::CreateGrid(int unit_size, int num_dims, 
		const IndexArray &size, const int halo_width)
{
	IndexArray halo_fw;				
	halo_fw.Set(halo_width);
	IndexArray halo_bw;
	halo_bw.Set(halo_width);

	for(int i = 0; i < num_dims; i++)
	{
		if(my_size[i] == 0 /*|| proc_size[i] == 1*/)	
		{
			halo_fw[i] = 0;
			halo_bw[i] = 0;
		}
	}

	Width2 halo = {halo_bw, halo_fw};

	GridMPI *g = GridMPI::Create(unit_size, num_dims, size, my_offset, my_size, halo); 

	//cout<<"my rank: "<<my_rank<<"; my idx: "<<my_idx<<"; my size: "<<my_size<<"; my offset: "<<my_offset<<endl;

	return g;
}

char *GridMPI::GetHaloPeerBuf(int dim, bool fw, unsigned width) 
{
  	if (dim == num_dims_ - 1) 
	{
    		IndexArray offset(0);

    		if (fw) 
		{
      			offset[dim] = my_real_size_[dim] - halo_.fw[dim];
    		} 

		else 
		{
      			offset[dim] = halo_.fw[dim] - width;
    		}

    		return data() + GridCalcOffset3D(offset, my_real_size_) * unit_size_;
  	} 

	else 
	{
    		if (fw) return (char *)(halo_peer_fw_[dim]->Get());
    		else  return (char *)(halo_peer_bw_[dim]->Get());
  	}
}


void Grid_view::ExchangeBoundariesAsync(GridMPI *grid, 
			int dim, unsigned halo_fw_width, 
			unsigned halo_bw_width, bool diagonal, 
			bool periodic, std::vector<MPI_Request> &requests) const
{
	int fw_peer = fw_neighbors[dim];
	int bw_peer = bw_neighbors[dim]; 
	cout<<"I am: "<<my_rank << " and my fw peer is: "<< fw_peer<<" my bw peer is: "<<bw_peer<<endl;
	size_t fw_size = grid->CalcHaloSize(dim, halo_fw_width) * grid->unit_size();
	size_t bw_size = grid->CalcHaloSize(dim, halo_bw_width) * grid->unit_size();
	int tag = 0;

   	if (halo_fw_width > 0 &&
      	(grid->my_offset()[dim] + grid->my_size()[dim]
       	< grid->size_[dim] ||
        (periodic && proc_size[dim] > 1))) 
	{
		//if(my_rank==0)
    		cout << "[" << my_rank << "] "
        	        << "Receiving halo of " << fw_size
        	        << " bytes for fw access from " << fw_peer << "\n";
    		MPI_Request req;
    		CHECK_MPI(MPI_Irecv(
        	grid->GetHaloPeerBuf(dim, true, halo_fw_width),
        	fw_size, MPI_BYTE, fw_peer, tag, MPI_COMM_WORLD, &req));
    		requests.push_back(req);
  	}

  	if (halo_bw_width > 0 &&
      	(grid->my_offset()[dim] > 0 ||
       	(periodic && proc_size[dim] > 1))) 
	{
		//if(my_rank==0)
    		cout << "[" << my_rank << "] "
                << "Receiving halo of " << bw_size
                << " bytes for bw access from " << bw_peer << "\n";
    		MPI_Request req;
    		CHECK_MPI(MPI_Irecv(
        	grid->GetHaloPeerBuf(dim, false, halo_bw_width),
        	bw_size, MPI_BYTE, bw_peer, tag, MPI_COMM_WORLD, &req));
    		requests.push_back(req);
  	}

	// Sends out the halo for forward access
  	if (halo_fw_width > 0 &&
      	(grid->my_offset()[dim] > 0 ||
       	(periodic && proc_size[dim] > 1))) 
	{
		//if(my_rank==0)
    		cout << "[" << my_rank << "] "
        	        << "Sending halo of " << fw_size << " bytes"
        	        << " for fw access to " << bw_peer << "\n";
    		LOG_DEBUG() << "grid: " << grid << "\n";
    		grid->CopyoutHalo(dim, halo_fw_width, true);
    		MPI_Request req;
    		CHECK_MPI(MPI_Isend(grid->halo_self_fw_[dim]->Get(), fw_size, MPI_BYTE,
        	                   bw_peer, tag, MPI_COMM_WORLD, &req));
  	}
	
	// Sends out the halo for backward access
  	if (halo_bw_width > 0 &&
      	(grid->my_offset()[dim] + grid->my_size()[dim]
       	< grid->size_[dim] ||
       	(periodic && proc_size[dim] > 1))) 
	{
		//if(my_rank==0)
    		cout  << "[" << my_rank << "] "
        	        << "Sending halo of " << bw_size << " bytes"
        	        << " for bw access to " << fw_peer << "\n";
    		grid->CopyoutHalo(dim, halo_bw_width, false);
    		MPI_Request req;
    		CHECK_MPI(MPI_Isend(grid->halo_self_bw_[dim]->Get(), bw_size, MPI_BYTE,
        	                   fw_peer, tag, MPI_COMM_WORLD, &req));
  	}

	return;
}

void Grid_view::ExchangeBoundaries(GridMPI *grid,
                                      int dim,
                                      unsigned halo_fw_width,
                                      unsigned halo_bw_width,
                                      bool diagonal,
                                      bool periodic) const 
{
  	std::vector<MPI_Request> requests;
  	ExchangeBoundariesAsync(grid, dim, halo_fw_width,
  	                        halo_bw_width, diagonal,
  	                        periodic, requests);
  	FOREACH (it, requests.begin(), requests.end()) 
	{
  	  	MPI_Request *req = &(*it);
  	  	CHECK_MPI(MPI_Wait(req, MPI_STATUS_IGNORE));
  	  	grid->CopyinHalo(dim, halo_bw_width, false);
  	  	grid->CopyinHalo(dim, halo_fw_width, true);
  	}

  	return;
}

// TODO: reuse is used?
void Grid_view::ExchangeBoundaries(GridMPI *g,
                                      const Width2 &halo_width,
                                      bool diagonal,
                                      bool periodic
                                      ) const {
  	LOG_DEBUG() << "GridSpaceMPI::ExchangeBoundaries\n";

  	for (int i = g->num_dims_ - 1; i >= 0; --i) 
	{
    		LOG_DEBUG() << "Exchanging dimension " << i << " data\n";

    		ExchangeBoundaries(g, i, halo_width.fw[i],
        	               halo_width.bw[i], diagonal, periodic);
  	}

  	return;
}


