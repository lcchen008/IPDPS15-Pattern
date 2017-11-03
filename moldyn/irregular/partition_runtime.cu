#include "irregular_runtime.h"

IrregularRuntime::IrregularRuntime(IRIndex num_edges, 
			IRIndex  num_nodes, 
			void *edge_data, 
			void *node_data,
			int edge_data_elm_size,
			int node_data_elm_size, int num_iters)
			:global_num_edges_(num_edges),
			global_num_nodes_(num_nodes),
			global_edge_data_(edge_data),
			global_node_data_(node_data),
			edge_data_elm_size_(edge_data_elm_size),
			node_data_elm_size_(node_data_elm_size),
			num_iters_(num_iters),
			current_iter_(0)
{
		
}

void IrregularRuntime::IrregularInit(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);	

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);

	this->num_gpus_ = GetGPUNumber();
}
