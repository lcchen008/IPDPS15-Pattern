#ifndef IRREGULAR_PARTITION_MPI_H_
#define IRREGULAR_PARTITION_MPI_H_

#include "partition.h"
#include <vector> 
#include "data_type.h"
#include "../lib/common.h"
using namespace std;

class partition_mpi:public partition
{
protected:
	int my_reduction_elm_num_;
	int *peer_nodes_start_number_;
	int my_own_node_number_;
	void *my_node_data_d_;

public:
	int my_reduction_elm_num(){return  my_reduction_elm_num_;}
	//reduction_array *my_reduction_array(){return my_reduction_array_;}
	int *peer_nodes_start_number(){return peer_nodes_start_number_;}
	void *my_node_data_d(){return my_node_data_d_;}
	int my_own_num_nodes(){return  my_own_node_number_;}

	partition_mpi(
			int total_nodes, 
			int my_num_nodes,
			int my_node_start, 
			void *node_data, 
			void *node_data_d,
			IRIndex num_edges,
			EDGE *edges,
			void *edge_data,  
			int edge_data_elm_size, 
			int node_data_elm_size,
			int reduction_elm_size):
			partition(total_nodes,
					my_node_start,
					reduction_elm_size,
					num_edges,
					edges,edge_data, 
					node_data, 
					edge_data_elm_size, 
					node_data_elm_size), 
			my_own_node_number_(my_num_nodes),
			my_node_data_d_(node_data_d)
	{}
};

#endif
