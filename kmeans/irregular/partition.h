#ifndef IRREGULAR_PARTITION_H_
#define IRREGULAR_PARTITION_H_

#include "../lib/common.h"
#include "data_type.h"
#include <stdlib.h>

class partition
{
protected:
	IRIndex  my_num_nodes_;
	IRIndex  my_node_start_;
	void *my_node_data_;


	IRIndex  my_num_edges_;
	EDGE *my_edges_;
	void *my_edge_data_;


	int edge_data_elm_size_;
	int node_data_elm_size_;
	int reduction_elm_size_;

public:
	partition(int my_num_nodes, 
			int my_node_start, 
			int reduction_elm_size, 
			IRIndex num_edges,
			EDGE *edges, 
			void *edge_data, 
			void *node_data, 
			int edge_data_elm_size, 
			int  node_data_elm_size):
			my_num_nodes_(my_num_nodes),
			my_node_start_(my_node_start),
			reduction_elm_size_(reduction_elm_size),
			my_num_edges_(num_edges),
			my_edges_(edges),
			my_edge_data_(edge_data),
			my_node_data_(node_data),
			edge_data_elm_size_(edge_data_elm_size),
			node_data_elm_size_(node_data_elm_size)
	{

	}
	//IRIndex global_num_nodes(){return global_num_nodes_;}
	//IRIndex global_num_edges(){return global_num_edges_;}
	IRIndex my_num_nodes(){return my_num_nodes_;}
	IRIndex my_node_start(){return my_node_start_;}
	int reduction_elm_size(){return reduction_elm_size_;}
	IRIndex my_num_edges(){return my_num_edges_;}
	void *my_node_data(){return my_node_data_;}
	void *my_edge_data(){return my_edge_data_;}
	EDGE *my_edges(){return my_edges_;}
	int node_data_elm_size(){return node_data_elm_size_;}
	int edge_data_elm_size(){return edge_data_elm_size_;}

	~partition()
	{
		//free(my_node_data_);		
		free(my_edges_);
		if(my_edge_data_)
			free(my_edge_data_);
	}
};

#endif
