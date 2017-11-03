#ifndef IRREGULAR_PARTITION_VIEW_H_
#define IRREGULAR_PARTITION_VIEW_H_

#include "data_type.h"
#include "../lib/common.h"
#include "partition_mpi.h"
using namespace std;

class partition_view
{
protected:
	IRIndex global_num_nodes_;
	IRIndex global_num_edges_;

	IRIndex my_num_nodes_; 
	int num_procs_;
	int my_rank_;	
	int my_node_start_;

	EDGE *global_edges_;
	void *global_edge_data_;
	void *global_node_data_;

	int edge_data_elm_size_;
	int node_data_elm_size_;
	int reduction_elm_size_;

	//num of nodes requested by remote nodes
	int *peer_request_num_nodes_;
	//accumulative num of nodes requested by peers
	int *peer_request_num_nodes_sum_;
	//node id of the nodes requested by each remote node
	int **peer_request_nodes_;
	//node data of the nodes requested by each remote node
	void **peer_request_node_data_;

	vector<int> *my_nodes;

	
	int **remote_nodes_array_;

	//required node from other partitions
	int *remote_node_sum_;
	int *remote_num_nodes_;


public:
	partition_view(int num_procs, 
			int my_rank, 
			IRIndex global_num_nodes, 
			IRIndex  global_num_edges, 
			EDGE *edges, 
			void *edge_data, 
			void *node_data, 
			int edge_data_elm_size, 
			int node_data_elm_size,
			int reduction_elm_size);

	int my_num_nodes(){return my_num_nodes_;}
	//tells the senders of the number of nodes wanted
	//so that the sender can prepare receive buffer 
	//(for receiving node id) and send buffer (for sending node data)
	void exchange_halo_size_info();

	//sends node ids 
	void exchange_halo_node_info();

	//send node data
	void exchange_halo_node_data(partition_mpi *p);

	//creates partition based on nodes
	partition_mpi *CreatePartition();

	int my_node_start(){return my_node_start_;}	
};

#endif
