#ifndef IRREGULAR_PARTITION_CUDA_H_
#define IRREGULAR_PARTITION_CUDA_H_


#include "partition.h"
#include "../lib/common.h"

class partition_cuda:public partition
{
	protected:
		void *my_node_data_device_;
		void *my_node_data_d_;
		void *my_edge_data_d_;
		void *my_edges_d_;
		void *my_parts_;
		int my_num_parts_;
		Part *my_parts_d_;

	public:
		partition_cuda(IRIndex my_num_nodes, 
				void *node_data, 
				void *node_data_d,
				IRIndex  my_num_edges, 
				EDGE *edges, 
				void *edge_data, 
				int node_data_elm_size, 
				int edge_data_elm_size, 
				int reduction_elm_size, 
				int num_parts, 
				Part *parts):
			partition(my_num_nodes, 
					0, 
					reduction_elm_size, 
					my_num_edges, 
					edges, 
					edge_data, 
					node_data, 
					edge_data_elm_size, 
					node_data_elm_size),
			my_num_parts_(num_parts), 
			my_node_data_d_(node_data_d),
			my_parts_(parts)
		{
			
		}
	
		//allocate all the data
		void Allocate();

		void *my_node_data_device(){return my_node_data_device_;}
		void *my_node_data_d(){return my_node_data_d_;}
		void *my_edge_data_d(){return my_edge_data_d_;}
		void *my_edges_d(){return my_edges_d_;}
		Part *my_parts_d(){return my_parts_d_;}
		int my_num_parts(){return my_num_parts_;}

		~partition_cuda();
};

#endif
