#ifndef IRREGULAR_PARTITION_CPU_H_
#define IRREGULAR_PARTITION_CPU_H_


#include "partition.h"
#include "../lib/common.h"

class partition_cpu:public partition
{
	protected:
		Part *my_parts_;
		int my_num_parts_;
	public:
		partition_cpu(IRIndex  my_num_nodes, 
				void *node_data, 
				IRIndex my_num_edges, 
				EDGE *edges, 
				void *edge_data, 
				int node_data_elm_size, 
				int edge_data_elm_size, 
				int reduction_elm_size,
				int my_num_parts,
				Part *my_parts):
			partition(my_num_nodes, 
					0,
					reduction_elm_size, 
					my_num_edges, 
					edges, 
					edge_data, 
					node_data, 
					edge_data_elm_size, 
					node_data_elm_size),
			my_num_parts_(my_num_parts),
			my_parts_(my_parts)
		{}
	
		//allocate all the data
		void Allocate();
		Part *my_parts(){return my_parts_;}
		int my_num_parts(){return my_num_parts_;}

		~partition_cpu()
		{
			free(my_parts_);	
		}
};

#endif
