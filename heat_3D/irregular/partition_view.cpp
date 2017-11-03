#include "partition_view.h"
#include "partition_util.h"
#include "data_type.h"
#include "partition_mpi.h"
#include "../lib/macro.h"
#include <mpi.h>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <unordered_map>
#include <map>
using namespace std;

partition_view::partition_view(int num_procs, 
		int my_rank, 
		IRIndex global_num_nodes, 
		IRIndex  global_num_edges, 
		EDGE *edges, 
		void *edge_data, 
		void *node_data, 
		int edge_data_elm_size, 
		int node_data_elm_size, 
		int reduction_elm_size):
		num_procs_(num_procs),
		my_rank_(my_rank), 
		global_num_nodes_(global_num_nodes),
		global_num_edges_(global_num_edges),
		global_edges_(edges),
		global_edge_data_(edge_data),
		global_node_data_(node_data),
		edge_data_elm_size_(edge_data_elm_size),
		node_data_elm_size_(node_data_elm_size),
		reduction_elm_size_(reduction_elm_size)
{
	peer_request_num_nodes_ = (int *)malloc(num_procs_*sizeof(int));
	peer_request_num_nodes_sum_ = (int *)malloc(num_procs_*sizeof(int));	
	peer_request_nodes_ = (int **)malloc(num_procs_ * sizeof(int *));		
	peer_request_node_data_ = (void **)malloc(num_procs_ * sizeof(void *));

	memset(peer_request_nodes_, 0, num_procs_ * sizeof(int *));
	memset(peer_request_node_data_, 0, num_procs_ * sizeof(void *));
}

partition_mpi *partition_view::CreatePartition()
{
	map<int, int> **remote_nodes;
	int average = global_num_nodes_/num_procs_; 
	my_node_start_ = my_rank_ * average;

	if(my_rank_ < num_procs_ - 1)
	{
		my_num_nodes_ = average; 
	}

	else if(my_rank_ == num_procs_ - 1)
	{
		my_num_nodes_ = global_num_nodes_ - average * my_rank_;	
	}

	//void *my_node_data = malloc(my_num_nodes_*node_data_elm_size_);	
	//memcpy(my_node_data, (char *)global_node_data_ + node_data_elm_size_* my_node_start_, my_num_nodes_*node_data_elm_size_);

	vector<EDGE> *my_edges = new vector<EDGE>;
	vector<int> edge_ids;
	//my_nodes = new vector<int>;
	
	//create a list of vectors containing remote nodes. each vector corresponds to a remote node
	//remote nodes are the nodes required by the crossing edges but resides on remote peers 
	printf("num_procs_: %d\n", num_procs_);

	remote_nodes = (map<int, int> **)malloc(num_procs_ * sizeof(unordered_map<int, int> *));
	
	for(int i = 0; i < num_procs_; i++)
	{
		remote_nodes[i] = new map<int, int>;	
	}

	remote_nodes_array_ = (int **)malloc(num_procs_ * sizeof(int *));

	remote_node_sum_ = (int *)malloc(sizeof(int)*num_procs_);
	memset(remote_node_sum_, 0, sizeof(int)*num_procs_);

	remote_num_nodes_ = (int *)malloc(sizeof(int)*num_procs_);
	memset(remote_num_nodes_, 0, sizeof(int)*num_procs_);

	//create edges
	//only process edges which have at least one node falling in my partition
	
	vector<EDGE> cross_edges;

	EDGE edge_tmp;
	//first store local edges and count remote edges
	printf("===========my rank: %d, my node start: %d global num edges: %ld\n", my_rank_, my_node_start_, global_num_edges_);

	int idx0, idx1, proc0, proc1;

	printf("before====================================\n");

	for(int i = 0; i < global_num_edges_; i++)
	{
		idx0 = global_edges_[i].idx0;				
		idx1 = global_edges_[i].idx1;				
		
		proc0 = get_proc_id(idx0, global_num_nodes_, num_procs_);
		proc1 = get_proc_id(idx1, global_num_nodes_, num_procs_);

		if(proc0==proc1&&proc0==my_rank_)
		{
			edge_tmp = global_edges_[i];
			edge_tmp.idx0 -= my_node_start_;
			edge_tmp.idx1 -= my_node_start_;

			my_edges->push_back(edge_tmp);	
			edge_ids.push_back(i);
		}

		else if(proc0==my_rank_)
		{
			edge_tmp = global_edges_[i];
			cross_edges.push_back(edge_tmp);

			(*remote_nodes[proc1])[idx1] = idx1;
			edge_ids.push_back(i);
		}

		else if(proc1==my_rank_)
		{
			edge_tmp = global_edges_[i];
			cross_edges.push_back(edge_tmp);

			(*remote_nodes[proc0])[idx0] = idx0;
			edge_ids.push_back(i);
		}
	}


	MPI_Barrier(MPI_COMM_WORLD);

	//if(my_rank_==0)
	//	printf("my_rank_: %d remote node size: %ld\n", my_rank_, remote_nodes[1]->size());
	//else
	//	printf("my_rank_: %d remote node size: %ld\n", my_rank_, remote_nodes[0]->size());
	//if(my_rank_==2)
	//for(int i = 0; i < num_procs_; i++)
	//{
	//	printf("my_rank_: %d remote node size~~~~~~~~~~~~~~~~~: %ld\n", my_rank_, remote_nodes[i]->size());			
	//}
	
	//allocate space for edge data
	char *edge_data = NULL;
	if(edge_data_elm_size_>0)
	{
		edge_data = (char *)malloc(edge_data_elm_size_ * my_edges->size());

		//copy edge_data
		for(int i = 0;i<my_edges->size();i++)
		{
			memcpy(edge_data + edge_ids[i] * edge_data_elm_size_, global_edge_data_, edge_data_elm_size_);	
		}
	}

	printf("after====================================\n");

	remote_node_sum_[0] = 0;

	for(int i = 1; i < num_procs_; i++)
	{
		remote_num_nodes_[i] = (*remote_nodes[i]).size();
		remote_node_sum_[i] = (*remote_nodes[i-1]).size() + remote_node_sum_[i-1];
	}

	printf("before processing crossing edges size: %ld my edge size: %ld==============================\n", cross_edges.size(), my_edges->size());	

	map<int, int> **pos_map = (map<int, int> **)malloc(sizeof(map<int, int>*)*num_procs_);;

	//===============retrieve crossing node ids from map
	int count;
	for(int i = 0; i < num_procs_; i++)
	{
		remote_nodes_array_[i] = (int *)malloc(sizeof(int)*(*remote_nodes[i]).size());
		pos_map[i] = new map<int, int>;
		
		//next, copy remote node id 
		count = 0;
	
		for(map<int,int>::iterator it = remote_nodes[i]->begin();it != remote_nodes[i]->end();++it)
		{
			int second = it->second;
			remote_nodes_array_[i][count] = second;	
			(*pos_map[i])[second] = count;
			count++;
		}
	}

	//================

	EDGE tmp;

	//next deal with cross edges
	for(int i = 0; i < cross_edges.size(); i++)
	{
		tmp = cross_edges[i];				
		int idx0 = tmp.idx0;
		int idx1 = tmp.idx1;
		int proc0 = get_proc_id(idx0, global_num_nodes_, num_procs_);
		int proc1 = get_proc_id(idx1, global_num_nodes_, num_procs_);

		if(proc0==my_rank_)	
		{
			int rank_in_proc = (*pos_map[proc1])[idx1];//distance(remote_nodes[proc1]->begin(), remote_nodes[proc1]->find(idx1)); 

			//if(my_rank_==1)
			//printf("edge_id: %d rank in proc: %d\n", i, rank_in_proc);

			cross_edges[i].idx1 = my_num_nodes_ + remote_node_sum_[proc1] + rank_in_proc; 	
			cross_edges[i].idx0 -= my_node_start_;

			my_edges->push_back(cross_edges[i]);
		}

		else
		{
			int rank_in_proc = (*pos_map[proc0])[idx0];//distance(remote_nodes[proc0]->begin(), remote_nodes[proc0]->find(idx0)); 

			//if(my_rank_==1)
			//printf("edge_id: %d rank in proc: %d\n", i, rank_in_proc);

			cross_edges[i].idx0 = my_num_nodes_ + remote_node_sum_[proc0] + rank_in_proc; 	
			cross_edges[i].idx1 -= my_node_start_;
			my_edges->push_back(cross_edges[i]);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	printf("crossing edge done.........\n");

	IRIndex total_nodes = my_num_nodes_ + remote_node_sum_[num_procs_-1] + (*remote_nodes[num_procs_-1]).size();

	//if(my_rank_ == 0)
	//printf("============>total remote nodes: %d\n", remote_node_sum_[num_procs_-1] + (*remote_nodes[num_procs_-1]).size());
	//printf("==============> total nodes are: %ld\n", total_nodes);

	void *new_node_data = malloc(total_nodes * node_data_elm_size_);

	//first, copy my local node
	memcpy(new_node_data, (char *)global_node_data_ + node_data_elm_size_ * my_node_start_, node_data_elm_size_ * my_num_nodes_);	

	// remote node data need to be copied at exchange time	
	
	//node data should be in zero-copy memory
	void *ptr = NULL;
	CUDA_SAFE_CALL(cudaHostAlloc((void **)&ptr, 
				node_data_elm_size_ * total_nodes, 
				cudaHostAllocPortable|cudaHostAllocMapped));	

	//copy node data into zero copy
	memcpy(ptr, new_node_data, node_data_elm_size_ * total_nodes);

	void *ptr_d = NULL;
	cudaHostGetDevicePointer((void **)&ptr_d, ptr, 0);

	free(new_node_data);

	printf("xxxxxxxxxxxxxxxxxnum of my edges: %ld\n", my_edges->size());

	//TODO
	partition_mpi *part = new partition_mpi(
			total_nodes, 
			my_num_nodes_, 
			my_node_start_, 
			ptr, 
			ptr_d,
			my_edges->size(),
			&(*my_edges)[0], 
			edge_data, 
			edge_data_elm_size_, 
			node_data_elm_size_,
			reduction_elm_size_ 
			);

	return part;
}

//tells others how many nodes I am requesting
void partition_view::exchange_halo_size_info()
{
	int tag = 0;
	vector<MPI_Request> requests;
	
	memset(peer_request_num_nodes_sum_, 0, sizeof(int) * num_procs_);

	//receive
	for(int i = 0; i < num_procs_; i++)	
	{
		if(i!=my_rank_)		
		{
			MPI_Request req;
			CHECK_MPI(MPI_Irecv(&peer_request_num_nodes_[i], 1, MPI_INT, i, tag, MPI_COMM_WORLD, &req));
			requests.push_back(req);
		}
	}

	//send
	for(int i = 0; i < num_procs_; i++)	
	{
		if(i!=my_rank_)
		{
			int num_nodes = remote_num_nodes_[i];
			//printf("I am %d and I am requesting %d for %d nodes *************\n", my_rank_, i, num_nodes);
			MPI_Request req;	
			CHECK_MPI(MPI_Isend(&num_nodes, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &req));
		}
	}

	//synchronize
	for(int i = 0; i < requests.size(); i++)
	{
		MPI_Request *req = &(requests[i]);	
		CHECK_MPI(MPI_Wait(req, MPI_STATUS_IGNORE));	
	}

	//calculate accumulative number requested nodes
	//for(int i = 1; i < num_procs_; i++)
	//{
	//	printf("requested............: %d\n", );
	//	//peer_request_num_nodes_sum_[i] = peer_request_num_nodes_sum_[i-1] + peer_request_num_nodes_[i-1];	
	//}

	//create buffer for sending node info, and sending node data buffer 
	for(int i = 0; i < num_procs_; i++)
	{
		if(i!=my_rank_)
		{
			//printf("=============================>I am %d, %d requesting: %d\n", my_rank_, i, peer_request_num_nodes_[i]);
			if(peer_request_nodes_[i]!=NULL)
				delete [] peer_request_nodes_[i];
			peer_request_nodes_[i] = new int[peer_request_num_nodes_[i]];		

			if(peer_request_node_data_[i]!=NULL)
				delete [] peer_request_node_data_[i];
			peer_request_node_data_[i] = new char[peer_request_num_nodes_[i]*node_data_elm_size_];
		}
	}
}

//tells others which nodes I am requesting
void partition_view::exchange_halo_node_info()
{
	int tag = 1;
	vector<MPI_Request> requests;
	//receive	
	for(int i = 0; i < num_procs_; i++)	
	{
		if(i!=my_rank_)	
		{
			MPI_Request req;
			CHECK_MPI(MPI_Irecv(peer_request_nodes_[i], peer_request_num_nodes_[i], MPI_INT, i, tag, MPI_COMM_WORLD, &req));	
			requests.push_back(req);
		}
	}

	//send
	for(int i = 0; i < num_procs_; i++)
	{
		if(i!=my_rank_)		
		{
			MPI_Request req;	
			int * nodes = remote_nodes_array_[i];
			CHECK_MPI(MPI_Isend(nodes, remote_num_nodes_[i], MPI_INT, i, tag, MPI_COMM_WORLD, &req));
		}
	}

	//synchronize
	for(int i = 0; i < requests.size(); i++)
	{
		MPI_Request *req = &(requests[i]);	
		CHECK_MPI(MPI_Wait(req, MPI_STATUS_IGNORE));	
	}
}

//send the request nodes to remote
void partition_view::exchange_halo_node_data(partition_mpi *p)
{
	int tag = 2;	
	vector<MPI_Request> requests;

	//receive node data from remote
	char *p_node_data = (char *)p->my_node_data(); 
	for(int i = 0; i < num_procs_; i++)
	{
		if(i!=my_rank_&&remote_num_nodes_[i]!=0)	
		{
			int cur = my_num_nodes_ + remote_node_sum_[i];	
			char *ptr = p_node_data + cur * node_data_elm_size_;
			MPI_Request req;	
			CHECK_MPI(MPI_Irecv(ptr, remote_num_nodes_[i] * node_data_elm_size_, MPI_BYTE, i, tag, MPI_COMM_WORLD, &req));	
			requests.push_back(req);
		}
	}
	
	//copy nodes into send buffer and send
	for(int i = 0; i < num_procs_; i++)
	{	
		//int node_start = peer_request_num_nodes_sum_[i];	
		int node_num = peer_request_num_nodes_[i];

		if(i!=my_rank_&&node_num!=0)		
		{
			//copy nodes into send buffer		
			
			char *node_data = (char *)peer_request_node_data_[i];
			int *node_ids = peer_request_nodes_[i];
			
			if(my_rank_==0)
			printf("num_nodes:~~~~~~~~%d~~~~~~~~~~\n", node_num);

			for(int j = 0; j < node_num; j++)
			{
				int node_id = node_ids[j] - my_node_start_;		
				char *data_tmp = p_node_data + node_id * node_data_elm_size_;
				memcpy(node_data + j * node_data_elm_size_, data_tmp, node_data_elm_size_);
			}

			MPI_Request req;
			CHECK_MPI(MPI_Isend(node_data, node_num * node_data_elm_size_, MPI_BYTE, i, tag, MPI_COMM_WORLD, &req));
		}
	}

	//synchronize
	for(int i = 0; i < requests.size(); i++)
	{
		MPI_Request *req = &(requests[i]);	
		CHECK_MPI(MPI_Wait(req, MPI_STATUS_IGNORE));	
	}
}
