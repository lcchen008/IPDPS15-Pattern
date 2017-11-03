#ifndef PARTITION
#define PARTITION
#include <vector>
#include <iostream>
#include "../lib/common.h"
#include "data_type.h"
#include "partition_cuda.h"
#include "partition_cpu.h"

using namespace std;

struct partition_args
{
	int cpu_gpu_split_point;
	int *cpu_partition;
	int *gpu_partition;
};

class Partitioner
{
public: 
	int rank_;
	int num_points;
	IRIndex num_edges;
	int cpu_new_edges;
	int *gpu_new_edges;
	int num_devices_;
	int num_gpus_;
	int reduction_elm_size_;

	double *speeds;

	int num_dims;

	int *cpu_partitions;
	int **gpu_partitions;

	int *pos_map; //used to record the change of position of points
	int *index_map; //used to record the change of position of points
	int *part_index;
	
	int cpu_total_num_nodes_;
	int *gpu_total_num_nodes_;

	int *device_num_nodes_sum_;

	int cpu_total_num_partitions_;
	int *gpu_total_num_partitions_;

	int cpu_nodes_per_partition;
	int gpu_nodes_per_partition;

	Part *cpu_parts;
	Part **gpu_parts;

	EDGE *cpu_edges;
	EDGE **gpu_edges;

	vector<struct edge> **cpu_edge_vectors;
	vector<struct edge> **gpu_edge_vectors;

	/**
	 * num_points: the total number of nodes
	 * num_edges: the total number of edges
	 * num_dims: the dimensionality of the coordinate for each point
	 * */
	Partitioner(int *part_index, int num_points, IRIndex num_edges, int num_devices, double *speeds, int reduction_elm_size, int node_num_dims, int rank);

	void reorder_edges(int *edges);

	void generate_device_edges(EDGE *edges);

	void sort_parts(Part *parts, int n);

	int get_cpu_num_edges();

	int *get_gpu_num_edges();

	EDGE *get_cpu_edges();

	EDGE **get_gpu_edges();

	int get_cpu_num_parts();

	int *get_gpu_num_parts();

	Part *get_cpu_parts();

	Part **get_gpu_parts();

	void gen_part_index();

	//split the points along the highest dim to each device
	
	template <class T>
	void exchange(T *coordinates, int i, int j);	

	template <class T>
	int partition_a_dim(T *coordinates, int dim, int p, int r);

	template <class T>
	int random_partition_a_dim(T *coordinates, int dim, int p, int r); 

	template <class T>
	int random_select(T *coordinates, int dim, int p, int r, int i);

	template <class T>
	void partition_device_nodes(T *coordinates);

	template <class T>
	void partition_points(T *coordinates, int *partitions, int dim, int start, int end);
	
	//get the device id for a specific node
	int get_device_id(size_t node_id);

	//get the partition id within device for a specific node
	int get_partition_id(size_t node_id, int device_id);

	partition_cpu *get_cpu_partition();

	partition_cuda **get_gpu_partitions();

	int get_node_start(int device_id);

	int get_node_sum(int device_id);

	void reorder_satellite_data(void *satellite_data, int elm_size);
};

#endif
