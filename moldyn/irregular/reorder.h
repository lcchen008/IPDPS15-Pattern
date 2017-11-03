#ifndef IRREGULAR_REORDER_H_ 
#define IRREGULAR_REORDER_H_
#include <vector>
#include <iostream>
#include "data_type.h"
using namespace std;

//struct edge
//{
//	int x; 
//	int y;
//};

class Reorder 
{
public: 
	int num_points;
	int num_edges;
	int new_edges;
	int num_dims;
	int *partitions;
	int *pos_map; //used to record the change of position of points
	int *index_map; //used to record the change of position of points
	int *part_index;
	int total_num_partitions;
	Part *parts;
	vector<struct edge> **edge_vectors;

	/**
	 * num_points: the total number of nodes
	 * num_edges: the total number of edges
	 * num_dims: the dimensionality of the coordinate for each point
	 * */
	Reorder(int num_points, int num_edges, int num_dims, int *partitions);

	template <class T>
	void partition(T *coordinates);

	template <class T>
	void partition_points(T *coordinates, int dim, int start, int end);

	void reorder_edges(EDGE *edges);

	//template <class T>
	void reorder_satellite_data(void *satellite_data, int elm_size);

	template <class T>
	int partition_a_dim(T *coordinates, int dim, int p, int r);

	template <class T>
	int random_partition_a_dim(T *coordinates, int dim, int p, int r);

	template <class T>
	void exchange(T *coordinates, int i, int j);

	//select the iTH largest element from the dimTH dimension
	template <class T>
	int random_select(T *coordinates, int dim, int p, int r, int i);

	template <class T>
	void print_map(T *coordinates);

	int get_num_edges();

	Part *get_parts();

	int *get_part_index();
};


#endif
