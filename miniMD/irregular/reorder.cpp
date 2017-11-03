#include "reorder.h" 
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
//#include <DS.h>

int gen_random(int p, int r)
{
	srand (time(NULL));
	return rand()%(r-p+1) + p;
}

Reorder::Reorder(int num_points, int num_edges, int num_dims, int *partitions)
{
	this->num_points = num_points;
	this->num_edges = num_edges;
	this->new_edges = 0;
	this->num_dims = num_dims;

	this->pos_map = (int *)malloc(sizeof(int) * num_points);
	this->index_map = (int *)malloc(sizeof(int) * num_points);
	for(int i = 0; i < num_points; i++)
	{
		(this->pos_map)[i] = i;	
		(this->index_map)[i] = i;	
	}

	this->partitions = (int *)malloc(num_dims * sizeof(int));
	total_num_partitions = 1;
	for(int i = 0; i < num_dims; i++)
	{
		(this->partitions)[i] = partitions[i];
		total_num_partitions *= partitions[i];
	}

	edge_vectors = new vector<struct edge> *[total_num_partitions];

	for(int i = 0; i < total_num_partitions; i++)
	{
		edge_vectors[i] = new vector<struct edge>; 
	}

	this->part_index = (int *)malloc(num_points*sizeof(int));
}

template <class T>
void Reorder::exchange(T *coordinates, int i, int j)
{
	//printf("exchange %d and %d\n", i, j);

	T *tmp = (T *)malloc(sizeof(T)*num_dims);	
	memcpy(tmp, coordinates + i * num_dims, sizeof(T)*num_dims);
	memcpy(coordinates + i * num_dims, coordinates + j * num_dims, sizeof(T)*num_dims);
	memcpy(coordinates + j * num_dims, tmp, sizeof(T)*num_dims);

	free(tmp);

	int tmp_pos = pos_map[i];
	pos_map[i] = pos_map[j];
	pos_map[j] = tmp_pos;
}

template <class T>
int  Reorder::partition_a_dim(T *coordinates, int dim, int p, int r)
{
	T *end = coordinates + dim + num_dims * r; 

	T x = *end;

	int i = p - 1;

	for(int j = p; j <= r - 1; j++)
	{
		if(*(coordinates + dim + num_dims * j)<=x)	
		{
			i++;	
			if(i!=j)
			exchange(coordinates, i, j);
		}
	}

	exchange(coordinates, i+1, r);

	return i + 1;
}

template <class T>
int Reorder::random_partition_a_dim(T *coordinates, int dim, int p, int r) 
{
	int i =  gen_random(p, r);	
	exchange(coordinates, r, i);
	return partition_a_dim(coordinates, dim, p, r);
}

template <class T>
int  Reorder::random_select(T *coordinates, int dim, int p, int r, int i)
{
	if(p==r)	
		return p;

	int q = random_partition_a_dim(coordinates, dim, p, r);

	int k = q - p + 1;

	//printf("q is: %d\n", q);

	if(i==k)
		return q;

	else if(i < k)
		return  random_select(coordinates, dim, p, q-1, i);
	else
		return  random_select(coordinates, dim, q+1, r, i - k);
}

template <class T>
void Reorder::partition_points(T *coordinates, int dim, int start, int end)
{
	int total = end - start + 1;	
	int num_points_per_part = ceil((double)total/partitions[dim]);

	for(int j = 0; j < partitions[dim]; j++)	
	{
		int st = start + j * num_points_per_part; 
		int ed = start + (j + 1) * num_points_per_part - 1;

		if(ed > end)
			ed = end;

		if(ed<end)
		int pos = random_select(coordinates, dim, st, end, num_points_per_part);				
		//further partition in a lower dimension
		if(dim < num_dims - 1)
		{
			partition_points(coordinates, dim + 1, st, ed);	
		}
	}
}

template <class T>
void  Reorder::partition(T *coordinates)
{
	partition_points(coordinates, 0, 0, num_points - 1);

	int count = 0;
	//generate index map based on position map
	for(int i = 0; i < num_points; i++)
	{
		index_map[pos_map[i]] = i;	
		//if(pos_map[i]!=i)
		//{
		//	printf("changed: %d ==> %d\n", pos_map[i], i);
		//	count++;
		//}
	}

	printf("count is: =====> %d", count);
}

void  Reorder::reorder_edges(EDGE *edges)
{
	int x, y;
	int count = 0;
	int count_0= 0,count_1=0;
	for(int i = 0; i < num_edges; i++)		
	{
		x = edges[i].idx0;//*(edges + 2 * i);	
		y = edges[i].idx1;//*(edges + 2 * i + 1);

		edges[i].idx0 = index_map[x];
		edges[i].idx1 = index_map[y];

		//struct edge cur_edge;

		//cur_edge.idx0 = index_map[x];
		//cur_edge.idx1 = index_map[y];

		//int part_x = 0;
		//int part_y = 0;

		//int points_per_partition = ceil((double)num_points/total_num_partitions);

		//part_x = cur_edge.idx0/points_per_partition;
		//part_y = cur_edge.idx1/points_per_partition;

		//if(part_x==part_y)
		//{
		//	edge_vectors[part_x]->push_back(cur_edge);	
		//	new_edges++;

		//	part_x==0?count_0++:count_1++;
		//}

		//else
		//{
		//	edge_vectors[part_x]->push_back(cur_edge);	
		//	edge_vectors[part_y]->push_back(cur_edge);	
		//	new_edges+=2;
		//	count++;
		//}
	}

	//printf("increased edges: %d\n", count);

	//printf("0: %d 1: %d\n", count_0, count_1);

	//parts = (Part *)malloc(sizeof(Part)*total_num_partitions);

	//int total_edges = get_num_edges();

	//struct edge *new_edges = (struct edge *)malloc(sizeof(struct edge) * total_edges);

	//int total = 0;

	//for(int i = 0; i < total_num_partitions; i++)	
	//{
	//	int edges_current = edge_vectors[i]->size();

	//	//printf("partition %d edges: %d\n", i, edges_current);

	//	parts[i].start = total; 
	//	parts[i].end = total + edges_current - 1; 
	//	parts[i].part_id = i;

	//	memcpy(new_edges + total, &(*(edge_vectors[i]))[0], edges_current * sizeof(struct edge));
	//	
	//	total += edges_current;

	//	delete edge_vectors[i];
	//}

	//free(edge_vectors);

	//return new_edges;
}

//template <class T>
void   Reorder::reorder_satellite_data(void *satellite_data, int elm_size)
{
	void *satellite_data_back = malloc(elm_size * num_points);		
	memcpy(satellite_data_back, satellite_data, elm_size * num_points);

	for(int i = 0; i < num_points; i++)
	{
		memcpy((char *)satellite_data + i * elm_size, (char *)satellite_data_back + pos_map[i] * elm_size, elm_size);
		//satellite_data[i] = satellite_data_back[pos_map[i]];
	}

	free(satellite_data_back);
}

template <class T>
void    Reorder::print_map(T *coordinates)
{
	for(int i = 0; i < num_points; i++)
	{
		printf("%d-->%d\n", i, pos_map[i]);	
		//printf("%f, %f, %f\n", i, coordinates[i*num_dims], coordinates[i*num_dims + 1], coordinates[i*num_dims + 2]);	
	}
}

int     Reorder::get_num_edges()
{
	return new_edges;
}

Part * Reorder::get_parts()
{
	return parts;
}

int * Reorder::get_part_index()
{
	for(int i = 0; i < num_points; i++)
	{
		int points_per_partition = ceil((double)num_points/total_num_partitions);
		part_index[i] = i/points_per_partition;
	}

	return part_index;
}
