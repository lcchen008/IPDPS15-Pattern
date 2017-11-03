#include "partitioner.h" 
#include <time.h>
#include <stdlib.h>
#include "../lib/common.h"
#include "partition_cpu.h"
#include "partition_cuda.h"
#include "data_type.h"
#include <cstring>
#include <stdio.h>
#include "parameters.h"
#include <math.h>

//returns a random number from p to r
int gen_rand(int p, int r)
{
	srand (time(NULL));
	return rand()%(r-p+1) + p;
}

Partitioner::Partitioner(int *part_index, int num_points, IRIndex num_edges, int num_devices, double *speeds, int reduction_elm_size, int node_num_dims, int rank)
{
	//num_points does not include the remote nodes
	this->num_points = num_points;
	this->num_devices_ = num_devices;
	this->num_gpus_ = num_devices_ - 1;
	this->reduction_elm_size_ = reduction_elm_size;
	this->num_edges = num_edges;
	this->num_dims = node_num_dims;
	this->rank_ = rank;

	this->cpu_new_edges = 0;
	this->gpu_new_edges = (int *)malloc(num_gpus_ * sizeof(int));
	memset(gpu_new_edges, 0, num_gpus_ * sizeof(int));

	this->speeds = speeds;

	this->pos_map = (int *)malloc(sizeof(int) * num_points);
	this->index_map = (int *)malloc(sizeof(int) * num_points);

	for(int i = 0; i < num_points; i++)
	{
		pos_map[i] = index_map[i] = i;	
	}
	
	this->part_index = part_index; 

	cpu_total_num_partitions_ = CPU_NUM_PARTS;

	double total_speed = 0;
	for(int i = 0; i < num_devices_; i++)
	{
		//printf("speed %d: ==> %f\n", i, speeds[i]);
		total_speed += speeds[i];	
	}

	size_t count = 0;
	cpu_total_num_nodes_ = speeds[0]/total_speed * num_points; 

	//if(rank_==0)
	//{
	//	printf("~~~~~~~~~~~~~~~TOTAL nodes: %d\n", num_points);
	//	printf("~~~~~~~~~~~~~~~CPU total nodes: %d\n", cpu_total_num_nodes_);
	//}

	count = cpu_total_num_nodes_;
	gpu_total_num_nodes_ = (int *)malloc(num_gpus_ * sizeof(int));

	cpu_partitions = (int *)malloc(sizeof(int)*num_dims);

	gpu_partitions = (int **)malloc(sizeof(int *)*num_gpus_);

	//count total number of nodes for each device
	for(int i = 0; i < num_gpus_ - 1; i++)
	{
		gpu_total_num_nodes_[i] = speeds[i+1]/total_speed * num_points;	
		count+= gpu_total_num_nodes_[i];
	}

	gpu_total_num_nodes_[num_gpus_ - 1] = num_points - count;

	device_num_nodes_sum_ = (int *)malloc(sizeof(int)*num_devices_);

	device_num_nodes_sum_[0] = 0;
	device_num_nodes_sum_[1] = cpu_total_num_nodes_;

	for(int i = 2; i < num_devices_; i++)
	{
		device_num_nodes_sum_[i] = device_num_nodes_sum_[i - 1] + gpu_total_num_nodes_[i-2]; 
	}

	//for(int i = 0; i < num_devices_; i++)
	//{
	//	printf("++++++++++++++device_num_node_sum %d: %d\n", i, device_num_nodes_sum_[i]);	
	//}

	//count total number of partitions for each device
	//gpu_parts = new Part*[num_gpus_];

	gpu_nodes_per_partition = floor((double)SHARED_SIZE/reduction_elm_size_); 

	gpu_total_num_partitions_ = (int *)malloc(sizeof(int)*num_gpus_);



	for(int i = 0; i < num_gpus_; i++)
	{
		gpu_partitions[i] = (int *)malloc(sizeof(int)*num_dims);
		gpu_total_num_partitions_[i] = ceil((double)gpu_total_num_nodes_[i]/gpu_nodes_per_partition);

		gpu_partitions[i][0] = GPU_X;
		gpu_partitions[i][1] = GPU_Y;
		gpu_partitions[i][2] = ceil(gpu_total_num_partitions_[i]/((double)GPU_X*GPU_Y));

		gpu_total_num_partitions_[i] = gpu_partitions[i][0] * gpu_partitions[i][1] * gpu_partitions[i][2];

		printf("gpu_total_num_partitons %d: %d\n", i, gpu_total_num_partitions_[i]);
	}

	//======================

	cpu_edge_vectors = new vector<struct edge> *[cpu_total_num_partitions_];

	gpu_edge_vectors = new vector<struct edge> *[num_gpus_];

	for(int i = 0; i < cpu_total_num_partitions_; i++)
	{
		cpu_edge_vectors[i] = new vector<EDGE>; 
	}

	for(int i = 0; i < num_gpus_; i++)
	{
		gpu_edge_vectors[i] = new vector<EDGE> [gpu_total_num_partitions_[i]];		
	}

	cpu_nodes_per_partition = ceil((double)(cpu_total_num_nodes_)/cpu_total_num_partitions_);
		
	//printf("cpu points per part: %d\n", cpu_nodes_per_partition);

	this->part_index = (int *)malloc(num_points*sizeof(int));
}

void Partitioner::generate_device_edges(EDGE *edges)
{
	int x, y;
	int device_x, device_y;
	int part_x, part_y;

	EDGE cur_edge;

	for(int i = 0; i < num_edges; i++)
	{
		cur_edge = edges[i];

		if(cur_edge.idx0 < num_points)
		cur_edge.idx0 = index_map[cur_edge.idx0];

		if(cur_edge.idx1 < num_points)
		cur_edge.idx1 = index_map[cur_edge.idx1];

		x = cur_edge.idx0;	
		y = cur_edge.idx1;	

		//first, process x
		//get device id
		device_x = get_device_id(x);
		//printf("===========>device id: %d\n", device_x);

		//get part id within device
		part_x = get_partition_id(x, device_x);

		if(device_x!=-1)
		{
			//if x is on CPU
			if(device_x == 0)
			{
				cpu_edge_vectors[part_x]->push_back(cur_edge);	
				cpu_new_edges++;
			}
			//on GPU
			else
			{
				(gpu_edge_vectors[device_x - 1])[part_x].push_back(cur_edge);
				gpu_new_edges[device_x - 1]++;
			}
		}

		//next, process y
		device_y = get_device_id(y);
		//printf("===========>device id: %d\n", device_y);

		//get part id within device
		part_y = get_partition_id(y, device_y);

		if(device_y!=-1)
		{
			//y is on CPU
			if(device_y == 0)
			{
				//x is not within the same partition with y
				if(!(device_x == device_y && part_x == part_y))	
				{
					//if x is on GPU, we need to replicate this edge
					//if x is on CPU, we don't need to, as hybrid partitioning
					//is being used
					if(device_x != 0)	
					{
						cpu_edge_vectors[part_y]->push_back(cur_edge);
						cpu_new_edges++;
					}
				}
			}

			//y is on GPU
			else
			{
				//x is not within the same partition with y			
				if(!(device_x == device_y && part_x == part_y))
				{
					(gpu_edge_vectors[device_y - 1])[part_y].push_back(cur_edge);
					gpu_new_edges[device_y - 1]++;
				}
			}
		}
	}

	cpu_parts = (Part *)malloc(sizeof(Part)*cpu_total_num_partitions_);
	gpu_parts = (Part **)malloc(sizeof(Part *)*num_gpus_);

	for(int i = 0; i < num_gpus_; i++)
	{
		gpu_parts[i] = (Part *)malloc(sizeof(Part)*gpu_total_num_partitions_[i]);
	}

	int cpu_total_edges = get_cpu_num_edges();
	int *gpu_total_edges = get_gpu_num_edges();

	cpu_edges = (EDGE *)malloc(sizeof(EDGE) * cpu_total_edges);
	gpu_edges = (EDGE **)malloc(sizeof(EDGE *) * num_gpus_);

	for(int i = 0; i < num_gpus_; i++)
	{
		gpu_edges[i] = (EDGE *)malloc(sizeof(EDGE) * gpu_total_edges[i]);	
	}

	int total = 0;

	//process the cpu edges
	for(int i = 0; i < cpu_total_num_partitions_; i++)
	{
		int edges_current = cpu_edge_vectors[i]->size();		
	
		//printf("xxxxxxxxxxxxxxxxxxxedges current: %d\n", edges_current);

		cpu_parts[i].start = total;
		cpu_parts[i].end = total + edges_current - 1;
		cpu_parts[i].part_id = i;
		memcpy(cpu_edges + total, &(*(cpu_edge_vectors[i]))[0], edges_current * sizeof(EDGE));
		total += edges_current;
		//since data has been copied, the vectors are no longer useful
		delete cpu_edge_vectors[i];
	}

	delete [] cpu_edge_vectors;

	printf("++++++++++++++cpu edges: =========> %d\n", total);

	//process the gpu edges
	for(int j = 0; j < num_gpus_; j++)
	{
		total = 0;
		EDGE *my_edges = gpu_edges[j];

		for(int i = 0; i < gpu_total_num_partitions_[j]; i++)
		{
			int edges_current = gpu_edge_vectors[j][i].size();		
			gpu_parts[j][i].start = total;
			gpu_parts[j][i].end = total + edges_current - 1;
			gpu_parts[j][i].part_id = i;

			//printf("==========>gpu %d edges_current: %d\n", j, edges_current);

			memcpy(my_edges + total, &((gpu_edge_vectors[j][i]))[0], edges_current * sizeof(EDGE));

			total += edges_current;
		}

		printf("++++++++++++++gpu %d edges: =========> %d\n", j, total);

		delete [] gpu_edge_vectors[j];	
	}

	printf("++++++++++++++ gpu edges copy done +++++++++++++ \n");

	delete [] gpu_edge_vectors;

	gen_part_index();
}

template <class T>
void Partitioner::exchange(T *coordinates, int i, int j)
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
int Partitioner::partition_a_dim(T *coordinates, int dim, int p, int r)
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
int Partitioner::random_partition_a_dim(T *coordinates, int dim, int p, int r) 
{
	int i =  gen_rand(p, r);	
	exchange(coordinates, r, i);
	return partition_a_dim(coordinates, dim, p, r);
}

template <class T>
int Partitioner::random_select(T *coordinates, int dim, int p, int r, int i)
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
void Partitioner::partition_device_nodes(T *coordinates)
{
	for(int i = 0; i < num_devices_; i++)
	{
		int device_start = device_num_nodes_sum_[i];
		int device_num_nodes = i==0? cpu_total_num_nodes_:gpu_total_num_nodes_[i - 1];

		//if(rank_==0)
		//printf("==============device_start %d: %d device_num_nodes: %d, num_points: %d before device partitioning================\n", i, device_start, device_num_nodes, num_points);

		//partition to devices
		random_select(coordinates, 0, device_start, num_points - 1, device_num_nodes);		



		//partition the nodes assigned to each device along the second dim
		if(i==0) //cpu
		{
			cpu_partitions[0] = CPU_X;
			cpu_partitions[1] = CPU_Y;
			cpu_partitions[2] = CPU_Z;

			partition_points(coordinates, cpu_partitions, 0, 0, cpu_total_num_nodes_ - 1);
		}

		else //gpu
		////for(int j = 0; j < gpu_total_num_partitions_[i-1] - 1; j++)
		{

			partition_points(coordinates, gpu_partitions[i - 1], 0, device_start, device_start + device_num_nodes - 1);
		//	//int st = device_start + gpu_nodes_per_partition * j;
		//	//int ed = device_start + gpu_total_num_nodes_[i-1] - 1;

		//	//random_select(coordinates, 1, st, ed, gpu_nodes_per_partition);
		}
	}

	printf("partition done.............................................\n");
	//generate index map based on position map
	for(int i = 0; i < num_points; i++)
	{
		index_map[pos_map[i]] = i;	
	}
}

template <class T>
void Partitioner::partition_points(T *coordinates, int *partitions, int dim, int start, int end)
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
                        partition_points(coordinates, partitions, dim + 1, st, ed);
                }
        }
}


void Partitioner::reorder_satellite_data(void *satellite_data, int elm_size)
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


//get the device id for a specific node
int Partitioner::get_device_id(size_t node_id)
{
	if(node_id >= num_points)
		return -1;

	for(int i = 0; i < num_devices_; i++)		
	{
		if(node_id < device_num_nodes_sum_[i])	
			return i - 1;
	}

	return num_devices_ - 1;
}

//get the partition id within device for a specific node
int Partitioner::get_partition_id(size_t node_id, int device_id)
{
	if(node_id >= num_points)
		return -1;

	//within CPU
	if(device_id == 0)
	{
		return (double)node_id/cpu_nodes_per_partition;	
	}

	//within GPU
	else
	{
		size_t start = node_id - device_num_nodes_sum_[device_id];	
		return (double)start/gpu_nodes_per_partition;
	}
}

void Partitioner::gen_part_index()
{
        for(int i = 0; i < num_points; i++)
        {
		int device_id = get_device_id(i);
                part_index[i] = get_partition_id(i, device_id);
        }

        //return part_index;
}

int Partitioner::get_cpu_num_edges()
{
	return cpu_new_edges;
}

int *Partitioner::get_gpu_num_edges()
{
	return gpu_new_edges;
}

int Partitioner::get_cpu_num_parts()
{
	return cpu_total_num_partitions_;	
}

int *Partitioner::get_gpu_num_parts()
{
	return gpu_total_num_partitions_;
}

Part *Partitioner::get_cpu_parts()
{
	return cpu_parts;
}

Part **Partitioner::get_gpu_parts()
{
	return gpu_parts;
}

EDGE *Partitioner::get_cpu_edges()
{
	return cpu_edges;
}

EDGE **Partitioner::get_gpu_edges()
{
	return gpu_edges;
}

int Partitioner::get_node_start(int device_id)
{
	return device_num_nodes_sum_[device_id];
}

int Partitioner::get_node_sum(int device_id)
{
	if(device_id==0)	
		return cpu_total_num_nodes_;
	else
		return gpu_total_num_nodes_[device_id - 1];
}
