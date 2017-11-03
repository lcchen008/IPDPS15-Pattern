#include "partition_util.h"
#include <stdio.h>
#include <math.h>

int get_proc_id(int node_id, int global_num_nodes, int num_procs)
{
	double average = ceil((double)global_num_nodes/num_procs);
	int ret = node_id/average;

	return ret;
}

