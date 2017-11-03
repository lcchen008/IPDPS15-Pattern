#ifndef GPU_KERNEL_TOOLS
#define GPU_KERNEL_TOOLS

#include "parameters.h"
#include "ros.h"
#include "../lib/common.h"

namespace FFGPU{

__device__ void copy_val(void *dst, void *src, unsigned size)
{
         char *d=(char*)dst;
         const char *s=(const char *)src;
         for(unsigned short i=0;i < size;i++)
                 d[i]=s[i];
}

__device__ unsigned int align(unsigned int size)
{
        return (size+ALIGN_SIZE-1)&(~(ALIGN_SIZE-1));
}

//round to expo of 2
__device__ unsigned int get_num_groups()
{
	unsigned int x = SHARED_SIZE/sizeof(SO);
	
	x = x | (x >> 1); 
   	x = x | (x >> 2); 
   	x = x | (x >> 4); 
   	x = x | (x >> 8); 
   	x = x | (x >>16); 

   	return x - (x >> 1); 
}

__device__ unsigned int get_group_id(int num_groups, int block_size, int tid)
{
	int ave = block_size/num_groups;	
	return tid/ave;
}

__device__ unsigned int get_gid(int num_groups, int block_size, int tid)
{
	return tid%(block_size/num_groups);	
}

__device__ unsigned int get_group_size(int num_groups, int block_size, int group_id)
{
	int ave = block_size/num_groups;	

	if(group_id < (num_groups - 1))
		return ave;
	else
		return GPU_THREADS - ave * (num_groups - 1);
}

__device__ bool get_lock(int *lockVal)
{
        return atomicCAS(lockVal, 0, 1) == 0;
}

__device__ void release_lock(int *lockVal)
{
        atomicExch(lockVal, 0);
}

__device__ unsigned int ANY_KEY()
{
	return threadIdx.x % R_NUM_BUCKETS_S;
}

__device__ int compare(void *&key1, unsigned short &key1_size, void *&key2, unsigned short &key2_size, void *&value1, unsigned short value1_size,  void *&value2, unsigned short value2_size)
{
    return 0;
}

}
#endif
