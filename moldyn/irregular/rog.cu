#define GPU_KERNEL
#ifndef ROG
#define ROG
#include "parameters.h"
#include "../irregular.h"
#include "stdio.h"
#include "rog.h"
#include "hash.h"
#include "gpu_kernel_tools.cu"

__device__ bool Reduction_Object_G::insert(KEY *key, VALUE *value, irreduce_fp reduce_ptr)
{
	unsigned int h = FGPU::hash(key);
	unsigned int index = h%NUM_BUCKETS_G;

	unsigned int finish = 0;
        bool do_work = true;
	bool ret = true;
	int stride = 1;

	while(true)
	{
		do_work = true;	
		while(do_work)
		{
			//if(FGPU::get_lock(&locks[index]))
                	{
				if(keys[index]==EMPTY_BUCKET_VALUE)
				{
					keys[index] = *key;	
					values[index] = *value;
					pairs_per_bucket[index] = 1;

					finish = 1;
					do_work = false;
					threadfence();
					//FGPU::release_lock(&locks[index]);
				}
				else
				{
					if(FGPU::equal(&keys[index], key))	
					{
						//FGPU::reduce(&values[index], value);					
						reduce_ptr(&values[index], value);					
	                                        do_work = false;
						finish = 1;
						ret = true;
						threadfence();
						//FGPU::release_lock(&locks[index]);
					}
					
					else
					{
                                        	do_work = false;
                                        	finish = 1;
                                        	//FGPU::release_lock(&locks[index]);
                                        	index = (index+stride)%NUM_BUCKETS_G;
					}
				}
			}
		}

		if(finish)
			return ret;
	}
}

__device__ void Reduction_Object_G::merges(Sobject* object, irreduce_fp reduce_ptr)
{
	const unsigned int tid = threadIdx.x;

	for(int index = tid; index < NUM_BUCKETS_S; index += blockDim.x)
	{
		insert(&(object->keys)[index], &(object->values)[index], reduce_ptr);	
	}
}

__device__ void Reduction_Object_G::mergeg(Reduction_Object_G *object, irreduce_fp reduce_ptr)
{
	const unsigned int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int global_size = gridDim.x * blockDim.x;

	for(int index = global_id; index < NUM_BUCKETS_G; index += global_size)
	{
		if(object->pairs_per_bucket[index]!=0)
			insert(&(object->keys)[index], &(object->values)[index], reduce_ptr);
	}
}

__device__ void Reduction_Object_G::init()
{
	const unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int global_size = gridDim.x * blockDim.x;

	for(int i = global_id; i < NUM_BUCKETS_G; i += global_size)
	{
		keys[i] = EMPTY_BUCKET_VALUE;	
		pairs_per_bucket[i] = 0;
	}
}
#endif
