#define GPU_KERNEL
#ifndef ROS
#define ROS
#include "ros.h"
//#include "../irregular.h"
#include "hash.h"
#include "gpu_kernel_tools.cu"
#include "parameters.h"

__device__ bool Reduction_Object_S::insert(KEY *key, VALUE *value, irreduce_fp reduce_ptr)
{
	unsigned int h = FGPU::hash(key);
	unsigned int index = h%NUM_BUCKETS_S;

	unsigned int finish = 0;
        bool do_work = true;
	bool ret = true;
	int stride = 1;

	if(remaining_buckets <= NUM_BUCKETS_S * 0.1)
		return false;

	while(true)
	{
		do_work = true;	
		while(do_work)
		{
			if(FGPU::get_lock(&locks[index]))
                	{
				if(keys[index]==EMPTY_BUCKET_VALUE)
				{
					keys[index] = *key;	
					values[index] = *value;
					atomicAdd(&remaining_buckets, -1);

					finish = 1;
					do_work = false;
					threadfence();
					FGPU::release_lock(&locks[index]);
				}
				else
				{
					if(FGPU::equal(&keys[index], key))	
					{
						reduce_ptr(&values[index], value);	
						//reduce(&values[index], value);	
	                                        do_work = false;
						finish = 1;
						ret = true;
						threadfence();
						FGPU::release_lock(&locks[index]);
					}
					
					else
					{
                                        	do_work = false;
                                        	finish = 1;
                                        	FGPU::release_lock(&locks[index]);
                                        	index = (index+stride)%NUM_BUCKETS_S;
					}
				}
			}
		}

		if(finish)
			return ret;
	}
}

__device__ void Reduction_Object_S::first_init()
{
        const unsigned int tid = threadIdx.x;
	if(tid == 0)
		remaining_buckets = NUM_BUCKETS_S;

        for(int index = tid; index < NUM_BUCKETS_S; index += blockDim.x)
        {
                keys[index] = EMPTY_BUCKET_VALUE;
                locks[index] = 0;
        }
}

__device__ void Reduction_Object_S::init()
{
        const unsigned int tid = threadIdx.x;
	if(tid == 0)
		remaining_buckets = NUM_BUCKETS_S;

        for(int index = tid; index < NUM_BUCKETS_S; index += blockDim.x)
        {
                keys[index] = EMPTY_BUCKET_VALUE;
        }
}
#endif
