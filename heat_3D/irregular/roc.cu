#ifndef ROC
#define ROC
#define CPU_KERNEL
#include "roc.h"
#include "hash.h"
#include "parameters.h"
#include "cpu_kernel_tools.h"
//#include "../irregular.h"
#include "stdio.h"

bool Reduction_Object_C::insert(KEY *key, VALUE *value, irreduce_fp reduce_ptr)
{
	unsigned int h = FCPU::hash(key);
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
			if(FCPU::get_lock(&locks[index]))
			{
				if(keys[index]==EMPTY_BUCKET_VALUE)
				{
					keys[index] = *key;	
					values[index] = *value;
					pairs_per_bucket[index] = 1;
					//printf("added\n");

					finish = 1;
					do_work = false;
					FCPU::release_lock(&locks[index]);
				}
				else
				{
					if(FCPU::equal(&keys[index], key))	
					{
						//FCPU::reduce(&values[index], value);
						reduce_ptr(&values[index], value);					
						do_work = false;
						finish = 1;
						ret = true;
						FCPU::release_lock(&locks[index]);
					}
					
					else
					{
						do_work = false;
						finish = 1;
						FCPU::release_lock(&locks[index]);
						index = (index+stride)%NUM_BUCKETS_G;
					}
				}
			}
		}

		if(finish)
			return ret;
	}
}

void Reduction_Object_C::mergec(Reduction_Object_C *object, int tid)
{
	for(int i = tid; i < NUM_BUCKETS_G; i+=CPU_THREADS) 
	{
		if(object->pairs_per_bucket[i]!=0)
		{
			//if(object->pairs_per_bucket[i]!=0)
				//insert(&(object->keys)[i], &(object->values)[i]);	
		}
	}
}

void Reduction_Object_C::init(int tid)
{
	for(int i = 0; i < NUM_BUCKETS_G; i ++)
	{
		keys[i] = EMPTY_BUCKET_VALUE;	
		pairs_per_bucket[i] = 0;
	}
}

#endif
