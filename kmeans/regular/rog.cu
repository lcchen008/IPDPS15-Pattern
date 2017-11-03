#define GPU_KERNEL
#include <stdio.h>
#ifndef ROG
#define ROG

#include "rog.h"
#include "ros.cu"
//#include "../regular.h"
#include "parameters.h"
#include "hash.h"
#include "gpu_kernel_tools.cu"

using namespace FFGPU;

__shared__ unsigned int global_object_offset[GPU_THREADS/WARP_SIZE];

__device__ void Reduction_Object_GPU::oma_init()
{
	unsigned int tid = threadIdx.x;
	if(tid%WARP_SIZE==0)
	{
		unsigned int bid = blockIdx.x;
		unsigned int global_id = bid * blockDim.x + tid;
		int local_group_id = tid/WARP_SIZE;
		global_object_offset[local_group_id] = offsets[global_id/WARP_SIZE];
	}
}

__device__ int Reduction_Object_GPU::omalloc(unsigned int size)
{
	size = FFGPU::align(size)/ALIGN_SIZE;
	//unsigned int bid = blockIdx.x;
	unsigned int tid = threadIdx.x;
	//unsigned int num_groups_g = NUM_THREADS/WARP_SIZE;
	unsigned int gid_g = tid/WARP_SIZE;

	unsigned int offset = atomicAdd(&global_object_offset[gid_g], size);

	return offset; 
}

__device__ void * Reduction_Object_GPU::oget_address(unsigned int index)
{
	return memory_pool + index;	
}

__device__ bool Reduction_Object_GPU::insert(void *key, unsigned short key_size, void *value, unsigned short value_size, reduce_fp reduce_ptr)
{
	//First, calculate the bucket index number


	unsigned int h = FFGPU::hash(key, key_size);	

//	int keytmp1;
//	FFGPU::copy_val(&keytmp1, key, key_size);
//	printf("key is: %d\n", keytmp1);

    //printf("hash is: %d\n", h);

	unsigned int index = h%R_NUM_BUCKETS_G;
	unsigned int finish = 0;
	unsigned long long int kvn = 0;
	
	bool DoWork = true;
	bool ret = true;
	int stride = 1;

	//lock only when new key is to be inserted
	while(true){

	DoWork = true;
	while(DoWork)
	{
		//Second, test whether the bucket is empty
		if(FFGPU::get_lock(&locks[index]))
		{
			if(pairs_per_bucket[index]==0)
			{
				//If the bucket is empty, the key has not appeared in the reduction object, create a new key-value pair
				//Copy the real data of key and value to shared memory
				int k  = omalloc(key_size);//The first byte stores size of the key, and the second byte stores the size of the val
				if(k==-1)//The shared memory pool is full
				ret = false;
						
				int v = omalloc(value_size);
				if(v==-1)
				ret = false;	

				//store the key index and value index to the temparary variable 
				FFGPU::copy_val((int *)&kvn, &k, sizeof(k));
				FFGPU::copy_val((int *)&kvn + 1, &v, sizeof(v));

				//The start address of the key data
				void *key_data_start = oget_address(k);
				//The start address of the value data
				void *value_data_start = oget_address(v); 
		
				//Copy the key data to shared memory
				FFGPU::copy_val(key_data_start,key,key_size);
				//Copy the value data to shared memory
				FFGPU::copy_val(value_data_start,value,value_size);

				buckets[index] = kvn;

				key_size_per_bucket[index] = key_size;	
				value_size_per_bucket[index] = value_size;
				pairs_per_bucket[index] = 1;

				unsigned short size = get_key_size(index);
				void *key_data = get_key_address(index);
				int keytmp ;
                		FFGPU::copy_val(&keytmp, key_data, size);

				//if(*(int *)key != keytmp)
				//printf("in key is: %d and keytmp is: %d\n", *(int *)key, keytmp);

				finish = 1;
				DoWork = false;
				threadfence();

				FFGPU::release_lock(&locks[index]);
			}

			else 
			{
				unsigned short size = get_key_size(index);
				void *key_data = get_key_address(index);

                	//	int keytmp ;
                	//	FFGPU::copy_val(&keytmp, key_data, size);

			//	int keytmp1;
			//	FGPU::copy_val(&keytmp1, key, key_size);
	
				if(FFGPU::equal(key_data, size, key, key_size))
				{
					//printf("equal..\n");
					//FFGPU::reduce(get_value_address(index), get_value_size(index), value, value_size);
					reduce_ptr(get_value_address(index), get_value_size(index), value, value_size);
					DoWork = false;
					finish = 1;
					ret = true;

					threadfence();
					FFGPU::release_lock(&locks[index]);
				}

				else
				{
					//int keytmp1;
					//FGPU::copy_val(&keytmp1, key, key_size);
                    			//printf("keysize: %d continue... %d, real key: %d\n",size, keytmp, keytmp1);
					//printf("continue..\n");
					DoWork = false;
                    			finish = 1;

					FFGPU::release_lock(&locks[index]);
					index = (index+stride)%R_NUM_BUCKETS_G;
				}
			}
			
		}
	}
	if(finish)
	return ret;

	}

	}

__device__ void * Reduction_Object_GPU::get_key_address(unsigned int bucket_index)
{
	unsigned int key_index = ((unsigned int *)&buckets[bucket_index])[0]; 
	char *key_size_address = (char *)oget_address(key_index);
	return key_size_address;
}

__device__ unsigned short Reduction_Object_GPU::get_key_size(unsigned int bucket_index)
{
	return key_size_per_bucket[bucket_index];
}

__device__ void * Reduction_Object_GPU::get_value_address(unsigned int bucket_index)
{
	unsigned int value_index = ((unsigned int *)&buckets[bucket_index])[1]; 
	return oget_address(value_index);
}

__device__ unsigned short Reduction_Object_GPU::get_value_size(unsigned int bucket_index)
{
	return value_size_per_bucket[bucket_index];
}

__device__ int Reduction_Object_GPU::get_compare_value(unsigned int bucket_index1, unsigned int bucket_index2)
{
	unsigned long long int bucket1 = buckets[bucket_index1];
	unsigned long long int bucket2 = buckets[bucket_index2];
	int compare_value = 0;

	if(bucket1==0&&bucket2==0)
	compare_value = 0;

	else if(bucket2==0)
	compare_value = -1;

	else if(bucket1==0)
	compare_value = 1;

	else
	{
		unsigned short key_size1 = get_key_size(bucket_index1);	
		unsigned short key_size2 = get_key_size(bucket_index2);	
		void *key_addr1 = get_key_address(bucket_index1);
		void *key_addr2 = get_key_address(bucket_index2);
		unsigned short value_size1 = get_value_size(bucket_index1);
		unsigned short value_size2 = get_value_size(bucket_index2);
		void *value_addr1 = get_value_address(bucket_index1);
		void *value_addr2 = get_value_address(bucket_index2);
		compare_value = FFGPU::compare(key_addr1, key_size1, key_addr2, key_size2, value_addr1, value_size1, value_addr2, value_size2);
	}

	return compare_value;
}

__device__ void Reduction_Object_GPU::swap(unsigned long long int &a, unsigned long long int &b)
{
	unsigned long long int tmp = a;
	a = b;
	b = tmp;
}

__device__ void swap_int(unsigned int &a, unsigned int &b)
{
	unsigned int tmp =a;	
	a = b;
	b = tmp;
}

/*The sort is based on the key value*/
__device__ void Reduction_Object_GPU::bitonic_merge(unsigned int *testdata, unsigned int k, unsigned int j)
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;
	const unsigned int id = bid*blockDim.x+tid;
	const unsigned int ixj = id ^ j;//j controls which thread to use 

	int compare = 0;
	if(ixj<R_NUM_BUCKETS_G && id<R_NUM_BUCKETS_G)
	if(ixj > id)
	{
		compare = get_compare_value(id, ixj);
		if((id & k) == 0)//k controls the direction
		{
			if(compare>0)
			{
				swap(buckets[id], buckets[ixj]);
				swap_int(key_size_per_bucket[id], key_size_per_bucket[ixj]);
				swap_int(value_size_per_bucket[id], value_size_per_bucket[ixj]);
				swap_int(pairs_per_bucket[id], pairs_per_bucket[ixj]);
			}
		}

		else
		{
			if(compare<0)
			{
				swap(buckets[id], buckets[ixj]);
				swap_int(key_size_per_bucket[id], key_size_per_bucket[ixj]);
				swap_int(value_size_per_bucket[id], value_size_per_bucket[ixj]);
				swap_int(pairs_per_bucket[id], pairs_per_bucket[ixj]);
			}
		}
	}
}

__device__ void Reduction_Object_GPU::merge(SO *object, reduce_fp reduce_ptr)
{
    const unsigned int tid = threadIdx.x;
    for(int i = tid; i < object->num_buckets; i+=GPU_THREADS)
    {
         if((object->buckets)[i]!=0)
         {
            void *key = object->get_key_address(i); 
            unsigned key_size = object->get_key_size(i);
            void *value = object->get_value_address(i);
            unsigned value_size = object->get_value_size(i);
            int key1;
            FFGPU::copy_val(&key1, key, sizeof(int));
            //printf("key size: %d\n", key1);
            insert(key, key_size, value, value_size, reduce_ptr);
         }
    }
}

__device__ void Reduction_Object_GPU::mergeg(GO *object, reduce_fp reduce_ptr)
{
    const unsigned int tid = threadIdx.x;
    for(int i = tid; i < object->num_buckets; i+=GPU_THREADS)
    {
         if((object->buckets)[i]!=0)
         {
            void *key = object->get_key_address(i); 
            unsigned key_size = object->get_key_size(i);
            void *value = object->get_value_address(i);
            unsigned value_size = object->get_value_size(i);
	    //printf("key: %d\n", *(int *)key);
            insert(key, key_size, value, value_size, reduce_ptr);
         }
    }
}

#endif
