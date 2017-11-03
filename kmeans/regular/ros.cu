#ifndef ROS 
#define ROS
#define GPU_KERNEL
#include "ros.h"
//#include "../regular.h"
#include "parameters.h"
#define GPU_KERNEL
#include "hash.h"
#include "gpu_kernel_tools.cu"

//typedef void (* reduce_fp)(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size);

/*if the key is already in the reduction object, conduct the operation defined 
		 *in reduce() to update the value. Otherwise, use a new bucket .*/
__device__ bool Reduction_Object::insert(void *key, unsigned short key_size, void *value, unsigned short value_size, reduce_fp reduce_ptr)
{
	int rehash = 0;
	if(remaining_buckets<=rehash)
	return false;

	//First, calculate the bucket index number
	unsigned int h = FFGPU::hash(key, key_size);	
	unsigned int index = h%R_NUM_BUCKETS_S;
	unsigned int finish = 0;
	int kvn = 0;
	
	bool DoWork = true;
	bool ret = true;
	int stride = 1;//NUM_BUCKETS/2;

	//lock only when new key is to be inserted

	while(1)
	{
		if(remaining_buckets==0)
		return false;
	
		DoWork = true;
	
		while(DoWork)
		{
			//Second, test whether the bucket is empty
			if(FFGPU::get_lock(&locks[index]))
			{
				if(buckets[index]==0)
				{
					//If the bucket is empty, the key has not appeared in the reduction object, create a new key-value pair
					//Copy the real data of key and value to shared memory
					short k  = omalloc(2+key_size);//The first byte stores size of the key
					if(k==-1)//The shared memory pool is full
					ret = false;
							
					short v = omalloc(value_size);
					if(v==-1)
					ret = false;	
	
					/*if the space is full, don't proceed to following steps*/
					if(ret==false)
					{
						FFGPU::release_lock(&locks[index]);
						finish = 1;
						break;
					}
	
					//store the key index and value index to the temparary variable 
					*((unsigned short *)&kvn)= k;
					*(1+(unsigned short *)&kvn)= v;
	
					char *key_size_address = (char *)oget_address(k);
					char *value_size_address = key_size_address + 1;
					*key_size_address = key_size; 
					*value_size_address = value_size;
					
					//The start address of the key data
					void *key_data_start = key_size_address + 2;
					//The start address of the value data
					void *value_data_start = oget_address(v); 
			
					//Copy the key data to shared memory
					FFGPU::copy_val(key_data_start,key,key_size);
					//Copy the value data to shared memory
					FFGPU::copy_val(value_data_start,value,value_size);
	
					atomicAdd(&buckets[index], kvn);
					atomicAdd(&remaining_buckets, -1);
					finish = 1;
					FFGPU::release_lock(&locks[index]);
					DoWork = false;
				}
	
				else 
				{
					unsigned short size = get_key_size(index);
					void *key_data = get_key_address(index);
	
					if(FFGPU::equal(key_data, size, key, key_size ))
					{
						//FFGPU::reduce(get_value_address(index), get_value_size(index), value, value_size);
						reduce_ptr(get_value_address(index), get_value_size(index), value, value_size);
						FFGPU::release_lock(&locks[index]);
						DoWork = false;
						finish = 1;
						ret = true;
					}

					else
					{
						FFGPU::release_lock(&locks[index]);
						DoWork = false;
						index = (index+stride)%R_NUM_BUCKETS_S;
					}
				}

			}
		}

		if(finish)
		return ret;
	}
}

//__device__ void Reduction_Object::oma_init()
//{
//	const unsigned int tid = threadIdx.x;
//	const unsigned int group_size = blockDim.x/NUM_GROUPS;
//
//	for(int index = tid%group_size; index < NUM_BUCKETS_S; index += group_size)
//	{
//		buckets[index] = 0;
//		locks[index] = 0;
//	}
//
//	if(tid%group_size==0)
//	{
//		num_buckets = NUM_BUCKETS_S;	
//		memory_offset = 0;
//		remaining_buckets = NUM_BUCKETS_S;
//	}
//}

__device__ void Reduction_Object::oma_init(int gid, int group_size)
{
	//const unsigned int tid = threadIdx.x;
	//const unsigned int group_size = blockDim.x/NUM_GROUPS;

	for(int index = gid; index < R_NUM_BUCKETS_S; index += group_size)
	{
		buckets[index] = 0;
		locks[index] = 0;
	}

	if(gid==0)
	{
		num_buckets = R_NUM_BUCKETS_S;	
		memory_offset = 0;
		remaining_buckets = R_NUM_BUCKETS_S;
	}
}

__device__ short Reduction_Object::omalloc(unsigned int size)
{
	size = FFGPU::align(size)/ALIGN_SIZE;
	unsigned int offset = atomicAdd(&memory_offset, size);

	if(offset+size>SHARED_POOL_SIZE)
	return -1; //-1 stands for fail

	else
	return offset; 
}

__device__ void * Reduction_Object::oget_address(unsigned short index)
{
	return memory_pool + index;
}


__device__ void * Reduction_Object::get_key_address(unsigned short bucket_index)
{
	if(buckets[bucket_index]==0)
	return 0;

	unsigned short key_index = ((unsigned short *)&buckets[bucket_index])[0]; 
	char *key_size_address = (char *)oget_address(key_index);
	return key_size_address + 2; //key data starts after the size byte
}

__device__ unsigned short Reduction_Object::get_key_size(unsigned short bucket_index)
{
	unsigned short key_index = /*get_key_index(bucket_index);*/((unsigned short *)&buckets[bucket_index])[0]; 
	return *(char *)oget_address(key_index);
}

__device__ unsigned short Reduction_Object::get_key_index(unsigned short bucket_index)
{
	unsigned short key_index = ((unsigned short *)&buckets[bucket_index])[0];
	return key_index;
}

__device__ void * Reduction_Object::get_value_address(unsigned short bucket_index)
{
	unsigned short value_index = ((unsigned short *)&buckets[bucket_index])[1]; 
	return oget_address(value_index);
}

__device__ unsigned short Reduction_Object::get_value_size(unsigned int bucket_index)
{
	unsigned short key_index = ((unsigned short *)&buckets[bucket_index])[0]; 
	return *((char *)oget_address(key_index) + 1);
}

__device__ unsigned short Reduction_Object::get_value_index(unsigned short bucket_index)
{
	return ((unsigned short *)&buckets[bucket_index])[1];
}

__device__ int Reduction_Object::get_compare_value(unsigned short bucket_index1, unsigned short bucket_index2)
{
	int bucket1 = buckets[bucket_index1];
	int bucket2 = buckets[bucket_index2];
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

__device__ void Reduction_Object::swap(int &a, int &b)
{
	int tmp = a;
	a = b;
	b = tmp;
}

__device__ void Reduction_Object::bitonic_sort(int *testdata)
{
	//const unsigned int id = threadIdx.x;
	const unsigned int tid = threadIdx.x;
	const unsigned int group_size = blockDim.x/NUM_GROUPS;
	const unsigned int id = tid%group_size; //the id in one group

	//*testdata = 0;
	__syncthreads();

	//atomicAdd(testdata, id);

	for(int k = 2; k <= R_NUM_BUCKETS_S; k=k << 1)
		for(int j = k/2; j>0; j = j >> 1)
		{
			unsigned int ixj = id ^ j;	
			if(id<R_NUM_BUCKETS_S && ixj<R_NUM_BUCKETS_S)
			if(ixj>id)
			{
				if((id & k)==0)	
				{
					if(get_compare_value(id, ixj)>0)
					swap(buckets[id], buckets[ixj]);
				}
				else
				{
					if(get_compare_value(id, ixj)<0)
					swap(buckets[id], buckets[ixj]);
				}
			}
			__syncthreads();
		}
}

__device__ inline void Reduction_Object::remove(unsigned short bucket_index)
{
	buckets[bucket_index] = 0;
}

//__device__ inline void Reduction_Object::merge(Reduction_Object* object, unsigned int k, unsigned int value_size)
//{
//	const unsigned int tid = threadIdx.x;
//	const unsigned int group_size = blockDim.x/NUM_GROUPS;
//	const unsigned int id = tid%group_size;
//
//	for(int i = id; i < k; i += group_size)
//	{
//		if((object->buckets)[i]!=0)
//		{
//			int key_size = object->get_key_size(i);
//			void *key = object->get_key_address(i);
//			void *value = object->get_value_address(i);
//			insert(key, key_size, value, value_size);
//		}
//	}
//}

//__device__ inline void Reduction_Object::merge(Reduction_Object* object, unsigned int k, unsigned int value_size)
//{
//	const unsigned int tid = threadIdx.x;
//	const unsigned int group_size = blockDim.x/NUM_GROUPS;
//	const unsigned int id = tid%group_size;
//
//	for(int i = id; i < k; i += group_size)
//	{
//		if((object->buckets)[i]!=0)
//		{
//			int key_size = object->get_key_size(i);
//			void *key = object->get_key_address(i);
//			void *value = object->get_value_address(i);
//			insert(key, key_size, value, value_size);
//		}
//	}
//}

__device__ inline void Reduction_Object::get_used_buckets(int *ret)
{
	*ret = 0;
	for(int i = 0; i<R_NUM_BUCKETS_S; i++)
	{
		if(buckets[i]!=0)
		*ret = *ret + 1;
	}
}

#endif
