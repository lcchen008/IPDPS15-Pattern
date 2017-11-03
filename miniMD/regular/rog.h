#ifndef ROGH 
#define ROGH
#include "parameters.h"
#include "ros.h"

struct Reduction_Object_GPU
{
	public:
		unsigned int num_buckets;
		int locks[R_NUM_BUCKETS_G];
		unsigned long long int buckets[R_NUM_BUCKETS_G]; 
		unsigned int memory_pool[GLOBAL_POOL_SIZE];
		unsigned int key_size_per_bucket[R_NUM_BUCKETS_G];
		unsigned int value_size_per_bucket[R_NUM_BUCKETS_G];
		unsigned int pairs_per_bucket[R_NUM_BUCKETS_G];
		unsigned int offsets[GPU_BLOCKS*GPU_THREADS/WARP_SIZE];
		unsigned int memory_offset;

		__device__ void oma_init();
		/*returns the index*/
		__device__ int omalloc(unsigned int size);
		__device__ void *oget_address(unsigned int index);

		__device__ bool insert(void *key, unsigned short key_size, void *value, unsigned short value_size, reduce_fp reduce_ptr);

		__device__ void * get_key_address(unsigned int bucket_index);
		__device__ unsigned short get_key_size(unsigned int bucket_index);

		__device__ void * get_value_address(unsigned int bucket_index);
		__device__ unsigned short get_value_size(unsigned int bucket_index);
		__device__ int get_compare_value(unsigned int bucket_index1, unsigned int bucket_index2);
		__device__ void swap(unsigned long long int &a, unsigned long long int &b);
		__device__ void bitonic_merge(unsigned int *testdata, unsigned int k, unsigned int j);
        	__device__ void merge(SO *ros, reduce_fp reduce_ptr);
		__device__ void mergeg(struct Reduction_Object_GPU *object, reduce_fp reduce_ptr);
};

typedef struct Reduction_Object_GPU GO;

#endif
