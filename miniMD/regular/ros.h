#ifndef ROSH 
#define ROSH
#include "parameters.h"
//#define GPU_KERNEL
//#include "../regular.h"
namespace FFGPU
{
	typedef void (* reduce_fp)(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size);
}

using namespace FFGPU;

struct Reduction_Object
{
	public:
		//unsigned int lock; //this lock is used to lock the whole object. when the object is being merged to another object, it should be locked
		unsigned int num_buckets;
		int remaining_buckets;
		int locks[R_NUM_BUCKETS_S]; //every bucket has a lock
		int buckets[R_NUM_BUCKETS_S]; //every bucket contains an integer, which is two shorts, indices of key and value 
		unsigned int memory_pool[SHARED_POOL_SIZE]; //the memory pool in each object
		unsigned int memory_offset;

		/*if the key is already in the reduction object, conduct the operation defined 
		 *in reduce() to update the value. Else, insert a new key_value node into the key_value list of 
		 *the appropriate bucket .*/

		__device__ void oma_init(int gid, int group_size);
		//__device__ void oma_init();
		__device__ short omalloc(unsigned int size);
		__device__ void * oget_address(unsigned short index);

		__device__ bool insert(void *key, unsigned short key_size, void *value, unsigned short value_size, reduce_fp reduce_ptr);

		__device__ void * get_key_address(unsigned short bucket_index);
		__device__ unsigned short get_key_size(unsigned short bucket_index);

		__device__ void * get_value_address(unsigned short bucket_index);
		__device__ unsigned short get_value_size(unsigned int bucket_index);

		__device__ unsigned short get_value_index(unsigned short bucket_index);
		__device__ unsigned short get_key_index(unsigned short bucket_index);
		__device__ int get_compare_value(unsigned short bucket_index1, unsigned short bucket_index2);
		__device__ void swap(int &a, int &b);
		__device__ void bitonic_sort(int *testdata);
		/*free one bucket from the reduction object*/
		__device__ inline void remove(unsigned short bucket_index);
		__device__ inline void merge(Reduction_Object* object, unsigned int k, unsigned int value_size);
		__device__ inline void get_used_buckets(int *ret);
};

typedef struct Reduction_Object SO;  
#endif
