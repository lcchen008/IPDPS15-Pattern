#ifndef REDUCTION_S
#define REDUCTION_S
#include "parameters.h"

namespace FGPU
{
	typedef void (* irreduce_fp)(VALUE *value1, VALUE *value2);
}

using namespace FGPU;

struct Reduction_Object_S
{
	public:
		//number of keys equals number of vals
		KEY keys[NUM_BUCKETS_S];
		VALUE values[NUM_BUCKETS_S];
		int locks[NUM_BUCKETS_S]; 
		int remaining_buckets;
		//hight-weight initilization
		__device__ void first_init();
		//light-weight initialization
		__device__ void init();
		__device__ bool insert(KEY *key, VALUE *value, irreduce_fp reduce_ptr);
};

typedef struct Reduction_Object_S Sobject;

#endif
