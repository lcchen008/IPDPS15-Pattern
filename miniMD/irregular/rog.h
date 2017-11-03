#ifndef REDUCTION_G
#define REDUCTION_G
#include "parameters.h"
#include "ros.h"

struct Reduction_Object_G
{
public:
	//number of keys equals number of vals
	KEY keys[NUM_BUCKETS_G];
	VALUE values[NUM_BUCKETS_G];
	int pairs_per_bucket[NUM_BUCKETS_G];
	int locks[NUM_BUCKETS_G];
	
	__device__ bool insert(KEY *key, VALUE *value, irreduce_fp reduce_ptr);
	//merge shared memory reduction object to itself
	__device__ void merges(Sobject *object, irreduce_fp reduce_ptr);
	//merge device memory reduction object to itself
	__device__ void mergeg(Reduction_Object_G *object, irreduce_fp reduce_ptr);
	//initialize the global reduction object
	__device__ void init();
};

typedef struct Reduction_Object_G Gobject;

#endif
