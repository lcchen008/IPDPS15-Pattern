#ifndef REDUCTION_C
#define REDUCTION_C
#include "parameters.h"

namespace FCPU
{
	typedef void (* irreduce_fp)(VALUE *value1, VALUE *value2);
}

struct Reduction_Object_C
{
public:
	//number of keys equals number of vals
	KEY keys[NUM_BUCKETS_G];
	VALUE values[NUM_BUCKETS_G];
	int pairs_per_bucket[NUM_BUCKETS_G];
	int locks[NUM_BUCKETS_G];
	
	bool insert(KEY *key, VALUE *value, irreduce_fp reduce_ptr);
	//merge shared memory reduction object to itself
	void mergec(Reduction_Object_C *object, int tid);
	//initialize the global reduction object
	void init(int tid);
};

typedef struct Reduction_Object_C Cobject;

#endif
