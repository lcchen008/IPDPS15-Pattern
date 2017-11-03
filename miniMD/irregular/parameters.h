#ifndef IRREGULAR_PARAMETERS_H_
#define IRREGULAR_PARAMETERS_H_

#define NUM_BUCKETS_S 2400 
#define NUM_BUCKETS_G 600000 

#define CPU_BLOCK_SIZE 100000
#define GPU_BLOCK_SIZE 1

#define CPU_X 4
#define CPU_Y 4
#define CPU_Z 6

#define CPU_NUM_PARTS 256 

#define GPU_X 4
#define GPU_Y 4

typedef int KEY;

struct value
{
	float x;
	float y;
	float z;
};

typedef struct value VALUE;

#define EMPTY_BUCKET_VALUE -1

#endif

