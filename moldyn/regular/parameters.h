#ifndef REGULAR_PARAMETERS_H_
#define REGULAR_PARAMETERS_H_


#define R_GPU_PREALLOC_SIZE 140000
#define R_GPU_BLOCK_SIZE 5000
#define R_CPU_PREALLOC_SIZE 100000
#define R_CPU_BLOCK_SIZE 10000

#define GLOBAL_POOL_SIZE 4096 
#define SHARED_POOL_SIZE 1200 
#define R_NUM_BUCKETS_S 41 
#define R_NUM_BUCKETS_G 41 

//#define GLOBAL_POOL_SIZE 4096 
//#define SHARED_POOL_SIZE 2048 
//#define R_NUM_BUCKETS_S 60 
//#define R_NUM_BUCKETS_G 60 


#define NUM_GROUPS 1

#define ALIGN_SIZE 4

#define USE_SHARED

#define WARP_SIZE 64

#define CPU_THREADS 12 

#define GPU_THREADS 256
#define GPU_BLOCKS 14 

//#define SHARED_SIZE 32768

#endif
