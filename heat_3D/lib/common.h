#ifndef LIB_COMMON_H_
#define LIB_COMMON_H_

#define CPU_ONLY //using only CPU to process

#include <stdint.h>
//#include "../regular/ros.h"

#define MAX_DIM 3
typedef int64_t PSIndex;
//#define PS_DEBUG

#define CPU_THREADS 12 
#define GPU_BLOCKS 14
#define GPU_THREADS 256

#define CPU_TILE_SIZE 16 
#define GPU_TILE_SIZE 32 

#define INTERNAL 0
#define BORDER 1

#define SHARED_SIZE 48128

typedef int64_t IRIndex;

//template <class T>

//typedef bool (*map_fp)(SO *object, void *global_data, int index, void *parameter, int device_type);

//typedef struct map_fp_tmp map_fp;

//typedef void (* reduce_fp)(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size);

#endif
