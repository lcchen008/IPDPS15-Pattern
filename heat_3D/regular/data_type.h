#ifndef REGULAR_DATA_TYPE_H_
#define REGULAR_DATA_TYPE_H_
#include <stdlib.h>

//#include "ros.h"
//#include "roc.h"

#if defined(GPU_KERNEL)
#define DEVICE __device__
#define OBJECT SO
#elif defined(CPU_KERNEL)
#define DEVICE
#define OBJECT CO
#endif

//typedef bool (* map_fp)(OBJECT *object, void *global_data, int index, void *parameter, int device_type);
//
//typedef void (* reduce_fp)(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size);

//the smallest data unit to be processed by a thread
struct element_t
{
	size_t offset;		
	size_t size;
};

typedef struct element_t Offset;

struct output
{
    char *output_keys;
    char *output_vals;
    unsigned int *key_index;
    unsigned int *val_index;
    unsigned int key_num;
    unsigned int key_size;
    unsigned int value_size;
};

#endif
