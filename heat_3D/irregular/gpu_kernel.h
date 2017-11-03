#ifndef GPU_KERNELH
#define GPU_KERNELH

__global__ void compute_gpu(
void *input,
void *output,
void *data_offset,      //used to split the input data
volatile int *task_offset,       //indicates the global task offset
int *device_offset,     //offset within each device
volatile int *base_offset,       //the base offset of preallocated task block
volatile int *token,
Status *status,
volatile int *has_task,
int offset_number,
int width,
unsigned int unit_size,
volatile int *arrayin1,
volatile int *arrayout1,
volatile int *arrayin2,
volatile int *arrayout2,
Gobject *object_g
);

__global__ void mergecgd(Gobject *object1, Gobject *object2);

#endif
