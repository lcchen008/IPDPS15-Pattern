#ifndef REGULAR_GPU_ARG_H_
#define REGULAR_GPU_ARG_H_

struct cpu_args_reg
{
	int tid;
	//pthread_mutex_t *mutex;
	RegularRuntime *runtime;
};

struct gpu_args_reg
{
	int gpu_id;
	RegularRuntime *runtime;
};

#endif
