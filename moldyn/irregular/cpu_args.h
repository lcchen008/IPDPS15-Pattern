#ifndef IRREGULAR_CPU_ARGS_H_
#define IRREGULAR_CPU_ARGS_H_
#include "irregular_runtime.h"

struct cpu_args
{
    int tid;
    pthread_mutex_t *mutex;
    IrregularRuntime *runtime;
};

typedef struct cpu_args CpuArgs;

struct init_args
{
	int tid;
	Cobject *roc;	
};

struct merge_args
{
	int tid;
	Cobject *roc1;
	Cobject *roc2;
};


#endif
