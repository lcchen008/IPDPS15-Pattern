#ifndef LIB_CPU_UTIL_H_
#define LIB_CPU_UTIL_H_
#include "stencil_runtime.h"
#include "DS.h"

struct cargs
{
	StencilRuntime *runtime;
	std::vector<struct Tile> *tiles;
	int tid;
};

typedef struct cargs Cargs;

#endif
