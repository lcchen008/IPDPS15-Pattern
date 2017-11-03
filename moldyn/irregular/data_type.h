#ifndef IRREGULAR_DATA_TYPE_H_
#define IRREGULAR_DATA_TYPE_H_
#include "../lib/common.h"

struct edge
{
	int idx0;	
	int idx1;
};

struct part
{
	int start;
	int end;
	int part_id;
};

typedef struct part Part;
typedef struct edge EDGE;

#endif
