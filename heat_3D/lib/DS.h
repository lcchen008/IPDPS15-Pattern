#ifndef LIB_DS_H_
#define LIB_DS_H_
#include "array.h"

struct Width2 
{
  	UnsignedArray bw;
  	UnsignedArray fw;
};

struct Tile
{
	IndexArray offset;
	IndexArray size;
};

#endif
