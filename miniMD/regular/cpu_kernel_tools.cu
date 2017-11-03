#include <stdio.h>
#include "parameters.h"
namespace FFCPU
{
	void copy_val(void *dst, void *src, unsigned size)
	{
	         char *d= (char *)dst;
	         const char *s = (const char *)src;
	         for(unsigned short i=0; i < size; i++)
	            d[i]=s[i];
	}
	
	unsigned int align(unsigned int size)
	{
	        return (size+ALIGN_SIZE-1)&(~(ALIGN_SIZE-1));
	}

	unsigned int ANY_KEY()
	{
		return 0;	
	}
}

