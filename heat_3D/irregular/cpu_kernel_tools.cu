#include <stdio.h>
#include "parameters.h"
namespace FCPU
{
	bool get_lock(volatile int *lock_value)
	{
	    return __sync_val_compare_and_swap(lock_value, 0, 1) == 0;
	}
	
	void release_lock(volatile int *lock_value)
	{
	    __sync_val_compare_and_swap(lock_value, 1, 0); 
	}
}
