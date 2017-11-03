#ifndef CPU_KERNEL_TOOLS
#define CPU_KERNEL_TOOLS
#include <stdio.h>
#include "parameters.h"
namespace FCPU
{

	//void pass_token(volatile int *token, volatile Status *status, int device_type);
	
	bool get_lock(volatile int *lock_value);
	
	void release_lock(volatile int *lock_value);
}
#endif
