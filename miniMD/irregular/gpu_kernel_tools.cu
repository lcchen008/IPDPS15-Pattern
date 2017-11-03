#ifndef GPU_KERNEL_TOOLS
#define GPU_KERNEL_TOOLS
namespace FGPU
{
	__device__ bool get_lock(int *lockVal)
	{
	        return atomicCAS(lockVal, 0, 1) == 0;
	}
	
	__device__ void release_lock(int *lockVal)
	{
	        atomicExch(lockVal, 0);
	}

}
#endif
