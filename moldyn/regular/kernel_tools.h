#ifndef KERNEL_TOOLS
#define KERNEL_TOOLS
#if defined(GPU_KERNEL)
#define DEVICE __device__
namespace FFGPU{
#elif defined(CPU_KERNEL)
#define DEVICE
namespace FFCPU{
#endif

DEVICE void copy_val(void *dst, void *src, unsigned size);

DEVICE unsigned int align(unsigned int size);

DEVICE unsigned int ANY_KEY();

}

#endif
