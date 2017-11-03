#ifndef HASH
#define HASH

#include "kernel_tools.h"

#if defined(GPU_KERNEL)
#define DEVICE __device__
namespace FFGPU{
#elif defined(CPU_KERNEL)
#define DEVICE
namespace FFCPU{
#endif

DEVICE unsigned int hash(void *key, unsigned short size)
{
    unsigned int m;
    copy_val(&m, key, size);
    return m;
}

DEVICE bool equal(void *key1, const unsigned short size1, void *key2, const unsigned short size2)
{
       	int keytmp;
	copy_val(&keytmp, key1, size1);
	int keytmp1;
	copy_val(&keytmp1, key2, size2);
	return keytmp==keytmp1;
}

}
#endif

