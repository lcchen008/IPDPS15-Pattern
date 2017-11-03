#ifndef HASH
#define HASH
#include "parameters.h"

#if defined(GPU_KERNEL)
#define DEVICE __device__
namespace FGPU{
#elif defined(CPU_KERNEL)
#define DEVICE
namespace FCPU{
#endif

DEVICE unsigned int hash(KEY *key)
{
    return (unsigned int)(*key);
}

DEVICE bool equal(KEY *key1, KEY *key2)
{
       return *key1==*key2; 
}

}
#endif

