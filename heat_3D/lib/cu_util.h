#ifndef LIB_CU_UTILS_H_
#define LIB_CU_UTILS_H_
#include <stdio.h>

int GetGPUNumber();

#define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void checkCUDAError(const char *msg);



#endif
