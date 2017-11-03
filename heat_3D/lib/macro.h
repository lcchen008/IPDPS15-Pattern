#ifndef LIB_MACRO_H_
#define LIB_MACRO_H_

#include "common.h"
#include "../regular/parameters.h"

#include </usr/local/cuda/5.0.35/include/cuda_runtime.h>

#define CUDA_SAFE_CALL(x) do {                                  \
    cudaError_t e = x;                                          \
    if (e != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA ERROR at " __FILE__ "#%d: %s\n",    \
              __LINE__, cudaGetErrorString(e));                 \
      exit(EXIT_FAILURE);                                       \
    }                                                           \
  } while (0)

#define CUDA_CHECK_ERROR(msg) do {                                      \
    cudaError_t e = cudaGetLastError();                                 \
    if (e != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA ERROR: %s at " __FILE__ "#%d: %s\n",        \
              msg, __LINE__, cudaGetErrorString(e));                    \
    }                                                                   \
    e = cudaDeviceSynchronize();                                        \
    if (e != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA ERROR: %s at " __FILE__ "#%d: %s\n",        \
              msg, __LINE__, cudaGetErrorString(e));                    \
    }                                                                   \
  } while (0)

#define CUDA_DEVICE_INIT(devid) do {                            \
    cudaDeviceProp dp;                                          \
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&dp, devid));        \
    LOG_INFO() << "Using device " << devid                      \
               << ": " << dp.name << "\n";                      \
    CUDA_SAFE_CALL(cudaSetDevice(devid));                       \
  } while (0)

#define FOREACH(it, it_begin, it_end)           \
  for (typeof(it_begin) it = (it_begin),        \
           _for_each_end = (it_end);            \
       it != _for_each_end; ++it)

#define PS_XFREE(p) do {                        \
	    if (p) {                                    \
		          free(p);                                  \
		          p = NULL;                                 \
		        } } while (0)

#define PS_XDELETE(p) do {                        \
	    delete (p);                                   \
	    p = NULL;                                     \
	  } while (0)

#define PS_XDELETEA(p) do {                         \
	    delete[] (p);                                   \
	    p = NULL;                                       \
	  } while (0)


#define CHECK_MPI(c) do {                       \
  int ret = c;                                  \
  if (ret != MPI_SUCCESS) {                     \
    std::cout << "MPI error\n";               \
    exit(1);                                 \
  }                                             \
  } while (0)


#define GET_FLOAT3(input, i, j, k) *((float *)input + size[0]*(j) + (i) * size[0]*size[1] + (k))

#define GET_DOUBLE3(input, i, j, k) *((double *)input + size[0]*(j) + (i) * size[0]*size[1] + (k))

#define GET_INT3(input, i, j, k) *((int *)input + size[0]*(j) + (i)*size[0]*size[1] + (k))

#define GET_FLOAT2(input, i, j) *((float *)input + size[0]*(i) + j)

#define GET_INT2(input, i, j) *((int *)input + size[0]*(i) + j)


#define CPU 0
#define GPU 1

#endif /* PHYSIS_RUNTIME_RUNTIME_COMMON_CUDA_H_ */
