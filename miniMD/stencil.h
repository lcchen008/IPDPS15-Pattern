#ifndef STENCIL
#define STENCIL
#define W 3

#include "lib/array.h"

#if defined(GPU_KERNEL)
#define DEVICE __device__
#define CONSTANT __constant__
namespace FGPU{
#elif defined(CPU_KERNEL)
#define DEVICE
#define CONSTANT
namespace FCPU{
#endif

DEVICE size_t get(size_t reference, int dims, int *size, int i, int j, int k)
{
	if(dims == 2)	
	{
		reference += i + j * size[0];	
	}
	else if(dims == 3)
	{
		reference += i + j * size[0] + k * size[0] * size[1];	
	}
	return reference;
}

//#define (void *input, void *output, int *offset) (void *input, void *output, int *offset, int *size)

typedef void (* stencil_fp)(void *input, void *output, int *offset, int *size);

DEVICE void stencil1(void *input, void *output, int *offset, int *size)
{
	int k = offset[0];	
	int j = offset[1];	
	int i = offset[2];	

	int count = 0;
	float total = 0;
	int m;
	/*X direction*/
	/*positive direction*/
	for(m = 1; m < W; m++)
	{
		count++;
		total += GET_FLOAT3(input, i, j, k+m);
	}

	/*negative direction*/
	for(m = 1; m < W; m++)
	{
		count++;
		total += GET_FLOAT3(input, i, j, k-m);
	}

	/*Y direction*/
	/*positive direction*/
	for(m = 1; m < W; m++)
	{
		count++;
		total += GET_FLOAT3(input, i, j+m, k);
	}

	/*negative direction*/
	for(m = 1; m < W; m++)
	{
		count++;
		total += GET_FLOAT3(input, i, j-m, k);
	}

	/*Z direction*/
	/*positive direction*/
	for(m = 1; m < W; m++)
	{
		count++;
		total += GET_FLOAT3(input, i+m, j, k);
	}

	/*negative direction*/
	for(m = 1; m < W; m++)
	{
		count++;
		total += GET_FLOAT3(input, i-m, j, k);
	}

	total /= count;
	
	GET_FLOAT3(output, i, j, k) = total;
}


DEVICE void stencil(void *input, void *output, int *offset, int *size)
{
	int k = offset[0];	
	int j = offset[1];	
	int i = offset[2];	

	int count = 0;
	float total = 0;
	int m;
	/*X direction*/
	/*positive direction*/
	for(int l = 0; l < 10; l++)
	{
	for(m = 1; m < W; m++)
	{
		count++;
		total += 2*GET_FLOAT3(input, i, j, k+m);
	}

	/*negative direction*/
	for(m = 1; m < W; m++)
	{
		count++;
		total += 2*GET_FLOAT3(input, i, j, k-m);
	}

	/*Y direction*/
	/*positive direction*/
	for(m = 1; m < W; m++)
	{
		count++;
		total += 2*GET_FLOAT3(input, i, j+m, k);
	}

	/*negative direction*/
	for(m = 1; m < W; m++)
	{
		count++;
		total += 2*GET_FLOAT3(input, i, j-m, k);
	}

	/*Z direction*/
	/*positive direction*/
	for(m = 1; m < W; m++)
	{
		count++;
		total += 2*GET_FLOAT3(input, i+m, j, k);
	}

	/*negative direction*/
	for(m = 1; m < W; m++)
	{
		count++;
		total += 2*GET_FLOAT3(input, i-m, j, k);
	}
	}

	total /= count;
	
	GET_FLOAT3(output, i, j, k) = total;
}

CONSTANT stencil_fp stencil_function_table[] = {stencil};

}
#endif
