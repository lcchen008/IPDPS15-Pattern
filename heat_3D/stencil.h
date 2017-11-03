#ifndef STENCIL
#define STENCIL
#define W 1

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

typedef void (* stencil_fp)(void *input, void *output, int *offset, int *size, void *parameter);

DEVICE void stencil1(void *input, void *output, int *offset, int *size, void *parameter)
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


DEVICE void stencil(void *input, void *output, int *offset, int *size, void *parameter)
{
	int k = offset[0];	
	int j = offset[1];	
	int i = offset[2];	

	//double hx , hy , hz , k0 , dt;
	////double hx = ((double *)parameter)[0], hy = ((double *)parameter)[1], hz = ((double *)parameter)[2], k0 = ((double *)parameter)[3], dt = ((double *)parameter)[4];

  	//double diagx, diagy, diagz, weightx, weighty, weightz, rk;

	//diagx = - 2.0 + hx*hx/(3*k0*dt);
  	//weightx = k0* dt/(hx*hx);
  	//diagy = - 2.0 + hy*hy/(3*k0*dt);
  	//weighty = k0* dt/(hy*hy);
  	//diagz = - 2.0 + hz*hz/(3*k0*dt);
  	//weightz = k0* dt/(hz*hz);

	GET_DOUBLE3(output, i, j, k) = 0.234 *(GET_DOUBLE3(input, i, j - 1, k) + 
			GET_DOUBLE3(input, i, j + 1, k) +
          GET_DOUBLE3(input, i, j, k)*0.234) + 0.234 *( GET_DOUBLE3(input, i, j, k - 1) + GET_DOUBLE3(input, i, j, k + 1) +
            GET_DOUBLE3(input, i, j, k)*0.234 ) + 0.234  *( GET_DOUBLE3(input, i-1, j, k) + GET_DOUBLE3(input, i+1, j, k) +
              GET_DOUBLE3(input, i, j, k)*0.234);
}

CONSTANT stencil_fp stencil_function_table[] = {stencil};

}
#endif
