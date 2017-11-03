#include <stdio.h>
#ifndef REDUCTION
#define REDUCTION

#include <math.h>
#include "kmeans.h"
#include "regular/kernel_tools.h"
#include "regular/ros.h"
#include "lib/common.h"

#if defined(GPU_KERNEL)
#define DEVICE __device__
#define OBJECT SO
#define CONSTANT __constant__
#include "regular/data_type.h"
namespace FFGPU{
#elif defined(CPU_KERNEL)
#define DEVICE
#define OBJECT CO
#define CONSTANT
#include "regular/data_type.h"
namespace FFCPU{
#endif

typedef void (* reduce_fp)(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size);

typedef bool (* map_fp)(OBJECT *object, void *global_data, unsigned long long index, void *parameter, int device_type, reduce_fp reducee_ptr);

//Define the way of emitting key-value pairs
DEVICE bool map(OBJECT *object, void *global_data, unsigned long long index, void *parameter, int device_type, reduce_fp reduce_ptr)
{
	//int data_start = offset->offset;
        //printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~offset is: %d\n", index);
	float vaverh = *(float *)parameter;
	double *start = (double *)((char *)global_data + index);

        double dim1 = start[0];
        double dim2 = start[1];
        double dim3 = start[2];

	//int k1 = ANY_KEY();
	//int k2 = ANY_KEY();
	//int k3 = ANY_KEY();
	int k = ANY_KEY();

	float value[2];
	value[0] = sqrt(dim1*dim1 + dim2 * dim2 + dim3 * dim3);
	value[1] = 0;
	if(value[0]>vaverh)
		value[1] = 1;

       	//double v1 = dim1 * dim1; 
	//double v2 = dim2 * dim2;
	//double v3 = dim3 * dim3;

        object->insert(&k, sizeof(int), value, sizeof(value), reduce_ptr);
        //object->insert(&k2, sizeof(int), &v2, sizeof(v2));
        //object->insert(&k3, sizeof(int), &v3, sizeof(v3));

	return true;
}

DEVICE void reduce(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size)
{
        //double v1 = 12;//*((double *)value1);
        //double v2 = 12;//*((double *)value2);

	//double temp = v1 + v2;
	((float *)value1)[0] += ((float *)value2)[0];
	((float *)value1)[1] += ((float *)value2)[1];
        
        //copy_val(value1, &temp, sizeof(double));
}

DEVICE bool map1(OBJECT *object, void *global_data, unsigned long long index, void *parameter, int device_type, reduce_fp reduce_ptr)
{
	float *start = (float *)((char *)global_data + index);

        float dim1 = start[0];
        float dim2 = start[1];
        float dim3 = start[2];


        unsigned int key = 0;

        float min_dist = 65536*65;
        float dist;
        //The first K points are the cluster centers
        for(int i = 0; i<K; i++)
        {
                dist = 0;
                float cluster_dim1 = ((float *)parameter)[KMEANS_DIM*i];
                float cluster_dim2 = ((float *)parameter)[KMEANS_DIM*i+1];
                float cluster_dim3 = ((float *)parameter)[KMEANS_DIM*i+2];

                dist = (cluster_dim1-dim1)*(cluster_dim1-dim1)+(cluster_dim2-dim2)*(cluster_dim2-dim2)+(cluster_dim3-dim3)*(cluster_dim3-dim3);
                dist = sqrt(dist);
                if(dist < min_dist)
                {
                        min_dist = dist;
                        key = i;
                }
        }

	//if(key == 23)
        //printf("data point: %f %f %f\n", dim1, dim2, dim3);
	//if(device_type == 1)
        //printf("key is: %d\n", key);

        float value[5];
        value[0] = dim1;
        value[1] = dim2;
        value[2] = dim3;
        value[3] = 1;   //The last element of value records the number of one point, i.e., 1
        value[4] = min_dist;

	//if(key!=23)
        return object->insert(&key, sizeof(key), value, sizeof(value), reduce_ptr);
}

DEVICE void reduce1(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size)
{
        float num_points1 = ((float *)value1)[3];
        float num_points2 = ((float *)value2)[3];
        float dist1 = ((float *)value1)[4];
        float dist2 = ((float *)value2)[4];
        float total_points = num_points1+num_points2;
        float temp[5];
        temp[0] = ((float *)value1)[0] + ((float *)value2)[0];
        temp[1] = ((float *)value1)[1] + ((float *)value2)[1];
        temp[2] = ((float *)value1)[2] + ((float *)value2)[2];
        temp[3] = total_points;
        temp[4] = dist1+dist2;

        copy_val(value1, temp, sizeof(float)*5);
}

//Define the way of emitting key-value pairs
DEVICE bool map2(OBJECT *object, void *global_data, unsigned long long index, void *parameter, int device_type, reduce_fp reduce_ptr)
{
	//int data_start = offset->offset;
        //printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~offset is: %d\n", index);
	double *cutsqp = (double *)parameter;
	double cutsq = *cutsqp;
	double *v = cutsqp + 1; 

	int *start = (int *)((char *)global_data + index);

	int i = start[0];
	int j = start[1];

	double delx = v[i * 3] - v[j*3]; 
	double dely = v[i * 3 + 1] - v[j * 3 + 1]; 
	double delz = v[i * 3 + 2] - v[j*3 + 2]; 

	double rsq = delx * delx + dely*dely + delz*delz;
	//int k1 = ANY_KEY();
	//int k2 = ANY_KEY();
	//int k3 = ANY_KEY();
	int k = ANY_KEY();

	if(rsq < cutsq)
	{
		double sr2 = 1.0/rsq;	
		double sr6 = sr2*sr2*sr2;
		double phi = sr6 * (sr6 - 1.0);
		
        	object->insert(&k, sizeof(int), &phi, sizeof(double), reduce_ptr);
	}

	return true;
}

DEVICE void reduce2(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size)
{
        //double v1 = 12;//*((double *)value1);
        //double v2 = 12;//*((double *)value2);

	//double temp = v1 + v2;
	((double *)value1)[0] += ((double *)value2)[0];
}

//CONSTANT map_fp map_function_table[] = {map, map1};
//
//CONSTANT reduce_fp reduce_function_table[] = {reduce, reduce1};

CONSTANT map_fp map_function_table[] = {map, map1};

CONSTANT reduce_fp reduce_function_table[] = {reduce, reduce1};


}
#endif
