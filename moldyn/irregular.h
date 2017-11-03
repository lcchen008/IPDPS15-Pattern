#ifndef IRREGULAR_H_ 
#define IRREGULAR_H_
#include "irregular/parameters.h"
#include <math.h>
#include "irregular/ros.h"
#include "stdio.h"
#include "moldyn.h"

#if defined(GPU_KERNEL)
#define DEVICE __device__
#define OBJECT Sobject
#define CONSTANT __constant__
namespace FGPU
{
#elif defined(CPU_KERNEL)
#define DEVICE
#define OBJECT Cobject
#define CONSTANT
namespace FCPU
{
#endif
	typedef void (* irreduce_fp)(VALUE *value1, VALUE *value2);

	typedef bool (* irmap_fp)(OBJECT *object, void *point_data, int *part_index, void *edge_data, void *parameter, unsigned int task_id, unsigned int part_id, irreduce_fp reduce_ptr);

	DEVICE bool map(OBJECT *object, void *point_data, int *part_index, void *edge_data, void *parameter, unsigned int task_id, unsigned int part_id, irreduce_fp reduce_ptr)
	{
		struct para *pa = (struct para *)parameter;

		struct point_data *input = (struct point_data *)point_data;
		int *edges = (int *)edge_data;

		double cutoffRadius = pa->cutoffRadius;

		double side = pa->side;

		double sideHalf = 0.5 * side;
		
		double cutoffSquare = cutoffRadius * cutoffRadius; 
		//double vir = 0.0;
		//double epot = 0.0;
		double xx,yy,zz;
		double rd;
		double rrd, rrd2, rrd3, rrd4, rrd6, rrd7, r148;
		VALUE forces;

		int i, j;
		i = (edges)[task_id * 2];	
		j = (edges)[task_id * 2 + 1];

		xx = (input->points)[3 * i] - (input->points)[3 * j];
		yy = (input->points)[3 * i + 1] - (input->points)[3 * j + 1];
		zz = (input->points)[3 * i + 2] - (input->points)[3 * j + 2];

		if(xx < -sideHalf) xx += side;
		if(yy < -sideHalf) yy += side;
		if(zz < -sideHalf) zz += side;
		if(xx > sideHalf) xx -= side;
		if(yy > sideHalf) yy -= side;
		if(zz > sideHalf) zz -= side;

		rd = (xx*xx + yy*yy + zz*zz);

		if(rd < cutoffSquare)
		{
			rrd   = 1.0/rd;
	 		rrd2  = rrd*rrd ;
	 		//rrd3  = rrd2*rrd ;
	 		rrd4  = rrd2*rrd2 ;
	 		rrd6  = rrd2*rrd4;
	 		rrd7  = rrd6*rrd ;
	 		r148  = rrd7 - 0.5 * rrd4 ;
			
			if(part_index[i] == part_id)
			{
	 			forces.x = xx*r148;
	 			forces.y = yy*r148;
	 			forces.z = zz*r148;

				object -> insert(&i, &forces, reduce_ptr);
				//printf("inserted....\n");
			}

			if(part_index[j] == part_id)
			{
				forces.x = -forces.x;
				forces.y = -forces.y;
				forces.z = -forces.z;

				object -> insert(&j, &forces, reduce_ptr);
				//printf("inserted....\n");
			}

	 	//	fx(i)  += forcex ;
	 	//	fy(i)  += forcey ;
	 	//	fz(i)  += forcez ;

	 	//	fx(j)  -= forcex ;
	 	//	fy(j)  -= forcey ;
	 	//	fz(j)  -= forcez ;

	 		//vir  -= rd*r148 ;
	 		//epot += (rrd6 - rrd3);
		}
			
		return true;
	}
	
	DEVICE void reduce(VALUE *value1, VALUE *value2)
	{
		value1->x += value2->x;
		value1->y += value2->y;
		value1->z += value2->z;
	}

	CONSTANT irmap_fp map_function_table[] = {map};

	CONSTANT irreduce_fp reduce_function_table[] = {reduce};
}

#endif
