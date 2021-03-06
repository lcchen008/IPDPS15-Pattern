#include <iostream>
#include <fstream>
using namespace std;
#define NUM_PROCS 2 
#include "kmeans.h"
#include "lib/time_util.h"
#define GRIDSZ 1000
#include "regular/regular_runtime.h"
#include "include/runtime.h"

struct kmeans_value
{
    float dim0;
    float dim1;
    float dim2;
    float num;
    float dist;
};

int main(int argc, char *argv[])
{
    	//ifstream input;
    	//char filename[] = "data";
    	//The first K points are cluster centers
    	float *points = (float *)malloc(sizeof(float)*KMEANS_DIM*BSIZE);
	float *centers = (float *)malloc(sizeof(float)*KMEANS_DIM*K);

    	srand(2006);
    	//Generate cluster centers
    	for(int i = 0; i<K; i++)
    	{
    	    for(int j = 0; j<KMEANS_DIM; j++)
    	    {
    	        centers[i*KMEANS_DIM+j]= rand()%GRIDSZ;
    	    }
    	}

    	//Every point has a offset relative to the start of the points data
    	Offset *offsets = (Offset *)malloc(sizeof(Offset)*BSIZE);

    	//input.open(filename);
    	//size_t exhaust;
    	//input.read((char *)&exhaust, sizeof(size_t));
    	//cout<<exhaust<<endl;
    	//input.read((char *)&exhaust, sizeof(size_t));
    	//cout<<exhaust<<endl;

    	//cout<<"The first two integers are exhausted..."<<endl<<endl;

    	//float tool;
    	 
    	cout<<"The BSIZE is: "<<BSIZE<<endl;
    	cout<<"Loading data... "<<endl;
    	double beforeload = rtclock();
    	for(size_t i = 0; i< BSIZE*KMEANS_DIM; i++)
    	{
    	    //input.read((char *)&tool, sizeof(float));
    	    points[i] = rand()%GRIDSZ;

    	    if(i%KMEANS_DIM==0)
	    {
    	    	offsets[i/KMEANS_DIM].offset = i*sizeof(float); //The DIM*K is the number of floats used in cluster centers
		offsets[i/KMEANS_DIM].size = sizeof(float) * KMEANS_DIM;
	    }
    	}

    	//input.close();
    	double afterload = rtclock();

    	cout<<"Load done..."<<endl;
    	cout<<"Load time: "<<(afterload-beforeload)<<endl<<endl;
    	cout<<"Doing mapreduce..."<<endl;

	RuntimeInit(argc, argv);	

	RegularRuntime rr(NUM_PROCS, points, sizeof(float)*KMEANS_DIM*BSIZE, offsets, BSIZE, centers, KMEANS_DIM*sizeof(float)*K);
	
	rr.RegularInit();
	rr.set_map_idx(1);
	rr.set_reduce_idx(1);

	cout<<"init done..."<<endl;

	rr.RegularStart();

	cout<<"after start..."<<endl;
}

