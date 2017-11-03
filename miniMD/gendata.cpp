#include <iostream>
using namespace std;
#include <fstream>
#include <stdlib.h>
#define DIM 3
#define BSIZE 200000000 
#define GRIDSZ 1000

int main()
{
	ofstream output;
	char filename[] = "data";
	output.open(filename);
	size_t padding1 = DIM;
	size_t padding2 = BSIZE;
	output.write((char *)&padding1, sizeof(size_t));
	output.write((char *)&padding2, sizeof(size_t));
	float tmp;

    srand(2006);

    for(size_t i = 0; i<BSIZE*DIM; i++)
	{
	    tmp = rand()%GRIDSZ;	
		output.write((char *)&tmp, sizeof(float));
	}
	output.close();
	return 0;
}
