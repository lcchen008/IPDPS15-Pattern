#include <iostream>
using namespace std;
#include <fstream>
#include <stdlib.h>
#define DIM 3
#define BSIZE 20000000 
#define GRIDSZ 1000

int main()
{
	ofstream output;
	char filename[] = "data.txt";
	output.open(filename);
	//size_t padding1 = DIM;
	//size_t padding2 = BSIZE;
	//output.write((char *)&padding1, sizeof(size_t));
	//output.write((char *)&padding2, sizeof(size_t));
	float tmp;

    srand(2006);

    for(size_t i = 0; i<BSIZE*DIM; i++)
	{
	    tmp = rand()%GRIDSZ;	
		//output.write((char *)&tmp, sizeof(float));
	    if(i%DIM==0)	    
	    {
		if(i!=0)
			output<<endl;

	   	output << (i/3 + 1);
         	output << " ";
	    }
		output << tmp <<" ";
	}
	output.close();
	return 0;
}
