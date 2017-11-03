#include "cu_util.h"

int GetGPUNumber()
{
        int count;

        cudaGetDeviceCount(&count);
	//    printf("Number of devices is: %d\n", count);

        //if(count==0)
        //{
        //        fprintf(stderr, "There is no device.\n");
        //        return 0;
        //}

	//int counter = 0;

        //int i;
        //for(i = 0; i<count; i++)
        //{
        //        cudaDeviceProp prop;
        //        if(cudaGetDeviceProperties(&prop, i)==cudaSuccess)
        //        {
        //                if(prop.major>=2)
        //                {
        //                    //check whether support zero-copy
        //                    bool whe = prop.unifiedAddressing;

        //                    if(whe)
        //                    {
	//			//printf("%d supports unified addressing\n");
	//			counter++;
        //                    }

        //                    //break;
        //                }
        //        }
	//}       

    	//cudaSetDevice(0);
	//return 1;
     	return count;      
}

void checkCUDAError(const char *msg)
{
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err)
        {
                fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
                exit(EXIT_FAILURE);
        }
}

