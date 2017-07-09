
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<iostream>
#include<chrono>
using namespace std;
using namespace std::chrono;

cudaError_t addWithCuda(float *c,  unsigned int size);

__global__ void addKernel(float *c)
{
    int i = blockIdx.x;
	int index = threadIdx.x + blockDim.x * blockIdx.x;
    c[index] = index * 0.6573 * 0.09876 * 78.89;
}


int main()
{
    //const int arraySize = 650000;
	//Using to full Block size(65535) and max threads per block(1024)
	const unsigned int arraySize = 67107840;
	//Parallel generated array
	//Allocating huge array on Heap instead of stack
    float* c = new float[arraySize];

	//Serially Generated array
	//Allocating huge array on Heap instead of stack
	float* s = new float[arraySize];

    // Add vectors in parallel.

	//Measure performance for parallel add funtion
	
	
	cudaError_t cudaStatus = addWithCuda(c, arraySize);
	
   

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	//Print the randomly generated array
	//Serial duration for generating the same array
	
	high_resolution_clock::time_point t3 = high_resolution_clock::now();
	for (int i = 0; i < arraySize; i++) {
		s[i] = i * 0.6573 * 0.09876 * 78.89;
	}
	high_resolution_clock::time_point t4 = high_resolution_clock::now();

	auto duration2 = duration_cast<microseconds>(t4 - t3).count();

	cout << "Duration to execute the serial portion is " << duration2 << endl;





	/*for (int i = 0; i < arraySize; i++) {
		cout << c[i] << " ";
	}*/



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float *c,  unsigned int size)
{
    
    float *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	
   
    // Launch a kernel on the GPU with one block for each element.
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	//Dividing the grid for 1000000
	//addKernel<<<65535,1024>>>(dev_c); 
	int blocksize = 1024;
	int gridsize = size / blocksize;
	printf("The gridsize is %d", gridsize);
	addKernel << <gridsize, blocksize >> >(dev_c);
	
	
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
	
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
	
	//After everything is syncronized

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(t2 - t1).count();

	cout << "Duration to execute the parallel portion is " << duration << endl;


    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);

	printf("No of elements in output array is %d", sizeof(c));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);    
    return cudaStatus;
}
