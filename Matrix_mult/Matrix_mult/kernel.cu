
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<iostream>
#include<chrono>
using namespace std;
using namespace std::chrono;

cudaError_t performNormalMatrixMultiplication();

__global__ void multiply(float *dev_a, float *dev_x, float *dev_b)
{
	int i = blockIdx.x;
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	dev_b[i] = dev_a[index] * dev_x[i]; //Row multiplication of matrix A with vector x
}

//Later init can be moved to GPU
void initArrays(float *a, float *x, float *b) 
{
	int index = 0;
	for (int i = 0; i < 32; i++) {
		x[i] = i*0.56;
		b[i] = 0.0;
		for (int j = 0; j < 32; j++) {
			a[index] = i * j * 0.045 * (index/89); //Generating a random number and storing in a[index]
			index++;
		}
		
	}
}

cudaError_t performNormalMatrixMultiplication()
{
	int size = 32;
	//Create Matrix Vectors
	float c[32];//To copy final result from device to host

	float *a = new float[1024]; //Total elements in one matrix 32 x 32
	float *x = new float[32]; //Vector to be multiplied
	float *b = new float[32]; //Resultant vector

	//For use on Device 
	float *dev_a, *dev_x, *dev_b;

	initArrays(a,x,b);
	cout << sizeof(*a);
	cout << sizeof(*b);
	cout << sizeof(*x);

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}



	// Allocate GPU buffers for three vectors (two input, one output)    
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, 1024 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_x, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, 1024 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	//To refer each element of the matrix we get 32 blocks with 32 threads
	int blocksize = 32;
	int gridsize = 32;
	printf("The gridsize is %d", gridsize);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	multiply <<<gridsize, blocksize >>>(a,x,b);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multiply launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy(c, &dev_b, 32 * sizeof(float), cudaMemcpyDeviceToHost);

	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_b);
	return cudaStatus;
}






int main()
{
	cudaError_t cudaStatus = performNormalMatrixMultiplication(); 



	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Matrix Multiply failed!");
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;

}

	
