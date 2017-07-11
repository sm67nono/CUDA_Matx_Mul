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
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	dev_b[index] = dev_a[index] * dev_x[index]; //Row multiplication of matrix A with vector x
}

//Later init can be moved to GPU
void initArrays(float *a, float *x, float *b)
{
	int index = 0;
	for (int i = 0; i < 32; i++) {
		x[i] = i*0.56f;
		b[i] = 0.0f;
		for (int j = 0; j < 32; j++) {
			a[index] = i * j * 0.045f * (index / 89); //Generating a random number and storing in a[index]
			index++;
		}

	}
}

void testInitilization(float *a, float *x, float *b)
{
	int index = 0;
	for (int i = 0; i < 32; i++) {
		//cout<< x[i] <<"  ";
		//cout << b[i] << "  ";
		for (int j = 0; j < 32; j++) {
			//cout<<	a[index] << "  ";
			index++;
		}
		cout << endl;

	}
}
#include <memory>

struct cuda_deleter
{
	void operator() (void * p) { cudaFree(p); }
};

template<typename T>
auto make_unique_cuda_array(std::size_t size)
{
	T * p = nullptr;
	if(auto err = cudaMalloc((void**)&p, size * sizeof(T)))
		throw std::bad_alloc();
	return std::unique_ptr<T[], cuda_deleter>(p);
}

cudaError_t performNormalMatrixMultiplication()
{
	const int size = 32;
	//Create Matrix Vectors
	auto c = std::make_unique<float[]>(size);//To copy final result from device to host
	auto a = std::make_unique<float[]>(size * size); //Total elements in one matrix 32 x 32
	auto x = std::make_unique<float[]>(size); //Vector to be multiplied
	auto b = std::make_unique<float[]>(size); //Resultant vector

	initArrays(a.get(), x.get(), b.get());

	//testInitilization(a, x, b);

	// Choose which GPU to run on, change this on a multi-GPU system.
	if(auto err = cudaSetDevice(0))
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return err;
	}

	//For use on Device 
	auto dev_a = make_unique_cuda_array<float>(size);
	auto dev_x = make_unique_cuda_array<float>(size);
	auto dev_b = make_unique_cuda_array<float>(size);

	// Copy input vectors from host memory to GPU buffers.
	cout << "memcopy 1 \n";
	if(auto err = cudaMemcpy(dev_a.get(), a.get(), size * sizeof(float), cudaMemcpyHostToDevice))
	{
		fprintf(stderr, "cudaMemcpy failed!");
		return err;
	}

	cout << "memcopy 2 \n";
	if(auto err = cudaMemcpy(dev_x.get(), x.get(), size * sizeof(float), cudaMemcpyHostToDevice))
	{
		fprintf(stderr, "cudaMemcpy failed!");
		return err;
	}


	//To refer each element of the matrix we get 32 blocks with 32 threads
	int threads = 32;
	int gridsize = 1;
	printf("The gridsize is %d \n", gridsize);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	multiply<<<gridsize, threads>>>(dev_a.get(), dev_x.get(), dev_b.get());

	// Check for any errors launching the kernel
	if(auto err = cudaGetLastError())
	{
		fprintf(stderr, "multiply launch failed: %s\n", cudaGetErrorString(err));
		return err;
	}


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	if(auto err = cudaDeviceSynchronize())
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
		return err;
	}

	//After everything is syncronized

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(t2 - t1).count();

	cout << "Duration to execute the parallel portion is \n" << duration << endl;


	// Copy output vector from GPU buffer to host memory.
	cout << "memcopy 3 \n";
	/*cudaStatus = cudaMemcpy(c, dev_b, 32 * sizeof(float), cudaMemcpyDeviceToHost);



	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! \n");
		goto Error;
	}*/

	cout << "all memcopy succeed \n";

	return cudaSuccess;
}

int main()
{
	cudaError_t cudaStatus = performNormalMatrixMultiplication();



	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Matrix Multiply failed! \n");
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed! \n");
		return 1;
	}

	return 0;

}
