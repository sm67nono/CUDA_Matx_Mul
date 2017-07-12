#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

#define IMUL(a,b) __mul24(a,b)

cudaError_t performNormalMatrixMultiplication();


__device__ __forceinline__ float multBandedMatrixVectorRow(
	const float *A0, const float *A1, const float *A2, const float *A3, const float *A4, const float *x,
	int idx, int2 pos, int2 dim)
{
	float res = 0.f;


	res += A2[idx] * x[idx];

	// left, right
	if (pos.x > 0)
		res += A1[idx] * x[idx - 1];
	if (pos.x < dim.x - 1)
		res += A3[idx] * x[idx + 1];

	// up, down
	if (pos.y > 0)
		res += A0[idx] * x[idx - dim.x];
	if (pos.y < dim.y - 1)
		res += A4[idx] * x[idx + dim.x];
	//#endif
	return res;
}



__global__ void multiply(float *dev_a, float *dev_x, float *dev_b, int stride)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	//Row multiplication of matrix A with vector x
	for (int i = 0; i < stride; i++) {
		dev_b[index] = dev_a[i + (index * stride)] * dev_x[index];
	}
}

template<bool addResult, typename Real>
__global__ void multMatrix_kernel(const float *A0, const float *A1, const float *A2, const float *A3, const float *A4, int2 dim, const Real * x, Real * y)
{
	int2 pos;

	pos.x = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	pos.y = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;

	int idx = pos.x + IMUL(pos.y, dim.x);
	//int tdx = IMUL(threadIdx.y, blockDim.x) + threadIdx.x;

	if (pos.x > 0 && pos.x < dim.x - 1 && pos.y > 0 && pos.y < dim.y - 1)
	{
		Real sum = multBandedMatrixVectorRow(A0, A1, A2, A3, A4, x, idx, pos, dim);
		if (addResult)
		{
			y[idx] += sum;
		}
		else
		{
			y[idx] = sum;
		}
	}
}

//Later init can be moved to GPU
void initArrays(float *a, float *x, float *b, int size)
{
	int index = 0;
	for (int i = 0; i < size; i++) {
		x[i] = i*0.56f;
		b[i] = 0.0f;
		for (int j = 0; j < size; j++) {
			a[index] = i * j * 0.045f * (index / 89); //Generating a random number and storing in a[index]
			index++;
		}

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
	const int size = 1024;
	//Create Matrix Vectors
	auto c = std::make_unique<float[]>(size);//To copy final result from device to host
	auto a = std::make_unique<float[]>(size * size); //Total elements in one matrix 32 x 32
	auto x = std::make_unique<float[]>(size); //Vector to be multiplied
	auto b = std::make_unique<float[]>(size); //Resultant vector

	

	initArrays(a.get(), x.get(), b.get(), size);

	//testInitilization(a, x, b);

	// Choose which GPU to run on, change this on a multi-GPU system.
	if(auto err = cudaSetDevice(0))
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return err;
	}

	//For use on Device 
	auto dev_a = make_unique_cuda_array<float>(size*size);
	auto dev_x = make_unique_cuda_array<float>(size);
	auto dev_b = make_unique_cuda_array<float>(size);

	// Copy input vectors from host memory to GPU buffers.
	cout << "memcopy 1 \n";
	if(auto err = cudaMemcpy(dev_a.get(), a.get(), size * size * sizeof(float), cudaMemcpyHostToDevice))
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
	int threads = 128;
	int gridsize = 8;
	printf("The gridsize is %d \n", gridsize);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	multiply<<<gridsize, threads>>>(dev_a.get(), dev_x.get(), dev_b.get(), size);

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
