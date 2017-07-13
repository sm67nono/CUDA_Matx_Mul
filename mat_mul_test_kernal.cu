#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

#define IMUL(a,b) __mul24(a,b)

cudaError_t performNormalMatrixMultiplication();
cudaError_t performJacobi();


/*__device__ __forceinline__ float multBandedMatrixVectorRow(
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




//Single GPU Kernel for Jacobi
/* template<bool addResult, typename Real>
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
} */


//Simple Jacobi iteration
__global__ void jacobi_Simple(const float *A0, const float *A1, const float *A2, const float *A3, const float *A4, int dim, float *x, const float *rhs)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	float result = rhs[index];

	//Get the boundaries

	int leftBoundaryElem = blockDim.x * blockIdx.x;

	int rightBoundaryElem = (dim - 1) + blockDim.x * blockIdx.x;

	int topBoundaryElem = threadIdx.x + blockDim.x * dim;

	int bottomBoundaryElem = threadIdx.x;

	//Carry out computations for boundary elements
	if (index == leftBoundaryElem && index == bottomBoundaryElem) // Bottom left Corner Element
	{
		
		//Top
		result -= A4[index] * x[index + dim];


		//Right 

		result -= A3[index] * x[index + 1];


		result /= A2[index];

		x[index] = result;

		return;
	}

	else if (index == rightBoundaryElem && index == bottomBoundaryElem) //Bottom Right Corner Element
	{

		//Top
		result -= A4[index] * x[index + dim];

		//Left
		result -= A1[index] * x[index - 1];


		result /= A2[index];

		x[index] = result;

		return;

	}
	else if (index == leftBoundaryElem && index == topBoundaryElem) //Top left Corner Element
	{
		//Bottom
		result -= A0[index] * x[index - dim];

		//Right 

		result -= A3[index] * x[index + 1];

		result /= A2[index];

		x[index] = result;

		return;

	}

	else if (index == leftBoundaryElem && index == topBoundaryElem) //Top Right Corner Element
	{
		//Bottom
		result -= A0[index] * x[index - dim];
		
		//Left
		result -= A1[index] * x[index - 1];

		result /= A2[index];

		x[index] = result;

		return;

	}

	


	else if (index == leftBoundaryElem)
	{
		//Bottom
		result -= A0[index] * x[index - dim];

		//Top
		result -= A4[index] * x[index + dim];


		//Right 

		result -= A3[index] * x[index + 1];

		result /= A2[index];
		
		x[index] = result;

		return;
	}

	else if (index == bottomBoundaryElem) {


		//Top
		result -= A4[index] * x[index + dim];

		//Left
		result -= A1[index] * x[index - 1];

		//Right 

		result -= A3[index] * x[index + 1];

		result /= A2[index];

		x[index] = result;

		return;


	}

	else if (index == rightBoundaryElem) {


		//Bottom
		result -= A0[index] * x[index - dim];

		//Top
		result -= A4[index] * x[index + dim];

		//Left
		result -= A1[index] * x[index - 1];


		result /= A2[index];

		x[index] = result;

		return;


	}


	else if (index == topBoundaryElem) {


		//Bottom
		result -= A0[index] * x[index - dim];


		//Left
		result -= A1[index] * x[index - 1];

		//Right 

		result -= A3[index] * x[index + 1];

		result /= A2[index];

		x[index] = result;

		return;


	}
		
	//For every other element not on the boundary
	else {
		//Bottom
		result -= A0[index] * x[index - dim];

		//Top
		result -= A4[index] * x[index + dim];

		//Left
		result -= A1[index] * x[index - 1];

		//Right 

		result -= A3[index] * x[index + 1];

		result /= A2[index];

		x[index] = result;

		return;
	}
	 
}



//Dense matrix multiplication Single GPU
__global__ void multiply(float *dev_a, float *dev_x, float *dev_b, int stride)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	//Row multiplication of matrix A with vector x
	for (int i = 0; i < stride; i++) {
		dev_b[index] = dev_a[i + (index * stride)] * dev_x[index];
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

//Init matrix Diagonals A0, A1, A2, A3, A4
void initDiag(float *A0, float *A1, float *A2, float *A3, float *A4, float *res, float * vec, int dim)
{
	//Not accounted for Obstacles

	for (unsigned int i = 0; i < dim; ++i)
	{
		for (unsigned int j = 0; j < dim; ++j)
		{
			int idx = j * dim + i;

			//Bottom
			if (i==0) {
				A0[idx] = 0.0f;
			}
			else{
				A0[idx] = 1.0f;
			}

			//Left 
			if(j==0)
			{
				A1[idx] = 0.0f;
			}
			else {
			
				A1[idx] = 1.0f;
			}
			

			//Right
			if (j == dim-1)
			{
				A3[idx] = 0.0f;
			}
			else {

				A3[idx] = 1.0f;
			}

			//Top
			if (i == dim - 1)
			{
				A4[idx] = 0.0f;
			}
			else {

				A4[idx] = 1.0f;
			}

			//Primary Diagonal 
			A2[idx] = 1.0f;

			//Result(RHS) and Vector init
			res[idx] = 0.0 + i * 0.15;
			vec[idx] = 0.0f;

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

cudaError_t performJacobi()
{


	//Fixed values to be changed later
	const int size = 1024;

	const int dim = 32;
	auto result = std::make_unique<float[]>(size);

	//Create Diagonal Vectors
	auto a0 = std::make_unique<float[]>(size);//To copy final result from device to host
	auto a1 = std::make_unique<float[]>(size); 
	auto a2 = std::make_unique<float[]>(size); 
	auto a3 = std::make_unique<float[]>(size); 
	auto a4 = std::make_unique<float[]>(size);
	auto d_Vec = std::make_unique<float[]>(size);
	auto d_Res = std::make_unique<float[]>(size);


	initDiag(a0.get(), a1.get(), a2.get(), a3.get(), a4.get(), d_Vec.get(), d_Res.get(), dim);


	//For use on Device 
	auto d_A0 = make_unique_cuda_array<float>(size);
	auto d_A1 = make_unique_cuda_array<float>(size);
	auto d_A2 = make_unique_cuda_array<float>(size);
	auto d_A3 = make_unique_cuda_array<float>(size);
	auto d_A4 = make_unique_cuda_array<float>(size);


	//cudamalloc the Diagonals
	cudaMalloc((void**)&d_A0, size * sizeof(float));
	cudaMalloc((void**)&d_A1, size * sizeof(float));
	cudaMalloc((void**)&d_A2, size * sizeof(float));
	cudaMalloc((void**)&d_A3, size * sizeof(float));
	cudaMalloc((void**)&d_A4, size * sizeof(float));
							
	//cudamalloc the Input Vector and Result vector
	cudaMalloc((void**)&d_Vec, size  * sizeof(float));
	cudaMalloc((void**)&d_Res, size  * sizeof(float));
							   

	cudaMemcpy(d_A0.get(), a0.get(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A1.get(), a1.get(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A2.get(), a2.get(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A3.get(), a3.get(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A4.get(), a4.get(), size * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_A4.get(), d_Vec.get(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A4.get(), d_Res.get(), size * sizeof(float), cudaMemcpyHostToDevice);


	//multMatrix(d_A0, d_A1, d_A2, d_A3, d_A4, myDim, d_vec, d_res);

	//Perform one Jacobi Step
	int blocksize = 32;
	int threads = 32;


	jacobi_Simple<<<blocksize,threads>>>(d_A0.get(), d_A1.get(), d_A2.get(), d_A3.get(), d_A4.get(), dim, d_Vec.get(), d_Res.get());


	cudaMemcpy(result.get(), d_Res.get(), size* sizeof(float), cudaMemcpyDeviceToHost);

	if (auto err = cudaGetLastError())
	{
		fprintf(stderr, "Jacobi launch failed: %s\n", cudaGetErrorString(err));
		return err;
	}


	cout << "One iteration successful";
	// Freeing memory auto done by cuda deleter

	/*cudaFree(d_A0.get());
	cudaFree(d_A1.get());
	cudaFree(d_A2.get());
	cudaFree(d_A3.get());
	cudaFree(d_A4.get());

	cudaFree(d_Vec.get());
	cudaFree(d_Res.get());*/

	return cudaSuccess;


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


	//To refer each element of the matrix we get 8 blocks with 128 threads
	int threads = 128;
	int blocksize = 8;
	printf("The BlockSize is %d \n", blocksize);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	multiply<<<blocksize, threads>>>(dev_a.get(), dev_x.get(), dev_b.get(), size);

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
	//cudaError_t cudaStatus = performNormalMatrixMultiplication();

	cudaError_t cudaStatus = performJacobi();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Computation failed! \n");
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed! \n");
		return 1;
	}

	return 0;

}
