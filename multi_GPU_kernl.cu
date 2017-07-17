#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <memory>
using namespace std;
using namespace std::chrono;

#define IMUL(a,b) __mul24(a,b)


cudaError_t performJacobi();


struct cuda_deleter
{
	void operator() (void * p) { cudaFree(p); }
};

template<typename T>
auto make_unique_cuda_array(std::size_t size)
{
	T * p = nullptr;
	if (auto err = cudaMalloc((void**)&p, size * sizeof(T)))
		throw std::bad_alloc();
	return std::unique_ptr<T[], cuda_deleter>(p);
}


//Simple Jacobi iteration
__global__ void jacobi_Simple(const float *A0, const float *A1, const float *A2, const float *A3, const float *A4, float *x_in, float *x_out, const float *rhs)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	float result = rhs[index];

	int dim = blockDim.x;
	int x_pos = blockIdx.x;
	int y_pos = threadIdx.x;

	//Get the boundaries

	int leftBoundaryElem = x_pos * (dim);

	int rightBoundaryElem = (x_pos * dim) +(dim-1);

	int topBoundaryElem = y_pos + (dim * (dim-1));

	int bottomBoundaryElem = y_pos;

	//Carry out computations for boundary elements
	if ((index == leftBoundaryElem) && (index == bottomBoundaryElem)) // Bottom left Corner Element
	{

		//Top
		result -= A4[index] * x_in[index + dim];


		//Right 

		result -= A3[index] * x_in[index + 1];


		result /= A2[index];

		x_out[index] = result;

		return;
	}

	else if ((index == rightBoundaryElem) && (index == bottomBoundaryElem)) //Bottom Right Corner Element
	{

		//Top
		result -= A4[index] * x_in[index + dim];

		//Left
		result -= A1[index] * x_in[index - 1];


		result /= A2[index];

		x_out[index] = result;

		return;

	}
	else if ((index == leftBoundaryElem) && (index == topBoundaryElem)) //Top left Corner Element
	{
		//Bottom
		result -= A0[index] * x_in[index - dim];

		//Right 

		result -= A3[index] * x_in[index + 1];

		result /= A2[index];

		x_out[index] = result;

		return;

	}

	else if ((index == rightBoundaryElem) && (index == topBoundaryElem)) //Top Right Corner Element
	{
		//Bottom
		result -= A0[index] * x_in[index - dim];

		//Left
		result -= A1[index] * x_in[index - 1];

		result /= A2[index];

		x_out[index] = result;

		return;

	}




	else if (index == leftBoundaryElem)
	{
		//Bottom
		result -= A0[index] * x_in[index - dim];

		//Top
		result -= A4[index] * x_in[index + dim];


		//Right 

		result -= A3[index] * x_in[index + 1];

		result /= A2[index];

		x_out[index] = result;

		return;
	}

	else if (index == bottomBoundaryElem) {


		//Top
		result -= A4[index] * x_in[index + dim];

		//Left
		result -= A1[index] * x_in[index - 1];

		//Right 

		result -= A3[index] * x_in[index + 1];

		result /= A2[index];

		x_out[index] = result;

		return;


	}

	else if (index == rightBoundaryElem) {


		//Bottom
		result -= A0[index] * x_in[index - dim];

		//Top
		result -= A4[index] * x_in[index + dim];

		//Left
		result -= A1[index] * x_in[index - 1];


		result /= A2[index];

		x_out[index] = result;

		return;


	}


	else if (index == topBoundaryElem) {


		//Bottom
		result -= A0[index] * x_in[index - dim];


		//Left
		result -= A1[index] * x_in[index - 1];

		//Right 

		result -= A3[index] * x_in[index + 1];

		result /= A2[index];

		x_out[index] = result;

		return;


	}

	//For every other element not on the boundary
	else { 
		//Bottom
		result -= A0[index] * x_in[index - dim];

		//Top
		result -= A4[index] * x_in[index + dim]; 

		//Left
		result -= A1[index] * x_in[index - 1];

		//Right 

		result -= A3[index] * x_in[index + 1];

		result /= A2[index];

		x_out[index] = result;

		return;
	}

}



//Init matrix Diagonals A0, A1, A2, A3, A4
void initDiag(float *A0, float *A1, float *A2, float *A3, float *A4, float *res, float * vec_in, float * vec_out, int dim)
{
	//Not accounted for Obstacles

	for (int i = 0; i < dim; ++i)
	{
		for (int j = 0; j < dim; ++j)
		{
			int idx = j + dim * i;

			//Bottom
			if (i == 0) {
				A0[idx] = 0.0f;
			}
			else {
				A0[idx] = 1.0f;
			}

			//Left 
			if (j == 0)
			{
				A1[idx] = 0.0f;
			}
			else {

				A1[idx] = 1.0f;
			}


			//Right
			if (j == dim - 1)
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
			res[idx] = 1.0f;
			vec_in[idx] = j+i;
			vec_out[idx] = 0.0f;


		}
	}

}




cudaError_t performJacobi()
{


	//Fixed values to be changed later

	const int dim = 16;

	const int size = dim * dim;

	auto result = std::make_unique<float[]>(size);

	//Create Diagonal Vectors
	auto a0 = std::make_unique<float[]>(size);
	auto a1 = std::make_unique<float[]>(size);
	auto a2 = std::make_unique<float[]>(size);
	auto a3 = std::make_unique<float[]>(size);
	auto a4 = std::make_unique<float[]>(size);
	auto vec_in = std::make_unique<float[]>(size);
	auto res = std::make_unique<float[]>(size);
	auto vec_out = std::make_unique<float[]>(size);


	initDiag(a0.get(), a1.get(), a2.get(), a3.get(), a4.get(), res.get(), vec_in.get(), vec_out.get(), dim);

	cout << "A0             ....";
	for (int i = 0; i < size; i++) {
		cout << a0[i] << " ";
	}
	cout << endl;

	cout << "A1             ....";
	for (int i = 0; i < size; i++) {
		cout << a1[i] << " ";
	}
	cout << endl;
	cout << "A2             ....";
	for (int i = 0; i < size; i++) {
		cout << a2[i] << " ";
	}
	cout << endl;
	cout << "A3             ....";
	for (int i = 0; i < size; i++) {
		cout << a3[i] << " ";
	}
	cout << endl;
	cout << "A4             ....";
	for (int i = 0; i < size; i++) {
		cout << a4[i] << " ";
	}
	cout << endl;

	cout << "RHS             ....";
	for (int i = 0; i < size; i++) {
		cout << res[i] << " ";
	}
	cout << endl;

	cout << "Vec In            ....";
	for (int i = 0; i < size; i++) {
		cout << vec_in[i] << " ";
	}
	cout << endl;



	//For use on Device 
	auto d_A0 = make_unique_cuda_array<float>(size);
	auto d_A1 = make_unique_cuda_array<float>(size);
	auto d_A2 = make_unique_cuda_array<float>(size);
	auto d_A3 = make_unique_cuda_array<float>(size);
	auto d_A4 = make_unique_cuda_array<float>(size);
	auto d_Vec_In = make_unique_cuda_array<float>(size);
	auto d_Res = make_unique_cuda_array<float>(size);
	auto d_Vec_Out = make_unique_cuda_array<float>(size);

	if (auto err = cudaGetLastError())
	{
		fprintf(stderr, "Jacobi launch failed: %s\n", cudaGetErrorString(err));
		return err;
	}

	//cudamalloc the Diagonals
	cudaMalloc((void**)&d_A0, size * sizeof(float));
	cudaMalloc((void**)&d_A1, size * sizeof(float));
	cudaMalloc((void**)&d_A2, size * sizeof(float));
	cudaMalloc((void**)&d_A3, size * sizeof(float));
	cudaMalloc((void**)&d_A4, size * sizeof(float));

	//cudamalloc the Input Vector and Result vector
	cudaMalloc((void**)&d_Vec_In, size * sizeof(float));
	cudaMalloc((void**)&d_Vec_Out, size * sizeof(float));
	cudaMalloc((void**)&d_Res, size * sizeof(float));

	if (auto err = cudaGetLastError())
	{
		fprintf(stderr, "Jacobi launch failed: %s\n", cudaGetErrorString(err));
		return err;
	}


	cudaMemcpy(d_A0.get(), a0.get(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A1.get(), a1.get(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A2.get(), a2.get(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A3.get(), a3.get(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A4.get(), a4.get(), size * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_Vec_In.get(), vec_in.get(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vec_Out.get(), vec_out.get(), size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Res.get(), res.get(), size * sizeof(float), cudaMemcpyHostToDevice);

	if (auto err = cudaGetLastError())
	{
		fprintf(stderr, "Jacobi launch failed: %s\n", cudaGetErrorString(err));
		return err;
	}

	//multMatrix(d_A0, d_A1, d_A2, d_A3, d_A4, myDim, d_vec, d_res);

	//Perform one Jacobi Step
	int blocksize = dim;
	int threads = dim;

	int iterations = 20;
	for(int i=0;i<iterations;i++)
	{ 
		jacobi_Simple <<<blocksize, threads >>>(d_A0.get(), d_A1.get(), d_A2.get(), d_A3.get(), d_A4.get(), d_Vec_In.get(), d_Vec_Out.get(), d_Res.get());
		cudaMemcpy(result.get(), d_Vec_Out.get(), size * sizeof(float), cudaMemcpyDeviceToHost);
		
	}
	if (auto err = cudaGetLastError())
	{
		fprintf(stderr, "Jacobi launch failed: %s\n", cudaGetErrorString(err));
		return err;
	}

	cudaMemcpy(result.get(), d_Vec_Out.get(), size * sizeof(float), cudaMemcpyDeviceToHost);

	if (auto err = cudaGetLastError())
	{
		fprintf(stderr, "Jacobi launch failed: %s\n", cudaGetErrorString(err));
		return err;
	}


	cout << "One iteration successful";

	//Print result
	for (int i= size-1; i >=0; i--) {


		if ((i+1) % dim == 0) { cout << endl; }

		cout << vec_in[i] << " ";
	}

	cout << endl << endl;
	for (int i = size-1; i >=0; i--) {


		if ((i + 1) % dim == 0) { cout << endl; }

		cout << result[i] << " ";
	}
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


int main()
{


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
