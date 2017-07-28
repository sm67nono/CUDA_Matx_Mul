#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <memory>
using namespace std;
using namespace std::chrono;

#define IMUL(a,b) __mul24(a,b)


cudaError_t performMultiGPUJacobi();


/*struct cuda_deleter
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
}*/


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

	int rightBoundaryElem = (x_pos * dim) + (dim - 1);

	int topBoundaryElem = y_pos + (dim * (dim - 1));

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
void initDiag(float *A0, float *A1, float *A2, float *A3, float *A4, float *rhs, float * vec_in, float * vec_out, int dim, float *halos, int numDevices)
{


	int halocounter = 0;
	int haloOffset = 0;
	int haloDivision = dim / numDevices; // In 1D

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
			A2[idx] = 1.0f; // sum of A0, A1, A3, A4 ... except all 0.0... 

							//Result(RHS) and Vector_In
			rhs[idx] = 1.0f;
			vec_in[idx] = 1.0f;
			vec_out[idx] = 0.0f;





			//Initial value of Halos should be same as vec_in
			//For GPU 1 the halo should be top row(North) of domain in GPU 2
			if(i==haloOffset){
				halos[halocounter] = halocounter; //vec_in[idx]
				halocounter++;
				
			}
			//For GPU 2 the halo should be bottom row(South) of domain in GPU 1
			if (i == (haloOffset - 1))
			{
				halos[halocounter] = halocounter; //vec_in[idx]
				halocounter++;
				
			}
			


		}
		haloOffset += haloDivision;
	}

	// (i1, j1) = dimX / 4, dimY / 2
	// (i2, j2) = 3 * dimX / 4, dimY / 2
	// rhs[i1, j1] = +5.0
	// rhs[i2, j2] = -5.0

}




cudaError_t performMultiGPUJacobi()
{


	//Fixed values to be changed later

	int dim = 8;

	//TODO: write a 2D domain decomposition method for more than 2 GPUs
	int size = dim * dim;

	auto result = std::make_unique<float[]>(size);

	//Create Diagonal Vectors
	auto a0 = std::make_unique<float[]>(size);
	auto a1 = std::make_unique<float[]>(size);
	auto a2 = std::make_unique<float[]>(size);
	auto a3 = std::make_unique<float[]>(size);
	auto a4 = std::make_unique<float[]>(size);
	auto vec_in = std::make_unique<float[]>(size);
	auto rhs = std::make_unique<float[]>(size);
	auto vec_out = std::make_unique<float[]>(size);


	//Get the total number of devices
	int numDevices;
	cudaGetDeviceCount(&numDevices);


	//For both the GPUs - One halo per device(for 2 GPUs down and up halos are needed) : TODO: division when exchanging with more than 2 GPUs
	//For 4 GPUs up, down, left right would be needed (1*dim  must be changed to 4 * dim)
	//1*dim*numDevices calculates total storage space needed for Halos 1:-Total halos needed per device. dim:-number of elements in the dimension
	int haloStorage = 1 * dim * numDevices;
	auto halos = std::make_unique<float[]>(haloStorage);


	initDiag(a0.get(), a1.get(), a2.get(), a3.get(), a4.get(), rhs.get(), vec_in.get(), vec_out.get(), dim, halos.get(),numDevices);

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
		cout << rhs[i] << " ";
	}
	cout << endl;

	cout << "Halos             ....";
	for (int i = 0; i < haloStorage; i++) {
		cout << halos[i] << " ";
	}
	cout << endl;

	cout << "Vec In            ...." << endl;

	for (int i = size - 1; i >= 0; i--) {


		if ((i + 1) % dim == 0) { cout << endl; }

		cout << vec_in[i] << " ";
	}
	cout << endl;





	/*  auto d_A0 = make_unique_cuda_array<float>(size);
	auto d_A1 = make_unique_cuda_array<float>(size);
	auto d_A2 = make_unique_cuda_array<float>(size);
	auto d_A3 = make_unique_cuda_array<float>(size);
	auto d_A4 = make_unique_cuda_array<float>(size);
	auto d_Vec_In = make_unique_cuda_array<float>(size);
	auto d_Rhs = make_unique_cuda_array<float>(size);
	auto d_Vec_Out = make_unique_cuda_array<float>(size); */

	
	cout << endl << "The total numeber of Devices: " << numDevices;
	//Allocate memory on the devices

	//Let the total number of GPU be 2 : has to be changed later
	//Computation divided into (size/2) on first and size-(size/2) on second
	int *domainDivision;
	domainDivision = new int[numDevices]; 



	//Logic for total chunk per device (Domain distribution)
	for (int i = 0; i < numDevices; i++) {
		//if(!(i==numDevices-1)){
			domainDivision[i] = size / numDevices;
			//size = (size - size / numDevices);
		//}
	}


	//For use on Device 
	float *d_A0[2], *d_A1[2], *d_A2[2], *d_A3[2], *d_A4[2], *d_Vec_In[2], *d_Vec_Out[2], *d_Rhs[2], *d_halos[2];

	/* The domain division is done in 1D rowise */
	for (int dev = 0; dev<numDevices; dev++)
	{
		//Setting the device before allocation
		cudaSetDevice(dev);

		//cudamalloc the Diagonals
		cudaMalloc((void**)&d_A0[dev], domainDivision[dev] * sizeof(float));
		cudaMalloc((void**)&d_A1[dev], domainDivision[dev] * sizeof(float));
		cudaMalloc((void**)&d_A2[dev], domainDivision[dev] * sizeof(float));
		cudaMalloc((void**)&d_A3[dev], domainDivision[dev] * sizeof(float));
		cudaMalloc((void**)&d_A4[dev], domainDivision[dev] * sizeof(float));

		//cudamalloc the Input Vector and Result vector
		cudaMalloc((void**)&d_Vec_In[dev], domainDivision[dev] * sizeof(float));
		cudaMalloc((void**)&d_Vec_Out[dev], domainDivision[dev] * sizeof(float));
		cudaMalloc((void**)&d_Rhs[dev], domainDivision[dev] * sizeof(float));

		//cudaMalloc Halos
		cudaMalloc((void**)&d_halos[dev], dim * sizeof(float));
	}




	/* The transfer of Data from Host to Device */

	for (int dev = 0, pos = 0, haloPos =0; dev<numDevices; pos += domainDivision[dev], haloPos+=dim, dev++)
	{
		//Setting the device before allocation
		cudaSetDevice(dev);

		//Copy the diagonals from host to device
		cudaMemcpy(d_A0[dev], a0.get() + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_A1[dev], a1.get() + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_A2[dev], a2.get() + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_A3[dev], a3.get() + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_A4[dev], a4.get() + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);

		//Copy in and out vectors and RHS
		cudaMemcpy(d_Vec_In[dev], vec_in.get() + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Vec_Out[dev], vec_out.get() + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Rhs[dev], rhs.get() + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);

		//Copy intial Halos : TODO more than 2 devices
		cudaMemcpy(d_halos[dev], halos.get() + haloPos, dim * sizeof(float), cudaMemcpyHostToDevice);
	}

	if (auto err = cudaGetLastError())
	{
		fprintf(stderr, "Jacobi launch failed: %s\n", cudaGetErrorString(err));
		return err;
	}



	if (auto err = cudaGetLastError())
	{
		fprintf(stderr, "Jacobi launch failed: %s\n", cudaGetErrorString(err));
		return err;
	}




	if (auto err = cudaGetLastError())
	{
		fprintf(stderr, "Jacobi launch failed: %s\n", cudaGetErrorString(err));
		return err;
	}

	//multMatrix(d_A0, d_A1, d_A2, d_A3, d_A4, myDim, d_vec, d_res);

	//Perform one Jacobi Step
	int blocksize = dim / numDevices; //TODO: make it to more than 2 GPUs
	int threads = dim;

	//Call to kernal
	int iterations = 4;
	for (int i = 0; i<iterations; i++)
	{
		cout << endl << endl << "Iteration : " << i + 1 << endl << endl << endl;
		for (int dev = 0, pos = 0; dev<numDevices; pos += domainDivision[dev], dev++)
		{
			cout << endl << "Kernal Execution on GPU : " << dev;
			cout << endl << "Position :" << pos;
			cudaSetDevice(dev);

			cout << endl << "Check Intermediate Result before it gets passed to kernal" << endl;

			cudaMemcpy(result.get() + pos, d_Vec_In[dev], domainDivision[dev] * sizeof(float), cudaMemcpyDeviceToHost);

			/*for (int i = domainDivision[dev]+pos-1; i>=0; i--) {


			if ((i + 1) % dim == 0) { cout << endl; }

			cout << "matrix_pos:" << i << " " << result[i] << "   ";
			}*/

			for (int i = size - 1; i >= 0; i--) {


				if ((i + 1) % dim == 0) { cout << endl; }

				cout << "matrix_pos:" << i << " " << result[i] << "   ";
			}

			jacobi_Simple << <blocksize, threads >> >(d_A0[dev], d_A1[dev], d_A2[dev], d_A3[dev], d_A4[dev], d_Vec_In[dev], d_Vec_Out[dev], d_Rhs[dev]);

			//TODO: Currently serial has to be done cudaMemcpyAsync using CUDA Streams

			//Copy the intermediate result from Device to Host memory
			cudaMemcpy(result.get() + pos, d_Vec_Out[dev], domainDivision[dev] * sizeof(float), cudaMemcpyDeviceToHost);
			//Copy the intermediate result from the Host memory to the Device memory

			//Print Intermediate result
			/* cout << endl <<"Intermediate Result";

			for (int i = domainDivision[dev]; i >= 0; i--) {


			if ((i + 1) % dim == 0) { cout << endl; }

			cout << "position:"<<i<<" "<<result[i] << "   ";
			}*/

			cudaMemcpy(d_Vec_In[dev], result.get() + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
		}

		cout << endl << "Exchanging Halos";
	}

	if (auto err = cudaGetLastError())
	{
		fprintf(stderr, "Jacobi launch failed: %s\n", cudaGetErrorString(err));
		return err;
	}

	cout << endl << "Iterations successful " << endl;

	//Copy the final result from all devices
	for (int dev = 0, pos = 0; dev < numDevices; pos += domainDivision[dev], dev++)
	{
		cudaMemcpy(result.get() + pos, d_Vec_Out[dev], domainDivision[dev] * sizeof(float), cudaMemcpyDeviceToHost);
	}



	if (auto err = cudaGetLastError())
	{
		fprintf(stderr, "Jacobi launch failed: %s\n", cudaGetErrorString(err));
		return err;
	}

	//Print result

	for (int i = size - 1; i >= 0; i--) {


		if ((i + 1) % dim == 0) { cout << endl; }

		cout << result[i] << " ";
	}
	// Freeing memory auto done by cuda deleter

	//Free memory on devices
	for (int dev = 0; dev<numDevices; dev++)
	{
		cudaFree(d_A0[dev]);
		cudaFree(d_A1[dev]);
		cudaFree(d_A2[dev]);
		cudaFree(d_A3[dev]);
		cudaFree(d_A4[dev]);
		cudaFree(d_Vec_In[dev]);
		cudaFree(d_Vec_Out[dev]);
		cudaFree(d_Rhs[dev]);
	}

	//Take care of dynamic mem location
	delete[] domainDivision;

	return cudaSuccess;


}


int main()
{


	cudaError_t cudaStatus = performMultiGPUJacobi();

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
