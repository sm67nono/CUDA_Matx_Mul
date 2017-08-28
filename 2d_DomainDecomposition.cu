#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "testMultiGPU_Jacobi2D_Decom.cuh"
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
using namespace std;
using namespace std::chrono;

#define IMUL(a,b) __mul24(a,b)


//cudaError_t performMultiGPUJacobi();

//Support for below c++14 on *nix
template<typename T, typename ...Args>
std::unique_ptr<T> make_unique(Args&& ...args)
{
	return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

struct create_Device
{
	int deviceID;

	//In a GPU topology set the GPU position
	int devicePosition_X;
	int devicePosition_Y;
	int devicePosition_Z;

	vector<float> eHalo;
	vector<float> wHalo;
	vector<float> nHalo;
	vector<float> sHalo;

};



//Simple Jacobi iteration
__global__ void jacobi_Simple(const float *A0, const float *A1, const float *A2, const float *A3, const float *A4, float *x_in, float *x_out, const float *rhs, float *nhalo, float *shalo, const int deviceID, const int numDevices)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	float result = rhs[index];

	int dim_x = blockDim.x;// dim across x
	int dim_y = gridDim.x;
	int x_pos = blockIdx.x;
	int y_pos = threadIdx.x;


	//result = nhalo[y_pos];
	//x_out[index] = result;
	//Get the boundaries

	int leftBoundaryElem = x_pos * (dim_x);

	int rightBoundaryElem = (x_pos * dim_x) + (dim_x - 1);

	int topBoundaryElem = y_pos + ((dim_y - 1) * (dim_x));

	int bottomBoundaryElem = y_pos;


	//Halo computation for 1D Decompostion: For the First and Last GPU Halo computation on both the sides(nhalo and shalo wont be needed)
	if (numDevices > 1)
	{
		//First GPU
		if (deviceID == 0) {
			//We need to use nhalos

			//Carry out computations for boundary elements
			if (index != leftBoundaryElem)
				//Left
				result -= A1[index] * x_in[index - 1];

			if (index != rightBoundaryElem)
				//Right 
				result -= A3[index] * x_in[index + 1];
			if (index != bottomBoundaryElem)
				//Bottom
				result -= A0[index] * x_in[index - dim_x];

			if (index != topBoundaryElem)
				//Top
				result -= A4[index] * x_in[index + dim_x];
			//The top boundary needs element from nhalo
			if (index == topBoundaryElem)
				//nHalos
				result -= A4[index] * nhalo[y_pos];




			result /= A2[index];

			x_out[index] = result;


			//Update Halo at the end of computation
			if (index == topBoundaryElem)
				//nHalos updated
				nhalo[y_pos] = result;

			return;

		}

		//Last GPU
		else if (deviceID == (numDevices - 1)) {
			//We need to use shalos

			//Carry out computations for boundary elements
			if (index != leftBoundaryElem)
				//Left
				result -= A1[index] * x_in[index - 1];

			if (index != rightBoundaryElem)
				//Right 
				result -= A3[index] * x_in[index + 1];
			if (index != bottomBoundaryElem)
				//Bottom
				result -= A0[index] * x_in[index - dim_x];
			//The Bottom boundary needs elements from shalo
			if (index == bottomBoundaryElem)
				//nHalos
				result -= A0[index] * shalo[y_pos];


			if (index != topBoundaryElem)
				//Top
				result -= A4[index] * x_in[index + dim_x];


			result /= A2[index];

			x_out[index] = result;

			//Update Halo at the end of computation
			if (index == bottomBoundaryElem)
				//sHalos updated
				shalo[y_pos] = result;

			return;

		}
		//For all the middle GPUs
		else
		{
			//We need to use both shalos and nhalos

			//Carry out computations for boundary elements
			if (index != leftBoundaryElem)
				//Left
				result -= A1[index] * x_in[index - 1];

			if (index != rightBoundaryElem)
				//Right 
				result -= A3[index] * x_in[index + 1];

			if (index != bottomBoundaryElem)
				//Bottom
				result -= A0[index] * x_in[index - dim_x];
			//The Bottom boundary needs elements from shalo
			if (index == bottomBoundaryElem)
				//nHalos
				result -= A0[index] * shalo[y_pos];


			if (index != topBoundaryElem)
				//Top
				result -= A4[index] * x_in[index + dim_x];
			//The top boundary needs element from nhalo
			if (index == topBoundaryElem)
				//nHalos
				result -= A4[index] * nhalo[y_pos];





			result /= A2[index];

			x_out[index] = result;



			//Update Halo at the end of computation
			if (index == bottomBoundaryElem)
				//sHalos updated
				shalo[y_pos] = result;

			//Update Halo at the end of computation
			if (index == topBoundaryElem)
				//nHalos updated
				nhalo[y_pos] = result;



			return;

		}

	}

	//For computations on a Machine with a single GPU
	else
	{
		{//For some reason order of computation (left,right,top and bottom) gives a different result

		 //Carry out computations for boundary elements
			if (index != leftBoundaryElem)
				//Left
				result -= A1[index] * x_in[index - 1];

			if (index != rightBoundaryElem)
				//Right 
				result -= A3[index] * x_in[index + 1];
			if (index != bottomBoundaryElem)
				//Bottom
				result -= A0[index] * x_in[index - dim_x];

			if (index != topBoundaryElem)
				//Top
				result -= A4[index] * x_in[index + dim_x];



			result /= A2[index];

			x_out[index] = result;

			return;
		}
	}



}




//====================================Creating Topology with the number of Devices available====================================

void generateGPUGRID(unsigned int numDevices, int &numberOfDevicesAlong_X, int &numberOfDevicesAlong_Y)
{
	//Finding GPU topology along x and y
	//Assumuing total number of devices is a perfect square(To be changed later)
	numberOfDevicesAlong_X = (int)sqrt(numDevices);
	numberOfDevicesAlong_Y =(int) numberOfDevicesAlong_X;
}


/* Creates a topology for a number of devices in a system
for ex. The devices are aware of left, right, top and bottom neigbours in 2D
1. It also decides the chunk per devices by determining x-dimension and y-dimensions for per chunk of data per device.
2. It also initializes halos for each devices which can be exchanged with the neighbours
*/

void createTopology(unsigned numDevices, vector<create_Device> &deviceArray, int numberOfDevicesAlong_X, int numberOfDevicesAlong_Y)
{

	deviceArray.resize(numDevices);
	unsigned int deviceCount = 0;
	for (int gridCount_X = 0; gridCount_X < numberOfDevicesAlong_X; gridCount_X++) {
		for (int gridCount_Y = 0; gridCount_Y < numberOfDevicesAlong_Y; gridCount_Y++) {
			deviceArray[deviceCount].deviceID = deviceCount;
			deviceArray[deviceCount].devicePosition_X = gridCount_X;
			deviceArray[deviceCount].devicePosition_Y = gridCount_Y;
			//devicePosition_Z to be changed later
			deviceArray[deviceCount].devicePosition_Z = 1;
			deviceCount++;
		}
	}


}
//==============================================================================================================================

//Init Halos: In 1D decomposition only North and South Halos are used. In 2D decomposition North, South, East and West Halo need to be initialized and computed
//In 3D decomposition North, South, East , West, Top and Bottom needs to be initialized and computed
void initHalos(int numDevices, vector<create_Device> &deviceArray, int dim_x, float *vec_in) {


	deviceArray.resize(numDevices);
	int chunksize = ((dim_x*dim_x) / numDevices);
	cout << "chunk size is :" << chunksize << endl;
	for (int i = 0, pos = chunksize; i < numDevices; pos += chunksize, i++) {

		deviceArray[i].deviceID = i;
		deviceArray[i].nHalo.resize(dim_x);
		//TODO: 2D halo exchange
		//TODO: deviceArray[i].eHalo.resize(dim_x);
		//TODO: deviceArray[i].wHalo.resize(dim_x);
		deviceArray[i].sHalo.resize(dim_x);

		if (numDevices == 1)
		{
			for (int count = 0; count < dim_x; count++)
			{

				deviceArray[i].nHalo[count] = 1.0f;
				deviceArray[i].sHalo[count] = 1.0f;
			}
			return;
		}

		//First Device needs only nHalo
		if (i == 0)
		{

			for (int k = pos, count = 0; count < dim_x; k++, count++)
			{
				cout << "Halo nPosition for first Device is : " << k << endl;
				deviceArray[i].nHalo[count] = vec_in[k];
			}

		}

		//Last device needs only sHalo
		else if (i == (numDevices - 1))
		{

			for (int k = pos - (chunksize + dim_x), count = 0; count < dim_x; count++, k++)
			{
				cout << "Halo sPosition for Last Device is : " << k << endl;
				deviceArray[i].sHalo[count] = vec_in[k];
			}

		}

		//All the other devices need both sHalo and nHalo
		else
		{


			for (int k = pos, count = 0; count < dim_x; count++, k++)
			{
				cout << "Halo nPosition for Mid Device " << i << " is : " << k << endl;
				deviceArray[i].nHalo[count] = vec_in[k];
			}
			for (int k = pos - (chunksize + dim_x), count = 0; count < dim_x; count++, k++)
			{
				cout << "Halo sPosition for Mid Device " << i << "  is : " << k << endl;
				deviceArray[i].sHalo[count] = vec_in[k];
			}


		}

	}


}

//TODO:Create a Halo Exchange Mechanism for 2D Multi GPU topology
void initHalos2D(create_Device &device, int chunk_X, int chunk_Y, float *vec_in, int maxdevicesAlong_X, int maxDevicesAlong_Y, int rowStartPos, int rowEndPos, int dim) {

	cout << endl <<"Inside Halo Computation 2D. printing Details" ;
	cout << endl <<"Device ID "<< device.deviceID;
	cout << endl <<"Device position X " << device.devicePosition_X;
	cout << endl <<"Device position Y " << device.devicePosition_Y;
	cout << endl << "Row Start " << rowStartPos;
	cout << endl << "Row End " << rowEndPos;

	//Checks provided for Boundary devices in GPU topology
	if ((device.devicePosition_Y - 1) >= 0) {
		cout << "West Halo needed ";
		device.wHalo.resize(chunk_Y);
		for (int rowNum = 0; rowNum < chunk_Y; rowNum++)
		{
			device.wHalo[rowNum] = vec_in[rowStartPos];
			//cout << rowStartPos << " ";
			rowStartPos += dim;	
		}

	}

	if ((device.devicePosition_Y + 1) < maxdevicesAlong_X) {
		cout << "East Halo needed  ";
		device.eHalo.resize(chunk_Y);
		for (int rowNum = 0; rowNum < chunk_Y; rowNum++)
		{
			device.eHalo[rowNum] = vec_in[rowEndPos];
			//cout << rowEndPos << " ";
			rowEndPos += dim;
		}
	}
	if ((device.devicePosition_X - 1) >= 0) {
		cout << "South Halo needed ";
		device.sHalo.resize(chunk_X);
		for (int rowNum = 0; rowNum < chunk_X; rowNum++)
		{
			device.sHalo[rowNum] = vec_in[rowStartPos];
			cout << rowStartPos << " ";
			rowStartPos ++;
		}
		
	}
	if ((device.devicePosition_X + 1) < maxDevicesAlong_Y) {
		cout << "North Halo needed  ";
		device.sHalo.resize(chunk_X);
		rowStartPos = rowStartPos + (dim * chunk_Y);
		for (int rowNum = 0; rowNum < chunk_X; rowNum++)
		{
			device.sHalo[rowNum] = vec_in[rowStartPos];
			cout << rowStartPos << " ";
			rowStartPos++;
		}
	}


}



//Domain Decompostion 2D:Data distribution


//Init matrix Diagonals A0, A1, A2, A3, A4
void copyValues(float *A0, float *A1, float *A2, float *A3, float *A4, float *rhs, float *vec_in, float *vec_out, int dim, float *val_A0, float *val_A1, float *val_A2, float *val_A3, float *val_A4, float *val_rhs, float *val_x_in)
{

	unsigned int size = dim * dim;

	for (unsigned int i = 0; i < size; i++)
	{
		A0[i] = i;// val_A0[i];
		A1[i] = val_A1[i];
		A2[i] = val_A2[i];
		A3[i] = val_A3[i];
		A4[i] = val_A4[i];
		rhs[i] = val_rhs[i];
		vec_in[i] = val_x_in[i];
		vec_out[i] = 0.0f;

	}



}


void getAllDeviceProperties() {

	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		cout << " Device Number: " << i << endl;
		cout << " Device name: " << prop.name << endl;
		cout << " Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
		cout << " Memory Bus Width (bits): " << prop.memoryBusWidth << endl;;
		cout << " Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6 << endl << endl << endl;
	}
}



cudaError_t performMultiGPUJacobi(unsigned int val_dim, unsigned int numJacobiIt, float* val_A0, float* val_A1, float* val_A2, float* val_A3, float* val_A4, float* val_rhs, float* val_x_in)
{
	//Fixed value changed later
	int dim = 8;
	if (val_dim != 0) {
		dim = val_dim;
	}


	//TODO: write a 2D domain decomposition method for more than 2 GPUs
	int size = dim * dim;

	//auto result = make_unique<float[]>(size);

	//Create Diagonal Vectors
	std::vector<float> a0(size);
	std::vector<float> a1(size);
	std::vector<float> a2(size);
	std::vector<float> a3(size);
	std::vector<float> a4(size);
	std::vector<float> vec_in(size);
	std::vector<float> vec_out(size);
	std::vector<float> rhs(size);
	std::vector<float> result(size);

	//Used for exchanging the Halos after each Jacobi Iteration
	std::vector<float> prev_nHalo(dim);
	std::vector<float> curr_sHalo(dim);

	//Get the total number of devices
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	cout << endl << "Total number of Devices in the System are :  " << numDevices << endl;

	getAllDeviceProperties();


	//Configuring the number of GPU's manually
	//numDevices=2;

	copyValues(&a0[0], &a1[0], &a2[0], &a3[0], &a4[0], &rhs[0], &vec_in[0], &vec_out[0], dim, &val_A0[0], &val_A1[0], &val_A2[0], &val_A3[0], &val_A4[0], &val_rhs[0], &val_x_in[0]);

	vector<create_Device> deviceArray;



	/* Distributed Compuation using Halos: Algorithm

	1. Init Halos.
	1.a) In 1D decomposition nhalo and shalo intialized from vector x_in
	1.b) In 2D decompsition nhalo,shalo, ehalo and whalo initialozed from vector x_in
	2. Pass the halos to Jacobi_kernal.
	3. Store the result computed at the boundary into the halo boundary positions.
	4. Swap nhalo and shalo pairs in 1D decompostion. Swap (nhalo,shalo) and (ehalo,whalo) in 2D.

	*/

	/*initHalos(numDevices, deviceArray, dim, &vec_in[0]);

	//Display Halos
	if (numDevices > 1) {
		cout << endl << "Halo Init.." << endl;

		for (int i = 0; i < numDevices; i++) {

			cout << "Device ID: " << deviceArray[i].deviceID;

			//First Device needs only nHalo
			if (i == 0)
			{
				cout << "First Device";
				for (int k = 0; k < dim; k++)
				{
					cout << deviceArray[i].nHalo[k];
				}

			}

			//Last device needs only sHalo
			else if (i == (numDevices - 1))
			{
				cout << "Last Device";
				for (int k = 0; k < dim; k++)
				{
					cout << deviceArray[i].sHalo[k];
				}

			}

			//All the other devices need both sHalo and nHalo
			else
			{

				cout << "Middle Device";
				for (int k = 0; k < dim; k++)
				{
					cout << deviceArray[i].nHalo[k];
				}

				for (int k = 0; k < dim; k++)
				{
					cout << deviceArray[i].sHalo[k];
				}


			}
			cout << endl;


		}

		cout << endl;
		cout << endl;
		cout << endl;

	}

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

	cout << "Vec In            ...." << endl;

	for (int i = size - 1; i >= 0; i--) {


		if ((i + 1) % dim == 0) { cout << endl; }

		cout << vec_in[i] << " ";
	}

	cout << endl;



	cout << "Made it here.."; */

	//=================================Domain Decomposition Logic Starts=================================================================

	/*Generating a GPU Grid with  multiple GPUs and creating a Topology*/

	int numberOfDevicesAlong_X = 1;
	int numberOfDevicesAlong_Y = 1;
	generateGPUGRID(numDevices, numberOfDevicesAlong_X, numberOfDevicesAlong_Y);
	cout << "GPU grid structure is : " << numberOfDevicesAlong_X << " X " << numberOfDevicesAlong_Y << endl;

	/* Creating a GPU topology with multiple devices*/
	createTopology(numDevices, deviceArray, numberOfDevicesAlong_X, numberOfDevicesAlong_Y);


	//Set Decomposition dimension 1D or 2D
	int decom_Dim = 2;

	//Allocate memory on the devices

	//Let the total number of GPU be 2 : has to be changed later
	//Computation divided into (size/2) on first and size-(size/2) on second
	int *domainDivision;
	domainDivision = new int[numDevices];



	//Logic for total chunk per device (Domain distribution)
	for (int i = 0; i < numDevices; i++) {
		//Chunk per GPU will be same irrepective of 1D or 2D decomposition
		domainDivision[i] = size / numDevices;
	}


	//For use on Device 
	float *d_A0[4],
		*d_A1[4],
		*d_A2[4],
		*d_A3[4],
		*d_A4[4],
		*d_Vec_In[4],
		*d_Vec_Out[4],
		*d_Rhs[4],
		*d_nhalos[4],
		*d_shalos[4],
		*d_ehalos[4],
		*d_whalos[4];

	/* The domain division is done in 1D rowise */
	for (int dev = 0; dev < numDevices; dev++)
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

		//cudaMalloc Halos: North and South--1D. TODO: East and West for 2D
		cudaMalloc((void**)&d_nhalos[dev], (dim / decom_Dim) * sizeof(float));
		cudaMalloc((void**)&d_shalos[dev], (dim / decom_Dim) * sizeof(float));
		cudaMalloc((void**)&d_ehalos[dev], (dim / decom_Dim) * sizeof(float));
		cudaMalloc((void**)&d_whalos[dev], (dim / decom_Dim) * sizeof(float));

	}




	/* The transfer of Data from Host to Device :  Domain Decomposition in 1D*/
	if (decom_Dim == 1) {

		for (int dev = 0, pos = 0; dev < numDevices; pos += domainDivision[dev], dev++)
		{
			//Setting the device before allocation
			cudaSetDevice(dev);

			//Copy the diagonals from host to device
			cudaMemcpy(d_A0[dev], &a0[0] + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A1[dev], &a1[0] + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A2[dev], &a2[0] + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A3[dev], &a3[0] + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A4[dev], &a4[0] + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);

			//Copy in and out vectors and RHS
			cudaMemcpy(d_Vec_In[dev], &vec_in[0] + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vec_Out[dev], &vec_out[0] + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Rhs[dev], &rhs[0] + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);

			//Copy intial Halos in 1D : TODO compute more than 1D
			if (dev == 0) {
				cudaMemcpy(d_nhalos[dev], &deviceArray[dev].nHalo[0], dim * sizeof(float), cudaMemcpyHostToDevice);
			}
			else if (dev == (numDevices - 1)) {
				cudaMemcpy(d_shalos[dev], &deviceArray[dev].sHalo[0], dim * sizeof(float), cudaMemcpyHostToDevice);
			}
			else {
				cudaMemcpy(d_nhalos[dev], &deviceArray[dev].nHalo[0], dim * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(d_shalos[dev], &deviceArray[dev].sHalo[0], dim * sizeof(float), cudaMemcpyHostToDevice);
			}
		}
	}



	/* The transfer of Data from Host to Device :  Domain Decomposition in 2D*/
	if (decom_Dim == 2) {
		
		//Total elements along each dim in 2D
		int chunk_X = dim / numberOfDevicesAlong_X;
		int chunk_Y = dim / numberOfDevicesAlong_Y;


		//Create Partial Diagonal Vectors
		//Size per GPU will be
		int chunkSize = chunk_X * chunk_Y;
		std::vector<float> partial_a0(chunkSize);
		std::vector<float> partial_a1(chunkSize);
		std::vector<float> partial_a2(chunkSize);
		std::vector<float> partial_a3(chunkSize);
		std::vector<float> partial_a4(chunkSize);
		std::vector<float> partial_vec_in(chunkSize);
		std::vector<float> partial_vec_out(chunkSize);
		std::vector<float> partial_rhs(chunkSize);
		std::vector<float> partial_result(chunkSize);


		for (int dev = 0; dev < numDevices; dev++)
		{

			//Test the properties of the device assigned
			cout << endl << "New Logical Device created " << deviceArray[dev].deviceID;
			cout << endl << "New Logical Device (X,Y) coord (" << deviceArray[dev].devicePosition_X << "," << deviceArray[dev].devicePosition_Y << ")";


			//==========Important: Logic for creation of Chunks to be allocated to GPUs==========================================

			//Important : Mention about the correlation between the topology and data position in the thesis
			cudaSetDevice(dev);
			int devicePosX = deviceArray[dev].devicePosition_X;
			int devicePosY = deviceArray[dev].devicePosition_Y;

			//Calculating data position based on device coords
			//numberOfDevicesAlong_X * Chunk_X * Chunk_Y : finds out the  total data per row of GPUs allocated
			int dataStartPos_X = (devicePosX * numberOfDevicesAlong_X * chunk_X * chunk_Y) + (devicePosY * chunk_X);
			int dataEndPos_X = dataStartPos_X + chunk_X;

			//One complete row across all GPU is dim in order to get the next element above an element we add (currentPosition + dim )
			int rowStartPos = dataStartPos_X;
			int rowEndPos = dataEndPos_X;
			int indexCounter = 0;
			//Initialize Halos
			initHalos2D(deviceArray[dev], chunk_X, chunk_Y, &vec_in[0], numberOfDevicesAlong_X, numberOfDevicesAlong_Y, rowStartPos, rowEndPos-1, dim);
			for (int rowNum = 0; rowNum < chunk_Y; rowNum++)
			{
				//Get one complete row for the GPU
				for (int pos = rowStartPos; pos < rowEndPos; pos++)
				{
					partial_a0[indexCounter] = a0[pos];
					partial_a1[indexCounter] = a1[pos];
					partial_a2[indexCounter] = a2[pos];
					partial_a3[indexCounter] = a3[pos];
					partial_a4[indexCounter] = a4[pos];
					partial_vec_in[indexCounter] = vec_in[pos];
					partial_vec_out[indexCounter] = vec_out[pos];
					partial_rhs[indexCounter] = rhs[pos];
					partial_result[indexCounter] = result[pos];
					indexCounter++;
				}
				rowStartPos += dim;
				rowEndPos += dim;
			}

		

			//==========Important: Logic for creation of Chunks to be allocated to GPUs Ends ==========================================
			
			//Testing if inputs are correct
			 /*cout << endl << endl;
			for (int i = 0; i < indexCounter; i++) {
				if ((i%chunk_X) == 0)cout << endl;
				cout << partial_a0[i]<<" ";
			}*/


			//Copy the diagonals from host to device : calling all at once instead of putting inside the for loop
			cudaMemcpy(d_A0[dev], &partial_a0, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A1[dev], &partial_a1, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A2[dev], &partial_a2, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A3[dev], &partial_a3, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A4[dev], &partial_a4, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);

			//Copy in and out vectors and RHS
			cudaMemcpy(d_Vec_In[dev], &partial_vec_in, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vec_Out[dev], &partial_vec_out, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Rhs[dev], &partial_rhs, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);

			//Copy intial Halos in 2D
			
		}
	}
	//=================================Domain Decomposition Logic Ends =================================================================


	if (auto err = cudaGetLastError())
	{
		cout << "Jacobi launch failed: " << cudaGetErrorString(err) << endl;
		return err;
	}



	if (auto err = cudaGetLastError())
	{
		cout << "Jacobi launch failed: " << cudaGetErrorString(err) << endl;
		return err;
	}




	if (auto err = cudaGetLastError())
	{
		cout << "Jacobi launch failed: " << cudaGetErrorString(err) << endl;
		return err;
	}

	//multMatrix(d_A0, d_A1, d_A2, d_A3, d_A4, myDim, d_vec, d_res);

	//Perform one Jacobi Step
	int blocksize = dim / numDevices; //TODO: make it to more than 2 GPUs
	int threads = dim;

	//Call to kernal
	int iterations = 4;
	if (numJacobiIt != 0) {
		iterations = numJacobiIt;
	}

	for (int i = 0; i < iterations; i++)
	{

		cout << endl << endl << "Iteration : " << i + 1 << endl << endl << endl;

		//TODO: optimization using kernel instead of For Loop
		for (int dev = 0, pos = 0; dev < numDevices; pos += domainDivision[dev], dev++)
		{
			cudaSetDevice(dev);
			cout << endl << endl << "Kernal Execution on GPU : " << dev;
			cout << endl << "Position :" << pos;


			cout << endl << "Check Intermediate Result before it gets passed to kernal" << endl;

			cudaMemcpy(&result[0] + pos, d_Vec_In[dev], domainDivision[dev] * sizeof(float), cudaMemcpyDeviceToHost);

			for (int i = size - 1; i >= 0; i--) {


				if ((i + 1) % dim == 0) { cout << endl; }

				cout << "#pos:" << i << " " << result[i] << "    ";
			}

			jacobi_Simple <<<blocksize, threads >>>(d_A0[dev], d_A1[dev], d_A2[dev], d_A3[dev], d_A4[dev], d_Vec_In[dev], d_Vec_Out[dev], d_Rhs[dev], d_nhalos[dev], d_shalos[dev], deviceArray[dev].deviceID, numDevices);

			//TODO: Currently serial has to be done cudaMemcpyAsync using CUDA Streams

			//Copy the intermediate result from Device to Host memory
			cudaMemcpy(&result[0] + pos, d_Vec_Out[dev], domainDivision[dev] * sizeof(float), cudaMemcpyDeviceToHost);
			//Copy the intermediate result from the Host memory to the Device memory
			cudaMemcpy(d_Vec_In[dev], &result[0] + pos, domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);


			/* Store Halo positions after iteration for exchanging */
			if (numDevices > 1)
			{
				if (dev == 0) {
					cudaMemcpy(&prev_nHalo[0], d_nhalos[dev], dim * sizeof(float), cudaMemcpyDeviceToHost);
				}
				else if (dev == (numDevices - 1)) {
					//Exchange Happens here
					cudaMemcpy(&curr_sHalo[0], d_shalos[dev], dim * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(d_shalos[dev], &prev_nHalo[0], dim * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_nhalos[dev - 1], &curr_sHalo[0], dim * sizeof(float), cudaMemcpyHostToDevice);

				}
				else {
					//Exchange Happens here
					cudaMemcpy(&curr_sHalo[0], d_shalos[dev], dim * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(d_shalos[dev], &prev_nHalo[0], dim * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(d_nhalos[dev - 1], &curr_sHalo[0], dim * sizeof(float), cudaMemcpyHostToDevice);
					//Store current North Boundary in prev_halo for exchanging in later step
					cudaMemcpy(&prev_nHalo[0], d_nhalos[dev], dim * sizeof(float), cudaMemcpyDeviceToHost);
				}
			}


		}

		//TODO: Using P2P to be done later
		//exchangeHalos(numDevices,result, d_Vec_In);
		//Exchange halo logic
		//1. Prev = current nhalo
		//2. On next  iteration shalo = Prev and, Prev = nhalo.


	}

	if (auto err = cudaGetLastError())
	{
		cout << "Jacobi launch failed: " << cudaGetErrorString(err) << endl;
		return err;
	}

	cout << endl << "Iterations successful " << endl;

	//Copy the final result from all devices
	for (int dev = 0, pos = 0; dev < numDevices; pos += domainDivision[dev], dev++)
	{
		cudaMemcpy(&result[0] + pos, d_Vec_Out[dev], domainDivision[dev] * sizeof(float), cudaMemcpyDeviceToHost);
	}



	if (auto err = cudaGetLastError())
	{
		cout << "Jacobi launch failed: " << cudaGetErrorString(err) << endl;
		return err;
	}

	//Print result

	for (int i = size - 1; i >= 0; i--) {


		if ((i + 1) % dim == 0) { cout << endl; }

		cout << result[i] << " ";
	}
	// Freeing memory auto done by cuda deleter

	//Free memory on devices
	for (int dev = 0; dev < numDevices; dev++)
	{
		cudaFree(d_A0[dev]);
		cudaFree(d_A1[dev]);
		cudaFree(d_A2[dev]);
		cudaFree(d_A3[dev]);
		cudaFree(d_A4[dev]);
		cudaFree(d_Vec_In[dev]);
		cudaFree(d_Vec_Out[dev]);
		cudaFree(d_nhalos[dev]);
		cudaFree(d_shalos[dev]);
		cudaFree(d_Rhs[dev]);
	}

	cout << endl << "Device Memory free successful.";
	//Take care of dynamic mem location
	delete[] domainDivision;

	return cudaSuccess;


}


int performJacobi_MultiGPU2D_Decom(unsigned int dim, unsigned int numJacobiIt, float* A0, float* A1, float* A2, float* A3, float* A4, float* rhs, float* x_in)
{


	cudaError_t cudaStatus = performMultiGPUJacobi(dim, numJacobiIt, &A0[0], &A1[0], &A2[0], &A3[0], &A4[0], &rhs[0], &x_in[0]);

	if (cudaStatus != cudaSuccess) {
		cout << "Computation failed: " << endl;
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		cout << "Cuda Device Reset failed: " << endl;
		return 1;
	}

	return 0;

}
