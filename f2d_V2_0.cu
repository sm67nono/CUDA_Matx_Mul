#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "testMultiGPU_Jacobi2D_Decom.cuh"
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <fstream>
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

	//Flags check the halos needed by the device
	int eHalo_flag = 0;
	int wHalo_flag = 0;
	int nHalo_flag = 0;
	int sHalo_flag = 0;

};

//Simple Jacobi iteration
__global__ void jacobi_Simple(const float *A0, const float *A1, const float *A2, const float *A3, const float *A4, float *x_in, float *x_out, const float *rhs, const int ehalo_flag, const int whalo_flag, const int nhalo_flag, const int shalo_flag, float *ehalo, float *whalo, float *nhalo, float *shalo, const int deviceID, const int numDevices, const int domain_Decom)
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
	if (domain_Decom == 1)
	{
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
	}
	else if (domain_Decom == 2) {


		//======Left Bounday Elem
		if (index != leftBoundaryElem)
			//Left
			result -= A1[index] * x_in[index - 1];
		//Computation using the Halos
		if (index == leftBoundaryElem) {
			if (whalo_flag == 1) {
				result -= A1[index] * whalo[x_pos];
			}
		}

		//======Right Bounday Elem
		if (index != rightBoundaryElem)
			//Right
			result -= A3[index] * x_in[index + 1];
		if (index == rightBoundaryElem) {
			if (ehalo_flag == 1) {
				result -= A3[index] * ehalo[x_pos];
			}
		}


		//======Bottom Bounday Elem
		if (index != bottomBoundaryElem)
			//Bottom
			result -= A0[index] * x_in[index - dim_x];

		if (index == bottomBoundaryElem) {
			if (shalo_flag == 1) {
				result -= A0[index] * shalo[y_pos];
			}
		}


		//======Top Bounday Elem
		if (index != topBoundaryElem)
			//Top
			result -= A4[index] * x_in[index + dim_x];
		if (index == topBoundaryElem) {
			if (nhalo_flag == 1) {
				result -= A4[index] * nhalo[y_pos];
			}
		}





		result /= A2[index];

		x_out[index] = result;




		//Updating Halos at the End of the computation
		if (index == topBoundaryElem) {
			if (nhalo_flag == 1) {
				nhalo[y_pos] = result;
			}
		}

		if (index == bottomBoundaryElem) {
			if (shalo_flag == 1) {
				shalo[y_pos] = result;
			}
		}

		if (index == leftBoundaryElem) {
			if (whalo_flag == 1) {
				whalo[x_pos] = result;
			}
		}

		if (index == rightBoundaryElem) {
			if (ehalo_flag == 1) {
				ehalo[x_pos] = result;
			}
		}
		return;

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

void generateGPUGRID(int numDevices, int &numberOfDevicesAlong_X, int &numberOfDevicesAlong_Y)
{
	//Finding GPU topology along x and y
	//Assumuing total number of devices is a perfect square(To be changed later)
	numberOfDevicesAlong_X = (int)sqrt(numDevices);
	numberOfDevicesAlong_Y = (int)numberOfDevicesAlong_X;
}


/* Creates a topology for a number of devices in a system
for ex. The devices are aware of left, right, top and bottom neigbours in 2D
1. It also decides the chunk per devices by determining x-dimension and y-dimensions for per chunk of data per device.
2. It also initializes halos for each devices which can be exchanged with the neighbours
*/

void createTopology(int numDevices, vector<create_Device> &deviceArray, int numberOfDevicesAlong_X, int numberOfDevicesAlong_Y)
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
//TODO:Create a Halo Exchange Mechanism for 2D Multi GPU topology
void initHalos2D(create_Device &device, int chunk_X, int chunk_Y, float *vec_in, int maxdevicesAlong_X, int maxDevicesAlong_Y, int rowStartPos, int rowEndPos, int dim) {

	/*cout << endl << "Inside Halo Computation 2D. printing Details";
	cout << endl << "Device ID " << device.deviceID;
	cout << endl << "Device position X " << device.devicePosition_X;
	cout << endl << "Device position Y " << device.devicePosition_Y;
	cout << endl << "Row Start " << rowStartPos;
	cout << endl << "Row End " << rowEndPos;*/

	//Assigning counter for each individual Halos. To prevent update of the same counter
	//int rowStartPosEast = rowStartPos;
	int rowStartPosWest = rowStartPos;
	int rowStartPosNorth = rowStartPos;
	int rowStartPosSouth = rowStartPos;

	int rowEndPosEast = rowEndPos;
	//int rowEndPosWest =  rowEndPos;
	//int rowEndPosNorth = rowEndPos;
	//int rowEndPosSouth = rowEndPos;


	//Checks provided for Boundary devices in GPU topology
	if ((device.devicePosition_Y - 1) >= 0) {
		//cout << "West Halo needed ";
		device.wHalo_flag = 1;
		device.wHalo.resize(chunk_Y);
		for (int rowNum = 0; rowNum < chunk_Y; rowNum++)
		{
			device.wHalo[rowNum] = vec_in[rowStartPosWest];
			//cout << rowStartPosWest << " ";
			rowStartPosWest += dim;
		}

	}

	if ((device.devicePosition_Y + 1) < maxdevicesAlong_X) {
		//cout << "East Halo needed  ";
		device.eHalo_flag = 1;
		device.eHalo.resize(chunk_Y);
		for (int rowNum = 0; rowNum < chunk_Y; rowNum++)
		{
			device.eHalo[rowNum] = vec_in[rowEndPosEast];
			//cout << rowEndPosEast << " ";
			rowEndPosEast += dim;
		}
	}
	if ((device.devicePosition_X - 1) >= 0) {
		//cout << "South Halo needed ";
		device.sHalo_flag = 1;
		device.sHalo.resize(chunk_X);
		for (int rowNum = 0; rowNum < chunk_X; rowNum++)
		{
			device.sHalo[rowNum] = vec_in[rowStartPosSouth];
			//cout << rowStartPosSouth << " ";
			rowStartPosSouth++;
		}

	}
	if ((device.devicePosition_X + 1) < maxDevicesAlong_Y) {
		//cout << "North Halo needed  ";
		device.nHalo_flag = 1;
		device.nHalo.resize(chunk_X);
		rowStartPosNorth = rowStartPosNorth + (dim * (chunk_Y - 1));
		for (int rowNum = 0; rowNum < chunk_X; rowNum++)
		{
			device.nHalo[rowNum] = vec_in[rowStartPosNorth];
			//cout << rowStartPosNorth << " ";
			rowStartPosNorth++;
		}
	}

}

//======================================Exchange Halos: on Host==============================================


int getDeviceIDfromCoord(int devCoord_x, int devCoord_y, int numberofDevicesAlong_X) {
	int devID = (devCoord_x * numberofDevicesAlong_X) + devCoord_y;
	return devID;
}


void exchangehalos_onHost(int numDevices, vector<create_Device> &deviceArray, int numberofDevicesAlong_X)
{
	//Halos exist in pairs so:
	//Important: A device exchanges North-to-South Pairs and East-to-West Pairs only. Not South-to-North pairs and West-to-East pairs
	//That way the number of exchanges are kept to minimum


	for (int dev = 0;dev < numDevices;dev++)
	{
		int getDevCoord_X = deviceArray[dev].devicePosition_X;
		int getDevCoord_Y = deviceArray[dev].devicePosition_Y;


		//Check if device is having a north Halo buffer
		if (deviceArray[dev].nHalo_flag == 1) {
			int devIDtoNorth = getDeviceIDfromCoord(getDevCoord_X + 1, getDevCoord_Y, numberofDevicesAlong_X);
			//Exchange Halos 
			(deviceArray[dev].nHalo).swap(deviceArray[devIDtoNorth].sHalo);
		}

		//Check if device is having a east Halo buffer
		if (deviceArray[dev].eHalo_flag == 1) {
			int devIDtoEast = getDeviceIDfromCoord(getDevCoord_X, getDevCoord_Y + 1, numberofDevicesAlong_X);
			//Exchange Halos 
			(deviceArray[dev].eHalo).swap(deviceArray[devIDtoEast].wHalo);
		}
	}

}


//===========================Exchange Halos: on Host Ends=====================================================


//Init matrix Diagonals A0, A1, A2, A3, A4
void copyValues(float *A0, float *A1, float *A2, float *A3, float *A4, float *rhs, float *vec_in, float *vec_out, int dim, float *val_A0, float *val_A1, float *val_A2, float *val_A3, float *val_A4, float *val_rhs, float *val_x_in)
{

	unsigned int size = dim * dim;

	for (unsigned int i = 0; i < size; i++)
	{
		A0[i] = val_A0[i];
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


/* Prints an output file for checking results */
void sendToPrint(float *partial_result, int devicePosition_X, int devicePosition_Y, int numberOfDevicesAlong_X, int chunk_X, int  chunk_Y, int dim, int totalSize, vector<float> &result, int numDevices, int currentIteration, int numberOfTotalIterations) {

	int devicePosX = devicePosition_X;
	int devicePosY = devicePosition_Y;

	//Calculating data position based on device coords
	//numberOfDevicesAlong_X * Chunk_X * Chunk_Y : finds out the  total data per row of GPUs allocated
	int dataStartPos_X = (devicePosX * numberOfDevicesAlong_X * chunk_X * chunk_Y) + (devicePosY * chunk_X);
	int dataEndPos_X = dataStartPos_X + chunk_X;

	//One complete row across all GPU is dim in order to get the next element above an element we add (currentPosition + dim )
	int rowStartPos = dataStartPos_X;
	int rowEndPos = dataEndPos_X;
	int indexCounter = 0;
	//cout << endl;
	for (int rowNum = 0; rowNum < chunk_Y; rowNum++)
	{
		//Get one complete row for the GPU
		for (int pos = rowStartPos; pos < rowEndPos; pos++)
		{
			result[pos] = partial_result[indexCounter];
			indexCounter++;
		}
		//cout << endl;
		rowStartPos += dim;
		rowEndPos += dim;
	}

	//Printing when the last device computation is done: Remove the check to check computation for each device
	int deviceID = getDeviceIDfromCoord(devicePosition_X, devicePosition_Y, numberOfDevicesAlong_X);
	if ((deviceID == (numDevices - 1)) && (currentIteration == (numberOfTotalIterations - 1)))
	{
		ofstream myfile;
		myfile.open("data2.txt");
		//Printing the values here
		for (int i = totalSize;i > 0; i--) {

			if (i%dim == 0) {
				myfile << endl;
			}
			myfile << result[i - 1] << " ";
		}
		myfile.close();
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

	//=================================Domain Decomposition Logic Starts=================================================================

	/*Generating a GPU Grid with  multiple GPUs and creating a Topology*/

	int numberOfDevicesAlong_X = 1;
	int numberOfDevicesAlong_Y = 1;
	generateGPUGRID(numDevices, numberOfDevicesAlong_X, numberOfDevicesAlong_Y);
	cout << "GPU grid structure is : " << numberOfDevicesAlong_X << " X " << numberOfDevicesAlong_Y << endl;

	//Set Decomposition dimension 1D or 2D: when decomposition is 0. Computation happens on a single GPU
	int decom_Dim = 2;

	//Total elements along each dim in 2D
	int chunk_X = dim / numberOfDevicesAlong_X;
	int chunk_Y = dim / numberOfDevicesAlong_Y;

	/* Creating a GPU topology with multiple devices*/
	createTopology(numDevices, deviceArray, numberOfDevicesAlong_X, numberOfDevicesAlong_Y);


	//Let the total number of GPU be 2 : has to be changed later
	//Computation divided into (size/2) on first and size-(size/2) on second

	std::vector<int> domainDivision(numDevices);



	//Logic for total chunk per device (Domain distribution)
	for (int i = 0; i < numDevices; i++) {
		//Chunk per GPU will be same irrepective of 1D or 2D decomposition
		domainDivision[i] = size / numDevices;
	}


	//For use on Device 
	std::vector<float*>d_A0(numDevices);
	std::vector<float*>d_A1(numDevices);
	std::vector<float*>d_A2(numDevices);
	std::vector<float*>d_A3(numDevices);
	std::vector<float*>d_A4(numDevices);
	std::vector<float*>d_Vec_In(numDevices);
	std::vector<float*>d_Vec_Out(numDevices);
	std::vector<float*>d_nhalos(numDevices);
	std::vector<float*>d_shalos(numDevices);
	std::vector<float*>d_ehalos(numDevices);
	std::vector<float*>d_whalos(numDevices);
	std::vector<float*>d_Rhs(numDevices);

	//Compute / store partial results on the Host
	//float *partial_result[4];


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


		//Using pinned memory as part of performance upgrade- Phase 2 of development

		//cudamalloc the Input Vector and Result vector
		cudaMalloc((void**)&d_Vec_In[dev], domainDivision[dev] * sizeof(float));
		cudaMalloc((void**)&d_Vec_Out[dev], domainDivision[dev] * sizeof(float));
		cudaMalloc((void**)&d_Rhs[dev], domainDivision[dev] * sizeof(float));

		//cudaMalloc Halos: North and South--1D. TODO: East and West for 2D
		cudaMalloc((void**)&d_nhalos[dev], chunk_X * sizeof(float));
		cudaMalloc((void**)&d_shalos[dev], chunk_X * sizeof(float));
		cudaMalloc((void**)&d_ehalos[dev], chunk_Y * sizeof(float));
		cudaMalloc((void**)&d_whalos[dev], chunk_Y * sizeof(float));


	}


	/* The transfer of Data from Host to Device :  Domain Decomposition in 2D*/
	if (decom_Dim == 2) {

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

			//cout << endl << "New Logical Device created " << deviceArray[dev].deviceID;
			//cout << endl << "New Logical Device (X,Y) coord (" << deviceArray[dev].devicePosition_X << "," << deviceArray[dev].devicePosition_Y << ")";


			//==========Important: Logic for creation of Chunks to be allocated to GPUs==========================================

			//Important : Mention about the correlation between the topology and data position in the thesis

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
			initHalos2D(deviceArray[dev], chunk_X, chunk_Y, &vec_in[0], numberOfDevicesAlong_X, numberOfDevicesAlong_Y, rowStartPos, rowEndPos - 1, dim);
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

			//Setting Cuda device
			cudaSetDevice(dev);

			//Copy the diagonals from host to device : calling all at once instead of putting inside the for loop
			cudaMemcpy(d_A0[dev], &partial_a0[0], domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A1[dev], &partial_a1[0], domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A2[dev], &partial_a2[0], domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A3[dev], &partial_a3[0], domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A4[dev], &partial_a4[0], domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);



			//Copy in and out vectors and RHS
			cudaMemcpy(d_Vec_In[dev], &partial_vec_in[0], domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Vec_Out[dev], &partial_vec_out[0], domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Rhs[dev], &partial_rhs[0], domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice);



		}

		if (auto err = cudaGetLastError())
		{
			cout << "Data copy failed 1: " << cudaGetErrorString(err) << endl;
			return err;
		}

		//Copy intial Halos in 2D
		//Initial Exchange Halos: Then do intial cudaMemcopies

		exchangehalos_onHost(numDevices, deviceArray, numberOfDevicesAlong_X);
		for (int dev = 0; dev < numDevices; dev++)
		{

			//Copying Halos to the device
			if (deviceArray[dev].nHalo_flag == 1)
			{
				cudaMemcpy(d_nhalos[dev], &deviceArray[dev].nHalo[0], chunk_X * sizeof(float), cudaMemcpyHostToDevice);
			}
			if (deviceArray[dev].sHalo_flag == 1)
			{
				cudaMemcpy(d_shalos[dev], &deviceArray[dev].sHalo[0], chunk_X * sizeof(float), cudaMemcpyHostToDevice);

			}
			if (deviceArray[dev].eHalo_flag == 1)
			{
				cudaMemcpy(d_ehalos[dev], &deviceArray[dev].eHalo[0], chunk_Y * sizeof(float), cudaMemcpyHostToDevice);
			}
			if (deviceArray[dev].wHalo_flag == 1)
			{
				cudaMemcpy(d_whalos[dev], &deviceArray[dev].wHalo[0], chunk_Y * sizeof(float), cudaMemcpyHostToDevice);
			}
		}

	}
	//=================================Domain Decomposition Logic Ends =================================================================


	int blocksize = chunk_X;
	int threads = chunk_Y;

	//cout << endl<<"blocksize" << blocksize;
	//cout << endl<<"thread" << threads;

	//Call to kernal
	int iterations = 0;
	if (numJacobiIt != 0) {
		iterations = numJacobiIt;
	}
	else
	{
		cout << endl << " No. of iterations is zero exiting... ";
		//return;
	}

	//===========================================CUDA Stream implementation for performance. Phase 2 of Development ====================================================

	cudaStream_t streams[4];//Possible to declare it dynamically ?
	for (int i = 0; i < numDevices; i++)
	{
		cudaStreamCreate(&streams[i]);
	}



	/*Using a pagable memory first*/
	//std::vector<float> partial_resultOnHost(chunk_X * chunk_Y);

	/*Using a pinned(page locked) memory for performance*/
	vector<float*>partial_resultOnHost(numDevices);
	for (int dev = 0;dev < numDevices;dev++)
	{
		cudaMallocHost((void**)&partial_resultOnHost[dev], (chunk_Y * chunk_Y) * sizeof(float));

	}


	//Check performance

	//For kernal exec time measurement
	high_resolution_clock::time_point kernel_t1;
	high_resolution_clock::time_point kernel_t2;

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	for (int i = 0; i < iterations; i++)
	{

		//cout << endl << endl << "Iteration : " << i + 1 << endl << endl << endl;

		//TODO: optimization using kernel instead of For Loop
		//Performance changes by launching kernal seperately
		kernel_t1 = high_resolution_clock::now();
		for (int dev = 0; dev < numDevices;dev++)
		{
			cudaSetDevice(dev);
			jacobi_Simple <<<blocksize, threads >>> (d_A0[dev], d_A1[dev], d_A2[dev], d_A3[dev], d_A4[dev], d_Vec_In[dev], d_Vec_Out[dev], d_Rhs[dev], deviceArray[dev].eHalo_flag, deviceArray[dev].wHalo_flag, deviceArray[dev].nHalo_flag, deviceArray[dev].sHalo_flag, d_ehalos[dev], d_whalos[dev], d_nhalos[dev], d_shalos[dev], deviceArray[dev].deviceID, numDevices, decom_Dim);

		}

		if (auto err = cudaGetLastError())
		{
			cout << "Kernal Execution failed: " << cudaGetErrorString(err) << endl;
			return err;
		}

		for (int dev = 0; dev < numDevices;dev++)
		{
			cudaSetDevice(dev);
			//partial_resultOnHost.clear();
			//cout << endl << endl << "Computation for Device  " << dev;

			//TODO: Performance Upgrade: Currently serial has to be done cudaMemcpyAsync using CUDA Streams

			//Copy the intermediate result from Device to Host memory
			cudaMemcpyAsync(&partial_resultOnHost[dev][0], d_Vec_Out[dev], domainDivision[dev] * sizeof(float), cudaMemcpyDeviceToHost, streams[dev]);

			/*if (i == (iterations-1))//Copy the results just for the final iteration
			{
				cudaMemcpyAsync(&partial_resultOnHost[dev][0], d_Vec_Out[dev], domainDivision[dev] * sizeof(float), cudaMemcpyDeviceToHost, streams[dev]);
				continue;
			}*/

			//swap(d_Vec_In[dev], d_Vec_Out[dev]);


			//Check Intermeduate Results and store it 

			//sendToPrint(&partial_resultOnHost[0], deviceArray[dev].devicePosition_X, deviceArray[dev].devicePosition_Y, numberOfDevicesAlong_X, chunk_X, chunk_Y, dim, size, result, numDevices, i, iterations);


			//Copy the intermediate result from the Host memory to the Device memory

			cudaMemcpyAsync(d_Vec_In[dev], &partial_resultOnHost[dev][0], domainDivision[dev] * sizeof(float), cudaMemcpyHostToDevice, streams[dev]);

			/* Store Halo positions after iteration for exchanging */
			if (numDevices > 1)
			{
				if (deviceArray[dev].nHalo_flag == 1)
				{
					cudaMemcpyAsync(&deviceArray[dev].nHalo[0], d_nhalos[dev], chunk_X * sizeof(float), cudaMemcpyDeviceToHost, streams[dev]);
				}
				if (deviceArray[dev].sHalo_flag == 1)
				{
					cudaMemcpyAsync(&deviceArray[dev].sHalo[0], d_shalos[dev], chunk_X * sizeof(float), cudaMemcpyDeviceToHost, streams[dev]);
				}
				if (deviceArray[dev].eHalo_flag == 1)
				{
					cudaMemcpyAsync(&deviceArray[dev].eHalo[0], d_ehalos[dev], chunk_Y * sizeof(float), cudaMemcpyDeviceToHost, streams[dev]);
				}
				if (deviceArray[dev].wHalo_flag == 1)
				{
					cudaMemcpyAsync(&deviceArray[dev].wHalo[0], d_whalos[dev], chunk_Y * sizeof(float), cudaMemcpyDeviceToHost, streams[dev]);
				}

			}

		}
		kernel_t2 = high_resolution_clock::now();
		if (auto err = cudaGetLastError())
		{
			cout << "Data copy failed 2: " << cudaGetErrorString(err) << endl;
			return err;
		}

		//Exchange Halos after each iteration except the last iteration
		if (i < iterations) {

			exchangehalos_onHost(numDevices, deviceArray, numberOfDevicesAlong_X);

			for (int dev = 0; dev < numDevices; dev++)
			{

				//Copying Halos to the device
				if (deviceArray[dev].nHalo_flag == 1)
				{
					cudaMemcpyAsync(d_nhalos[dev], &deviceArray[dev].nHalo[0], chunk_X * sizeof(float), cudaMemcpyHostToDevice, streams[dev]);
				}
				if (deviceArray[dev].sHalo_flag == 1)
				{
					cudaMemcpyAsync(d_shalos[dev], &deviceArray[dev].sHalo[0], chunk_X * sizeof(float), cudaMemcpyHostToDevice, streams[dev]);

				}
				if (deviceArray[dev].eHalo_flag == 1)
				{
					cudaMemcpyAsync(d_ehalos[dev], &deviceArray[dev].eHalo[0], chunk_Y * sizeof(float), cudaMemcpyHostToDevice, streams[dev]);
				}
				if (deviceArray[dev].wHalo_flag == 1)
				{
					cudaMemcpyAsync(d_whalos[dev], &deviceArray[dev].wHalo[0], chunk_Y * sizeof(float), cudaMemcpyHostToDevice, streams[dev]);
				}
			}
		}

	}

	if (auto err = cudaGetLastError())
	{
		cout << "Data copy failed 3: " << cudaGetErrorString(err) << endl;
		return err;
	}

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(t2 - t1).count();

	auto duration1 = duration_cast<microseconds>(kernel_t2 - kernel_t1).count();
	cout << endl << "Iterations successful. Time taken for single iteration in microseconds :" << duration1 << endl;
	cout << endl << "Iterations successful. Time taken  in microseconds :" << duration << endl;


	for (int i = 0; i < numDevices; ++i)
	{
		//Synchro the streams 
		cudaStreamSynchronize(streams[i]);
		cudaStreamDestroy(streams[i]);
	}

	//Results copied to disk
	for (int dev = 0; dev < numDevices;dev++)
	{
		sendToPrint(&partial_resultOnHost[dev][0], deviceArray[dev].devicePosition_X, deviceArray[dev].devicePosition_Y, numberOfDevicesAlong_X, chunk_X, chunk_Y, dim, size, result, numDevices, iterations - 1, iterations);
	}



	//==========================================Performance using CUDA stream ends===========================================================================


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
		cudaFree(d_ehalos[dev]);
		cudaFree(d_whalos[dev]);
		cudaFree(d_Rhs[dev]);
		cudaFreeHost(partial_resultOnHost[dev]);
	}


	cout << endl << "Device Memory free successful." << endl;
	//Take care of dynamic mem location
	//delete[] domainDivision;

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
