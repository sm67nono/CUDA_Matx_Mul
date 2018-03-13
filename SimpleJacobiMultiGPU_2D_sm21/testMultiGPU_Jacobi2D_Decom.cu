//P2P Synchronization using events and Seperate streams. Coupling-Overlapping several Exchanges together.
#include <omp.h>
#include "tinyxml.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "testMultiGPU_Jacobi2D_Decom.cuh"
#include "multiGPU_topology_optimize.cuh"
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
//#include<map>
#include <fstream>
#include <vector_types.h>



#define IMUL(a,b) __mul24(a,b)
#define DIVRND(a,b) ((a+b-1)/b)
#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 16

using namespace std;
using namespace std::chrono;




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
__global__ void jacobi_Simple(const float *A0, const float *A1, const float *A2, const float *A3, const float *A4, float *x_in, float *x_out, const float *rhs, const int ehalo_flag, const int whalo_flag, const int nhalo_flag, const int shalo_flag, float *ehalo, float *whalo, float *nhalo, float *shalo, const int deviceID, const int numDevices, const int domain_Decom, int2 dim)
{
	int2 pos = make_int2(
		blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y
	);

	int index = (pos.y * dim.x) + pos.x;


	//int index = threadIdx.x + blockDim.x * blockIdx.x;

	float result = rhs[index];

	int dim_x = dim.x;
	int dim_y = dim.y;

	//X_pos and Y_pos are just to understand the thread layout. Can be named to any suitable variable names
	int x_pos = pos.y;
	int y_pos = pos.x;


	//result = nhalo[y_pos];
	//x_out[index] = result;

	//Get the boundaries

	int leftBoundaryElem = x_pos * (dim_x);


	int rightBoundaryElem = (x_pos * dim_x) + (dim_x - 1);

	int topBoundaryElem = y_pos + ((dim_y - 1) * (dim_x));

	int bottomBoundaryElem = y_pos;


	/*if((deviceID==2)&&(index==leftBoundaryElem))
	{
	printf("For Device %d index is : %d\n", deviceID, index);
	printf("For Device %d leftBoundaryElem is : %d\n", deviceID, leftBoundaryElem);
	printf("rightBoundaryElem is : %d\n", rightBoundaryElem);
	printf("topBoundaryElem is : %d\n", topBoundaryElem);
	printf("bottomBoundaryElem is : %d\n", bottomBoundaryElem);
	}*/

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


//========================MultiGPU utility functions============================================================================
// load the named file and dump its structure to STDOUT
void getConfiguration(const char* pFilename, int &numDevices, int &domain_decom)
{
	TiXmlDocument doc(pFilename);
	bool loadOkay = doc.LoadFile();
	if (loadOkay)
	{
		cout <<"\nFile Loaded successfully\n" ;

		TiXmlElement *pRoot = doc.RootElement();
		TiXmlElement *element = pRoot->FirstChildElement();
		while (element)
		{
			string elementName = element->Value();
			string attribute = element->Attribute("name"); //Gets you the time variable
			string value = element->GetText();
			cout << "\n The attribute is "<<attribute;
			cout << "\n The elementName is " << elementName;
			cout << "\n The element Value is " << value;
			if (attribute=="numDevices") {
				numDevices = stoi(value);
			}
			if (attribute == "decomposition") {
				domain_decom = stoi(value);
			}
			element = element->NextSiblingElement();
		}


	}
	else
	{
		cout << "\nCould not load config file\n";
	}
}

void checkP2Paccess(int numGPUs)
{

	for (int i = 0; i<numGPUs; i++)
	{
		cudaSetDevice(i);

		for (int j = 0; j<numGPUs; j++)
		{
			int access;
			if (i != j)
			{
				cudaDeviceCanAccessPeer(&access, i, j);
				if (auto err = cudaGetLastError())
				{
					cout << "P2P Operations failed : " << cudaGetErrorString(err) << endl;
					return;
				}

			}
		}
	}
	cout << "\n***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.\nSo you can see lesser Bandwidth (GB/s) in those cases.\n\n";
}

bool enableP2P(int numGPUs)
{
	for (int i = 0; i<numGPUs; i++)
	{
		cudaSetDevice(i);

		for (int j = 0; j<numGPUs; j++)
		{
			int access;
			cudaDeviceCanAccessPeer(&access, i, j);
			if (auto err = cudaGetLastError())
			{
				cout << "P2P Operations failed while enabling: " << cudaGetErrorString(err) << endl;
				return false;
			}

			if (access)
			{
				cudaDeviceEnablePeerAccess(j, 0);
				if (auto err = cudaGetLastError())
				{
					cout << "P2P Operations failed while enabling: " << cudaGetErrorString(err) << endl;
					return false;
				}

			}
		}
	}
	return true;
}
void disableP2P(int numGPUs)
{
	for (int i = 0; i<numGPUs; i++)
	{
		cudaSetDevice(i);

		for (int j = 0; j<numGPUs; j++)
		{
			int access;
			cudaDeviceCanAccessPeer(&access, i, j);
			if (auto err = cudaGetLastError())
			{
				cout << "P2P Operations failed while disabling : " << cudaGetErrorString(err) << endl;
				return;
			}

			if (access)
			{
				cudaDeviceDisablePeerAccess(j);
				if (auto err = cudaGetLastError())
				{
					cout << "P2P Operations failed while disabling: " << cudaGetErrorString(err) << endl;
					return;
				}
			}
		}
	}
}


void performFactorPairing(int numDevices, int &fact_x, int &fact_y)
{

	int i;

	//Check if numDevices is Prime
	bool isPrime = true;

	for (i = 2; i < numDevices / 2; ++i)
	{
		if (numDevices % i == 0)
		{
			isPrime = false;
			break;
		}
	}
	if (isPrime)
	{
		fact_x = numDevices;
		fact_y = 1;
	}
	else
	{
		//Finding the appropriate factor pairs to divide the grid
		for (i = 2; i < numDevices / 2; ++i)
		{
			if (numDevices % i == 0) {
				fact_x = i;
				fact_y = numDevices / i;
			}
		}


	}

}

//===============================================================================================================================

//====================================Creating Topology with the number of Devices available====================================

void generateGPUGRID(int numDevices, int &numberOfDevicesAlong_X, int &numberOfDevicesAlong_Y, int domainDecomType)
{
	//Finding GPU topology along x and y
	//Assumuing total number of devices is a perfect square(To be changed later)
	if(domainDecomType==1) 
	{
		numberOfDevicesAlong_X = numDevices;
		numberOfDevicesAlong_Y = 1;
	}

	else
	{
		int val = -1;
		val = (int)sqrt(numDevices);

		if ((val*val) == numDevices)
		{
			numberOfDevicesAlong_X = val;
			numberOfDevicesAlong_Y = val;
		}
			else
		{
			int fact_x = 1;
			int fact_y = 1;
			performFactorPairing(numDevices, fact_x, fact_y);
			numberOfDevicesAlong_X = fact_x;
			numberOfDevicesAlong_Y = fact_y;
		}
	}
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
	for (int gridCount_Y = 0; gridCount_Y < numberOfDevicesAlong_Y; gridCount_Y++) {
		for (int gridCount_X = 0; gridCount_X < numberOfDevicesAlong_X; gridCount_X++) {
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
void initHalos2D(create_Device &device, int chunk_X, int chunk_Y, float *vec_in, int maxdevicesAlong_X, int maxDevicesAlong_Y, int rowStartPos, int rowEndPos, int dim) 
{

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
	if ((device.devicePosition_X - 1) >= 0) {
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

	if ((device.devicePosition_X + 1) < maxdevicesAlong_X) {
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
	if ((device.devicePosition_Y - 1) >= 0) {
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
	if ((device.devicePosition_Y + 1) < maxDevicesAlong_Y) {
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
	int devID = (devCoord_y * numberofDevicesAlong_X) + devCoord_x;
	return devID;
}


void exchangehalos_onHost(int numDevices, vector<create_Device> &deviceArray, int numberofDevicesAlong_X)
{
	//Halos exist in pairs so:
	//Important: A device exchanges North-to-South Pairs and East-to-West Pairs only. Not South-to-North pairs and West-to-East pairs
	//That way the number of exchanges are kept to minimum


	for (int dev = 0; dev < numDevices; dev++)
	{
		int getDevCoord_X = deviceArray[dev].devicePosition_X;
		int getDevCoord_Y = deviceArray[dev].devicePosition_Y;


		//Check if device is having a north Halo buffer
		if (deviceArray[dev].nHalo_flag == 1) {
			int devIDtoNorth = getDeviceIDfromCoord(getDevCoord_X , getDevCoord_Y+1, numberofDevicesAlong_X);
			//Exchange Halos 
		
			(deviceArray[dev].nHalo).swap(deviceArray[devIDtoNorth].sHalo);
		}

		//Check if device is having a east Halo buffer
		if (deviceArray[dev].eHalo_flag == 1) {
			int devIDtoEast = getDeviceIDfromCoord(getDevCoord_X+1, getDevCoord_Y , numberofDevicesAlong_X);
			//Exchange Halos 
			(deviceArray[dev].eHalo).swap(deviceArray[devIDtoEast].wHalo);
		}
	}

}


bool exchangehalos_onHostPinned(int numDevices, vector<create_Device> &deviceArray, int numberofDevicesAlong_X, vector<float*> &nHalosPinned, vector<float*> &sHalosPinned, vector<float*> &eHalosPinned, vector<float*> &wHalosPinned)
{
	//Halos exist in pairs so:
	//Important: A device exchanges North-to-South Pairs and East-to-West Pairs only. Not South-to-North pairs and West-to-East pairs
	//That way the number of exchanges are kept to minimum


	for (int dev = 0; dev < numDevices; dev++)
	{
		int getDevCoord_X = deviceArray[dev].devicePosition_X;
		int getDevCoord_Y = deviceArray[dev].devicePosition_Y;


		//Check if device is having a north Halo buffer
		if (deviceArray[dev].nHalo_flag == 1) {
			int devIDtoNorth = getDeviceIDfromCoord(getDevCoord_X , getDevCoord_Y+1, numberofDevicesAlong_X);
			//Exchange Halos 
			//(deviceArray[dev].nHalo).swap(deviceArray[devIDtoNorth].sHalo);
			swap(nHalosPinned[dev], sHalosPinned[devIDtoNorth]);
		}

		//Check if device is having a east Halo buffer
		if (deviceArray[dev].eHalo_flag == 1) {
			int devIDtoEast = getDeviceIDfromCoord(getDevCoord_X+1, getDevCoord_Y, numberofDevicesAlong_X);
			//Exchange Halos 
			//(deviceArray[dev].eHalo).swap(deviceArray[devIDtoEast].wHalo);
			swap(eHalosPinned[dev], wHalosPinned[devIDtoEast]);
		}
	}
	return true;

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
	//int dataStartPos_X = (devicePosX * numberOfDevicesAlong_X * chunk_X * chunk_Y) + (devicePosY * chunk_X);
	int dataStartPos_X = (devicePosY * dim * chunk_Y) + (devicePosX * chunk_X);
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
		rowEndPos = rowStartPos + chunk_X;
	}

	//Printing when the last device computation is done: Remove the check to check computation for each device
	int deviceID = getDeviceIDfromCoord(devicePosition_X, devicePosition_Y, numberOfDevicesAlong_X);
	if ((deviceID == (numDevices - 1)) && (currentIteration == (numberOfTotalIterations - 1)))
	{
		ofstream myfile;
		myfile.open("data2.txt");
		//Printing the values here
		for (int i = totalSize; i > 0; i--) {

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
	int numDevices = -1;
	cudaGetDeviceCount(&numDevices);
	//numDevices = 2;
	



	//Set Decomposition dimension 1D or 2D: when decomposition is 0. Computation happens on a single GPU
	int decom_Dim = 2;

	//Set Values for Domain Decompostion type 1D or 2D
	int domainDecom_Dim = decom_Dim;

	//Read the custom config defined in file "multiGPUConfig.xml"
	getConfiguration("multiGPUConfig.xml", numDevices, domainDecom_Dim);


	cout << endl << "Total number of Devices in the System are :  " << numDevices << endl;

	getAllDeviceProperties();

	
	//Creating a Default topology mapping (without optimization) with the number of Devices specified in the configuration
	map<int, int> gpuTopology;
	for (int dev=0;dev<numDevices;dev++)
	{
		gpuTopology[dev] = dev;
	}


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

	
	generateGPUGRID(numDevices, numberOfDevicesAlong_X, numberOfDevicesAlong_Y, domainDecom_Dim);
	cout << "GPU grid structure is : " << numberOfDevicesAlong_X << " X " << numberOfDevicesAlong_Y << endl;

	

	//Total elements along each dim in 2D
	int chunk_X = dim / numberOfDevicesAlong_X;
	int chunk_Y = dim / numberOfDevicesAlong_Y;

	/* Creating a GPU topology with multiple devices*/
	createTopology(numDevices, deviceArray, numberOfDevicesAlong_X, numberOfDevicesAlong_Y);

	/*Optimize topology 1D or 2D. Depending upon the required domain division */
	

	if (domainDecom_Dim == 1)
	{
		gpuTopology=outputLatencyMatrix(numDevices, false, numberOfDevicesAlong_X, numberOfDevicesAlong_Y, domainDecom_Dim);
	}
	else//for 2D domain decomposition or higher
	{
		gpuTopology = outputLatencyMatrix(numDevices, false, numberOfDevicesAlong_X, numberOfDevicesAlong_Y, domainDecom_Dim);
	}

	//Enable Peer-to-Peer access across all GPUs : Done on phase 2 of development
	bool p2penabled = false;
	p2penabled = enableP2P(numDevices);



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

	//Device Buffers for parallel communication using streams: Concept of Front and Back Buffer Oct 30, 2017
	std::vector<float*>x_buffer_north(numDevices);
	std::vector<float*>x_buffer_south(numDevices);
	std::vector<float*>y_buffer_west(numDevices);
	std::vector<float*>y_buffer_east(numDevices);

	//Note: Using Pinned memory on Host for Halos -> Performance Approach 1


	vector<float*>nHalo_pinned(numDevices);
	vector<float*>sHalo_pinned(numDevices);
	vector<float*>wHalo_pinned(numDevices);
	vector<float*>eHalo_pinned(numDevices);

	for (int dev = 0; dev < numDevices; dev++)
	{
		cudaSetDevice(gpuTopology[dev]);

		cudaMallocHost((void**)&nHalo_pinned[dev], (chunk_X) * sizeof(float));
		cudaMallocHost((void**)&sHalo_pinned[dev], (chunk_X) * sizeof(float));
		cudaMallocHost((void**)&wHalo_pinned[dev], (chunk_Y) * sizeof(float));
		cudaMallocHost((void**)&eHalo_pinned[dev], (chunk_Y) * sizeof(float));

	}




	for (int dev = 0; dev < numDevices; dev++)
	{
		//Setting the device before allocation
		cudaSetDevice(gpuTopology[dev]);


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

		//Buffer memory used for p2p exchange
		cudaMalloc((void**)&x_buffer_north[dev], chunk_X * sizeof(float));
		cudaMalloc((void**)&x_buffer_south[dev], chunk_X * sizeof(float));
		cudaMalloc((void**)&y_buffer_west[dev], chunk_Y * sizeof(float));
		cudaMalloc((void**)&y_buffer_east[dev], chunk_Y * sizeof(float));

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

			//cout << endl << "For Device ID " << deviceArray[dev].deviceID << endl;
			//cout << endl << "Device pos X " << devicePosX << endl;
			//cout << endl << "Device pos Y " << devicePosY << endl;
			//cout << endl << "Chunk X " << chunk_X << endl;
			//cout << endl << "Chunk Y " << chunk_Y << endl;

			//cout << endl << "Number of device along X " << numberOfDevicesAlong_X << endl;
			//cout << endl << "Number of device along Y " << numberOfDevicesAlong_Y << endl;

			//Calculating data position based on device coords
			//numberOfDevicesAlong_X * Chunk_X * Chunk_Y : finds out the  total data per row of GPUs allocated
			//int dataStartPos_X = (devicePosX * numberOfDevicesAlong_X * chunk_X * chunk_Y) + (devicePosY * chunk_X);
			int dataStartPos_X =  (devicePosY * dim * chunk_Y) + (devicePosX * chunk_X);
			int dataEndPos_X = dataStartPos_X + chunk_X;


			//cout << endl << "Data Start Pos is " << dataStartPos_X << endl;
			//cout << endl << "Data End Pos is " << dataEndPos_X << endl;

			//One complete row across all GPU is dim in order to get the next element above an element we add (currentPosition + dim )
			int rowStartPos = dataStartPos_X;
			int rowEndPos = dataEndPos_X;
			int indexCounter = 0;
			//Initialize Halos
			initHalos2D(deviceArray[dev], chunk_X, chunk_Y, &vec_in[0], numberOfDevicesAlong_X, numberOfDevicesAlong_Y, rowStartPos, rowEndPos - 1, dim);
			

			for (int rowNum = 0; rowNum < chunk_Y; rowNum++)
			{
				//cout << endl << "Data Start Pos is " << rowStartPos << endl;
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

				//cout << endl << "Data End Pos is " << rowEndPos << endl;
				rowStartPos += dim;
				rowEndPos = rowStartPos+chunk_X;
				
			}

			

			//==========Important: Logic for creation of Chunks to be allocated to GPUs Ends ==========================================

			

			//Setting Cuda device
			cudaSetDevice(gpuTopology[dev]);

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
			cudaSetDevice(gpuTopology[dev]);
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


		if (auto err = cudaGetLastError())
		{
			cout << "Halo Copy Failed " << cudaGetErrorString(err) << endl;
			return err;
		}
		


		//Development phase 2 changes : For p2p operation communication initialize buffers

		for (int dev = 0; dev < numDevices; dev++)
		{
			cudaSetDevice(gpuTopology[dev]);
			//Copying Halos to the device
			if (deviceArray[dev].nHalo_flag == 1)
			{
				//cout << "Device ID for nHaloFlag is : " << deviceArray[dev].deviceID<<endl;
				cudaMemcpy(x_buffer_north[dev], &deviceArray[dev].nHalo[0], chunk_X * sizeof(float), cudaMemcpyHostToDevice);
			}
			if (deviceArray[dev].sHalo_flag == 1)
			{
				//cout << "Device ID for sHaloFlag is : " << deviceArray[dev].deviceID << endl;
				cudaMemcpy(x_buffer_south[dev], &deviceArray[dev].sHalo[0], chunk_X * sizeof(float), cudaMemcpyHostToDevice);
			}
			if (deviceArray[dev].eHalo_flag == 1)
			{
				//cout << "Device ID for eHaloFlag is : " << deviceArray[dev].deviceID << endl;
				cudaMemcpy(y_buffer_east[dev], &deviceArray[dev].eHalo[0], chunk_Y * sizeof(float), cudaMemcpyHostToDevice);
			}

			if (deviceArray[dev].wHalo_flag == 1)
			{
				//cout << "Device ID for wHaloFlag is : " << deviceArray[dev].deviceID << endl;
				cudaMemcpy(y_buffer_west[dev], &deviceArray[dev].wHalo[0], chunk_Y * sizeof(float), cudaMemcpyHostToDevice);
			}
		}

	}


	


	
	//=================================Domain Decomposition Logic Ends =================================================================


	//=================================Setting up the grids and blocks for kernel launch================================================
	//int blocksize = -1;
	//int threads = -1;

	int2 myDim;
	myDim.x = chunk_X;
	myDim.y = chunk_Y;

	dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y);
	dim3 grid(DIVRND(myDim.x, BLOCKSIZE_X), DIVRND(myDim.y, BLOCKSIZE_Y));





	//==================================================================================================================================


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

	//===========Algorithm Improvement: Identify the neighbours so that they could be launched together and the exchange can take place. Without having to wait for computation across all devices============================

	//cudaStream_t streams[4];//Possible to declare it dynamically ? Yes. Using Vectors.
	vector<cudaStream_t> streams(numDevices);

	//Create seperate streams for each Halo Exchange
	vector<cudaStream_t> nHaloExchange(numDevices);
	vector<cudaStream_t> sHaloExchange(numDevices);
	vector<cudaStream_t> eHaloExchange(numDevices);
	vector<cudaStream_t> wHaloExchange(numDevices);
	//cudaStream_t nHaloExchange[4];
	//cudaStream_t sHaloExchange[4];
	//cudaStream_t eHaloExchange[4];
	//cudaStream_t wHaloExchange[4];

	

	//Note: Default stream for a device is always syncronizing so creating seperate streams for each device
	for (int i = 0; i < numDevices; i++)
	{
		cudaSetDevice(gpuTopology[i]);
		cudaStreamCreate(&streams[i]);
		if (p2penabled) {
			cudaStreamCreate(&nHaloExchange[i]);
			cudaStreamCreate(&sHaloExchange[i]);
			cudaStreamCreate(&eHaloExchange[i]);
			cudaStreamCreate(&wHaloExchange[i]);
		}

	}


	//For explicit synchornizing p2p transfers and async memcopies
	//cudaEvent_t events[4];
	vector<cudaEvent_t> events(numDevices);

	vector<cudaEvent_t> nHaloEvent(numDevices);
	vector<cudaEvent_t> sHaloEvent(numDevices);
	vector<cudaEvent_t> eHaloEvent(numDevices);
	vector<cudaEvent_t> wHaloEvent(numDevices);

	//cudaEvent_t nHaloEvent[4];
	//cudaEvent_t sHaloEvent[4];
	//cudaEvent_t eHaloEvent[4];
	//cudaEvent_t wHaloEvent[4];

	for (int i = 0; i < numDevices; i++)
	{
		cudaSetDevice(gpuTopology[i]);
		cudaEventCreate(&events[i]);
		if (p2penabled) {
			cudaEventCreate(&nHaloEvent[i]);
			cudaEventCreate(&sHaloEvent[i]);
			cudaEventCreate(&eHaloEvent[i]);
			cudaEventCreate(&wHaloEvent[i]);
		}

	}




	/*Using a pagable memory first*/
	//std::vector<float> partial_resultOnHost(chunk_X * chunk_Y);

	/*Using a pinned(page locked) memory for performance*/
	vector<float*>partial_resultOnHost(numDevices);
	for (int dev = 0; dev < numDevices; dev++)
	{
		cudaSetDevice(gpuTopology[dev]);
		cudaMallocHost((void**)&partial_resultOnHost[dev], (chunk_X * chunk_Y) * sizeof(float));

	}



	//For OMP data race prevention pointer copies for Halos. TODO: change to vector of pointers

	

	vector<float*> nhalo_ptr(numDevices);
	vector<float*> shalo_ptr(numDevices);
	vector<float*> ehalo_ptr(numDevices);
	vector<float*> whalo_ptr(numDevices);




	vector<float*> x_buffer_north_ptr_write(numDevices);
	vector<float*> x_buffer_south_ptr_write(numDevices);
	vector<float*> y_buffer_east_ptr_write(numDevices);
	vector<float*> y_buffer_west_ptr_write(numDevices);

	//initialize the ptr to null

	for (int dev = 0; dev < numDevices; dev++)
	{
		//For OMP thread safety
		nhalo_ptr[dev] = d_nhalos[dev];
		shalo_ptr[dev] = d_shalos[dev];
		ehalo_ptr[dev] = d_ehalos[dev];
		whalo_ptr[dev] = d_whalos[dev];


		x_buffer_north_ptr_write[dev]=x_buffer_north[dev];
		x_buffer_south_ptr_write[dev]=x_buffer_south[dev];
		y_buffer_east_ptr_write[dev]= y_buffer_east[dev];
		y_buffer_west_ptr_write[dev]= y_buffer_west[dev];

	} 

	//==============================================================
	//Check performance

	cudaError_t status = cudaGetLastError();

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	#pragma omp parallel num_threads(numDevices) 
	{
		int dev = omp_get_thread_num();
		
		//cudaSetDevice(omp_get_thread_num());

		for (int i = 0; i < iterations; i++)
		{
			cudaSetDevice(gpuTopology[dev]);
			//#pragma omp barrier

			if ((i>0))
			{
				//As this is not a run on a single host thread race conditions occurs. So have to manage the pointer swapping by creating a copy

				//Check if device is having a north Halo buffer
				if (deviceArray[dev].nHalo_flag == 1)
				{
					swap(x_buffer_north[dev], d_nhalos[dev]);

				}

				//Check if device is having a south Halo buffer
				if (deviceArray[dev].sHalo_flag == 1)
				{

					swap(x_buffer_south[dev], d_shalos[dev]);

				}


				//Check if device is having a east Halo buffer
				if (deviceArray[dev].eHalo_flag == 1)
				{

					swap(y_buffer_east[dev], d_ehalos[dev]);
				}


				//Check if device is having a west Halo buffer
				if (deviceArray[dev].wHalo_flag == 1)
				{
					swap(y_buffer_west[dev], d_whalos[dev]);
				}

			}


			#pragma omp barrier
			jacobi_Simple << <grid, block, 0, streams[dev] >> >(d_A0[dev], d_A1[dev], d_A2[dev], d_A3[dev], d_A4[dev], d_Vec_In[dev], d_Vec_Out[dev], d_Rhs[dev], deviceArray[dev].eHalo_flag, deviceArray[dev].wHalo_flag, deviceArray[dev].nHalo_flag, deviceArray[dev].sHalo_flag, d_ehalos[dev], d_whalos[dev], d_nhalos[dev], d_shalos[dev], deviceArray[dev].deviceID, numDevices, decom_Dim, myDim);

			//For Synchronizing while Halo Exchange start
			cudaEventRecord(events[dev], streams[dev]);
	

			if (i == (iterations - 1))//Copy the results just for the final iteration
			{
				cudaMemcpyAsync(&partial_resultOnHost[dev][0], d_Vec_Out[dev], domainDivision[dev] * sizeof(float), cudaMemcpyDeviceToHost, streams[dev]);
				continue;
			}

			
			swap(d_Vec_In[dev], d_Vec_Out[dev]);


			//Store Halo positions after iteration for exchanging
			if (!p2penabled) {

				if (numDevices > 1)
				{
					if (deviceArray[dev].nHalo_flag == 1)
					{
						cudaStreamWaitEvent(nHaloExchange[dev], events[dev], 0);
						cudaMemcpyAsync(nHalo_pinned[dev], d_nhalos[dev], chunk_X * sizeof(float), cudaMemcpyDeviceToHost, nHaloExchange[dev]);
						if (auto err = cudaGetLastError())
						{
							cout << "d_nhalos copy failed D2H: " << cudaGetErrorString(err) << endl;
							//return err;
						}
					}
					if (deviceArray[dev].sHalo_flag == 1)
					{
						cudaStreamWaitEvent(sHaloExchange[dev], events[dev], 0);
						cudaMemcpyAsync(sHalo_pinned[dev], d_shalos[dev], chunk_X * sizeof(float), cudaMemcpyDeviceToHost, sHaloExchange[dev]);
						if (auto err = cudaGetLastError())
						{
							cout << "d_shalos copy failed D2H: " << cudaGetErrorString(err) << endl;
							//return err;
						}
					}
					if (deviceArray[dev].eHalo_flag == 1)
					{
						cudaStreamWaitEvent(eHaloExchange[dev], events[dev], 0);
						cudaMemcpyAsync(eHalo_pinned[dev], d_ehalos[dev], chunk_Y * sizeof(float), cudaMemcpyDeviceToHost, eHaloExchange[dev]);
						if (auto err = cudaGetLastError())
						{
							cout << "d_ehalos copy failed D2H: " << cudaGetErrorString(err) << endl;
							//return err;
						}
					}
					if (deviceArray[dev].wHalo_flag == 1)
					{
						cudaStreamWaitEvent(wHaloExchange[dev], events[dev], 0);
						cudaMemcpyAsync(wHalo_pinned[dev], d_whalos[dev], chunk_Y * sizeof(float), cudaMemcpyDeviceToHost, wHaloExchange[dev]);
						if (auto err = cudaGetLastError())
						{
							cout << "d_whalos copy failed D2H " << cudaGetErrorString(err) << endl;
							//return err;
						}
					}

				}
			}



			if (auto err = cudaGetLastError())
			{
				cout << "Data copy failed 2: " << cudaGetErrorString(err) << endl;
				//return err;
			}




			//Exchange Halos after each iteration except the last iteration
			if ((i < (iterations - 1)))
			{
				//cudaStreamSynchronize(streams[dev]);
				//cudaDeviceSynchronize();

				if ((!p2penabled)) {


					if (auto err = cudaGetLastError())
					{
						cout << "Stream " << dev << " synchronize error  for iteration : " << i << ". ERROR IS: " << cudaGetErrorString(err) << endl;
						//return err;
					}


					bool exchangeComplete = false;
					//Note: Using Pinned memory on Host for Halos -> Performance Approach 1

					//exchangehalos_onHost(numDevices, deviceArray, numberOfDevicesAlong_X);
					exchangeComplete = exchangehalos_onHostPinned(numDevices, deviceArray, numberOfDevicesAlong_X, nHalo_pinned, sHalo_pinned, eHalo_pinned, wHalo_pinned);
					if (exchangeComplete) {
						for (int dev = 0; dev < numDevices; dev++)
						{
							//Swap input output vectors for all devices
							swap(d_Vec_In[dev], d_Vec_Out[dev]);

							cudaSetDevice(gpuTopology[dev]);
							//Copying Halos to the device
							if (deviceArray[dev].nHalo_flag == 1)
							{
								cudaMemcpyAsync(d_nhalos[dev], nHalo_pinned[dev], chunk_X * sizeof(float), cudaMemcpyHostToDevice, nHaloExchange[dev]);
							}
							if (auto err = cudaGetLastError())
							{
								cout << "d_nhalos copy failed H2D: " << cudaGetErrorString(err) << endl;
								//return err;
							}
							if (deviceArray[dev].sHalo_flag == 1)
							{
								cudaMemcpyAsync(d_shalos[dev], sHalo_pinned[dev], chunk_X * sizeof(float), cudaMemcpyHostToDevice, sHaloExchange[dev]);

							}
							if (auto err = cudaGetLastError())
							{
								cout << "d_shalos copy failed H2D: " << cudaGetErrorString(err) << endl;
								//return err;
							}
							if (deviceArray[dev].eHalo_flag == 1)
							{
								cudaMemcpyAsync(d_ehalos[dev], eHalo_pinned[dev], chunk_Y * sizeof(float), cudaMemcpyHostToDevice, eHaloExchange[dev]);
							}
							if (auto err = cudaGetLastError())
							{
								cout << "d_ehalos copy failed H2D: " << cudaGetErrorString(err) << endl;
								//return err;
							}
							if (deviceArray[dev].wHalo_flag == 1)
							{
								cudaMemcpyAsync(d_whalos[dev], wHalo_pinned[dev], chunk_Y * sizeof(float), cudaMemcpyHostToDevice, wHaloExchange[dev]);
							}
							if (auto err = cudaGetLastError())
							{
								cout << "d_whalos copy failed H2D: " << cudaGetErrorString(err) << endl;
								//return err;
							}
						}
					}
				}

				else {

					//============Important: Before copying to buffers make sure the kernel on the respective GPU(s) finished execution using cudaStreamWaitEvent=======================

					int getDevCoord_X = deviceArray[dev].devicePosition_X;
					int getDevCoord_Y = deviceArray[dev].devicePosition_Y;


					
					nhalo_ptr[dev] = d_nhalos[dev];
					shalo_ptr[dev] = d_shalos[dev];
					ehalo_ptr[dev] = d_ehalos[dev];
					whalo_ptr[dev] = d_whalos[dev];

					x_buffer_north_ptr_write[dev] = x_buffer_north[dev];
					x_buffer_south_ptr_write[dev] = x_buffer_south[dev];
					y_buffer_east_ptr_write[dev] = y_buffer_east[dev];
					y_buffer_west_ptr_write[dev] = y_buffer_west[dev];
					

					#pragma omp barrier // Important: To make sure all threads assign proper values to duplicate pointers before Halo Exchange Begins
					
					


					//Check if device is having a north Halo buffer
					if (deviceArray[dev].nHalo_flag == 1)
					{
						cudaSetDevice(gpuTopology[dev]);
						int devIDtoNorth = getDeviceIDfromCoord(getDevCoord_X, getDevCoord_Y+1, numberOfDevicesAlong_X);
						//Exchange Halos 

						//Send to the device
						cudaStreamWaitEvent(nHaloExchange[dev], events[dev], 0);

						cudaMemcpyPeerAsync(x_buffer_south[devIDtoNorth], devIDtoNorth, d_nhalos[dev], dev, chunk_X * sizeof(float), nHaloExchange[dev]);
						cudaEventRecord(nHaloEvent[dev], nHaloExchange[dev]);

						//Postpone the next iteration kernel execution till the p2p transfers complete
						cudaSetDevice(gpuTopology[devIDtoNorth]);
						cudaStreamWaitEvent(streams[devIDtoNorth], nHaloEvent[dev], 0);


					}

					//Check if device is having a south Halo buffer
					if (deviceArray[dev].sHalo_flag == 1)
					{

						cudaSetDevice(gpuTopology[dev]);

						int devIDtoSouth = getDeviceIDfromCoord(getDevCoord_X, getDevCoord_Y-1, numberOfDevicesAlong_X);
						//Exchange Halos 

						//Send to the device
						cudaStreamWaitEvent(sHaloExchange[dev], events[dev], 0);

						cudaMemcpyPeerAsync(x_buffer_north[devIDtoSouth], devIDtoSouth, d_shalos[dev], dev, chunk_X * sizeof(float), sHaloExchange[dev]);
						cudaEventRecord(sHaloEvent[dev], sHaloExchange[dev]);

						//Postpone the next iteration kernel execution till the p2p transfers complete
						cudaSetDevice(gpuTopology[devIDtoSouth]);
						cudaStreamWaitEvent(streams[devIDtoSouth], sHaloEvent[dev], 0);


					}


					//Check if device is having a east Halo buffer
					if (deviceArray[dev].eHalo_flag == 1)
					{
						cudaSetDevice(gpuTopology[dev]);

						int devIDtoEast = getDeviceIDfromCoord(getDevCoord_X+1, getDevCoord_Y, numberOfDevicesAlong_Y);
						//Exchange Halos 

						//Send to the device
						cudaStreamWaitEvent(eHaloExchange[dev], events[dev], 0);

						cudaMemcpyPeerAsync(y_buffer_west[devIDtoEast], devIDtoEast, d_ehalos[dev], dev, chunk_Y * sizeof(float), eHaloExchange[dev]);
						cudaEventRecord(eHaloEvent[dev], eHaloExchange[dev]);

						//Postpone the next iteration kernel execution till the p2p transfers complete
						cudaSetDevice(gpuTopology[devIDtoEast]);
						cudaStreamWaitEvent(streams[devIDtoEast], eHaloEvent[dev], 0);


					}


					//Check if device is having a west Halo buffer
					if (deviceArray[dev].wHalo_flag == 1)
					{
						cudaSetDevice(gpuTopology[dev]);

						int devIDtoWest = getDeviceIDfromCoord(getDevCoord_X-1, getDevCoord_Y, numberOfDevicesAlong_Y);
						//Exchange Halos 

						//Send to the device
						cudaStreamWaitEvent(wHaloExchange[dev], events[dev], 0);

						cudaMemcpyPeerAsync(y_buffer_east[devIDtoWest], devIDtoWest, d_whalos[dev], dev, chunk_Y * sizeof(float), wHaloExchange[dev]);
						cudaEventRecord(wHaloEvent[dev], wHaloExchange[dev]);

						//Postpone the next iteration kernel execution till the p2p transfers complete
	 					cudaSetDevice(gpuTopology[devIDtoWest]);
						cudaStreamWaitEvent(streams[devIDtoWest], wHaloEvent[dev], 0);

					}

					/*if (auto err = cudaGetLastError())
					{
						cout << "Halo Exchange Error: " << cudaGetErrorString(err) << endl;
					}*/




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

	cout << endl << "Iterations successful. Time taken  in microseconds :" << duration << endl;



	//Sync and Destroy streams and events
	for (int i = 0; i < numDevices; ++i)
	{
		cudaSetDevice(gpuTopology[i]);

		//Destroy Events
		cudaEventDestroy(events[i]);
		cudaEventDestroy(nHaloEvent[i]);
		cudaEventDestroy(sHaloEvent[i]);
		cudaEventDestroy(eHaloEvent[i]);
		cudaEventDestroy(wHaloEvent[i]);


		//Synchro the streams 


		cudaStreamSynchronize(streams[i]);
		cudaStreamDestroy(streams[i]);

		cudaStreamSynchronize(nHaloExchange[i]);
		cudaStreamDestroy(nHaloExchange[i]);

		cudaStreamSynchronize(sHaloExchange[i]);
		cudaStreamDestroy(sHaloExchange[i]);

		cudaStreamSynchronize(eHaloExchange[i]);
		cudaStreamDestroy(eHaloExchange[i]);

		cudaStreamSynchronize(wHaloExchange[i]);
		cudaStreamDestroy(wHaloExchange[i]);
	}

	//Results copied to disk
	for (int dev = 0; dev < numDevices; dev++)
	{
		sendToPrint(&partial_resultOnHost[dev][0], deviceArray[dev].devicePosition_X, deviceArray[dev].devicePosition_Y, numberOfDevicesAlong_X, chunk_X, chunk_Y, dim, size, result, numDevices, iterations - 1, iterations);
	}



	//==========================================Performance using CUDA stream ends===========================================================================

	//Done in phase 2 of development: Disble P2P across devices
	if (p2penabled) {
		disableP2P(numDevices);
	}

	//Free memory on device
	for (int dev = 0; dev < numDevices; dev++)
	{
		cudaSetDevice(gpuTopology[dev]);

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
		cudaFree(x_buffer_south[dev]);
		cudaFree(x_buffer_north[dev]);
		cudaFree(y_buffer_west[dev]);
		cudaFree(y_buffer_east[dev]);
		cudaFreeHost(partial_resultOnHost[dev]);
		cudaFreeHost(nHalo_pinned[dev]);
		cudaFreeHost(sHalo_pinned[dev]);
		cudaFreeHost(wHalo_pinned[dev]);
		cudaFreeHost(eHalo_pinned[dev]);
		cudaDeviceReset();
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


	if (cudaStatus != cudaSuccess) {
		cout << "Cuda Device Reset failed: " << endl;
		return 1;
	}

	return 0;

}

