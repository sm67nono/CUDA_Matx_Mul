#include <cstdio>
#include <vector>
//#include <map>
#include <iostream>
#include "multiGPU_topology_optimize.cuh"


//#include <helper_cuda.h>

using namespace std;

const char *sSampleName = "MultiGPU Latency Matrix analysis and Topology Optimization (Units in microseconds.)";

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

__global__ void delay(int * null) {
  float j=threadIdx.x;
  for(int i=1;i<10000;i++)
      j=(j+1)/j;

  if(threadIdx.x == j) null[0] = j;
}



//Sorting 1D Optimized topology
std::vector<int> getOptimizedTopology1(vector<vector <float> > devices)
{
	vector<int> sorted;
	sorted.resize(devices.size());
	
	//Set availability flags
	vector<int> used;
	used.resize(devices.size());
	
	
	for(int k=0;k<static_cast<int>(devices.size());k++)
	{
		used[k]=0;
	}
	
	//First device is device 0 as the starting point
	sorted[0]=0;
	used[0]=1;

	int currentDevice=0;
		
	for(int i=0;i<static_cast<int>(devices.size()-1);i++)
	{
		//Loop over Each device latency data and find the next device with the lowest  average bidirectional latency
		float lowestLatency=999999.99;
		int nextDevice=-1;
		//j is a placeHolder for the next device
		for(int j=0;j<static_cast<int>(devices[i].size());j++)
		{
			//Dont check for the latency on itself
			if(currentDevice != j && used[j]==0)
			{
				float latencyD2D_outgoing=devices[currentDevice][j];
				float latencyD2D_incoming=devices[j][currentDevice];
				
				//Average current Latency less than the previous latency
				if(((latencyD2D_outgoing+latencyD2D_incoming)/2)<lowestLatency)
				{
					lowestLatency=(latencyD2D_outgoing+latencyD2D_incoming)/2;
					nextDevice=j;
				}
				
			}
		}
		//add to the Device list
		sorted[i+1]=nextDevice;
		//Prevent device reassign in future slots
		used[nextDevice]=1;
		currentDevice=nextDevice;
					
		
	}
		
	return sorted;
}

//Sorting 2D Optimized topology
std::vector<int> getOptimizedTopology2(vector<vector<float> > devices, int numberOfDevicesAlong_X, int numberOfDevicesAlong_Y)
{
	vector<int> sorted;
	sorted.resize(devices.size());
	
	
	//Set availability flags
	vector<int> used;
	used.resize(devices.size());
	for(int k=0;k<static_cast<int>(devices.size());k++)
	{
		used[k]=0;
	}
	
	
	//First device is device 0 as the starting point
	sorted[0]=0;
	used[0]=1;
	//Considering Left and Bottom devices  Averages Bidirectional Latency calculations: In 3D bottom, front and left would be considered
				
	int currentDeviceLeft=0;
	int currentDeviceBottom=0;
	

	
	for(int i=0;i<static_cast<int>(devices.size()-1);i++)
	{
		//Loop over Each device latency data and find the next device with the lowest  average bidirectional latency
		
		float lowestLatencyLeft=999999.99;
		float lowestLatencyBottom=999999.99;
		float combinedLowestLatency=999999.99*2;
		
		int nextDevice=-1;
		//j is a placeHolder for the next device
		for(int j=0;j<static_cast<int>(devices[i].size());j++)
		{
			//Dont check for the latency on itself
			if(currentDeviceLeft != j && currentDeviceBottom != j && used[j]==0)
			{
				//Left
				//Ignoring the first column : Where there is no left device
				if((i%numberOfDevicesAlong_X)!=0)
				{
					float l_latencyD2D_outgoing=devices[currentDeviceLeft][j];
					float l_latencyD2D_incoming=devices[j][currentDeviceLeft];
					
					//Average current Latency less than the previous latency
					if(((l_latencyD2D_outgoing+l_latencyD2D_incoming)/2)<lowestLatencyLeft)
					{
						lowestLatencyLeft=(l_latencyD2D_outgoing+l_latencyD2D_incoming)/2;
						
					}
				
				}
				
				//Bottom
				//Ignoring the last row: Where there is no bottom Device
				if(i>=numberOfDevicesAlong_X)
				{
					
					float b_latencyD2D_outgoing=devices[currentDeviceBottom][j];
					float b_latencyD2D_incoming=devices[j][currentDeviceBottom];
					
					//Average current Latency less than the previous latency
					if(((b_latencyD2D_outgoing+b_latencyD2D_incoming)/2)<lowestLatencyBottom)
					{
						lowestLatencyBottom=(b_latencyD2D_outgoing+b_latencyD2D_incoming)/2;
						
					}
					
					
				}
				
				if(((lowestLatencyBottom+lowestLatencyLeft)/2)<combinedLowestLatency)
				{
					combinedLowestLatency=(lowestLatencyBottom+lowestLatencyLeft)/2;
					nextDevice=j;
				}
				
				
				
				
			}
		}
		//add to the Device list
		sorted[i+1]=nextDevice;
		used[nextDevice]=1;
		
		
		//Set Device Left
		if((i%numberOfDevicesAlong_X)!=0)
		{
			currentDeviceLeft=nextDevice;
		}
		
		//Set Device Bottom
		if(i>=numberOfDevicesAlong_X)
		{
			currentDeviceBottom=sorted[(i-numberOfDevicesAlong_X)+1];
		}			
			
	}
	return sorted;
}




map<int,int> outputLatencyMatrix(int numGPUs, bool p2p, int numberOfDevicesAlong_X, int numberOfDevicesAlong_Y, int domainDecom_Dim)
{
	cout<<endl<<sSampleName<<endl;
    int repeat=10000;
    vector<int *> buffers(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(d);
        cudaMalloc(&buffers[d],1);
        cudaCheckError();
        cudaEventCreate(&start[d]);
        cudaCheckError();
        cudaEventCreate(&stop[d]);
        cudaCheckError();
    }

    vector<double> latencyMatrix(numGPUs*numGPUs);

    for (int i=0; i<numGPUs; i++)
    {
        cudaSetDevice(i);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if(p2p) {
                cudaDeviceCanAccessPeer(&access,i,j);
                if (access)
                {
                    cudaDeviceEnablePeerAccess(j,0);
                    cudaCheckError();
                }
            }
            cudaDeviceSynchronize();
            cudaCheckError();
            delay<<<1,1>>>(NULL);
            cudaEventRecord(start[i]);

            for (int r=0; r<repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[i],i,buffers[j],j,1);
            }

            cudaEventRecord(stop[i]);
            cudaDeviceSynchronize();
            cudaCheckError();

            float time_ms;
            cudaEventElapsedTime(&time_ms,start[i],stop[i]);

            latencyMatrix[i*numGPUs+j]=time_ms*1e3/repeat;
            if(p2p && access)
            {
                cudaDeviceDisablePeerAccess(j);
            }
        }
    }

    printf("   D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        printf("%6d ", j);
    }

    printf("\n");

	
	//Each Device stores the latency vector to other devices
	std::vector< vector<float> >device(numGPUs, vector<float> (numGPUs));

    for (int i=0; i<numGPUs; i++)
    {
		//Select GPUs here.
        printf("%6d ",i);

        for (int j=0; j<numGPUs; j++)
        {
			//Add latency to other GPUs here
			device[i][j]=latencyMatrix[i*numGPUs+j];
			
            printf("%6.02f ", device[i][j]);
        }

        printf("\n");
    }
	
	//Test matrix 8x8 for 8 GPUs
	
	//P2P=Enabled Latency Matrix (us)
	//D\D     0      1      2      3      4      5      6      7 
     //0   6.18  21.22  21.36  25.07  37.26  41.03  26.83  41.13 
     //1  27.48   6.09  21.28  24.82  37.67  40.89  26.81  40.91 
     //2  27.53  21.44   6.13  24.95  37.73  40.88  26.53  40.92 
     //3  27.51  21.38  21.16   5.77  38.09  41.10  27.16  41.10 
     //4  40.41  33.29  33.10  36.86   6.45  24.77   9.97  28.11 
     //5  40.42  33.15  32.84  36.62  24.83   6.07   9.93  28.17 
     //6  39.25  33.27  32.99  37.13  24.93  24.93   6.07  28.08 
     //7  40.48  33.66  33.53  36.75  24.85  24.90  10.05   6.37
	 
	 
	 //2D - 4 x 2 order should be
	 
	 
	 
	 
	/* std::vector< vector<float> >deviceTest(8, vector<float> (8));
	 
	 
	 
	 deviceTest[0][0]=6.18;
	 deviceTest[0][1]=21.22;
	 deviceTest[0][2]=21.36;
	 deviceTest[0][3]=25.07;
	 deviceTest[0][4]=37.26;
	 deviceTest[0][5]=41.03;
	 deviceTest[0][6]=26.83;
	 deviceTest[0][7]=41.13;
	 
	 deviceTest[1][0]=27.48;
	 deviceTest[1][1]=6.09;
	 deviceTest[1][2]=21.28;
	 deviceTest[1][3]=24.82;
	 deviceTest[1][4]=37.67;
	 deviceTest[1][5]=40.89;
	 deviceTest[1][6]=26.81;
	 deviceTest[1][7]=40.91;
	 
	 deviceTest[2][0]=27.53;
	 deviceTest[2][1]= 21.44;
	 deviceTest[2][2]=6.13;
	 deviceTest[2][3]=24.95;
	 deviceTest[2][4]=37.73;
	 deviceTest[2][5]=40.88;
	 deviceTest[2][6]=26.53;
	 deviceTest[2][7]=40.92;
	 
	 deviceTest[3][0]=27.51;
	 deviceTest[3][1]=21.38;
	 deviceTest[3][2]=21.16;
	 deviceTest[3][3]=5.77;
	 deviceTest[3][4]=38.09;
	 deviceTest[3][5]=41.10;
	 deviceTest[3][6]=27.16;
	 deviceTest[3][7]=41.10;
	 
	 
	 //4  40.41  33.29  33.10  36.86   6.45  24.77   9.97  28.11 
     //5  40.42  33.15  32.84  36.62  24.83   6.07   9.93  28.17 
     //6  39.25  33.27  32.99  37.13  24.93  24.93   6.07  28.08 
     //7  40.48  33.66  33.53  36.75  24.85  24.90  10.05   6.37
	 
	 deviceTest[4][0]=40.41;
	 deviceTest[4][1]=33.29;
	 deviceTest[4][2]=33.10;
	 deviceTest[4][3]=36.86;
	 deviceTest[4][4]=6.45;
	 deviceTest[4][5]=24.77;
	 deviceTest[4][6]=9.97;
	 deviceTest[4][7]=28.11;
	 
	 deviceTest[5][0]=40.42;
	 deviceTest[5][1]=33.15;
	 deviceTest[5][2]=32.84;
	 deviceTest[5][3]=36.62;
	 deviceTest[5][4]=24.83;
	 deviceTest[5][5]=6.07;
	 deviceTest[5][6]=9.93;
	 deviceTest[5][7]=28.17;
	 
	 deviceTest[6][0]=39.25;
	 deviceTest[6][1]=33.27;
	 deviceTest[6][2]=32.99;
	 deviceTest[6][3]=37.13;
	 deviceTest[6][4]=24.93;
	 deviceTest[6][5]=24.93;
	 deviceTest[6][6]=6.07;
	 deviceTest[6][7]=28.08;
	 
	 
	 deviceTest[7][0]= 40.48;
	 deviceTest[7][1]=33.66;
	 deviceTest[7][2]=33.53;
	 deviceTest[7][3]=36.75;
	 deviceTest[7][4]=24.85;
	 deviceTest[7][5]=24.90;
	 deviceTest[7][6]=10.05;
	 deviceTest[7][7]=6.37;*/
	
	 
	
	
	//Sort The Devices according to best performance
	map<int,int> optimized;
	//Initialize
	for(int init=0;init<numGPUs;init++)
	{
		optimized[init]=init;
	}
	
	
	
	if(domainDecom_Dim==1)
	{
		vector<int> getSortOrder_1D=getOptimizedTopology1(device);
		cout<<"1D Sort Order"<<endl;
	
		for(int j=0;j<static_cast<int>(getSortOrder_1D.size());j++)
		{
			//Dont check for the latency on itself
			cout<<"Device "<<j+1<<"   "<<getSortOrder_1D[j]<<endl;
			optimized[j]=getSortOrder_1D[j];
		}
	}
	else//For 2D domain and higher
	{
		vector<int> getSortOrder_2D=getOptimizedTopology2(device,numberOfDevicesAlong_X,numberOfDevicesAlong_Y);
		cout<<"2D Sort Order"<<endl;
		for(int j=0;j<static_cast<int>(getSortOrder_2D.size());j++)
		{
			//Dont check for the latency on itself
			cout<<"Device "<<j+1<<"   "<<getSortOrder_2D[j]<<endl;
			optimized[j]=getSortOrder_2D[j];
		}
	}
	
	
	
	
		

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(d);
        cudaFree(buffers[d]);
        cudaCheckError();
        cudaEventDestroy(start[d]);
        cudaCheckError();
        cudaEventDestroy(stop[d]);
        cudaCheckError();
    }
	
	
	
	return optimized;
}

/*int main(int argc, char **argv)
{

    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    printf("[%s]\n", sSampleName);

    //output devices
    for (int i=0; i<numGPUs; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);
        printf("Device: %d, %s, pciBusID: %x, pciDeviceID: %x, pciDomainID:%x\n",i,prop.name, prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
    }

    checkP2Paccess(numGPUs);

    //Check peer-to-peer connectivity
    printf("P2P Connectivity Matrix\n");
    printf("     D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        printf("%6d", j);
    }
    printf("\n");

    for (int i=0; i<numGPUs; i++)
    {
        printf("%6d\t", i);
        for (int j=0; j<numGPUs; j++)
        {
            if (i!=j)
            {
               int access;
               cudaDeviceCanAccessPeer(&access,i,j);
               printf("%6d", (access) ? 1 : 0);
            }
            else
            {
                printf("%6d", 1);
            }
        }
        printf("\n");
    }

 

    printf("P2P=Enabled Latency Matrix (us)\n");
    outputLatencyMatrix(numGPUs, true);

   
    exit(EXIT_SUCCESS);
}*/
