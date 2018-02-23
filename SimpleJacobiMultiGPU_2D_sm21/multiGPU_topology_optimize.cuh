#ifndef MULTIGPU_TOPOLOGY_OPTIMIZE_H
#define MULTIGPU_TOPOLOGY_OPTIMIZE_H
#include<map>


std::map<int,int> outputLatencyMatrix(int numGPUs, bool p2p, int numberOfDevicesAlong_X, int numberoOfDevicesAlong_Y, int domainDecom_Dim);

#endif