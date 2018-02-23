#ifndef TESTMULTI_GPU_JACOBI_H
#define TESTMULTI_GPU_JACOBI_H

int performJacobi_MultiGPU(unsigned int dim, unsigned int numJacobiIt, float* A0, float* A1, float* A2, float* A3,float* A4, float* rhs, float* x_in);

#endif