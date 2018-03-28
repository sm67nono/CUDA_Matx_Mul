#ifndef TESTMULTI_GPU2D_JACOBI_DECOM_H
#define TESTMULTI_GPU2D_JACOBI_DECOM_H

int performJacobi_MultiGPU2D_Decom(unsigned int dim, unsigned int numJacobiIt, float* A0, float* A1, float* A2, float* A3, float* A4, float* rhs, float* x_in);

#endif