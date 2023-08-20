#include <curand.h>
#include <iostream>
#include "cublas_v2.h"
#include <cuda_runtime.h>

#ifndef HELPERS_H
#define HELPERS_H
#define BLOCK_SIZE 512

void CUDA_CHECK(cudaError_t cudaError);

void CHECK_CUBLAS(cublasStatus_t status);

// Matrix functions
__global__ void elementwiseMult(int m, int n, double* matrix, double* matrix2);

__global__ void fill_A_B(int m, int n, double* matrix);

__global__ void sumMatrix(int m, int n, double* matrix, float* result);

void print_matrix(double *mat, int m, int n);

void print_device(double *dev, int m, int n);

void createRandomMatrix(int m, int n, double* matrix);
#endif 


