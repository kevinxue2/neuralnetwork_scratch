#include "helpers.h"
#include <unistd.h> 
#include <cuda_runtime.h>

void CUDA_CHECK(cudaError_t cudaError) {
    if (cudaError != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(cudaError));
        exit(1);
    }
}

void CHECK_CUBLAS(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS error: %d\n", status);
        exit(1);
    }
}

// Matrix functions

__global__ void elementwiseMult(int m, int n, double* matrix, double* matrix2){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * m + col;
    if (idx < m * n) {
        matrix[idx] *= matrix2[idx];
    }
}

__global__ void fill_A_B(int m, int n, double* matrix){
    // Fill matrix with values between A and B
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * m + col;
    if (idx < m * n) {
        matrix[idx] = matrix[idx] * 2 - 1;
    }
}

void print_matrix(double *mat, int m, int n){
    
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            printf("%f ", min(double(1000), max(double(-1000), mat[m * j + i])));
        }
        printf("\n");
    }
    printf("\n");
}

void print_device(double *dev, int m, int n){
    return;
    double *temp = (double *)calloc(sizeof(double), m * n);
    CUDA_CHECK(cudaMemcpy(temp, dev, sizeof(double) * m * n, cudaMemcpyDeviceToHost));
    print_matrix(temp, m, n);
    free(temp);
}

void createRandomMatrix(int m, int n, double* matrix) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, 123);
    curandGenerateUniformDouble(prng, matrix, m * n);
    curandDestroyGenerator(prng);
    dim3 threads(16,16);
    dim3 blocks((m + threads.x - 1)  / threads.x, (n + threads.y - 1)  / threads.y);
    fill_A_B<<<blocks, threads>>>(m, n, matrix);
}