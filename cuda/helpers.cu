#include "helpers.h"

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < m * n && matrix[idx] < 0) {
    if (idx < m * n) {
        matrix[idx] *= matrix2[idx];
    }
}

__global__ void fill_A_B(int m, int n, double* matrix){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        matrix[idx] = matrix[idx] * 2 - 1;
    }
}

void print_matrix(double *mat, int m, int n){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            printf("%f ", min(double(6900000), max(double(-6900000), mat[m * j + i])));
        }
        printf("\n");
    }
}

void print_device(double *dev, int m, int n){
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
    int nBlocks = (m * n)/BLOCK_SIZE;
    fill_A_B<<<512, BLOCK_SIZE>>>(m, n, matrix);
}