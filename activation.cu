#include "activation.h"

namespace Activation {
    __global__ void reluActivation(int m, int n, double* matrix){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < m * n && matrix[idx] < 0) {
            matrix[idx] = 0;
        }
    }
    __global__ void sigmoidActivation(int m, int n, double* matrix){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < m * n) {
            matrix[idx] = 1.0f / (1.0f + expf(-matrix[idx]));
        }
    }

    __global__ void sigmoidDerivative(int m, int n, double* matrix, double* deriv){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < m * n) {
            double sig = 1.0f / (1.0f + expf(-matrix[idx]));
            deriv[idx] = sig * (1 - sig);
        }
    }

    __global__ void reluDerivative(int m, int n, double* matrix, double* deriv){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < m * n && matrix[idx] < 0) {
            if (matrix[idx] < 0){
                deriv[idx] = 0;
            }
            else{
                deriv[idx] = 1;
            }
        }
    }
}