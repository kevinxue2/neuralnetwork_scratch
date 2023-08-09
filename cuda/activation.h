#include <cuda_runtime.h>

namespace Activation {
    __global__ void reluActivation(int m, int n, double* matrix);
    
    __global__ void sigmoidActivation(int m, int n, double* matrix);

    __global__ void sigmoidDerivative(int m, int n, double* matrix, double* deriv);

    __global__ void reluDerivative(int m, int n, double* matrix, double* deriv);
}