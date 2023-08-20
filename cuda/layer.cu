#include "layer.h"
#include "activation.h"
#include "helpers.h"
#include <chrono>

Layer::Layer(int u, const char* activation_string){
    cublasCreate(&handle);
    units = u;
    init = false;
    activation = activation_string;
    activation_f.emplace("sigmoid", &Activation::sigmoidActivation);
    deriv_f.emplace("sigmoid", &Activation::reluDerivative);
    activation_f.emplace("relu", &Activation::reluActivation);
    deriv_f.emplace("relu", &Activation::reluDerivative);
        
}

void Layer::forward_prop(double* A, int A_X, int A_Y){
    // Initialize CUDA device memory
    if (!init){
        
        CUDA_CHECK(cudaMalloc(&W, units * A_X * sizeof(double)));
        createRandomMatrix(units, A_X, W);
        CUDA_CHECK(cudaMalloc(&dW, units * A_X * sizeof(double)));
        cudaMemset(dW, 0, units * A_X * sizeof(double));
        CUDA_CHECK(cudaMalloc(&b, units * 1 * sizeof(double)));
        createRandomMatrix(units, 1, b);
        CUDA_CHECK(cudaMalloc(&db, units * 1 * sizeof(double)));
        cudaMemset(db, 0, units * 1 * sizeof(double));
        // Assume train dataset called first and A_Y never greater than initial
        CUDA_CHECK(cudaMalloc(&Z, units * A_Y * sizeof(double)));
        cudaMemset(Z, 0, units * A_Y * sizeof(double));
        CUDA_CHECK(cudaMalloc(&dZ, units * A_Y * sizeof(double)));
        cudaMemset(dZ, 0, units * A_Y * sizeof(double));
        // Create 1-vector
        CUDA_CHECK(cudaMalloc(&one_d, A_Y * sizeof(double)));
        cudaMemset(one_d, 1, A_Y * sizeof(double));
        init = true; 
    }
    // Set sizes
    A_x = A_X;
    // Can set multiple times for different dataset sizes 
    A_y = A_Y;
    // Z = W x A
    double alpha = 1.0;
    double beta = 0.0;
    CHECK_CUBLAS(cublasDgemm(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    units, A_y, A_x,
    &alpha, 
    W, units,
    A, A_x, 
    &beta, 
    Z, units));

    // Z += b x 1-vector (b repeated)
    beta = 1.0;
    CHECK_CUBLAS(cublasDgemm(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    units, A_y, 1,
    &alpha, 
    b, units,
    one_d, 1, 
    &beta, 
    Z, units));
    
    dim3 threads(16,16);
    dim3 blocks((units + threads.x - 1)  / threads.x, (A_y + threads.y - 1)  / threads.y);
    // Activation Kernel
    activation_f[activation]<<<blocks, threads>>>(units, A_y, Z);
    cudaDeviceSynchronize();
}

void Layer::backward_prop(Layer *layer_prev, Layer* layer_next, bool is_first, double *input){
    double alpha = 1.0;
    double beta = 0.0;
    double normal = 1.0/A_y;
    int nBlocks = (units * A_y) / BLOCK_SIZE;
    // Special case for last layer
    if (!is_first){
        // dZ = 2(Y - Z)
        alpha = 2.0;
        beta = -2.0;
        CHECK_CUBLAS(cublasDgeam(handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        units, A_y, 
        &alpha, 
        Z, units,
        &beta,
        input, units,
        dZ, units));

        // dW = 1/m dZ x dZ_x-1
        beta = 0.0;
        CHECK_CUBLAS(cublasDgemm(handle, 
        CUBLAS_OP_N, CUBLAS_OP_T, 
        units, layer_next->units, A_y, 
        &normal, 
        dZ, units, 
        layer_next->Z, layer_next->units, 
        &beta, 
        dW, units)); 
    }
    // Case for all other layers
    else {
        // dZ = W_T x dZ_x+1
        CHECK_CUBLAS(cublasDgemm(handle, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        layer_prev->A_x, layer_prev->A_y, layer_prev->units,
        &alpha, 
        layer_prev->W, layer_prev->units, 
        layer_prev->dZ, layer_prev->units, 
        &beta, 
        dZ, layer_prev->A_x));

        // Calculate derivative of Z
        double *deriv;
        dim3 threads(16,16);
        dim3 blocks((units + threads.x - 1)  / threads.x, (A_y + threads.y - 1)  / threads.y);
        CUDA_CHECK(cudaMalloc(&deriv, sizeof(double) * units * A_y));
        cudaMemset(deriv, 0, sizeof(double) * units * A_y);
        deriv_f[activation]<<<blocks, threads>>>(units, A_y, Z, deriv);
        // dZx *= deriv
        elementwiseMult<<<blocks, threads>>>(units, A_y, dZ, deriv);
        cudaFree(deriv);

        // first layer: dW = 1/m dZ x X
        // else: dW = 1/m dZ x Z_x-1
        CHECK_CUBLAS(cublasDgemm(handle, 
        CUBLAS_OP_N, CUBLAS_OP_T,
        units, A_x, A_y,
        &normal,
        dZ, units,
        input, A_x,
        &beta,
        dW, units));
    }
    // db = 1/m dZ * 1-vector (sum(dZ))
    CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, units, A_y, &normal, dZ, units, one_d, 1, &beta, db, 1)); 
    cudaDeviceSynchronize();
}

void Layer::update_parameters(double learn_rate){
    double alpha = 1.0;
    double beta = -1 * learn_rate;
    // W -= learn rate * dW
    CHECK_CUBLAS(cublasDgeam(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    units, A_x, 
    &alpha, 
    W, units, 
    &beta,
    dW, units,
    W, units));
    // b -= learn rate * db
    CHECK_CUBLAS(cublasDgeam(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    units, 1, 
    &alpha, 
    b, units, 
    &beta,
    db, units, 
    b, units));
    cudaDeviceSynchronize();
}

void Layer::free_all(){
    // free all device memory
    if (init){
        cudaFree(W);
        cudaFree(b);
        cudaFree(Z);
        cudaFree(dW);
        cudaFree(db);
        cudaFree(dZ);
        cudaFree(one_d);
    }
    // mark layer as uninitialized
    init = false;
}