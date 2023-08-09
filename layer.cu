#include "layer.h"
#include "activation.h"
#include "helpers.h"

Layer::Layer(int u, const char* activation_string){
    cublasCreate(&handle);
    units = u;
    init = false;
    activation = activation_string;
    activation_f.emplace("sigmoid", &Activation::sigmoidActivation);
    deriv_f.emplace("sigmoid", &Activation::sigmoidDerivative);
    activation_f.emplace("relu", &Activation::reluActivation);
    deriv_f.emplace("relu", &Activation::reluDerivative);
        
}

void Layer::forward_prop(double* A, int A_X, int A_Y){
    if (!init){
        //init A
        CUDA_CHECK(cudaMalloc(&W, units * A_X * sizeof(double)));
        createRandomMatrix(units, A_X, W);
        //init B
        CUDA_CHECK(cudaMalloc(&b, units * 1 * sizeof(double)));
        createRandomMatrix(units, 1, b);
        CUDA_CHECK(cudaMalloc(&Z, units * A_Y * sizeof(double)));
        cudaMemset(Z, 0, units * A_Y * sizeof(double));
        //deriv
        CUDA_CHECK(cudaMalloc(&dW, units * A_X * sizeof(double)));
        cudaMemset(dW, 0, units * A_X * sizeof(double));
        CUDA_CHECK(cudaMalloc(&db, units * 1 * sizeof(double)));
        cudaMemset(db, 0, units * 1 * sizeof(double));
        CUDA_CHECK(cudaMalloc(&dZ, units * A_Y * sizeof(double)));
        cudaMemset(dZ, 0, units * A_Y * sizeof(double));
        init = true;
    }
    double alpha = 1.0;
    double beta = 1.0;
    A_x = A_X;
    A_y = A_Y;
    printf("%d %d %d\n",units, A_y, A_x );
    CHECK_CUBLAS(cublasDgemm(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    units, A_y, A_x,
    &alpha, 
    W, units,
    A, A_x, 
    &beta, 
    Z, units));
    // printf("dafsdf\n");
    // print_device(W, units, min(10,A_x));
    // printf("dasdfasdfasfsdf\n");
    // print_device(A, min(10,A_x), min(10,A_Y));
    // printf("dafasdfasdfasfafsdafsdf\n");
    // print_device(Z, min(10,A_x), min(10,A_Y));
    for (int r = 0; r < A_y; r++){
        CHECK_CUBLAS(cublasDaxpy(handle, units, &alpha, b, 1, &Z[r*units], 1));
    }
    // activation(z)
    // int nBlocks = (units * A_y) / BLOCK_SIZE;
    activation_f[activation]<<<512, BLOCK_SIZE>>>(units, A_y, Z);
    
}

// function for first and last case
void Layer::backward_prop(Layer *layer_prev, Layer* layer_next, bool is_first, double *input){
    double alpha = 1.0;
    double beta = 1.0;
    double normal = 1.0/A_y;
    int nBlocks = (units * A_y) / BLOCK_SIZE;
    if (!is_first){
        // Y - Z
        beta = -1.0;
        // printf("bp 4\n");
        CHECK_CUBLAS(cublasDgeam(handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        units, A_y, 
        &alpha, 
        Z, units,
        &beta,
        input, units,
        dZ, units));
        // printf("bp 5\n");
        beta = 1.0;
        CHECK_CUBLAS(cublasDgemm(handle, 
        CUBLAS_OP_N, CUBLAS_OP_T, 
        units, layer_next->units, A_y, 
        &normal, 
        dZ, units, 
        layer_next->Z, layer_next->units, 
        &beta, 
        dW, units)); //1/m dzx*dzx-1t
    }
    else {
        // printf("bp 6\n");
        // CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, units, layer_prev->A_y, layer_prev->A_x, &alpha, layer_prev->W, units, layer_prev->dZ, layer_prev->A_y, &beta, dZ, units)); // Wt*dZx+1
        
        CHECK_CUBLAS(cublasDgemm(handle, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        layer_prev->A_x, layer_prev->A_y, layer_prev->units,
        &alpha, 
        layer_prev->W, layer_prev->units, 
        layer_prev->dZ, layer_prev->units, 
        &beta, 
        dZ, layer_prev->A_x)); // Wt*dZx+1
        // calculate deriv
        double *deriv;
        CUDA_CHECK(cudaMalloc(&deriv, sizeof(double) * units * A_y));
        cudaMemset(deriv, 0, sizeof(double) * units * A_y);
        deriv_f[activation]<<<512, BLOCK_SIZE>>>(units, A_y, dZ, deriv);
        // write element-wise
        // print_device(deriv, units, A_y);
        elementwiseMult<<<512, BLOCK_SIZE>>>(units, A_y, dZ, deriv); // dZx*deriv
        
        cudaFree(deriv);
        // printf("bp 7\n");
        CHECK_CUBLAS(cublasDgemm(handle, 
        CUBLAS_OP_N, CUBLAS_OP_T,
        units, A_x, A_y,
        &normal,
        dZ, units,
        input, A_x,
        &beta,
        dW, units)); //1/m dzx*X

    }
    // // db average of output write function*
    // // 1 vector
    double *one_h = (double *)calloc(A_y, sizeof(double));
    double *one_d;
    CUDA_CHECK(cudaMalloc(&one_d, A_y * sizeof(double)));
    cudaMemset(one_d, 0, A_y * sizeof(double));
    for (int i = 0; i < A_y; i++){
        one_h[i] = 1;
    }
    
    CUDA_CHECK(cudaMemcpy(one_d, one_h, A_y * sizeof(double), cudaMemcpyHostToDevice));
    // // // printf("bp 8\n");
    free(one_h);
    //broken
    CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, units, A_y, &normal, dZ, units, one_d, 1, &beta, db, 1)); // 1/'m sum(dz)
    
    cudaFree(one_d);
}

void Layer::update_parameters(double learn_rate){
    //update W
    // printf("here\n");
    double alpha = 1.0;
    double beta = -1 * learn_rate;
    CHECK_CUBLAS(cublasDgeam(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    units, A_x, 
    &alpha, 
    W, units, 
    &beta,
    dW, units,
    W, units));
    //update b
    CHECK_CUBLAS(cublasDgeam(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    units, 1, 
    &alpha, 
    b, units, 
    &beta, 
    db, units, 
    b, units));
    // printf("complete\n");
}

void Layer::free_all(){
    if (init){
        cudaFree(W);
        cudaFree(b);
        cudaFree(Z);
        cudaFree(dW);
        cudaFree(db);
        cudaFree(dZ);
    }
}