#include "model.h"
#include "helpers.h"
#include <chrono>
using namespace std::chrono;

Model::Model(Layer* l, int num_l){
    layers = l;
    num_layers = num_l;
}

void Model::set_data(double *X_p, double* Y_p, int X_size, int data_len){
    X = X_p;
    Y = Y_p;
    // m size of X
    m = X_size;
    n = data_len;
}

void Model::set_Y(double *Z, double* Y_b, int batch_size, int b_num){
    // Transform Y matrix to size 10
    double *temp_Y = (double *)calloc(sizeof(double), 10*batch_size);
    double *Y_h = (double *)calloc(batch_size, sizeof(double));
    CUDA_CHECK(cudaMemcpy(Y_h, Y_b, batch_size * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < batch_size; i++){
        temp_Y[(int)(i*10 + Y_h[i])] = 1.0;
    }
    free(Y_h);
    CUDA_CHECK(cudaMemcpy(Z, temp_Y, batch_size * 10 * sizeof(double), cudaMemcpyHostToDevice));
    free(temp_Y);
}

void Model::fit(int epoch){
    int batch_size = n;
    double* y_hat;
    double *X_batch;
    double *Y_batch;
    // copy dataset from host to deivce memory
    CUDA_CHECK(cudaMalloc(&y_hat, sizeof(double) * 10 * batch_size));
    CUDA_CHECK(cudaMalloc(&X_batch, m * batch_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&Y_batch, batch_size * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(X_batch, X, m * batch_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Y_batch, Y, batch_size * sizeof(double), cudaMemcpyHostToDevice));
    set_Y(y_hat, Y_batch, batch_size, 0);
    
    for (int i = 0; i < epoch; i++){
        printf("Iteration %d\n", i);
        double *x_mat = X_batch;
        int x_size = m;
        // Forward prop all layers
        for (int l = 0; l < num_layers; l++){
            layers[l].forward_prop(x_mat, x_size, batch_size);
            x_mat = layers[l].Z;
            x_size = layers[l].units;
        }
        
        // Back prop all layers in reverse order
        for (int i = num_layers-1; i >= 0; i--){
            // Last layer
            if (i == num_layers-1){
                layers[i].backward_prop(&layers[i+1], &layers[i-1], false, y_hat);
            }
            // First layer - use X instead of next layer
            else if (i == 0) {
                layers[i].backward_prop(&layers[i+1], &layers[i-1], true, X_batch);
            }
            // Middle layers
            else{
                layers[i].backward_prop(&layers[i+1], &layers[i-1], true, layers[i-1].Z);
            }
        }

        // Update weights and bias
        for (int i = 0; i < num_layers; i++){
            layers[i].update_parameters(0.15);
        }
    }
    // Free dataset on device
    CUDA_CHECK(cudaFree(y_hat));
    CUDA_CHECK(cudaFree(X_batch));
    CUDA_CHECK(cudaFree(Y_batch));
}

void Model::predict(double *inp_X, double *inp_Y, int data_len){
    // Todo: 
    double *X_device;
    CUDA_CHECK(cudaMalloc(&X_device, m * data_len * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(X_device, inp_X, m * data_len * sizeof(double), cudaMemcpyHostToDevice));
    double *x_mat = X_device;
    int x_size = m;
    double total = 0;
    for (int l = 0; l < num_layers; l++){
        layers[l].forward_prop(x_mat, x_size, data_len);
        x_mat = layers[l].Z;
        x_size = layers[l].units;
    }
    CUDA_CHECK(cudaFree(X_device));
}

int Model::accuracy(double *Z, double* Y_check, int num){
    // Measure correct predictions
    double *Z_host = (double *)malloc(sizeof(double) * num * layers[num_layers-1].units);
    int temp_max = 0;
    int count = 0;
    CUDA_CHECK(cudaMemcpy(Z_host, Z, sizeof(double) * num * layers[num_layers-1].units, cudaMemcpyDeviceToHost));
    for (int i = 0; i < num; i++){
        temp_max = 0;
        for (int j = 0; j < 10; j++){
            if (Z_host[j + i * 10] > Z_host[temp_max + i * 10]){
                temp_max = j;
            }
        }
        if ((double)temp_max == Y_check[i]){
            count += 1;
        }
        
    }
    free(Z_host);
    return count;
}