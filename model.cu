#include "model.h"
#include "helpers.h"

Model::Model(Layer* l, int num_l){
    layers = l;
    num_layers = num_l;
}

void Model::compile(double *X_p, double* Y_p, int X_size, int data_len){
    X = X_p;
    Y = Y_p;
    // m size of X
    m = X_size;
    n = data_len;
}

void Model::set_Y(double *Z, double* Y_b, int batch_size, int b_num){
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
    // n = 512;
    int batch_size = 512;
    int num_batch = 1;
    double *test_h = (double *)calloc(10 * n * sizeof(double),1);
    double* y_hat;
    double *X_batch;
    double *Y_batch;
    CUDA_CHECK(cudaMalloc(&y_hat, sizeof(double) * 10 * batch_size));
    for (int i = 0; i < epoch; i++){
        for (int b = 0; b < num_batch; b++){
            printf("Iteration %d %d\n", i, num_layers);
            CUDA_CHECK(cudaMalloc(&X_batch, m * batch_size * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&Y_batch, batch_size * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(X_batch, &X[m * batch_size * b], m * batch_size * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(Y_batch, &Y[batch_size * b], batch_size * sizeof(double), cudaMemcpyHostToDevice));
            // double *x_mat = X;
            double *x_mat = X_batch;
            int x_size = m;
            printf("Forward prop\n");
            for (int l = 0; l < num_layers; l++){
                printf("Forward prop %d %d\n", l, num_layers);
                layers[l].forward_prop(x_mat, x_size, batch_size);
                x_mat = layers[l].Z;
                x_size = layers[l].units;
                // print_device(layers[l].W, layers[l].units, min(20,layers[l].A_x));
                // print_device(layers[l].b, layers[l].units, 1);
                // print_device(layers[l].Z, layers[l].units, min(20,layers[l].A_y));
            }
            
            set_Y(y_hat, Y_batch, batch_size, b);
            
            // print_device(y_hat, 10, 10);

            printf("Back prop\n");
            cudaError_t cudaStatus;
            for (int i = num_layers-1; i >= 0; i--){
                printf("bp layer %d\n", i);
                if (i == num_layers-1){
                    //first iteration use Y
                    layers[i].backward_prop(&layers[i+1], &layers[i-1], false, y_hat);
                    
                }
                else if (i == 0) {
                    //last iteration use X
                    layers[i].backward_prop(&layers[i+1], &layers[i-1], true, X_batch);
                }
                else{
                    //normal
                    layers[i].backward_prop(&layers[i+1], &layers[i-1], true, layers[i-1].Z);
                    // layers[i].backward_prop(&layers[i+1], &layers[i-1]);
                    
                }
                // print_device(layers[i].dW, layers[i].units, min(20,layers[i].A_x));
                // print_device(layers[i].db, layers[i].units, 1);
                // print_device(layers[i].dZ, layers[i].units, min(20,layers[i].A_y));
            }
            
            printf("Update\n");
            for (int i = 0; i < num_layers; i++){
                layers[i].update_parameters(0.15);
                // print_device(layers[i].W, layers[i].units, min(10,layers[i].A_x));
                // print_device(layers[i].b, layers[i].units, 1);
            }
            CUDA_CHECK(cudaFree(X_batch));
            CUDA_CHECK(cudaFree(Y_batch));
        }
        print_device(layers[num_layers-1].Z, 10, 10);
    }
    cudaFree(y_hat);
}

double Model::predict(double *inp_X, double *inp_Y){
    int batch_size = 512;
    int num_batch = 8;
    double *x_mat = X;
    int x_size = m;
    double total = 0;
    printf("Forward prop\n");
    for (int l = 0; l < num_layers; l++){
        printf("Forward prop %d %d\n", l, num_layers);
        layers[l].forward_prop(x_mat, x_size, batch_size);
        x_mat = layers[l].Z;
        x_size = layers[l].units;
        // print_device(layers[l].W, layers[l].units, min(20,layers[l].A_x));
        // print_device(layers[l].b, layers[l].units, 1);
        // print_device(layers[l].Z, layers[l].units, min(20,layers[l].A_y));
        total += accuracy(layers[num_layers-1].Z, Y, batch_size);
        printf("total [%f] (%d)\n", total, l);
    }
    return total/n;
    
}

int Model::accuracy(double *Z, double* Y, int num){
    // num = 10;
    double *Z_host = (double *)malloc(sizeof(double) * num * layers[num_layers-1].units);
    double *Y_host = (double *)malloc(sizeof(double) * num);
    int temp_max = 0;
    int count = 0;
    CUDA_CHECK(cudaMemcpy(Z_host, Z, sizeof(double) * min(num, 512) * layers[num_layers-1].units, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Y_host, Y, sizeof(double) * num, cudaMemcpyDeviceToHost));
    for (int i = 0; i < num; i++){
        temp_max = 0;
        for (int j = 0; j < 10; j++){
            if (Z_host[j + i * 10] > Z_host[temp_max + i * 10]){
                temp_max = j;
            }
        }
        // printf(" end %d\n", temp_max);
        if ((double)temp_max == Y_host[i]){
            count += 1;
        }
        
    }
    free(Z_host);
    free(Y_host);
    return count;
}