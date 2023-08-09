#include <iostream>
#include <fstream>
#include <cstdio>
#include <arpa/inet.h>
#include <cassert>
#include "layer.h"
#include "model.h"
#include "helpers.h"

uint reverseEndian(uint n) {
        
        n = ((n & 0xffff0000) >> 16) | ((n & 0x0000ffff) << 16);
        n = ((n & 0xff00ff00) >> 8) | ((n & 0x00ff00ff) << 8);
        return n;
        
    }

double* load_data(const char* f_name, bool is_image){
    std::ifstream f(f_name, std::ios::out | std::ios::binary);
    uint magic;
    uint data_len;
    uint num_rows = 1;
    uint num_col = 1;
    // read meta data
    //little endian
    f.read((char *) &magic, sizeof(int));
    f.read((char *) &data_len, sizeof(int));
    if (is_image){
        f.read((char *) &num_rows, sizeof(int));
        f.read((char *) &num_col, sizeof(int));
        num_rows = reverseEndian(num_rows);
        num_col = reverseEndian(num_col);
    }
    magic = reverseEndian(magic);
    data_len = reverseEndian(data_len)/8;
    printf("%d %d %d %d\n", magic, data_len, num_rows, num_col);
    int *data_h = (int *)calloc(data_len * num_rows * num_col * sizeof(double),1);
    double *data_f = (double *)calloc(data_len * num_rows * num_col * sizeof(double),1);
    // read images
    for (int i = 0; i < data_len; i++){
        for (int j = 0; j < num_rows * num_col; j++){
            f.read((char *) &data_h[i*num_rows*num_col+j], 1);
            data_f[i*num_rows*num_col+j] = static_cast<double>(data_h[i*num_rows*num_col+j]);
            if (is_image){
                data_f[i*num_rows*num_col+j] /= 255;
            }
        }
    }
    if(!f.good()) {
      printf("error\n");
      free(data_h);
      free(data_f);
      f.close();
      return NULL;
    }
    f.close();
    // double *data_d;
    // CUDA_CHECK(cudaMalloc(&data_d, data_len * num_rows * num_col * sizeof(double)));
    // CUDA_CHECK(cudaMemcpy(data_d, data_f, data_len * num_rows * num_col * sizeof(double), cudaMemcpyHostToDevice));
    free(data_h);
    // free(data_f);
    return data_f;
}



int main(){
    double *data_x;
    cudaMalloc(&data_x, 10 * 10 * sizeof(double));
    createRandomMatrix(10, 10, data_x);
    data_x = load_data("../dataset/t10k-images-idx3-ubyte", true);
    double *data_y = load_data("../dataset/t10k-labels-idx1-ubyte", false);
    
    Layer layers[2] = {Layer(128, "relu"), Layer(10, "sigmoid")};
    Model m = Model(layers, 2);
    m.compile(data_x, data_y, 784, 512);
    printf("Training start %d\n", sizeof(layers));
    m.fit(20);
    printf("Training end\n");
    print_device(layers[1].Z, 10, 20);
    // print_device(data_y, 10, 1);
    // printf("%d\n", m.accuracy(layers[1].Z, data_y, 100));

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error appropriately (e.g., cleanup and return)
    }

}