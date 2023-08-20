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
    data_len = reverseEndian(data_len);
    printf("%s %d %d %d %d\n", f_name, magic, data_len, num_rows, num_col);
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
    free(data_h);
    return data_f;
}


int main(){
    // load train and test datasets
    double *data_x = load_data("../dataset/train-images-idx3-ubyte", true);
    double *data_y = load_data("../dataset/train-labels-idx1-ubyte", false);
    double *test_x = load_data("../dataset/t10k-images-idx3-ubyte", true);
    double *test_y = load_data("../dataset/t10k-labels-idx1-ubyte", false);
    
    int num_layers = 2;
    Layer layers[2] = {Layer(128, "relu"), Layer(10, "sigmoid")};
    Model m = Model(layers, num_layers);
    m.set_data(data_x, data_y, 784, 60000);
    printf("Training start\n");
    m.fit(300);
    printf("Training end\n");
    printf("Train Data: %d\n", m.accuracy(layers[num_layers-1].Z, data_y, 60000));
    // Predict with test data
    m.predict(test_x, test_y, 10000);
    // Check accuracy of test data
    printf("Test Data: %d\n", m.accuracy(layers[num_layers-1].Z, test_y, 10000));

    free(data_x);
    free(data_y);
    free(test_x);
    free(test_y);
    for (int i = 0; i < num_layers; i++){
        layers[i].free_all();
    }
    // Check for errors missed
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
    }

}