#include <string>
#include <map>
#include "cublas_v2.h"
#include <cuda_runtime.h>

#ifndef LAYER_H
#define LAYER_H
class Layer{
public:
    cublasHandle_t handle;
    int units;
    bool init;
    int A_x;
    int A_y;
    double *W;
    double *b;
    double *Z;
    double *dW;
    double *db;
    double *dZ;
    double *one_d;
    std::map<std::string, void (*)(int, int, double*)> activation_f;
    std::map<std::string, void (*)(int, int, double*, double*)> deriv_f;
    std::string activation;
    Layer(int u, const char* activation_string);

    void forward_prop(double* A, int A_X, int A_Y);
                                                                    
    void backward_prop(Layer *layer_prev, Layer* layer_next, bool is_first, double *input);

    void update_parameters(double learn_rate);

    void free_all();
};
#endif