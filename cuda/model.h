#include "layer.h"

class Model{
public:
    Layer *layers;
    int num_layers;
    double *X;
    double *Y;
    int m;
    int n;
    Model(Layer* l, int num_l);

    void set_data(double *X_p, double* Y_p, int X_size, int data_len);

    void set_Y(double *Z, double* Y_b, int batch_size, int b_num);

    void fit(int epoch);

    double predict(double *inp_X, double *inp_Y);

    int accuracy(double *Z, double* Y, int num);
};