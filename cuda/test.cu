#include <iostream>
#include <fstream>
#include <cstdio>
#include <arpa/inet.h>
#include <cassert>
#include "layer.h"
#include "model.h"
#include "helpers.h"
#include "activation.h"

void test_forward(int a, int b, int c){
    double *mat_a;
    CUDA_CHECK(cudaMalloc(&mat_a, b * c * sizeof(double)));
    createRandomMatrix(b, c, mat_a);
    Layer l = Layer(a, "sigmoid");
    l.forward_prop(mat_a, b, c);
    printf("W\n");
    print_device(l.W, a, b);
    printf("b\n");
    print_device(l.b, a, 1);
    printf("Z\n");
    print_device(l.Z, a, c);
    CUDA_CHECK(cudaFree(mat_a));
    l.free_all();
}

void test_element_wise(int a, int b){
    double *mat_a;
    CUDA_CHECK(cudaMalloc(&mat_a, a * b * sizeof(double)));
    createRandomMatrix(a, b, mat_a);
    double *mat_b;
    CUDA_CHECK(cudaMalloc(&mat_b, a * b * sizeof(double)));
    createRandomMatrix(a, b, mat_b);
    printf("A\n");
    print_device(mat_a, a, b);
    printf("B\n");
    print_device(mat_b, a, b);
    elementwiseMult<<<512, BLOCK_SIZE>>>(a, b, mat_a, mat_b); // dZx*deriv
    printf("A after\n");
    print_device(mat_a, a, b);
}

void test_relu_deriv(int a, int b){
    double *mat_a;
    CUDA_CHECK(cudaMalloc(&mat_a, a * b * sizeof(double)));
    createRandomMatrix(a, b, mat_a);
    double *deriv;
    CUDA_CHECK(cudaMalloc(&deriv, sizeof(double) * a * b));
    cudaMemset(deriv, 0, sizeof(double) * a * b);
    Activation::reluDerivative<<<512, BLOCK_SIZE>>>(a, b, mat_a, deriv);
    printf("A\n");
    print_device(mat_a, a, b);
    printf("deriv\n");
    print_device(deriv, a, b);

}

void test_update(int a, int b){
    int c = 10;
    double *mat_a;
    CUDA_CHECK(cudaMalloc(&mat_a, b * c * sizeof(double)));
    createRandomMatrix(b, c, mat_a);
    Layer l = Layer(a, "sigmoid");
    l.forward_prop(mat_a, b, c);
    printf("W\n");
    print_device(l.W, a, b);
    printf("b\n");
    print_device(l.b, a, 1);
    createRandomMatrix(a, b, l.dW);
    createRandomMatrix(a, 1, l.db);
    l.update_parameters(0.5);
    printf("After W\n");
    print_device(l.W, a, b);
    printf("After b\n");
    print_device(l.b, a, 1);
    CUDA_CHECK(cudaFree(mat_a));
    l.free_all();
}

int main(){
    printf("Test Forward Prop\n");
    test_forward(15, 10, 20);
    printf("Test Element Wise\n");
    test_element_wise(8, 12);
    printf("Test Relu deriv\n");
    test_relu_deriv(8, 12);
    printf("Test Update Parameters\n");
    test_update(15, 10);

}