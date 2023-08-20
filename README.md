# CUDA Neural Network Framework
Simple Neural Network Implementation that can be use to create fully connected neural network models. The framework is designed to utilized NVIDIA GPUs through CUDA programming to speedup training and inference performance.

## Sample
The current code includes a sample implementation of the MNIST handwritten digits dataset. The dataset consists of 60000 28x28 grayscale images of handwritten digits along with the corresponding labels. The dataset was taken from http://yann.lecun.com/exdb/mnist/

## Build Steps
### Prerequisites
- NVIDIA GPU
- CUDA
### Build
1. cd ./cuda
2. make
3. ./model
