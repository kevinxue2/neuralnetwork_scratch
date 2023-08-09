import numpy as np
from keras.datasets import mnist
import pickle   
from nn_class_py import Layer, Model


if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = (train_X.reshape((60000,784)) / 255)
    test_X = np.around(test_X.reshape((10000,784)) / 255)
    m, n = train_X.shape
    model = Model([
        Layer(300, 'relu'),
        Layer(100, 'relu'),
        Layer(10, 'softmax'),
        ])
    model.compile(train_X.T, train_y)
    epoch = 300
    model.fit(0.15, epoch)

    with open(f'300-100-10-softmax-e{epoch}.pkl', 'wb') as p:
        pickle.dump(model, p)
        print('saved')

