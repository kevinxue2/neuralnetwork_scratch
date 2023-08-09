import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
class Layer:
    def __init__(self, units, activation):
        self.units = units
        self.init = 0
        self.w = None
        self.b = None
        self.z = None
        self.activation_dict = {
            'relu': self.relu,
            'softmax': self.softmax,
        }
        self.deriv_dict = {
            'relu': self.relu_deriv,
            'softmax': self.relu_deriv,
        }
        self.activation = self.activation_dict[activation]
        self.deriv = self.deriv_dict[activation]
    
    def forward_prop(self, A):
        if not self.init:
            self.w = np.random.rand(self.units,A.shape[0]) - 0.5
            self.b = np.random.rand(self.units,1) - 0.5
            self.init = 1
        self.z = self.w.dot(A) + self.b
        return self.activation(self.z)

    def back_prop(self, dZ_prev, Z, next_A, m):
        dZ = self.w.T.dot(dZ_prev)
        dZ *= self.deriv(Z)
        dW = 1/m * dZ.dot(next_A.T)
        db = 1/m * np.sum(dZ, 1)
        return dZ, dW, db

    def relu(self, Z):
        A = np.maximum(Z, 0)
        return A
    
    def relu_deriv(self, Z):
        return Z > 0

    def softmax(self, Z):
        exp = np.exp(Z - np.max(Z)) 
        return exp / exp.sum(axis=0)

    def update_param(self, alpha, dW, db):
        self.w -= alpha * dW
        self.b -= alpha * np.reshape(db, (self.units,1))


class Model:
    def __init__(self, layers):
        self.layers = layers
        self.z_lst = []

    def compile(self, X, Y):
        self.X = X
        self.Y = Y
        self.m = self.X.shape[1]

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.max()+1,Y.size))
        one_hot_Y[Y,np.arange(Y.size)] = 1
        return one_hot_Y

    def fit(self, alpha, epoch):
        y_mat = self.one_hot(self.Y)
        for x in range(epoch):
            x_mat = self.X
            self.z_lst = []
            for y in self.layers:
                x_mat = y.forward_prop(x_mat)
                self.z_lst.append(x_mat)
            #back prop
            dz_lst = []
            dZ = 2*(self.z_lst[-1] - y_mat)
            dW = 1/self.m * dZ.dot((self.X if len(self.layers) < 2 else self.z_lst[-2]).T)
            db = 1/self.m * np.sum(dZ, 1)
            dz_lst.append((dZ, dW, db))
            for i in range(len(self.layers)-1,0,-1):
                tmp = self.z_lst[i-2] if i-1 > 0 else self.X
                dZ, dW, db = self.layers[i].back_prop(dZ, self.layers[i-1].z, tmp, self.m)
                dz_lst.append((dZ, dW, db))
            for i in range(len(self.layers)):
                dW = dz_lst[-(i+1)][1]
                db = dz_lst[-(i+1)][2]
                self.layers[i].update_param(alpha, dW, db)
            # test
            if x % 5 == 0:
                print("Iteration: ", x)
                predictions = self.get_predictions(self.z_lst[-1])
                print(self.get_accuracy(predictions, self.Y))

    def get_predictions(self, z):
        return np.argmax(z, 0)

    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def make_predictions(self, X):
        x_mat = X
        z_lst = []
        for y in self.layers:
            x_mat = y.forward_prop(x_mat)
            z_lst.append(x_mat)
        predictions = self.get_predictions(z_lst[-1])
        return predictions
    
    def test_prediction(self, index):
        current_image = self.X[:, index, None]
        # prediction = self.make_predictions(X_train[:, index, None])
        label = self.Y[index]
        # print("Prediction: ", prediction)
        print("Label: ", label)
        
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()