from random import randrange
from re import S
import numpy as np
import sys

class MyLogisticRegression():
    def __init__(self,alpha=2e-4):
        self.theta = 0
        self.alpha = alpha

    def sigmoid(self, x, theta):
        return 1 / (1 + np.exp(-np.dot(x, theta)))

    def logistic_predict_(self, x, theta):
        y_hat = self.sigmoid(x, theta)
        return y_hat

    def loss_(self, y, y_hat, eps=1e-15):
        loss = -np.dot(y * np.log(y_hat + eps)) + np.dot((1 - y) * np.log(1 - y_hat + eps))
        return loss

    def cross_entropy(self, y, y_hat, eps=1e-10):
        return -self.loss_(y, y_hat, eps) / len(y_hat)

    def log_gradient(self, x, y, theta):
        predictions = self.logistic_predict_(x, theta)
        d = np.matmul(np.transpose(x), (predictions - y)) / len(y)
        return d, predictions

    def accuracy(self, y_hat, y):
        accuracy = (y == y_hat).sum() / len(y)
        print("Current accuracy is: ", accuracy * 100, "%")

    def evaluate(self, x, y):
        predictions = self.logistic_predict_(np.c_[np.ones(x.shape[0]), x], self.theta)
        print('Calculating accurac of test set...')
        self.accuracy((predictions > 0.5).astype(int), y)
    
    def predict(self, x):
        predictions = self.logistic_predict_((np.c_[np.ones(x.shape[0]), x]), self.theta)
        return predictions

    def get_random_batch(self, max_len, batch_size=40):
        indexes = []
        while len(indexes) != batch_size:
            indexes.append(randrange(int(max_len)))
        return indexes
    def fit(self, X, y, alpha=2e-5, iterations=70000):
        self.alpha = alpha


        
        self.theta = np.ones(X.shape[1] + 1)
        for n in range(iterations):
            list_of_batch = self.get_random_batch(len(y))
            X1 = X.take(list_of_batch)
            y1 = y.take(list_of_batch)
            X_train = np.c_[np.ones(X1.shape[0]), X1]
            y_train = np.array(y1).astype('float64')
            delta, predictions = self.log_gradient(X_train, y_train, self.theta)
            self.theta = self.theta - self.alpha * delta
            if n % 1000 == 0:
                print('Epoch: ', n)
                self.accuracy((predictions > 0.5).astype(int), y_train)
    
    def get_thetas(self):
        return self.theta

    def load_weights(self, weights_array):
        self.theta = weights_array
