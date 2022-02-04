import pandas as pd
import numpy as np
import math
from pandas.api.types import is_numeric_dtype

    
class MyLogisticRegression():
    def __init__(self, X, y, alpha=2e-4):
        self.theta = np.ones(X.shape[1] + 1)
        self.X = np.c_[np.ones(X.shape[0]), X]
        self.y = np.array(y).astype('float64')
        self.m = len(y)
        print(self.m)
        self.alpha = alpha

    def sigmoid(self, x, theta):
        return 1 / (1 + np.exp(-np.dot(x, theta)))

    def logistic_predict_(self, x, theta):
        y_hat = self.sigmoid(x, theta)
        return y_hat

    def loss_(self, y, y_hat, eps=1e-15):
        loss = -np.dot(y * log(y_hat + eps)) + np.dot((1 - y) * log(1 - y_hat + eps))
        return loss

    def cross_entropy(self, y, y_hat, eps=1e-15):
        return -self.loss_(y, y_hat, eps) / len(y_hat)

    def log_gradient(self, x, y, theta):
        predictions = self.logistic_predict_(x, theta)
        self.accuracy((predictions > 0.5).astype(int), y)
        print(y)
        print(predictions)
        d = np.matmul(np.transpose(x), (predictions - y)) / self.m
        return d

    def accuracy(self, y_hat, y):
        accuracy = (y == y_hat).sum() / self.m
        print(accuracy)

    def fit(self, iterations=70000):
        print('theta = ', self.theta)
        for n in range(iterations):
            delta = self.log_gradient(self.X, self.y, self.theta)
            self.theta = self.theta - self.alpha * delta

    def get_thetas(self):
        return self.theta

house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

def prepare_dataset(df, house_name):
    df['label'] = df['Hogwarts House'].apply(lambda x: x == house_name).astype(int)
    df = df.drop(['Care of Magical Creatures', 'Arithmancy', 'Astronomy'], axis=1)
    final_df = df.drop(df.columns[df.apply(lambda col: not is_numeric_dtype(col))], axis=1)
    final_df = pd.concat([df['Hogwarts House'], final_df], axis=1)
    print(final_df)
    x = final_df.drop(['label', 'Hogwarts House'], axis=1)
    y = final_df['label']
    return x, y

df = pd.read_csv('datasets/dataset_train.csv')
df2 = df.drop(df.columns[df.apply(lambda col: not is_numeric_dtype(col))], axis=1).drop('Index', axis=1)
df2 = pd.concat([df['Hogwarts House'], df2], axis=1)

### Fill Nan values with mean value for each house for each course ###
for column in df2:
    if column != 'Hogwarts House':
        for hm in house_names:
            df2[column].fillna(df2.groupby('Hogwarts House')[column].transform("mean"), inplace=True)

list_of_thetas = []
for house in house_names:
    print(house)
    x, y = prepare_dataset(df2, house)
    lr_g = MyLogisticRegression(x, y)
    lr_g.fit()
    list_of_thetas.append(lr_g.get_thetas())
list_of_thetas
f=open('weights.csv','ab')
np.savetxt(f, list_of_thetas[0], fmt='%s', delimiter=',')
np.savetxt(f, list_of_thetas[1], fmt='%s', delimiter=',')
np.savetxt(f, list_of_thetas[2], fmt='%s', delimiter=',')
np.savetxt(f, list_of_thetas[3], fmt='%s', delimiter=',')
f.close()
