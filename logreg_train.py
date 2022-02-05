import pandas as pd
import numpy as np
import math
import os
from pandas.api.types import is_numeric_dtype
from MyLogisticRegression import MyLogisticRegression
from data_preparation import data_preparation

house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
list_of_thetas = []
regs = []
for house in house_names:
    print("Training ", house, "vs all")
    x, x_val, y, y_val = data_preparation('datasets/dataset_train.csv', house)
    lr_g = MyLogisticRegression()
    lr_g.fit(x, y, alpha=2e-4, iterations=50000)
    list_of_thetas.append(lr_g.get_thetas())
    lr_g.evaluate(x_val, y_val)
    regs.append(lr_g)


print(list_of_thetas)

if os.path.exists('weights.csv'):
    os.remove('weights.csv')
theta0 = list_of_thetas[0].reshape(1, 11)
theta1 = list_of_thetas[1].reshape(1, 11)
theta2 = list_of_thetas[2].reshape(1, 11)
theta3 = list_of_thetas[3].reshape(1, 11)
f=open('weights.csv','ab')
print('saving model to weights.csv')
f=open('weights.csv','ab')
np.savetxt(f, theta0, fmt='%s', delimiter=',')
np.savetxt(f, theta1, fmt='%s', delimiter=',')
np.savetxt(f, theta2, fmt='%s', delimiter=',')
np.savetxt(f, theta3, fmt='%s', delimiter=',')
f.close()
