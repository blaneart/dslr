import pandas as pd
import numpy as np
from MyLogisticRegression import MyLogisticRegression
from pandas.api.types import is_numeric_dtype
import os
house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

path = "datasets/dataset_test.csv"

weights_file = 'weights.csv'


df = pd.read_csv(path)


weights = []
with open(weights_file) as file:
    lines = file.readlines()
    for line in lines:
        weights.append(np.array(line[:-1].split(',')).reshape(-1, 1))
weights = np.array(weights)
lrs = []

for theta in weights:
    theta = theta.astype('float64').reshape(11,)
    l = MyLogisticRegression()
    l.load_weights(theta)
    lrs.append(l)
df = df.drop(['Care of Magical Creatures', 'Arithmancy', 'Astronomy', 'Hogwarts House'], axis=1)
df2 = df.drop(df.columns[df.apply(lambda col: not is_numeric_dtype(col))], axis=1).drop('Index', axis=1)

for column in df2:
    if column != 'Hogwarts House':
        df2[column].fillna(df2[column].mean(), inplace=True)
y_hats=[]
for lr in lrs:
    y_hat = lr.predict(df2)
    y_hats.append(y_hat)
res = (np.stack(y_hats, axis=1))
res = np.argmax(res, axis=-1)
if os.path.exists('houses.csv'):
    os.remove('houses.csv')
f=open('houses.csv','ab')
col_names = np.array([['Index','Hogwarts House']])
np.savetxt(f, col_names, fmt='%s', delimiter=',')
for i in range(len(df['Index'])):
    np.savetxt(f, np.array([[i, house_names[res[i]]]]), fmt='%s', delimiter=',')
f.close()
