from math import inf
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr

### +1 - Complete positive correlation ###
### +0.8 - Strong positive correlation ###
### +0.6 - Moderate positive correlation ###
### 0 - no correlation whatsoever ###
### -0.6 - Moderate negative correlation ###
### -0.8 - Strong negative correlation ###
### -1 - Complete negative correlation ###
### cov(X, Y) = (sum (x - mean(X)) * (y - mean(Y)) ) * 1/(n-1) ###
### Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y)) ###

def ft_count(frame):
    counter = 0
    for i in frame:
        counter += 1
    return counter

def ft_max(frame):
    max_val = -inf
    for i in frame:
        if max_val < i:
            max_val = i
    return max_val

def ft_min(frame):
    min_val = inf
    for i in frame:
        if min_val > i:
            min_val = i
    return min_val

def ft_mean(frame):
    return np.sum(frame) / ft_count(frame)

def ft_std(frame):
    return np.sqrt(np.sum(np.power((frame - ft_mean(frame)), 2)) / (ft_count(frame) - 1))
df = pd.read_csv('datasets/dataset_train.csv')

def ft_pearson(data1, data2):
    data1 = data1.fillna(ft_mean(data1))
    data2 = data2.fillna(ft_mean(data2))

    
    print(np.corrcoef(data1, data2))
    covar = np.sum(data1 - ft_mean(data1) * (data2 - ft_mean(data2))) /(ft_count(data1) - 1)
    pear = covar / (ft_std(data1) * ft_std(data2))
    print(pear, pearsonr(data1, data2))
    return 1

list_of_columns = list(df.columns)
print(list_of_columns)
collection_of_coef = {
}
for i in range(len(list_of_columns)):
    column = list_of_columns.pop()
    if column != 'Index' and  is_numeric_dtype(df[column]):
        for second_column in list_of_columns:
            if second_column != 'Index' and  is_numeric_dtype(df[second_column]) and column != second_column:
                coef = ft_pearson(df[column], df[second_column])
                collection_of_coef[column + " " + second_column] = coef
print(collection_of_coef)
