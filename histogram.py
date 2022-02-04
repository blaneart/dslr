import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

df = pd.read_csv('datasets/dataset_train.csv')

for i, column in enumerate(df):
    if column != 'Index' and  is_numeric_dtype(df[column]):
        plt.figure()
        sns.histplot(df, x=df[column], hue="Hogwarts House")
plt.show()
