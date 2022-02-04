from math import inf
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/dataset_train.csv')
final_df = df.drop(df.columns[df.apply(lambda col: not is_numeric_dtype(col))], axis=1).drop('Index', axis=1)
final_df = pd.concat([df['Hogwarts House'], final_df], axis=1)
sns.pairplot(final_df, hue='Hogwarts House')
plt.show()

### From this we can see that we can drop Care of Magical Creatures (too homogeneus), Arithmancy (too homogeneus) and Astronomy (or Defense Againt the Dark Arts) (correlation too high)