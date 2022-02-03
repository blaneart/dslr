import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

def get_color_for_house(name):
    switcher = {
        'Gryffindor': 'r',
        'Ravenclaw': 'b',
        'Slytherin': 'g',
        'Hufflepuff': 'y' 
    }
    return switcher[name]
df = pd.read_csv('datasets/dataset_train.csv')
# fig, axs = plt.subplots(19)
for i, column in enumerate(df):
    if column != 'Index' and  is_numeric_dtype(df[column]):
        plt.figure()
        sns.histplot(df, x=df[column], hue="Hogwarts House")
        # for name, df_house in houses:
            # axs[i].hist(df_house[column], bins=30, alpha=0.5, label = name, color = get_color_for_house(name))
            # axs[i].legend(loc = 'upper right')
            # axs[i].title(column)
plt.show()
