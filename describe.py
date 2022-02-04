from cmath import inf
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import math

class describe:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.description()

    def description(self):
        description_df = pd.DataFrame(index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
        for column in self.df:
            if is_numeric_dtype(self.df[column]):
                c_column = self.df[column]
                description_df[column] = self.describe_frame(c_column)
        print(description_df)

    def ft_count(self, frame):
        counter = 0
        for i in frame:
            counter += 1
        return counter

    def ft_max(self, frame):
        max_val = -inf
        for i in frame:
            if max_val < i:
                max_val = i
        return max_val

    def ft_min(self, frame):
        min_val = inf
        for i in frame:
            if min_val > i:
                min_val = i
        return min_val

    def ft_percentile(self, frame, percent, count):
        frame = np.sort(frame)
        k = (count - 1) * percent
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return frame[int(k)]
        d0 = frame[int(f)] * (c-k)
        d1 = frame[int(c)] * (k-f)
        return d0+d1

    def describe_frame(self, frame):
        frame = frame[~np.isnan(frame)]
        count = self.ft_count(frame)
        mean = np.sum(frame) / count
        std = np.sqrt(np.sum(np.power((frame - mean), 2)) / (count - 1))
        dmin = self.ft_min(frame)
        dmax = self.ft_max(frame)
        perc = self.ft_percentile(frame, 0.25, count)
        prec2 = self.ft_percentile(frame, 0.5, count)
        perc3 = self.ft_percentile(frame, 0.75, count)
        return [count, mean, std, dmin, perc, prec2, perc3, dmax]


        
if __name__ == '__main__':
    describe('datasets/dataset_train.csv')
