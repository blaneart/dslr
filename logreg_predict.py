import pandas as pd
import numpy as np

def sigmoid(x):
    return 1 / (1 + (math.e ** (theta * x)))
    
def logistic_predict_(x, theta):
    y_hat = sigmoid(x, theta)