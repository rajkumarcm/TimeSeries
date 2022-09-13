import numpy as np
import pandas as pd

def difference(x):
    return x.diff(periods=1)

def collapse_diff(x_d, x_dd, order):
    n = (len(x_d) + (order-1))
    y = [np.NaN] * n
    for i in range(order, n):
        y[i] = x_dd.iloc[i-order] + x_d.iloc[i-order]
    return(y)
