import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

def rolling_mean(x):
    n = len(x)
    roll_mean = np.zeros([n, 2])
    for i in range(1, n+1):
        roll_mean[i-1, 0] = i
        roll_mean[i-1, 1] = x.iloc[:i].mean()
    return roll_mean

def rolling_var(x):
    n = len(x)
    roll_var = np.zeros([n, 2])
    for i in range(1, n+1):
        roll_var[i-1, 0] = i
        roll_var[i-1, 1] = x.iloc[:i].var()
    return roll_var

def Cal_rolling_mean_var(x):
    rm = rolling_mean(x)
    rv = rolling_var(x)
    return rm[1], rv[1]

def Plot_Rolling_Mean_Var(x, name):
    rm = rolling_mean(x)
    rv = rolling_var(x)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    axes[0].plot(rm[:, 0], rm[:, 1], lw=2)
    axes[0].set_title(f'Rolling Mean -{name}')
    axes[0].set_xlabel('Number of Samples')
    axes[0].set_ylabel('Magnitude')
    axes[0].grid(True)

    axes[1].plot(rv[:, 0], rv[:, 1], lw=2)
    axes[1].set_title(f'Rolling Variance -{name}')
    axes[1].set_xlabel('Number of Samples')
    axes[1].set_ylabel('Magnitude')
    axes[1].grid(True)
    plt.show()

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

