#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels import api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from Utilities.Correlation import Correlation as Corr
import re
np.random.seed(123)

#%%
ar = [1, 0.5]
ma = [1]
arma_process = ArmaProcess(ar, ma)

#%% Q1.
def arma_generate_data():
    T = int(input('Enter the number of data samples'))
    mu = float(input('Enter the mean of white noise'))
    var = float(input('Enter the variance of the white noise'))
    std = np.sqrt(var)
    ar_coeffs = None
    ma_coeffs = None
    pattern = "[\-]*[0-9]+[\.0-9]*"
    while True:
        na = int(input('Enter the AR order'))
        coeffs_txt = input('Enter the coefficients of AR - Separate them by commas')
        ar_coeffs = re.findall(pattern, coeffs_txt)
        ar_coeffs = list(map(lambda x: float(x), ar_coeffs))
        if (len(ar_coeffs) == na + 1):
            break

    while True:
        nb = int(input('Enter the MA order'))
        coeffs_txt = input('Enter the coefficients of MA - Separate them by commas')
        ma_coeffs = re.findall(pattern, coeffs_txt)
        ma_coeffs = list(map(lambda x: float(x), ma_coeffs))
        if (len(ma_coeffs) == nb + 1):
            break

    arma_process = ArmaProcess(ar_coeffs, ma_coeffs)
    return arma_process.generate_sample(T, scale=std) + mu
#
# data = arma_generate_data()
# print('breakpoint...')

#%%
# Q2 and 4 combined. GPAC table

def gpac_table(acf_vals, na, nb):

    def gpac(acf_vals, j, k):
        num = np.zeros([k, k])
        for p in range((k - 1)):  # for all columns except the last one
            offset = j - p
            tmp_col = []
            for q in range(offset, offset + k):
                # tmp_col.append(np.abs(q)) # FOR DEBUGGING PURPOSES
                tmp_col.append(acf_vals[np.abs(q)])
            num[:, p] = tmp_col

        den = np.copy(num)
        den[:, k-1] = acf_vals[np.abs(list(range(j-k+1, j+1)))]
        tmp_col = []
        for q in range(j+1, j+k+1):
            tmp_col.append(acf_vals[np.abs(q)])
            # tmp_col.append(np.abs(q))
        num[:, k-1] = tmp_col
        return np.linalg.det(num)/np.linalg.det(den)

    corr = Corr()
    # acf_vals, _ = corr.acf(x=x, max_lag=na+nb, plot=False, return_acf=True)
    Q = np.zeros([nb+1, na+1])
    for j in range(nb+1):
        for k in range(1, na+1):
            if k == 1:
                Q[j, k] = acf_vals[j+1]/acf_vals[j]
                pass
            else:
                # gpac(acf_vals, 1, 3) # FOR DEBUGGING PURPOSE...
                Q[j, k] = gpac(acf_vals, j, k)
    return Q[:, 1:] # because k=0 does not exist

def arma_gpac(max_j, max_k, plot=True):
    T = int(input('Enter the number of data samples'))
    mu = float(input('Enter the mean of white noise'))
    var = float(input('Enter the variance of the white noise'))
    std = np.sqrt(var)
    ar_coeffs = None
    ma_coeffs = None
    pattern = "[\-]*[0-9]+[\.0-9]*"
    while True:
        na = int(input('Enter the AR order'))
        coeffs_txt = input('Enter the coefficients of AR - Separate them by commas')
        ar_coeffs = re.findall(pattern, coeffs_txt)
        ar_coeffs = list(map(lambda x: float(x), ar_coeffs))
        if(len(ar_coeffs)==na+1):
            break

    while True:
        nb = int(input('Enter the MA order'))
        coeffs_txt = input('Enter the coefficients of MA - Separate them by commas')
        ma_coeffs = re.findall(pattern, coeffs_txt)
        ma_coeffs = list(map(lambda x: float(x), ma_coeffs))
        if(len(ma_coeffs)==nb+1):
            break

    arma_process = ArmaProcess(ar_coeffs, ma_coeffs)
    x = arma_process.generate_sample(T, scale=std) + mu
    acf_vals = arma_process.acf(lags=max_j+max_k+1)
    gpac_vals = gpac_table(acf_vals, max_k, max_j - 1)
    if plot:
        plt.figure()
        sns.heatmap(gpac_vals, annot=True)
        plt.title('Generalized Partial Autocorrelation (GPAC) Table')
        plt.show()
    return gpac_vals

# print('breakpoint...')

#%% Q3 is the same as Q1, but need to be manually fed with different parameters. Hence the
# result will be on the report.

#%% Q5
gpac_vals = arma_gpac(7, 7)

#%% Q6

