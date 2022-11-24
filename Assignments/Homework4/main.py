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
# ar = [1, 0.5]
# ma = [1]
# arma_process = ArmaProcess(ar, ma)

#%% Q1.
def arma_generate_data():
    T = int(input('Enter the number of data samples'))
    mu_e = float(input('Enter the mean of white noise'))
    var_e = float(input('Enter the variance of the white noise'))
    std_e = np.sqrt(var_e)
    ar_coeffs = None
    ma_coeffs = None
    pattern = "[\-]*[0-9]+[\.0-9]*"
    while True:
        na = int(input('Enter the AR order'))
        coeffs_txt = input('Enter the coefficients of AR - Separate them by commas. Note: You should include 1 for y(t)')
        ar_coeffs = re.findall(pattern, coeffs_txt)
        ar_coeffs = list(map(lambda x: float(x), ar_coeffs))
        if (len(ar_coeffs) == na + 1):
            break

    while True:
        nb = int(input('Enter the MA order'))
        coeffs_txt = input('Enter the coefficients of MA - Separate them by commas. Note: You should include 1 for e(t)')
        ma_coeffs = re.findall(pattern, coeffs_txt)
        ma_coeffs = list(map(lambda x: float(x), ma_coeffs))
        if (len(ma_coeffs) == nb + 1):
            break

    arma_process = ArmaProcess(ar_coeffs, ma_coeffs)
    mu_y = (mu_e * np.sum(ma_coeffs))/(1 + np.sum(np.array(ar_coeffs[1:])))
    return arma_process.generate_sample(T, scale=std_e) + mu_y, arma_process
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
    mu_e = float(input('Enter the mean of white noise'))
    var_e = float(input('Enter the variance of the white noise'))
    std_e = np.sqrt(var_e)
    ar_coeffs = None
    ma_coeffs = None
    pattern = "[\-]*[0-9]+[\.0-9]*"
    while True:
        na = int(input('Enter the AR order'))
        coeffs_txt = input('Enter the coefficients of AR - Separate them by commas. Note: You should include 1 for y(t)')
        ar_coeffs = re.findall(pattern, coeffs_txt)
        ar_coeffs = list(map(lambda x: float(x), ar_coeffs))
        if(len(ar_coeffs)==na+1):
            break

    while True:
        nb = int(input('Enter the MA order'))
        coeffs_txt = input('Enter the coefficients of MA - Separate them by commas. Note: You should include 1 for e(t)')
        ma_coeffs = re.findall(pattern, coeffs_txt)
        ma_coeffs = list(map(lambda x: float(x), ma_coeffs))
        if(len(ma_coeffs)==nb+1):
            break

    arma_process = ArmaProcess(ar_coeffs, ma_coeffs)
    mu_y = (mu_e * np.sum(ma_coeffs)) / (1 + np.sum(np.array(ar_coeffs[1:])))
    x = arma_process.generate_sample(T, scale=std_e) + mu_y
    acf_vals = arma_process.acf(lags=max_j+max_k+1)
    gpac_vals = gpac_table(acf_vals, max_k, max_j - 1)
    if plot:
        plt.figure()
        sns.heatmap(gpac_vals, annot=True)
        plt.xticks(ticks=list(range(1, max_k+1)), labels=list(range(1, max_k+1)))
        plt.title('Generalized Partial Autocorrelation (GPAC) Table')
        plt.xlabel('AR Order')
        plt.ylabel('MA Order')
        plt.show()
    return x, arma_process, gpac_vals

# print('breakpoint...')

#%% Q3 and 5

# If you meant to ask manual method, here is the solution
data_q3, arma_process, gpac_vals = arma_gpac(7, 7, plot=True)

#%% Q4. Generate theoretical ACF
acf_vals = arma_process.acf(lags=15)

#%% Q6
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def ACF_PACF_Plot(y,lags):
     acf = sm.tsa.stattools.acf(y, nlags=lags)
     pacf = sm.tsa.stattools.pacf(y, nlags=lags)
     fig = plt.figure()
     plt.subplot(211)
     plt.title('ACF/PACF of the raw data')
     plot_acf(y, ax=plt.gca(), lags=lags)
     plt.subplot(212)
     plot_pacf(y, ax=plt.gca(), lags=lags)
     fig.tight_layout(pad=3)
     plt.show()

ACF_PACF_Plot(data_q3, lags=20)

#%% Q7 and Q8
data, arma_process = arma_generate_data()
ArmaProcess




















