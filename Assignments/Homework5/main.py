#%%

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import dlsim
from statsmodels import api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from Utilities.WhitenessTest import WhitenessTest as WT
from Utilities.Correlation import Correlation as Corr
from Utilities.GPAC import gpac_table
import pandas as pd
import seaborn as sns
# np.random.seed(12345)

#%%
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

def seasonal_differencing(y, seasonal_period):
    m = seasonal_period
    s_diff = []
    for t in range(m, len(y)):
        s_diff.append(y[t] - y[t-m])
    return s_diff

#%%
def one_step_pred_ex2(y, t):
    return (0.5 * y[t-2]) - (0.6 * y[t-5])

def h_step_pred_ex2(h, y, yhat, T, t):
    lag_3 = t+h-3
    lag_6 = t+h-6

    if lag_3 > T:
        lag3_val = yhat[lag_3]
    else:
        lag3_val = y[t+h-3]

    if lag_6 > T:
        lag6_val = yhat[lag_6]
    else:
        lag6_val = y[t+h-6]

    return (0.5 * lag3_val) - (0.6 * lag6_val)

#%%

def sarima():
    # Q1.
    T = int(input('Enter the number of data samples:'))
    mu = float(input('Enter the mean of white noise:'))
    var = float(input('Enter the variance of the white noise'))
    na = int(input('Enter the AR order:'))
    nb = int(input('Enter the MA order:'))

    ar_coeffs = []
    tmp_ar_coeffs = []
    for i in range(na):
        tmp_coeff = float(input(f'Enter the a{i+1} coefficient:'))
        tmp_ar_coeffs.append(tmp_coeff)
    tmp_ar_coeffs = np.array(tmp_ar_coeffs)

    tmp_ma_coeffs = []
    for i in range(nb):
        tmp_coeff = float(input(f"Enter the b{i + 1} coefficient:"))
        tmp_ma_coeffs.append(tmp_coeff)
    tmp_ma_coeffs = np.array(tmp_ma_coeffs)

    if na > 0:
        seasonal_indices = np.where(tmp_ar_coeffs != 0)[0]
        seasonal_indices = list(map(lambda x: x+1, seasonal_indices))
        ar_seasonal_period = 1
        if len(seasonal_indices) == 1:
            ar_seasonal_period = seasonal_indices[0]
        else:
            if seasonal_indices[1] == 2 * seasonal_indices[0]:
                ar_seasonal_period = seasonal_indices[0]
        # else:
        #     # Multiplicative model
        #     seasonal_index =
    else:
        ar_seasonal_period = 0

    if nb > 0:
        seasonal_indices = np.where(tmp_ma_coeffs != 0)[0]
        seasonal_indices = list(map(lambda x: x + 1, seasonal_indices))
        ma_seasonal_period = 1
        if len(seasonal_indices) == 1:
            ma_seasonal_period = seasonal_indices[0]
        else:
            if seasonal_indices[1] == 2 * seasonal_indices[0]:
                ma_seasonal_period = seasonal_indices[0]
    else:
        ma_seasonal_period = 0


    seasonal_period = np.max([ar_seasonal_period, ma_seasonal_period])

    max_order = np.max([na, nb, seasonal_period])

    ar_coeffs = np.r_[1, tmp_ar_coeffs]
    ar_coeffs = np.r_[ar_coeffs, [0] * (max_order-na)]

    ma_coeffs = np.r_[1, tmp_ma_coeffs]
    ma_coeffs = np.r_[ma_coeffs, [0] * (max_order - nb)]

    # Q3.
    wn = np.random.normal(loc=mu, scale=np.sqrt(var), size=T)
    _, y_out = dlsim((ma_coeffs, ar_coeffs, 1), wn)
    y_out = y_out[:, 0]

    # Q4.
    ACF_PACF_Plot(y_out, lags=20)

    # Q5.
    wt = WT(x=pd.Series(y_out))
    print("ADF test on raw dataset:\n")
    pval = wt.ADF_Cal()
    raw_pval = pval
    wt.Plot_Rolling_Mean_Var(name='y')

    # Q6.
    raw_y = np.copy(y_out)
    # If the original dataset is stationary, then skip the 6th step
    count = 0
    if raw_pval >= 0.01:
        while pval >= 0.01:
            # Non-Stationary
            if count == 0:
                y_out = seasonal_differencing(y_out, seasonal_period)
            else:
                y_out = seasonal_differencing(y_out, seasonal_period=1)
            print('ADF on differenced data - ignore this...')
            wt = WT(x=pd.Series(y_out))
            pval = wt.ADF_Cal()
            count += 1

        ACF_PACF_Plot(y_out, lags=20)

        # Q7.
        wt = WT(x=pd.Series(y_out))
        print('ADF on the differenced dataset')
        wt.ADF_Cal()
        wt.Plot_Rolling_Mean_Var(name='Differenced dataset')

    seasonal_period = np.max([seasonal_period, count])

    # Q8. GPAC
    corr = Corr()
    if raw_pval < 0.01:
        acf_vals_y, _ = corr.acf(raw_y, max_lag=na+nb+10, plot=False, return_acf=True)
        gpac_vals = gpac_table(acf_vals_y, na=na+3, nb=nb+3, plot=True)

    else:
        # y_out here is differenced dataset
        acf_vals_y_diff, _ = corr.acf(y_out, max_lag=na+nb+10, plot=False, return_acf=True)
        gpac_vals = gpac_table(acf_vals_y_diff, na=na+3, nb=nb+3, plot=True)
    plt.figure(figsize=(13, 10))
    sns.heatmap(gpac_vals, annot=True)
    plt.xticks(ticks=np.array(list(range(na+3))) + .5, labels=list(range(1, na+3+1)))
    plt.title('Generalized Partial Autocorrelation (GPAC) Table')
    plt.xlabel('AR Order')
    plt.ylabel('MA Order')
    plt.tight_layout()
    plt.show()

    # Q9.
    plt.figure()
    plt.plot(raw_y[:500], '-b', label='Original data')
    if raw_pval > 0.01:
        plt.plot(range(seasonal_period, 500+seasonal_period), y_out[:500], color='orange',
                 label='Seasonally differenced data')
    plt.xlabel('Time (t)')
    plt.ylabel('Value')
    plt.title('Original data vs Differenced Data')
    plt.tight_layout()
    plt.legend()
    plt.show()
    # print('stop me here...')

    # Q10 - will be on report


sarima()

#%% Questions 12 - 14
ar_coeffs = [1, 0, 0, -0.5, 0, 0, 0.6]
ma_coeffs = [1, 0, 0, 0, 0, 0, 0]
system = (ma_coeffs, ar_coeffs, 1)
T = 10000
wn = np.random.normal(loc=0, scale=1, size=T)
_, y_out = dlsim(system, wn)
y_out = y_out[:, 0]

#%%
y_hat = []
for t in range(5, 105):
    y_hat.append(one_step_pred_ex2(y_out, t))

#%%
plt.plot(y_out[6:106], '-b', label='Raw data')
plt.plot(y_hat, color='orange', label='1-step prediction')
plt.xlabel('Time (t)')
plt.ylabel('Value')
plt.title('$ARIMA(2,0,0)_{3}$')
plt.legend()
plt.tight_layout()
plt.show()

#%%
residual = y_out[6:106] - y_hat
corr = Corr()
acf_residual, _ = corr.acf(x=residual, max_lag=20, plot=True, return_acf=True)

#%%

import statsmodels.api as sm
Q = T * (acf_residual[1:].T @ acf_residual[1:])
alpha = 0.01
DOF = 20 - 6
lbvalue, pvalue = sm.stats.acorr_ljungbox(residual, lags=[20])



#%%

y_hat_new = [0]*5
for t in range(5, 1001):
    y_hat_new.append(one_step_pred_ex2(y_out, t))

#%% Question 15
y_hat_50 = []
for t in range(901, 951):
    y_hat_50.append(h_step_pred_ex2(h=50, y=y_out, yhat=y_hat_new, t=t, T=950))

#%%
plt.plot(y_out[952:1002], '-b', label='Raw data')
plt.plot(y_hat_50, color='orange', label='50-step prediction')
plt.xlabel('Time (t)')
plt.ylabel('Value')
plt.title('$ARIMA(2,0,0)_{3}$')
plt.legend()
plt.tight_layout()
plt.show()

#%%
print(f"Variance of test set vs predicted {np.var(y_out[952:1002])/np.var(y_hat_50)}")

























