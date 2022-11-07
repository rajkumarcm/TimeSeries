#%% Load the libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
from Utilities.WhitenessTest import WhitenessTest

#%% Load the data
data = pd.read_csv('daily-min-temperatures.csv', header=0)
data.head()

#%%
def ma(y, m=None, sm=None, sf=False):
    y_hat = None
    """
    T_hat_t = 1/m * sum(y_{t+j})
    """
    if not sf and m is None:
        while True:
            m = int(input('Enter the ma order:\n'))
            if m > 2:
                break

    if m%2 != 0:
        # Odd number
        y_hat = np.zeros([len(y), 1])
        k = (m+1)//2 # actual order
        rem = m - k
        for t in range(k-1, len(y)-rem):
            y_hat[t] = np.mean(y[t-(k-1):t+k])
        return y_hat[k-1:(len(y)-rem)], k-1, len(y)-rem

    else:
        # Even number
        if sf:
            y_hat = np.zeros([len(y), 1])
        else:
            y_hat = np.zeros([len(y), 2])
        k = m//2 - 1
        rem = m - (k+1)
        if m == 2:
            for t in range(1, len(y)-1):
                y_hat[t, 0] = np.mean(y[t-1: t+1])
        else:
            for t in range(k, len(y)-rem):
                y_hat[t, 0] = np.mean(y[t-(k-1): t+(k+1)+1])

        if sf:
            if m == 2:
                return y_hat[1:(len(y)-1), 0], 1, len(y)-1
            else:
                return y_hat[k:(len(y)-rem), 0], k, len(y)-rem
        else:
            sf_ma_order = sm
            if sm is None:
                while True:
                    sf_ma_order = int(input('Enter the order of the second fold:\n'))
                    if sf_ma_order > 0:
                        break
            if m == 2:
                return ma(y_hat[1:(len(y) - 1), 0], m=sf_ma_order, sf=True)
            else:
                return ma(y_hat[k:(len(y) - rem), 0], m=sf_ma_order, sf=True)

y = data.Temp.values
yhat, start, end = ma(y)
# sample = [18872.2, 1906.1, 2019.8, 2180.3, 2388.1, 2449.0, 2180.3, 2388.1, 2449.0]
# tmp_yhat, start, end = ma(sample)
#%%


#%% Q2
ma_orders = [3,5,7,9]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)
r_idx = 0
c_idx = 0
max_samples = 50
for i in range(len(ma_orders)):
    yhat, start, end = ma(y=y, m=ma_orders[i])
    k = ma_orders[i]//2 + 1
    axes[r_idx, c_idx].plot(data.Date.iloc[:max_samples],
                            y[:max_samples], color='blue', label='Original')
    axes[r_idx, c_idx].plot(data.Date.iloc[start:start+max_samples],
                            yhat[:max_samples], color='orange', label='Smoothed')
    axes[r_idx, c_idx].set_ylabel('Smoothed value')
    axes[r_idx, c_idx].set_xlabel('Time t')
    axes[r_idx, c_idx].set_title(f'MA({ma_orders[i]})')
    if c_idx == 1:
        r_idx += 1
        c_idx = 0
    else:
        c_idx += 1
plt.legend()
plt.show()

#%% Q3
ma_orders = [(2,4), (2,6), (2,8), (2,10)]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
r_idx = 0
c_idx = 0
max_samples = 50
for i in range(len(ma_orders)):
    ma_f_order = ma_orders[i][1]
    ma_s_order = ma_orders[i][0]
    yhat, start, end = ma(y=y, m=ma_f_order, sm=ma_s_order)
    axes[r_idx, c_idx].plot(data.Date.iloc[:max_samples],
                            y[:max_samples], color='blue', label='Original')
    axes[r_idx, c_idx].plot(data.Date.iloc[start:start+max_samples],
                            yhat[:max_samples], color='orange', label='Smoothed')
    axes[r_idx, c_idx].set_ylabel('Smoothed value')
    axes[r_idx, c_idx].set_xlabel('Time t')
    axes[r_idx, c_idx].set_title(f'MA({ma_orders[i]})')
    if c_idx == 1:
        r_idx += 1
        c_idx = 0
    else:
        c_idx += 1
plt.legend()
plt.show()

#%% Q4
yhat = ma(y)
WT = WhitenessTest(y)
WT.ADF_Cal()
WT = WhitenessTest(yhat)
WT.ADF_Cal()

#%% Q5.
data['Date'] = pd.DatetimeIndex(data.Date)
data = data.set_index('Date').asfreq('d')
stl = STL(data)
res = stl.fit()
fig = res.plot()
plt.show()











