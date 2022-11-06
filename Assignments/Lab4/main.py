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
        for t in range(k-1, len(y)):
            y_hat[t] = np.mean(y[t-(k-1):t+k])
        return y_hat[k-1:]

    else:
        # Even number
        if sf:
            y_hat = np.zeros([len(y), 1])
        else:
            y_hat = np.zeros([len(y), 2])
        k = m//2

        if m == 2:
            for t in range(k, len(y)):
                y_hat[t, 0] = np.mean(y[t-k: t+k])
        else:
            for t in range(k-1, len(y)):
                y_hat[t, 0] = np.mean(y[t-(k-1): t+k+1])

        if sf:
            if m == 2:
                return y_hat[k:, 0]
            else:
                return y_hat[(k-1):, 0]
        else:
            sf_ma_order = sm
            if sm is None:
                while True:
                    sf_ma_order = int(input('Enter the order of the second fold:\n'))
                    if sf_ma_order > 0:
                        break
            tmp_so_ma = ma(y_hat[(k-1):, 0], m=sf_ma_order, sf=True)
            y_hat[-len(tmp_so_ma):, 1] = tmp_so_ma
            return y_hat[:, 1]

y = data.Temp.values
yhat = ma(y)

#%%


#%% Q2
ma_orders = [3,5,7,9]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)
r_idx = 0
c_idx = 0
max_samples = 50
for i in range(len(ma_orders)):
    yhat = ma(y=y, m=ma_orders[i])
    k = ma_orders[i]//2 + 1
    axes[r_idx, c_idx].plot(y[(k-1):((k-1)+max_samples)], color='blue', label='Original')
    axes[r_idx, c_idx].plot(yhat[:max_samples], color='orange', label='Smoothed')
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
    yhat = ma(y=y, m=ma_f_order, sm=ma_s_order)
    k = ma_f_order//2 + 1
    axes[r_idx, c_idx].plot(y[(k - 1):((k - 1) + max_samples)], color='blue', label='Original')
    axes[r_idx, c_idx].plot(yhat[(k-1):((k-1)+max_samples)], color='orange', label='Smoothed')
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











