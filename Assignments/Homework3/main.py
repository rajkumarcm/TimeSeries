#%% Load the libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
from Utilities.WhitenessTest import WhitenessTest

#%% Load the data
data = pd.read_csv('daily-min-temperatures.csv', header=0)
print(data.head())

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
        return y_hat[k-1:(len(y)-rem)], k-1

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
                tmp, _, _ = ma(y_hat[1:(len(y) - 1), 0], m=sf_ma_order, sf=True)
                return tmp, k+1
            else:
                tmp, tmp_start, tmp_end = ma(y_hat[k:(len(y) - rem), 0], m=sf_ma_order, sf=True)
                return tmp, k+tmp_start

y = data.Temp.values
y_diff = data.Temp.diff().iloc[1:].values
yhat, start = ma(y)
# sample = [18872.2, 1906.1, 2019.8, 2180.3, 2388.1, 2449.0, 2180.3, 2388.1, 2449.0]
# tmp_yhat, start, end = ma(sample)
#%%

spacing=18
xticks = []
dates_spaced = []
for i in range(0, 50):
    if i % spacing == 0:
        xticks.append(i)
        dates_spaced.append(data.Date.values[i])

xticks.append(50)
dates_spaced.append(data.Date.values[49])


#%% Q2
ma_orders = [3,5,7,9]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)
r_idx = 0
c_idx = 0
max_samples = 50

for i in range(len(ma_orders)):
    yhat, start = ma(y=y, m=ma_orders[i])
    k = ma_orders[i]//2 + 1
    axes[r_idx, c_idx].plot(data.Date.iloc[:max_samples],
                            y[:max_samples], color='blue', label='Original')
    axes[r_idx, c_idx].plot(data.Date.iloc[start:start+max_samples],
                            yhat[:max_samples], color='orange', label='Smoothed')
    axes[r_idx, c_idx].plot(data.Date.iloc[1:max_samples+1],
                            y_diff[:max_samples], color='maroon', label='Detrended')
    axes[r_idx, c_idx].set_ylabel('Smoothed value')
    axes[r_idx, c_idx].set_xlabel('Time t')
    axes[r_idx, c_idx].set_title(f'MA({ma_orders[i]})')
    axes[r_idx, c_idx].set_xticks(ticks=xticks, labels=dates_spaced)
    axes[r_idx, c_idx].legend()
    if c_idx == 1:
        r_idx += 1
        c_idx = 0
    else:
        c_idx += 1
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
    yhat, start = ma(y=y, m=ma_f_order, sm=ma_s_order)
    axes[r_idx, c_idx].plot(data.Date.iloc[:max_samples],
                            y[:max_samples], color='blue', label='Original')
    axes[r_idx, c_idx].plot(data.Date.iloc[start:start+max_samples],
                            yhat[:max_samples], color='orange', label='Smoothed')
    axes[r_idx, c_idx].plot(data.Date.iloc[1:max_samples + 1],
                            y_diff[:max_samples], color='maroon', label='Detrended')
    axes[r_idx, c_idx].set_ylabel('Smoothed value')
    axes[r_idx, c_idx].set_xlabel('Time t')
    axes[r_idx, c_idx].set_title(f'MA({ma_orders[i]})')
    axes[r_idx, c_idx].set_xticks(ticks=xticks, labels=dates_spaced)
    axes[r_idx, c_idx].legend()
    if c_idx == 1:
        r_idx += 1
        c_idx = 0
    else:
        c_idx += 1

plt.show()

#%% Q4
# Original data
print('ADF of the original data')
WT = WhitenessTest(y)
WT.ADF_Cal()

# Original data using MA
print('ADF of the original data using MA')
yhat_ma, _ = ma(y)
WT = WhitenessTest(yhat_ma)
WT.ADF_Cal()

# De-trended data using MA
print('ADF of the de-trended data using MA')
y_diff_ma, _ = ma(y_diff)
WT = WhitenessTest(y_diff_ma)
WT.ADF_Cal()


#%% Q5.
data['Date'] = pd.DatetimeIndex(data.Date)
data = data.set_index('Date')#.asfreq('d')
stl = STL(data, period=10)
res = stl.fit()
fig = res.plot()
plt.show()

#%%
T = res.trend
S = res.seasonal
R = res.resid
#%% Q6.
# seasonal = S.set_axis(axis=0, labels=data.index)
sa_data = data.Temp.subtract(S)
# sa_data = res.resid + res.trend
plt.figure()
data.plot()
sa_data.plot(color='orange')
plt.xlabel('Time t')
plt.ylabel('Temperature')
plt.title('Temperature over time')
plt.show()

#%%
def str_trend_seasonal(T, S, R):
    Var_R = R.var()
    strength_of_trend = np.max([0, 1-(Var_R/(T+R).var())])
    strength_of_seasonality = np.max([0, 1-(Var_R/(S+R).var())])
    print(f'Strength of the trend {100 * strength_of_trend:.2f}')
    print(f'Strength of the seasonality {100 * strength_of_seasonality:.2f}')

str_trend_seasonal(T, S, R)

#%%




