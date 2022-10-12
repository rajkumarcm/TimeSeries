import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pandas_datareader as web

# Q1 is answered and will be attached the final report.

# Q2. Generate white noise and plot it
wn = np.random.normal(0, 1, size=1000)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(range(1000)), wn)
axes[0].set_title('White Noise')
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('White noise')
axes[1].hist(wn, rwidth=0.9)
axes[1].set_title('White Noise Histogram')
axes[1].set_xlabel('White noise')
axes[1].set_ylabel('White noise frequency')
print(f"The sample has a mean of {np.mean(wn)} and a standard deviation of {np.std(wn)}")
plt.show()

# Q3. ACF
def acf(x, max_lag, ax=None):
    def __acf(x, lag):
        acf_val = 0
        mu = np.mean(x)
        for t in range(lag, len(x)):
            acf_val += (x[t] - mu) * (x[t-lag] - mu)
        return acf_val

    acf_lags = np.zeros([max_lag+1])
    for k in range(max_lag+1):
        x_var = __acf(x, 0)
        acf_lags[k] = __acf(x, k)/x_var

    neg_acf = list(acf_lags[::-1])
    pos_acf = list(acf_lags[1:])
    neg_acf.extend(pos_acf)
    entire_acf = neg_acf
    m = 1.96/np.sqrt(len(x))
    if ax is None:
        plt.figure()
        plt.stem(np.arange(-max_lag, max_lag+1, 1), entire_acf, markerfmt='ro')
        plt.axhspan(-m, m, alpha=0.2, color='blue')
        plt.xlabel('Lag')
        plt.ylabel(r'$\rho(x)$')
        plt.title('Auto Correlation')
        plt.show()
    else:
        ax.stem(np.arange(-max_lag, max_lag+1, 1), entire_acf, markerfmt='ro')
        ax.axhspan(-m, m, alpha=0.2, color='blue')

# Q3. a
x = [3, 9, 27, 81, 243]
acf(x, max_lag=4)

# Q3. b
acf(wn, max_lag=20)

# Q3. c
# Write down your observations about the ACF plot of stationary dataset.
# This is answered in the final report and will be attached for the submission.

# Q4.
from datetime import date
stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
stocks_name = ['APPLE', 'ORACLE', 'TESLA', 'IBM', 'YELP', 'MICROSOFT']
close_df = {}
for stock_symb in stocks:
    tmp_df = web.DataReader(stock_symb, data_source='yahoo', start='2000-01-01', end=date.today().strftime("%Y-%m-%d"))
    close_df[stock_symb] = tmp_df.Close
close_df = pd.DataFrame(close_df)

# Q4.a Plot close value vs time
spacing = 365 * 5
fig, axes = plt.subplots(3, 2, sharey=True, figsize=(16, 8))
plt.title('Closing price of several firm\'s stock')
nrow = 3
ncol = 2
r_idx = 0
c_idx = 0
for stock_name, stock_sym in zip(stocks_name, stocks):
    close_val = close_df.loc[:, stock_sym]
    close_val.dropna(inplace=True)
    close_val.plot(ax=axes[r_idx, c_idx])
    axes[r_idx, c_idx].set_xlabel('Date')
    axes[r_idx, c_idx].set_ylabel('Closing Price')
    axes[r_idx, c_idx].set_title('Closing Price of ' + stock_name)
    axes[r_idx, c_idx].grid(True)
    if c_idx == 1: # index of 2nd column
        c_idx = 0
        r_idx += 1
    else:
        c_idx += 1
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.1,
                    hspace=0.7)
plt.show()

# Q4.b ACF of the closing stock price
fig, axes = plt.subplots(3, 2, figsize=(16, 8), sharey=True)
nrow = 3
ncol = 2
r_idx = 0
c_idx = 0
for stock_name, stock_sym in zip(stocks_name, stocks):
    close_val = close_df.loc[:, stock_sym]
    close_val.dropna(inplace=True)
    acf(close_val.to_numpy(), max_lag=50, ax=axes[r_idx, c_idx])
    axes[r_idx, c_idx].set_xlabel('Lag')
    axes[r_idx, c_idx].set_title(f'Autocorrelation of {stock_name}\'s Closing value')
    axes[r_idx, c_idx].set_ylabel('ACF value')
    if c_idx == 1: # index of 2nd column
        c_idx = 0
        r_idx += 1
    else:
        c_idx += 1
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.1,
                    hspace=0.7)
plt.show()

# Q5 is answered in the final report and will be attached for submission.

# Just some experiment to generate a non-stationary signal
np.random.seed(0)
noise = np.random.normal(0, 1, 100)
y = []
for i in range(100):
    y.append(np.sum(noise[:(i+1)]))

y_diff = pd.Series(y).diff(periods=1)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes[0, 0].plot(y)
axes[0, 0].set_title('NonStationary signal')
axes[0, 0].set_xlabel('Sample')
axes[0, 0].set_ylabel('Noise')
axes[0, 0].grid(True)
acf(y, 50, ax=axes[0, 1])
axes[0, 1].set_title('ACF of nonstationary signal')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('ACF value')
axes[1, 0].plot(y_diff)
axes[1, 0].set_title('Differenced nonstationary signal')
axes[1, 0].set_xlabel('Sample')
axes[1, 0].set_ylabel('Differenced value')
axes[1, 0].grid(True)
acf(y_diff[1:].values.reshape([-1]), max_lag=50, ax=axes[1, 1])
axes[1,1].set_title('ACF of differenced nonstationary signal')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('ACF value')
plt.show()











