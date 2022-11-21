#%%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import dlsim
from Utilities.Correlation import Correlation as Corr
np.random.seed(123)

#%% Q1.b
num = [1, 0, 0]
den = [1, -0.5, -0.2]
system = (num, den, 1)
mean = 2
std = 1
T = 100
e = np.random.normal(loc=mean, scale=std, size=T)
_, y = dlsim(system, e)
plt.figure()
plt.plot(list(range(T)), y[:, 0])
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.title('AR(2) process')
plt.show()

#%% Q1.c
# The experimental mean, and the variance converges to the theoretical mean and the variance
# as T gets closer to Infinite.
print(f'Experimental mean of y: {np.mean(y)}')
print(f'Experimental mean of y: {np.var(y)}')

#%% Q1.d
T = 1000
e = np.random.normal(loc=mean, scale=std, size=T)
_, y = dlsim(system, e)
mean_1000 = np.mean(y)
var_1000 = np.var(y)

T = 10000
e = np.random.normal(loc=mean, scale=std, size=T)
_, y = dlsim(system, e)
mean_10000 = np.mean(y)
var_10000 = np.var(y)

print(f'Experimental mean of y with 1000 samples: {mean_1000}')
print(f'Experimental variance of y with 1000 samples: {var_1000}')
print(f'Experimental mean of y with 10000 samples: {mean_10000}')
print(f'Experimental variance of y with 10000 samples: {var_10000}')

#%% Q1.e
pd.DataFrame({'theoretical':{'mean':2/0.3, 'variance':0.8/0.468},
              'experimental_1000':{'mean':mean_1000, 'variance':var_1000},
              'experimental_10000':{'mean':mean_10000, 'variance':var_10000}
              })

#%% Q1.f ACF of y
lags = [20, 40, 80]
corr = Corr()
fig, axes = plt.subplots(len(lags), 1, figsize=(12, 9))
for i, lag in enumerate(lags):
    corr.acf(x=y, max_lag=lag, ax=axes[i])
    axes[i].set_title(f'ACF of AR(2) with lag {lag}')
plt.tight_layout()
plt.show()

#%% Q2.b
# 𝑦(𝑡) = 𝑒(𝑡) + 0.1𝑒(𝑡 − 1) + 0.4𝑒(𝑡 − 2)
num = [1, 0.1, 0.4]
den = [1, 0, 0]
system = (num, den, 1)
_, y = dlsim(system, e)

plt.figure()
plt.plot(range(T), y[:, 0])
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.title('MA(2)')
plt.show()

#%% Q2.c
print(f'Experimental mean of y: {np.mean(y)}')
print(f'Experimental variance of y: {np.var(y)}')

#%% Q2.d
T = 1000
e = np.random.normal(loc=mean, scale=std, size=T)
_, y = dlsim(system, e)
mean_1000 = np.mean(y)
var_1000 = np.var(y)

T = 10000
e = np.random.normal(loc=mean, scale=std, size=T)
_, y = dlsim(system, e)
mean_10000 = np.mean(y)
var_10000 = np.var(y)

print(f'Experimental mean of y with 1000 samples: {mean_1000}')
print(f'Experimental variance of y with 1000 samples: {var_1000}')
print(f'Experimental mean of y with 10000 samples: {mean_10000}')
print(f'Experimental variance of y with 10000 samples: {var_10000}')

#%% Q2.e
pd.DataFrame({'theoretical':{'mean':3, 'variance':1.17},
              'experimental_1000':{'mean':mean_1000, 'variance':var_1000},
              'experimental_10000':{'mean':mean_10000, 'variance':var_10000}
              })

#%% Q2.f
lags = [20, 40, 80]
corr = Corr()
fig, axes = plt.subplots(len(lags), 1, figsize=(12, 9))
for i, lag in enumerate(lags):
    corr.acf(x=y, max_lag=lag, ax=axes[i])
    axes[i].set_title(f'ACF of MA(2) with lag {lag}')
plt.tight_layout()
plt.show()

#%% Q3.b
num = [1, 0.1, 0.4]
den = [1, -0.5, -0.2]
system = (num, den, 1)
_, y = dlsim(system, e)
plt.figure()
plt.plot(range(T), y[:, 0])
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.title('ARMA(2,2)')
plt.show()

#%% Q3.c
print(f'Experimental mean of y: {np.mean(y)}')
print(f'Experimental variance of y: {np.var(y)}')

#%%
T = 1000
e = np.random.normal(loc=mean, scale=std, size=T)
_, y = dlsim(system, e)
mean_1000 = np.mean(y)
var_1000 = np.var(y)

T = 10000
e = np.random.normal(loc=mean, scale=std, size=T)
_, y = dlsim(system, e)
mean_10000 = np.mean(y)
var_10000 = np.var(y)

print(f'Experimental mean of y with 1000 samples: {mean_1000}')
print(f'Experimental variance of y with 1000 samples: {var_1000}')
print(f'Experimental mean of y with 10000 samples: {mean_10000}')
print(f'Experimental variance of y with 10000 samples: {var_10000}')

#%% Q3.e
pd.DataFrame({'theoretical':{'mean':10, 'variance':3},
              'experimental_1000':{'mean':mean_1000, 'variance':var_1000},
              'experimental_10000':{'mean':mean_10000, 'variance':var_10000}
              })


#%%
lags = [20, 40, 80]
corr = Corr()
fig, axes = plt.subplots(len(lags), 1, figsize=(12, 9))
for i, lag in enumerate(lags):
    corr.acf(x=y, max_lag=lag, ax=axes[i])
    axes[i].set_title(f'ACF of ARMA(2,2) with lag {lag}')
plt.tight_layout()
plt.show()