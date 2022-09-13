import pandas_datareader as web
from Utilities.hypothesis_tests import *
from Utilities.basic_stats import *
import numpy as np

df = web.DataReader('AAPL', data_source='yahoo', start='2000-01-01', end='2022-09-07')

print(df.head())

plt.figure(figsize=(12,4))
df.Close.plot(lw=4)
plt.ylabel('USD($)')
plt.title('APPLE Closing Stock')
plt.grid(True)
plt.show()

# Plot rolling mean and variance

print('\nADF Results---------------------------')
ADF_Cal(df.Close)
print('\n\n')
print('KPSS Results---------------------------')
kpss_test(df.Close)

# Plot_Rolling_Mean_Var(df.Close)

noise = np.random.normal(loc=0, scale=1, size=1000)
noise = pd.Series(noise)
Plot_Rolling_Mean_Var(noise)
