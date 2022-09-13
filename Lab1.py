from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
from Utilities.basic_stats import *
from Utilities.transformation import *

# Load the dataset
df = pd.read_csv('../data/tute1.csv', header=None, sep=',',
                 names=['Date', 'Sales', 'AdBudget', 'GDP'],
                 skiprows=[0], on_bad_lines='warn')

# Parse date column
date_col = []
date_col_old = df.Date
for i in range(df.shape[0]):
    year = re.findall('[0-9]+', date_col_old[i])[0]
    year = int(year)
    if year >= 81 and year <= 99:
        year = '19' + str(year)
    else:
        year += 2000
    month = re.findall('[A-z]+', date_col_old[i])[0]
    date_col.append(month + '-' + str(year))
df.Date = date_col

# Set the date column as index
df.set_index('Date', inplace=True)

# 1. Plot
spacing=22
dates_spaced = []
for i in range(0, df.shape[0]):
    if i % spacing == 0:
        dates_spaced.append(df.index.values[i])
    else:
        dates_spaced.append('')
fig, ax = plt.subplots()
df.plot(ax=ax)
ax.legend(loc='upper right')
ax.set_ylabel('USD($)')
ax.set_title('Quarterly sales of a small-scale enterprise')
ax.set_xlabel('Period', labelpad=7)
ax.set_xticks(np.arange(len(dates_spaced)))
ax.set_xticklabels(dates_spaced)
# ax.set_xticks(np.arange(len(df.index.values)))
# ax.set_xticklabels(df.index.values)
# for label in ax.xaxis.get_ticklabels()[::spacing]:
#     label.set_visible(False)
plt.show()

# 2. Time Series Statistics
print(f'The Sales mean is {round(df.Sales.mean(), 2)} and has a variance of {round(df.Sales.var(), 2)} with a standard deviation of ' 
      f'{round(df.Sales.std(), 2)} Median: {round(df.Sales.median(), 2)}')
print(f'The AdBudget mean is {round(df.AdBudget.mean(), 2)} and has a variance of {round(df.AdBudget.var(), 2)} with a standard deviation of ' 
      f'{round(df.AdBudget.std(), 2)} Median: {round(df.AdBudget.median(), 2)}')
print(f'The GDP mean is {round(df.GDP.mean(), 2)} and has a variance of {round(df.GDP.var(), 2)} with a standard deviation of ' 
      f'{round(df.GDP.std(), 2)} Median: {round(df.GDP.median(), 2)}')

# Simple statistics plot - not part of the assignment requirement
plt.figure()
df.boxplot()
plt.ylabel('USD($)')
plt.title('Statistics of Quarterly Sales')
plt.show()

# 3. Plot Rolling Mean, Variance
Plot_Rolling_Mean_Var(df.Sales, 'Sales') # Fix the grid problem in the plot--------------------------
Plot_Rolling_Mean_Var(df.AdBudget, 'AdBudget') # Fix the grid problem in the plot--------------------------
Plot_Rolling_Mean_Var(df.GDP, 'GDP') # Fix the grid problem in the plot--------------------------

"""
4. Write down your observation about the plot of the mean and variance in the previous step. Is
   Sales, GDP and AdBudget stationary or not? Explain why?
   
   Although Sales data shows some sign of fluctuation, overall it appears that all three data - Sales, AdBudget, and GDP
   are stationary, given the rolling mean and the variance stabilises as more samples are included.
   
"""

#--------------EXPLAIN-------------------------------------------------------------

#5. Perform an ADF test
# a. Sales
print('\nADF results on Sales:')
print(f'{ADF_Cal(df.Sales)}\n')

# b. AdBudget
print('\nADF results on AdBudget:')
print(f'{ADF_Cal(df.AdBudget)}')

# c. GDP
print('\nADF results on GDP:')
print(f'{ADF_Cal(df.GDP)}')

"""
At confidence interval 95%, the Sales and the GPD appears stationary since their p-values are less than the 
significance threshold 0.05. In case of AdBudget, the p-value is greater than the significance threshold, which 
doesn't allow us to reject the null hypothesis so I would say AdBudget is non-stationary at 95% confidence.

At confidence interval 99%, all three data - Sales, AdBudget and GDP would be non-stationary given the p-values
for all three data are beyond the significance threshold 0.01. This means we fail to reject the null hypothesis, which
states for ADF test that the series has a unit root. In other words, the data is non-stationary.
"""

#6. KPSS test
# a. Sales
print('\nKPSS results on Sales:')
print(f'{kpss_test(df.Sales)}\n')

# b. AdBudget
print('\nKPSS results on AdBudget:')
print(f'{kpss_test(df.AdBudget)}')

# c. GDP
print('\nKPSS results on GDP:')
print(f'{kpss_test(df.GDP)}')

#-------------------------DO THE 7TH AND 8TH STEPS-----------------------------

# 7.1 - Plot the data
pass_df = pd.read_csv('../data/AirPassengers.csv', header=0, sep=',',
                      on_bad_lines='warn', names=['Year_Month', 'N_Passengers'], index_col=0)
spacing = 12
dates_spaced = []
for i in range(pass_df.shape[0]):
    if i%spacing == 0:
        dates_spaced.append(pass_df.index.values[i])
    else:
        dates_spaced.append(None)
fig, ax = plt.subplots(figsize=(12, 9))
pass_df.plot(ax=ax, rot=50)
ax.set_ylabel('Number of Passengers')
ax.set_title('Number of Air Passengers flew between 1949 - 1960')
ax.set_xticks(np.arange(len(dates_spaced)))
ax.set_xticklabels(dates_spaced)
plt.show()

# 7.2 - Display average, variance and standard deviation
print(f'Number of air passengers travelled each month on average would be {round(pass_df.N_Passengers.mean(), 2)} '
      f'and has a variance of {round(pass_df.N_Passengers.var(), 2)} with a standard deviation of ' 
      f'{round(pass_df.N_Passengers.std(), 2)} Median: {round(pass_df.N_Passengers.median(), 2)}')

# 7.3 - Plot the rolling mean and the variance
Plot_Rolling_Mean_Var(pass_df, 'Air Passengers')

# 7.4 -
"""
Write down your observation about the plot of the mean and variance in the previous step. Is
AirPassengers data stationary or not? Explain why?

The rolling mean and the variance of the AirPassengers time-series data does not appear to stabilise and continues
to increase. Hence the data is not stationary.
"""

# 7.5
print('\nADF results on the number of air passengers:')
print(f'{ADF_Cal(pass_df.N_Passengers)}\n')

"""
The p-value of the ADF test result is higher than the significance threshold for both 95% and 99% confidence interval.
Hence we fail to reject the null hypothesis, which states the time-series is non-stationary.
This result is intuitive and aligns with visual interpretation from the 7.1 and 7.2 steps showing the plot of the 
time-series itself and the rolling mean & variance respectively.
"""

print('\nKPSS results on the number of air passengers:')
print(f'{kpss_test(pass_df.N_Passengers)}')

"""
The p-value of the KPSS test result is 0.01
Although we could argue that 0.01 is not necessarily less than 0.01 and there is very less chance that subtle difference
could lead to the value being higher than 0.01. Hence my conclusion is that as per KPSS test results, the time-series
is non-stationary as we reject the null hypothesis in favor of the alternate hypothesis. In KPSS test, the null 
hypothesis and the alternate hypothesis are the reverse of ADF. Here the null hypothesis states the data is stationary
where as the alternate hypothesis signifies the time-series is a non-stationary signal.
"""

print('breakpoint...')
diff1 = difference(pass_df.N_Passengers)
rm, rv = Cal_rolling_mean_var(diff1)













