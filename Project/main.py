#%% Load the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from os import chdir
from os.path import abspath, join
from statsmodels.tsa import holtwinters as ETS
from Utilities.WhitenessTest import WhitenessTest as WT
from Utilities.Correlation import Correlation as Corr

#%% Load the data
uspoll = pd.read_csv('pollution_us_2000_2016.csv', header=0, index_col=0)
uspoll['Date Local'] = pd.to_datetime(uspoll['Date Local'])

#%% Dtypes
print(uspoll.dtypes)

#%% First and the last date
first_date_by_state = uspoll[['State', 'Date Local']].groupby(['State']).agg('first')
last_date_by_state = uspoll[['State', 'Date Local']].groupby(['State']).agg('last')

#%%
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].stem(first_date_by_state.index.values, first_date_by_state['Date Local'])
axes[0].axhline(y=first_date_by_state.min(), color='r')
axes[0].tick_params(labelrotation=90)
axes[0].set_ylabel('Start Date')
axes[0].set_title('Start date in each state')

axes[1].stem(last_date_by_state.index.values, last_date_by_state['Date Local'])
# axes[1].axhline(y=last_date_by_state.max(), color='r')
axes[1].tick_params(labelrotation=90)
axes[1].set_ylabel('Last Date')
axes[1].set_title('Last date in each state')
plt.show()

#%% Replace NaN's in CO AQI
co_aqi = uspoll[['State', 'Date Local', 'CO AQI']]\
                    .groupby(['State', 'Date Local'])\
                    .agg('max').reset_index()

#%%
non_cat_vars = uspoll.dtypes[uspoll.dtypes != 'object']
uspoll_agg = uspoll[list(np.setdiff1d(non_cat_vars.index.values, 'CO AQI')) + ['State']]\
                    .groupby(['State', 'Date Local'])\
                    .agg('mean').reset_index()

#%%
uspoll_agg = pd.merge(uspoll_agg, co_aqi, on=['State', 'Date Local'],
                      how='inner')

#%% List of States that are maximally sized
fdbs = first_date_by_state
ldbs = last_date_by_state
states_with_min_sy = (fdbs.loc[((fdbs == fdbs.min()).reset_index(drop=True)['Date Local']).values].reset_index())['State']
states_with_max_ey = (ldbs.loc[((ldbs == ldbs.mode().values[0]).reset_index(drop=True)['Date Local']).values].reset_index())['State']
states_equally_sized = np.intersect1d(states_with_min_sy, states_with_max_ey)

#%%
# It is my understanding that the using one of the states from the equally sized set
# will not affect the analysis by any means
chosen_state = states_equally_sized[0]
uspoll_state = uspoll_agg.loc[uspoll_agg.State == chosen_state]
print(uspoll_state.head())

#%%
original_date_range = pd.date_range(start='2000-01-01', end='2016-03-31',
                                    freq='D', inclusive='both')
original_date_range = pd.DataFrame({'Date Local':original_date_range})

uspoll_state = pd.merge(left=original_date_range, right=uspoll_state, how='left',
                        on='Date Local', sort=False, copy=True)

#%%
missing_dates = uspoll_state.loc[uspoll_state["CO AQI"].isnull(), "Date Local"]
print(f'List of dates at which values are missing:\n'
      f'{missing_dates}')

#%% Let's plot, and visualize the data until the point in time the values are incomplete
subset_nmissing = uspoll_state.loc[uspoll_state['Date Local'] < missing_dates.min(), ['Date Local', 'CO AQI']]
plt.figure()
subset_nmissing['CO AQI'].plot()
plt.show()


#%%
subset_nmissing = subset_nmissing.set_index('Date Local').asfreq('D')

#%%
mse_list = [np.Inf, np.Inf]
for i in range(2, 380):
    ets = ETS.ExponentialSmoothing(subset_nmissing, trend=None, damped_trend=False,
                                   seasonal="mul", seasonal_periods=i).fit()
    y_true = subset_nmissing.reset_index(drop=True).values.reshape([-1])
    y_pred = ets.fittedvalues.reset_index(drop=True).values.reshape([-1])
    diff = y_true - y_pred
    mse = 1/len(diff) * (diff.T @ diff)
    mse_list.append(mse)

#%%
seasonality_period = np.argmin(mse_list) + 1 # + 1 due to indices starting from 0

ets = ETS.ExponentialSmoothing(subset_nmissing, trend=None, damped_trend=False,
          seasonal="mul", seasonal_periods=375).fit()
imputed_data = ets.forecast(steps=missing_dates.shape[0])

plt.figure()
plt.plot(subset_nmissing.reset_index(drop=True), '-b')
plt.plot(ets.fittedvalues.reset_index(drop=True), '-r')
# plt.xlim([0, 30])
plt.show()

#%%
















