#%% Load the libraries
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from os import chdir
from os.path import abspath, join
from statsmodels.tsa import holtwinters as ETS
from Utilities.WhitenessTest import WhitenessTest as WT
from Utilities.Correlation import Correlation as Corr
from statsmodels.tsa.seasonal import STL

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
axes[1].axhline(y=last_date_by_state.mode(), color='r')
axes[1].tick_params(labelrotation=90)
axes[1].set_ylabel('Last Date')
axes[1].set_title('Last date in each state')
fig.tight_layout()
plt.show()

#%% Drop columns
"""
We will aggregate the values from all address within a State as the pollutant values
are not expected to vary by much.
"""
uspoll = uspoll.drop(columns=['State Code', 'County Code', 'Site Num',
                              'Address', 'County', 'City'])

#%% Replace NaN's in CO AQI
co_aqi = uspoll[['State', 'Date Local', 'CO AQI']]\
                    .groupby(['State', 'Date Local'])\
                    .agg('max').reset_index()

#%% Replace NaN's in SO2 AQI
so2_aqi = uspoll[['State', 'Date Local', 'SO2 AQI']]\
                .groupby(['State', 'Date Local'])\
                .agg('max').reset_index()

#%%
o3_aqi = uspoll[['State', 'Date Local', 'O3 AQI']]\
                .groupby(['State', 'Date Local'])\
                .agg('max').reset_index()

#%%
no2_aqi = uspoll[['State', 'Date Local', 'NO2 AQI']]\
                .groupby(['State', 'Date Local'])\
                .agg('max').reset_index()

#%%
non_cat_vars = uspoll.dtypes[uspoll.dtypes != 'object']
aqi_vars = ['CO AQI', 'NO2 AQI', 'SO2 AQI', 'O3 AQI']
uspoll_agg = uspoll[list(np.setdiff1d(non_cat_vars.index.values, aqi_vars)) + ['State']]\
                    .groupby(['State', 'Date Local'])\
                    .agg('mean').reset_index()

#%%
uspoll_agg = pd.merge(uspoll_agg, co_aqi, on=['State', 'Date Local'],
                      how='inner')
uspoll_agg = pd.merge(uspoll_agg, so2_aqi, on=['State', 'Date Local'],
                      how='inner')
uspoll_agg = pd.merge(uspoll_agg, no2_aqi, on=['State', 'Date Local'],
                      how='inner')
uspoll_agg = pd.merge(uspoll_agg, o3_aqi, on=['State', 'Date Local'],
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
print(f'List of dates for which values are missing:\n'
      f'{missing_dates}')

#%% Let's plot, and visualize the data until the point in time the values are incomplete
subset_nmissing = uspoll_state.loc[uspoll_state['Date Local'] < missing_dates.min(),
                                          ['Date Local'] + aqi_vars]\
                              .set_index('Date Local')\
                              .asfreq('D')

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
axes[0, 0].plot(subset_nmissing['CO AQI'])
axes[0, 0].set_ylabel('CO AQI')
axes[0, 0].set_title('Carbon Monoxide Air Quality Index')
axes[0, 1].plot(subset_nmissing['NO2 AQI'])
axes[0, 1].set_ylabel('NO2 AQI')
axes[0, 1].set_title('Nitrogen Dioxide Air Quality Index')
axes[1, 0].plot(subset_nmissing['SO2 AQI'])
axes[1, 0].set_ylabel('SO2 AQI')
axes[1, 0].set_title('Sulfur Dioxide Air Quality Index')
axes[1, 1].plot(subset_nmissing['O3 AQI'])
axes[1, 1].set_ylabel('O3 AQI')
axes[1, 1].set_title('Ground Level Ozone Air Quality Index')
fig.tight_layout()
plt.show()

#%%

co_aqi = subset_nmissing['CO AQI']
so2_aqi = subset_nmissing['SO2 AQI']
o3_aqi = subset_nmissing['O3 AQI']
no2_aqi = subset_nmissing['NO2 AQI']

#%%
mse_list = [np.Inf, np.Inf]
n = co_aqi.shape[0]
train_ratio = 0.7
train_len = int(train_ratio * n)
co_train, co_test = subset_nmissing['CO AQI'].iloc[:train_len], \
                    subset_nmissing['CO AQI'].iloc[train_len:]

for i in range(2, 400):
    ets_co = ETS.ExponentialSmoothing(co_train, trend=None, damped_trend=False,
                                      seasonal="mul", seasonal_periods=i).fit()
    # y_true = co_train.reset_index(drop=True).values.reshape([-1])
    # y_pred = ets_co.fittedvalues.reset_index(drop=True).values.reshape([-1])
    y_true = co_test.reset_index(drop=True).values.reshape([-1])
    y_pred = ets_co.forecast(steps=co_test.shape[0])
    diff = y_true - y_pred
    mse = 1/len(diff) * (diff.T @ diff)
    mse_list.append(mse)

#%%
seasonality_period = np.argmin(mse_list) + 1  # + 1 due to indices starting from 0

ets_co = ETS.ExponentialSmoothing(subset_nmissing['CO AQI'], trend=None, damped_trend=False,
                                  seasonal="mul", seasonal_periods=seasonality_period).fit()
co_imputed_data = ets_co.forecast(steps=missing_dates.shape[0])

plt.figure()
plt.plot(subset_nmissing['CO AQI'].reset_index(drop=True), '-b')
plt.plot(ets_co.fittedvalues.reset_index(drop=True), '-r', alpha=0.8)
plt.ylabel('CO AQI')
plt.title('Carbon Monoxide Air Quality Index after Imputation')
plt.show()

#%%
# second_part_nmissing_co = uspoll_state.loc[uspoll_state['Date Local'] > missing_dates.max(),
#                                            ['Date Local', 'CO AQI']]
# second_part_nmissing_co = second_part_nmissing_co.reset_index(drop=True).set_index('Date Local')
# imputed_data_o3_aqi_mod = pd.DataFrame({'Date Local':missing_dates.reset_index(drop=True),
#                                         'O3 AQI':imputed_data_o3_aqi.reset_index()[0]})
# imputed_data_o3_aqi_mod = imputed_data_o3_aqi_mod.set_index('Date Local')
# collated_o3_aqi = pd.concat([subset_nmissing_o3_aqi,
#                              imputed_data_o3_aqi_mod, second_part_nmissing_o3_aqi], axis=0)

uspoll_state = uspoll_state.set_index('Date Local').asfreq('D')
missing_idx_start = missing_dates.reset_index()['Date Local'].min()
missing_idx_end = missing_dates.reset_index()['Date Local'].max()
# reset_index()[0] should return CO AQI col
uspoll_state.loc[missing_idx_start:missing_idx_end, 'CO AQI'] = co_imputed_data.reset_index()[0].values
uspoll_state.loc[missing_idx_start:missing_idx_end, 'State'] = 'Arizona'

#%%
# stl = STL(o3_aqi, period=13)
# res = stl.fit()
# R = res.resid
# T = res.trend
# S = res.seasonal
# strength_of_trend = np.max([0, (1-(R.var()/(T+R).var()))])
# strength_of_seasonality = np.max([0, (1-(R.var()/(S+R).var()))])
# print(f'Strength of the trend {100 * strength_of_trend:.2f}')
# print(f'Strength of the seasonality {100 * strength_of_seasonality:.2f}')


#%% Impute the O3 AQI
#Holts Winter Optimization Loop

mse_list_o3_aqi = [np.Inf, np.Inf]
o3_train, o3_test = subset_nmissing['CO AQI'].iloc[:train_len], \
                    subset_nmissing['CO AQI'].iloc[train_len:]
for i in range(2, 400):
    ets_o3 = ETS.ExponentialSmoothing(o3_aqi, trend=None, damped_trend=False,
                                      seasonal="additive", seasonal_periods=i).fit()
    y_true = o3_aqi.reset_index(drop=True).values.reshape([-1])
    y_pred = ets_o3.fittedvalues.reset_index(drop=True).values.reshape([-1])
    diff = y_true - y_pred
    mse = 1/len(diff) * (diff.T @ diff)
    mse_list_o3_aqi.append(mse)


#%% O3 Holts Winter Seasonal Fit
seasonality_period_o3_aqi = np.argmin(mse_list_o3_aqi) + 1 # + 1 due to indices starting from 0

ets_o3_aqi = ETS.ExponentialSmoothing(subset_nmissing_o3_aqi, trend=None, damped_trend=False,
               seasonal="additive", seasonal_periods=seasonality_period_o3_aqi).fit()
imputed_data_o3_aqi = ets_o3_aqi.forecast(steps=missing_dates.shape[0])

plt.figure()
plt.plot(subset_nmissing_o3_aqi, '-b')
plt.plot(ets_o3_aqi.fittedvalues, '-r')
plt.title('O3 Air Quality Index')
plt.ylabel('Air Quality Index')
plt.show()

#%% O3 AQI Holt Winter Seasonal Imputation
second_part_nmissing_o3_aqi = uspoll_state.loc[uspoll_state['Date Local'] > missing_dates.max(), ['Date Local', 'O3 AQI']]
second_part_nmissing_o3_aqi = second_part_nmissing_o3_aqi.reset_index(drop=True).set_index('Date Local')
imputed_data_o3_aqi_mod = pd.DataFrame({'Date Local':missing_dates.reset_index(drop=True),
                                        'O3 AQI':imputed_data_o3_aqi.reset_index()[0]})
imputed_data_o3_aqi_mod = imputed_data_o3_aqi_mod.set_index('Date Local')
collated_o3_aqi = pd.concat([subset_nmissing_o3_aqi,
                             imputed_data_o3_aqi_mod, second_part_nmissing_o3_aqi], axis=0)



#%% Merge two dataframes
weather_df = pd.read_csv('arizona_weather.csv', header=0)
weather_df['Date Local'] = pd.to_datetime(weather_df['Date Local'])
joined_df = pd.merge(left=weather_df, right=collated_o3_aqi.reset_index(), how='inner',
                     on='Date Local', sort=False)

#%% Correlation plot

fig = plt.figure(figsize=(15, 15))
sns.heatmap(joined_df.corr(method='pearson'), linewidths=0.8)
plt.tick_params(axis='both', labelsize=20)
plt.title('Correlation Plot', fontsize=20)
fig.tight_layout()
plt.show()

#%%
plt.figure(figsize=(12, 8))
plt.plot(subset_nmissing_o3_aqi, '-b', label='Original Data')
plt.plot(imputed_data_o3_aqi_mod, '-r', label='Imputed Data')
plt.plot(second_part_nmissing_o3_aqi, '-b', label='Original Data')
plt.ylabel('AQI Value')
plt.title('Arizona Air Quality Index from 2000 - March 2016')
plt.show()


#%%
wt = WT(collated_o3_aqi)
wt.Plot_Rolling_Mean_Var(name='O3 Air Quality Index')

#%% De-Seasonalize the data
stl = STL(collated_o3_aqi)
res = stl.fit()
T = res.trend
S = res.seasonal
R = res.resid
res.plot()
Var_R = R.var()
Var_T = T.var()
Var_S = S.var()
strength_of_trend = np.max([0, 1-(Var_R/(T+R).var())])
strength_of_seasonality = np.max([0, 1-(Var_R/(S+R).var())])
print(f'Strength of the trend {100 * strength_of_trend:.2f}')
print(f'Strength of the seasonality {100 * strength_of_seasonality:.2f}')

#%%
deseasoned_o3_aqi = collated_o3_aqi - S.reset_index().set_axis(axis=1, labels=['Date Local', 'O3 AQI']).set_index('Date Local')
plt.figure()
deseasoned_o3_aqi.plot()
plt.show()

#%%
deseasoned_o3_aqi = np.log(collated_o3_aqi)
deseasoned_o3_aqi = deseasoned_o3_aqi.diff()


#%%
wt = WT(deseasoned_o3_aqi['O3 AQI'].iloc[1:])
fig, axes = plt.subplots(3, 1, figsize=(9, 15))
deseasoned_o3_aqi.plot(ax=axes[0])
axes[0].set_ylabel('O3 AQI Value')
axes[0].set_title('Stationary O3 AQI data')

wt.Plot_Rolling_Mean_Var(name='Stationary O3 AQI', axes=axes, start_idx=1)
plt.show()

#%%
corr = Corr()
corr.acf(x=deseasoned_o3_aqi['O3 AQI'].iloc[1:], max_lag=100)

#%% Second order differencing
deseasoned_o3_aqi_2nd_order = deseasoned_o3_aqi.diff().iloc[2:]
corr.acf(x=deseasoned_o3_aqi_2nd_order['O3 AQI'].iloc[1:], max_lag=100)

#%%
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
weather_df[['Rel_Humdity_Max', 'Rel_Humidity_Min', 'Rel_Humidity_Mean']].plot.box(showfliers=False, ax=axes[0])
axes[0].set_title('Relative Humidity')
axes[0].set_ylabel('Rel Humidity Value')

weather_df[['Air_Temp_Max', 'Air_Temp_Min', 'Air_Temp_Mean']].plot.box(showfliers=False, ax=axes[1])
axes[1].set_title('Air Temperature')
axes[1].set_ylabel('Degree Centigrade')
plt.show()










