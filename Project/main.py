#%% Load the libraries
from matplotlib import pylab
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os import chdir
from os.path import abspath, join
from statsmodels.tsa import holtwinters as ETS
from Utilities.WhitenessTest import WhitenessTest as WT
from Utilities.Correlation import Correlation as Corr
from statsmodels.tsa.seasonal import STL
import seaborn as sns

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

#%% Replace NaN's in O3 AQI
o3_aqi = uspoll[['State', 'Date Local', 'O3 AQI']]\
                .groupby(['State', 'Date Local'])\
                .agg('max').reset_index()

#%% Replace NaN's in NO2 AQI
no2_aqi = uspoll[['State', 'Date Local', 'NO2 AQI']]\
                .groupby(['State', 'Date Local'])\
                .agg('max').reset_index()

#%%
"""
We are aggregating values other than AQI since columns other than AQI have multiple records
for the same day, and the differences between the instances are subtle. Hence, an average of those
values would yield an equally sized daily data.
"""
non_cat_vars = uspoll.dtypes[uspoll.dtypes != 'object']
aqi_vars = ['CO AQI', 'NO2 AQI', 'SO2 AQI', 'O3 AQI']
uspoll_agg = uspoll[list(np.setdiff1d(non_cat_vars.index.values, aqi_vars)) + ['State']]\
                    .groupby(['State', 'Date Local'])\
                    .agg('mean').reset_index()

#%%
"""
Adding back all the AQI columns that we ignored in the aggregation step (last step) since
under the multiple records each day, only one of them has the value while the rest were filled
by NaN. This is why removed NaN prior to the aggregation, and now we join all AQIs with the
averaged data frame. 
"""
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
"""
We do an left outer join between the filtered uspoll_state data frame and the 
data frame that contain only the dates between the start and last date found in uspoll_state.
Just in case there are dates that have missing values under any column. This would allow us
to identify the dates on which no information was registered. Perhaps we could use forecasting
techniques to impute the data.
"""
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
# acronym nmissing correspond to not missing
co_aqi = subset_nmissing['CO AQI']
so2_aqi = subset_nmissing['SO2 AQI']
o3_aqi = subset_nmissing['O3 AQI']
no2_aqi = subset_nmissing['NO2 AQI']

#%%

""" --- FOR CO AQI
Iterate over different seasonality period and try fitting Holts Winter Seasonal method. Pick the 
seasonality period that yields the minimal MSE value.
"""

mse_list_co = [np.Inf, np.Inf]
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
    mse_list_co.append(mse)

#%%
""" --- FOR CO AQI
Using the seasonality period that produced the minimal MSE value, fit the data to the model as
we did not retain the best model from the previous step.
"""
seasonality_period = np.argmin(mse_list_co) + 1  # + 1 due to indices starting from 0

ets_co = ETS.ExponentialSmoothing(subset_nmissing['CO AQI'], trend=None, damped_trend=False,
                                  seasonal="mul", seasonal_periods=seasonality_period).fit()
"""
Use forecast method to extrapolate CO AQI to period the values were not registered.
"""
co_imputed_data = ets_co.forecast(steps=missing_dates.shape[0])

plt.figure()
plt.plot(subset_nmissing['CO AQI'].reset_index(drop=True), '-b')
plt.plot(ets_co.fittedvalues.reset_index(drop=True), '-r', alpha=0.8)
plt.ylabel('CO AQI')
plt.title('Carbon Monoxide Air Quality Index after Imputation')
plt.show()

#%%
"""
Now add the imputed values back to the CO AQI data frame
"""
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
""" --- FOR O3 AQI
Iterate over different seasonality period and try fitting Holts Winter Seasonal method. Pick the 
seasonality period that yields the minimal MSE value.
"""

mse_list_o3_aqi = [np.Inf, np.Inf]
# I believe using the test data to check for minimum MSE is not required at this stage
# as we are imputing just 5 values.
# o3_train, o3_test = o3_aqi.iloc[:train_len], \
#                     o3_aqi.iloc[train_len:]
for i in range(2, 400):
    ets_o3 = ETS.ExponentialSmoothing(o3_aqi, trend=None, damped_trend=False,
                                      seasonal="additive", seasonal_periods=i).fit()
    y_true = o3_aqi.reset_index(drop=True).values.reshape([-1])
    y_pred = ets_o3.fittedvalues.reset_index(drop=True).values.reshape([-1])
    diff = y_true - y_pred
    mse = 1/len(diff) * (diff.T @ diff)
    mse_list_o3_aqi.append(mse)


#%% O3 Holts Winter Seasonal Fit
""" --- FOR O3 AQI
Using the seasonality period that produced the minimal MSE value, fit the data to the model as
we did not retain the best model from the previous step.
"""
seasonality_period_o3_aqi = np.argmin(mse_list_o3_aqi) + 1 # + 1 due to indices starting from 0

ets_o3_aqi = ETS.ExponentialSmoothing(o3_aqi, trend=None, damped_trend=False,
               seasonal="additive", seasonal_periods=seasonality_period_o3_aqi).fit()
"""
Use forecast method to extrapolate O3 AQI to period the values were not registered.
"""
imputed_data_o3_aqi = ets_o3_aqi.forecast(steps=missing_dates.shape[0])

plt.figure()
plt.plot(o3_aqi, '-b')
plt.plot(ets_o3_aqi.fittedvalues, '-r')
plt.title('O3 Air Quality Index')
plt.ylabel('Air Quality Index')
plt.show()

#%% O3 AQI Holt Winter Seasonal Imputation
"""
Now add the imputed values back to the O3 AQI data frame
"""
uspoll_state.loc[missing_idx_start:missing_idx_end, 'O3 AQI'] = imputed_data_o3_aqi.reset_index()[0].values

#%% Impute the NO2 AQI
""" --- FOR NO2 AQI
Iterate over different seasonality period and try fitting Holts Winter Seasonal method. Pick the 
seasonality period that yields the minimal MSE value.
"""

mse_list_no2_aqi = [np.Inf, np.Inf]

for i in range(2, 400):
    ets_no2 = ETS.ExponentialSmoothing(no2_aqi, trend=None, damped_trend=False,
                                      seasonal="additive", seasonal_periods=i).fit()
    y_true = no2_aqi.reset_index(drop=True).values.reshape([-1])
    y_pred = ets_no2.fittedvalues.reset_index(drop=True).values.reshape([-1])
    diff = y_true - y_pred
    mse = 1/len(diff) * (diff.T @ diff)
    mse_list_no2_aqi.append(mse)


#%% NO2 Holts Winter Seasonal Fit
""" --- FOR NO2 AQI
Using the seasonality period that produced the minimal MSE value, fit the data to the model as
we did not retain the best model from the previous step.
"""
seasonality_period_no2_aqi = np.argmin(mse_list_no2_aqi) + 1 # + 1 due to indices starting from 0

ets_no2_aqi = ETS.ExponentialSmoothing(no2_aqi, trend=None, damped_trend=False,
               seasonal="additive", seasonal_periods=seasonality_period_no2_aqi).fit()
"""
Use forecast method to extrapolate NO2 AQI to period the values were not registered.
"""
imputed_data_no2_aqi = ets_no2_aqi.forecast(steps=missing_dates.shape[0])

plt.figure()
plt.plot(no2_aqi, '-b')
plt.plot(ets_no2_aqi.fittedvalues, '-r')
plt.title('NO2 Air Quality Index')
plt.ylabel('Air Quality Index')
plt.show()

#%% NO2 AQI Holt Winter Seasonal Imputation
"""
Now add the imputed values back to the NO2 AQI data frame
"""
uspoll_state.loc[missing_idx_start:missing_idx_end, 'NO2 AQI'] = imputed_data_no2_aqi.reset_index()[0].values

#%% Impute the SO2 AQI
""" --- FOR SO2 AQI
Iterate over different seasonality period and try fitting Holts Winter Seasonal method. Pick the 
seasonality period that yields the minimal MSE value.
"""

mse_list_so2_aqi = [np.Inf, np.Inf]

for i in range(2, 400):
    ets_so2 = ETS.ExponentialSmoothing(so2_aqi, trend=None, damped_trend=False,
                                      seasonal="additive", seasonal_periods=i).fit()
    y_true = so2_aqi.reset_index(drop=True).values.reshape([-1])
    y_pred = ets_so2.fittedvalues.reset_index(drop=True).values.reshape([-1])
    diff = y_true - y_pred
    mse = 1/len(diff) * (diff.T @ diff)
    mse_list_so2_aqi.append(mse)


#%% SO2 Holts Winter Seasonal Fit
""" --- FOR SO2 AQI
Using the seasonality period that produced the minimal MSE value, fit the data to the model as
we did not retain the best model from the previous step.
"""
seasonality_period_so2_aqi = np.argmin(mse_list_so2_aqi) + 1 # + 1 due to indices starting from 0

ets_so2_aqi = ETS.ExponentialSmoothing(so2_aqi, trend=None, damped_trend=False,
               seasonal="additive", seasonal_periods=seasonality_period_so2_aqi).fit()
"""
Use forecast method to extrapolate SO2 AQI to period the values were not registered.
"""
imputed_data_so2_aqi = ets_so2_aqi.forecast(steps=missing_dates.shape[0])

plt.figure()
plt.plot(so2_aqi, '-b')
plt.plot(ets_so2_aqi.fittedvalues, '-r')
plt.title('SO2 Air Quality Index')
plt.ylabel('Air Quality Index')
plt.show()

#%% SO2 AQI Holt Winter Seasonal Imputation
"""
Now add the imputed values back to the SO2 AQI data frame
"""
uspoll_state.loc[missing_idx_start:missing_idx_end, 'SO2 AQI'] = imputed_data_so2_aqi.reset_index()[0].values

#%%
"""
Weather Dataset is now included to aid in the process of regression.
"""

#%% Merge two dataframes
weather_df = pd.read_csv('arizona_weather.csv', header=0)
weather_df['Date Local'] = pd.to_datetime(weather_df['Date Local'])
joined_df = pd.merge(left=weather_df, right=uspoll_state.reset_index(), how='inner',
                     on='Date Local', sort=False)

# %%
"""
CORRELATION AND WHITENESS TEST-----------------------------------------------------------------
"""
#%% Correlation plot

fig = plt.figure(figsize=(15, 15))
sns.heatmap(joined_df.corr(method='pearson'), linewidths=0.8)
plt.tick_params(axis='both', labelsize=20)
plt.title('Correlation Plot', fontsize=20)
fig.tight_layout()
plt.show()


#%%
wt = WT(joined_df['NO2 AQI'])
wt.Plot_Rolling_Mean_Var(name='NO2 Air Quality Index')

wt = WT(joined_df['CO AQI'])
wt.Plot_Rolling_Mean_Var(name='CO Air Quality Index')

wt = WT(joined_df['SO2 AQI'])
wt.Plot_Rolling_Mean_Var(name='SO2 Air Quality Index')

wt = WT(joined_df['O3 AQI'])
wt.Plot_Rolling_Mean_Var(name='O3 Air Quality Index')

#%%
"""--------------------------------------------------------------------------
Make the dependent variable(s) stationary
--------------------------------------------------------------------------"""
# CO AQI
# stl = STL(endog=joined_df['CO AQI'], period=12).fit()
# trend = stl.trend
# seasonal = stl.seasonal
# residual = stl.resid
#
# de_seasonal = trend * residual
"""
RESUME WORK HERE \/\/\/\/\/\/\\/\/\/\/\/\/\\/\/\/\/\/\/\\/\/\/\/\/\/\\/\/\/\/\/\/\\/\/\/\/\/\/\
"""
co_log = np.log(joined_df['CO AQI'])
co_log_diff = co_log.diff()

plt.figure()
plt.plot(co_log_diff.diff()[2:])
plt.show()

wt = WT(co_log_diff.diff()[2:])
wt.Plot_Rolling_Mean_Var(name='CO AQI log transformed and differenced')
plt.show()


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










