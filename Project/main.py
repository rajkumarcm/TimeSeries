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
from numpy.testing import assert_equal

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
def seasonal_naive_forecast(df, vars, m):
    def forecast(X, last_tr_index, test_length):
        # Keep everything as indices
        T = X.loc[:last_tr_index].shape[0] - 1
        for h in range(1, test_length + 1):
            k = int((h - 1) / m)
            index = T + h - (m * (k + 1))
            X.iloc[T + h] = X.iloc[index]
        return X

    indices = df.index
    for var in vars:
        missing_indices = np.where(df.loc[:, var].isna())[0]
        if len(missing_indices) > 0:
            prev_time_index = indices[missing_indices[0] - 1]
            # train_length = len(indices[:(missing_indices[0]-1)])
            df.loc[:, var] = forecast(df[var], last_tr_index=prev_time_index,
                                      test_length=len(missing_indices))
    return df

uspoll_state = uspoll_state.set_index('Date Local')
uspoll_state.loc[:, aqi_vars] = seasonal_naive_forecast(df=uspoll_state, vars=aqi_vars, m=375)
# print('debug...')

#%%
#
# """ --- FOR CO AQI
# Iterate over different seasonality period and try fitting Holts Winter Seasonal method. Pick the
# seasonality period that yields the minimal MSE value.
# """
#
# mse_list_co = [np.Inf, np.Inf]
# n = co_aqi.shape[0]
# train_ratio = 0.7
# train_len = int(train_ratio * n)
# co_train, co_test = subset_nmissing['CO AQI'].iloc[:train_len], \
#                     subset_nmissing['CO AQI'].iloc[train_len:]
#
# for i in range(2, 400):
#     ets_co = ETS.ExponentialSmoothing(co_train, trend=None, damped_trend=False,
#                                       seasonal="mul", seasonal_periods=i).fit()
#     # y_true = co_train.reset_index(drop=True).values.reshape([-1])
#     # y_pred = ets_co.fittedvalues.reset_index(drop=True).values.reshape([-1])
#     y_true = co_test.reset_index(drop=True).values.reshape([-1])
#     y_pred = ets_co.forecast(steps=co_test.shape[0])
#     diff = y_true - y_pred
#     mse = 1/len(diff) * (diff.T @ diff)
#     mse_list_co.append(mse)
#
# #%%
# """ --- FOR CO AQI
# Using the seasonality period that produced the minimal MSE value, fit the data to the model as
# we did not retain the best model from the previous step.
# """
# seasonality_period = np.argmin(mse_list_co) + 1  # + 1 due to indices starting from 0
#
# ets_co = ETS.ExponentialSmoothing(subset_nmissing['CO AQI'], trend=None, damped_trend=False,
#                                   seasonal="mul", seasonal_periods=seasonality_period).fit()
# """
# Use forecast method to extrapolate CO AQI to period the values were not registered.
# """
# co_imputed_data = ets_co.forecast(steps=missing_dates.shape[0])
#
# plt.figure()
# plt.plot(subset_nmissing['CO AQI'].reset_index(drop=True), '-b')
# plt.plot(ets_co.fittedvalues.reset_index(drop=True), '-r', alpha=0.8)
# plt.ylabel('CO AQI')
# plt.title('Carbon Monoxide Air Quality Index after Imputation')
# plt.show()
#
# #%%
# """
# Now add the imputed values back to the CO AQI data frame
# """
# uspoll_state = uspoll_state.set_index('Date Local').asfreq('D')
# missing_idx_start = missing_dates.reset_index()['Date Local'].min()
# missing_idx_end = missing_dates.reset_index()['Date Local'].max()
# # reset_index()[0] should return CO AQI col
# uspoll_state.loc[missing_idx_start:missing_idx_end, 'CO AQI'] = co_imputed_data.reset_index()[0].values
# uspoll_state.loc[missing_idx_start:missing_idx_end, 'State'] = 'Arizona'
#
# #%%
# # stl = STL(o3_aqi, period=13)
# # res = stl.fit()
# # R = res.resid
# # T = res.trend
# # S = res.seasonal
# # strength_of_trend = np.max([0, (1-(R.var()/(T+R).var()))])
# # strength_of_seasonality = np.max([0, (1-(R.var()/(S+R).var()))])
# # print(f'Strength of the trend {100 * strength_of_trend:.2f}')
# # print(f'Strength of the seasonality {100 * strength_of_seasonality:.2f}')
#
#
# #%% Impute the O3 AQI
# """ --- FOR O3 AQI
# Iterate over different seasonality period and try fitting Holts Winter Seasonal method. Pick the
# seasonality period that yields the minimal MSE value.
# """
#
# mse_list_o3_aqi = [np.Inf, np.Inf]
# # I believe using the test data to check for minimum MSE is not required at this stage
# # as we are imputing just 5 values.
# # o3_train, o3_test = o3_aqi.iloc[:train_len], \
# #                     o3_aqi.iloc[train_len:]
# for i in range(2, 400):
#     ets_o3 = ETS.ExponentialSmoothing(o3_aqi, trend=None, damped_trend=False,
#                                       seasonal="additive", seasonal_periods=i).fit()
#     y_true = o3_aqi.reset_index(drop=True).values.reshape([-1])
#     y_pred = ets_o3.fittedvalues.reset_index(drop=True).values.reshape([-1])
#     diff = y_true - y_pred
#     mse = 1/len(diff) * (diff.T @ diff)
#     mse_list_o3_aqi.append(mse)
#
#
# #%% O3 Holts Winter Seasonal Fit
# """ --- FOR O3 AQI
# Using the seasonality period that produced the minimal MSE value, fit the data to the model as
# we did not retain the best model from the previous step.
# """
# seasonality_period_o3_aqi = np.argmin(mse_list_o3_aqi) + 1 # + 1 due to indices starting from 0
#
# ets_o3_aqi = ETS.ExponentialSmoothing(o3_aqi, trend=None, damped_trend=False,
#                seasonal="additive", seasonal_periods=seasonality_period_o3_aqi).fit()
# """
# Use forecast method to extrapolate O3 AQI to period the values were not registered.
# """
# imputed_data_o3_aqi = ets_o3_aqi.forecast(steps=missing_dates.shape[0])
#
# plt.figure()
# plt.plot(o3_aqi, '-b')
# plt.plot(ets_o3_aqi.fittedvalues, '-r')
# plt.title('O3 Air Quality Index')
# plt.ylabel('Air Quality Index')
# plt.show()
#
# #%% O3 AQI Holt Winter Seasonal Imputation
# """
# Now add the imputed values back to the O3 AQI data frame
# """
# uspoll_state.loc[missing_idx_start:missing_idx_end, 'O3 AQI'] = imputed_data_o3_aqi.reset_index()[0].values
#
# #%% Impute the NO2 AQI
# """ --- FOR NO2 AQI
# Iterate over different seasonality period and try fitting Holts Winter Seasonal method. Pick the
# seasonality period that yields the minimal MSE value.
# """
#
# mse_list_no2_aqi = [np.Inf, np.Inf]
#
# for i in range(2, 400):
#     ets_no2 = ETS.ExponentialSmoothing(no2_aqi, trend=None, damped_trend=False,
#                                       seasonal="additive", seasonal_periods=i).fit()
#     y_true = no2_aqi.reset_index(drop=True).values.reshape([-1])
#     y_pred = ets_no2.fittedvalues.reset_index(drop=True).values.reshape([-1])
#     diff = y_true - y_pred
#     mse = 1/len(diff) * (diff.T @ diff)
#     mse_list_no2_aqi.append(mse)
#
#
# #%% NO2 Holts Winter Seasonal Fit
# """ --- FOR NO2 AQI
# Using the seasonality period that produced the minimal MSE value, fit the data to the model as
# we did not retain the best model from the previous step.
# """
# seasonality_period_no2_aqi = np.argmin(mse_list_no2_aqi) + 1 # + 1 due to indices starting from 0
#
# ets_no2_aqi = ETS.ExponentialSmoothing(no2_aqi, trend=None, damped_trend=False,
#                seasonal="additive", seasonal_periods=seasonality_period_no2_aqi).fit()
# """
# Use forecast method to extrapolate NO2 AQI to period the values were not registered.
# """
# imputed_data_no2_aqi = ets_no2_aqi.forecast(steps=missing_dates.shape[0])
#
# plt.figure()
# plt.plot(no2_aqi, '-b')
# plt.plot(ets_no2_aqi.fittedvalues, '-r')
# plt.title('NO2 Air Quality Index')
# plt.ylabel('Air Quality Index')
# plt.show()
#
# #%% NO2 AQI Holt Winter Seasonal Imputation
# """
# Now add the imputed values back to the NO2 AQI data frame
# """
# uspoll_state.loc[missing_idx_start:missing_idx_end, 'NO2 AQI'] = imputed_data_no2_aqi.reset_index()[0].values
#
# #%% Impute the SO2 AQI
# """ --- FOR SO2 AQI
# Iterate over different seasonality period and try fitting Holts Winter Seasonal method. Pick the
# seasonality period that yields the minimal MSE value.
# """
#
# mse_list_so2_aqi = [np.Inf, np.Inf]
#
# for i in range(2, 400):
#     ets_so2 = ETS.ExponentialSmoothing(so2_aqi, trend=None, damped_trend=False,
#                                       seasonal="additive", seasonal_periods=i).fit()
#     y_true = so2_aqi.reset_index(drop=True).values.reshape([-1])
#     y_pred = ets_so2.fittedvalues.reset_index(drop=True).values.reshape([-1])
#     diff = y_true - y_pred
#     mse = 1/len(diff) * (diff.T @ diff)
#     mse_list_so2_aqi.append(mse)
#
#
# #%% SO2 Holts Winter Seasonal Fit
# """ --- FOR SO2 AQI
# Using the seasonality period that produced the minimal MSE value, fit the data to the model as
# we did not retain the best model from the previous step.
# """
# seasonality_period_so2_aqi = np.argmin(mse_list_so2_aqi) + 1 # + 1 due to indices starting from 0
#
# ets_so2_aqi = ETS.ExponentialSmoothing(so2_aqi, trend=None, damped_trend=False,
#                seasonal="additive", seasonal_periods=seasonality_period_so2_aqi).fit()
# """
# Use forecast method to extrapolate SO2 AQI to period the values were not registered.
# """
# imputed_data_so2_aqi = ets_so2_aqi.forecast(steps=missing_dates.shape[0])
#
# plt.figure()
# plt.plot(so2_aqi, '-b')
# plt.plot(ets_so2_aqi.fittedvalues, '-r')
# plt.title('SO2 Air Quality Index')
# plt.ylabel('Air Quality Index')
# plt.show()
#
# #%% SO2 AQI Holt Winter Seasonal Imputation
# """
# Now add the imputed values back to the SO2 AQI data frame
# """
# uspoll_state.loc[missing_idx_start:missing_idx_end, 'SO2 AQI'] = imputed_data_so2_aqi.reset_index()[0].values

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
Step 7 - Make the dependent variable(s) stationary
--------------------------------------------------------------------------"""

co_log = np.log(joined_df['CO AQI'])
co_log_diff = co_log.diff()[1:]
co_log_diff = co_log_diff.diff()[2:]

plt.figure()
plt.plot(co_log_diff)
plt.title('CO AQI Log transformed and differenced')
wt = WT(co_log_diff)
wt.Plot_Rolling_Mean_Var(name='CO AQI log transformed and 2nd order differenced')
plt.show()

no2_log = np.log(joined_df['NO2 AQI'])
no2_log_diff = no2_log.diff()[1:]

plt.figure()
plt.plot(no2_log_diff)
plt.title('NO2 AQI Log transformed and differenced')
wt = WT(no2_log_diff)
wt.Plot_Rolling_Mean_Var(name='NO2 AQI log transformed and differenced')
plt.show()

stl = STL(joined_df['SO2 AQI'], period=12).fit()
so2_trend = stl.trend
so2_seasonal = stl.seasonal
so2_residual = stl.resid
sadj_so2 = so2_trend + so2_residual
stl = STL(sadj_so2, period=12).fit()
so2_trend = stl.trend
so2_seasonal = stl.seasonal
so2_residual = stl.resid
so2_stationary = so2_seasonal + so2_residual

plt.figure()
plt.plot(so2_stationary)
plt.title('SO2 AQI Deseasonalized, and detrended using STL')
wt = WT(so2_stationary)
wt.Plot_Rolling_Mean_Var(name='SO2 AQI Deseasonalized, and detrended using STL')
plt.show()

o3_log = np.log(joined_df['O3 AQI'])
o3_log_diff = o3_log.diff()[1:]

plt.figure()
plt.plot(o3_log_diff)
plt.title('O3 AQI Log transformed and differenced')
wt = WT(o3_log_diff)
wt.Plot_Rolling_Mean_Var(name='O3 AQI log transformed and differenced')
plt.show()

#%%
"""
ACF of the dependent variables
"""
corr = Corr()
fig, axes = plt.subplots(4, 1, figsize=(13, 12))
corr.acf(x=co_log_diff.reset_index(drop=True), max_lag=40, name='CO Log transformed and 2nd order differenced',
         ax=axes[0])
corr.acf(x=no2_log_diff.reset_index(drop=True), max_lag=40, name='NO2 Log transformed', ax=axes[1])
corr.acf(x=so2_stationary.reset_index(drop=True), max_lag=40, name='SO2 Log transformed', ax=axes[2])
corr.acf(x=o3_log_diff.reset_index(drop=True), max_lag=40, name='O3 Log transformed', ax=axes[3])
fig.tight_layout()
plt.show()

#%%
"""
PACF of the dependent variables
"""
from statsmodels.graphics.tsaplots import plot_pacf
fig, axes = plt.subplots(4, 1, figsize=(13, 12))
plot_pacf(co_log_diff, ax=axes[0], title='PACF of CO Log transformed and 2nd order differenced')
plot_pacf(no2_log_diff, ax=axes[1], title='PACF of NO2 Log transformed and differenced')
plot_pacf(so2_stationary, ax=axes[2], title='PACF of NO2 Log transformed and differenced')
plot_pacf(o3_log_diff, ax=axes[3], title='PACF of O3 Log transformed and differenced')
fig.tight_layout()
plt.show()

#%%
"""
ADF and KPSS test
"""
print("CO AQI stationarized data's ADF test")
co_wt = WT(x=co_log_diff)
co_wt.ADF_Cal()
print('As per the ADF test on CO AQI, the signal is stationary since the p-value (0) is less than '
      'the significance threshold that is 0.05, for 95% confidence level.')
print("\nCO AQI stationarized data's KPSS test")
co_wt.kpss_test()
print('As per the KPSS test on CO AQI, the signal is stationary since the p-value (0.1) is above '
      'the significance threshold that is 0.05, for 95% confidence level.')

print("\n\nNO2 AQI stationarized data's ADF test")
no2_wt = WT(x=no2_log_diff)
no2_wt.ADF_Cal()
print('As per the ADF test on NO2 AQI, the signal is stationary since the p-value (0) is less than '
      'the significance threshold that is 0.05, for 95% confidence level.')
print("\nNO2 AQI stationarized data's KPSS test")
no2_wt.kpss_test()
print('As per the KPSS test on NO2 AQI, the signal is stationary since the p-value (0.1) is above '
      'the significance threshold that is 0.05, for 95% confidence level.')

print("\n\nSO2 AQI stationarized data's ADF test")
so2_wt = WT(x=so2_stationary)
so2_wt.ADF_Cal()
print('As per the ADF test on SO2 AQI, the signal is stationary since the p-value (0) is less than '
      'the significance threshold that is 0.05, for 95% confidence level.')
print("\nSO2 AQI stationarized data's KPSS test")
so2_wt.kpss_test()
print('As per the KPSS test on SO2 AQI, the signal is stationary since the p-value (0.1) is above '
      'the significance threshold that is 0.05, for 95% confidence level.')

print("\n\nO3 AQI stationarized data's ADF test")
o3_wt = WT(x=o3_log_diff)
o3_wt.ADF_Cal()
print('As per the ADF test on O3 AQI, the signal is stationary since the p-value (0) is less than '
      'the significance threshold that is 0.05, for 95% confidence level.')
print("\nO3 AQI stationarized data's KPSS test")
o3_wt.kpss_test()
print('As per the KPSS test on O3 AQI, the signal is stationary since the p-value (0.1) is above '
      'the significance threshold that is 0.05, for 95% confidence level.')


#%%
"""
Step 8 - Time Series Decomposition
"""
co_stl = STL(endog=co_aqi, period=12).fit()
co_trend = co_stl.trend
co_seasonal = co_stl.seasonal
co_residual = co_stl.resid
co_stl.plot()
plt.show()

plt.figure()
plt.plot(co_trend * co_residual)
plt.title('Seasonally adjusted CO AQI data')
plt.show()

wt = WT(x=co_trend * co_residual)
wt.Plot_Rolling_Mean_Var(name='Seasonally adjusted data (STL)')
wt.ADF_Cal()
wt.kpss_test()


corr.acf(pd.Series(co_trend*co_residual).diff()[1:], max_lag=40)

no2_stl = STL(endog=no2_aqi, period=12).fit()
no2_trend = no2_stl.trend
no2_seasonal = no2_stl.seasonal
no2_residual = no2_stl.resid
no2_stl.plot()
plt.show()

so2_stl = STL(endog=so2_aqi, period=12).fit()
so2_trend = so2_stl.trend
so2_seasonal = so2_stl.seasonal
so2_residual = so2_stl.resid
so2_stl.plot()
plt.show()

o3_stl = STL(endog=o3_aqi, period=12).fit()
o3_trend = o3_stl.trend
o3_seasonal = o3_stl.seasonal
o3_residual = o3_stl.resid
o3_stl.plot()
plt.show()

#%%
"""
Implement Holt-Winters Method
|\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/|
"""

#%%
def naive_iterpolate(df, vars):
    df = df.set_index('Date Local')
    indices = df.index
    for var in vars:
        missing_indices = np.where(df.loc[:, var].isna())[0]
        for m_index in missing_indices:
            prev_time_index = indices[m_index - 1]
            time_index = indices[m_index]
            df.loc[time_index, var] = df.loc[prev_time_index, var]
    return df

#%%
"""
Impute one or very few missing values in feature columns
"""
data_types = joined_df.dtypes.reset_index(drop=True)
float_indices = np.where(data_types == "float64")
float_vars = joined_df.dtypes.reset_index().iloc[float_indices[0], 0]

joined_df.loc[:, float_vars.values] = naive_iterpolate(joined_df, float_vars.values).reset_index()
joined_df = joined_df.set_index('Date Local')


#%%
"""
Feature Selection - PCA
"""

"""Remove Year and Station Number from float vars"""
float_vars = float_vars[float_vars != "Year"]
float_vars = float_vars[float_vars != "Station_Number"]

from sklearn.decomposition import PCA

pca = PCA()
feature_cols = np.setdiff1d(float_vars, aqi_vars)
pca.fit(joined_df[feature_cols], so2_stationary)
# The following expression on the right side is expected to return values sorted in descending order
n_components = len(pca.explained_variance_ratio_[pca.explained_variance_ratio_ >= 0.01])

pca = PCA(n_components=n_components)
X = pca.fit_transform(joined_df[feature_cols], so2_stationary)

#%% Split into train, and test subsets
T = joined_df.shape[0]
test_ratio = 0.2
train_ratio = 1 - test_ratio
train_size = int(T * train_ratio)
test_size = int(T - train_size)

date_indices = joined_df.index
X_tr = joined_df.loc[date_indices[:train_size]]
X_ts = joined_df.loc[date_indices[:test_size]]

#%%
def avg_forecast(df, tr_size, ts_size):
    prediction = []
    for i in range(1, tr_size+1): # used as length rather than as an index
        prediction.append(np.mean(df.iloc[:i]))
    forecast = [prediction[-1]] * ts_size

    if assert_equal(tr_size, len(prediction)):
        pass

    return prediction, forecast

co_avg_pred_forecast = avg_forecast(joined_df['CO AQI'], train_size, test_size)
indices = joined_df.index
train_indices = indices[:train_size]
test_indices = indices[train_size:]
plt.figure()
plt.plot(joined_df.iloc[:train_size]['CO AQI'], '-b', label='Training')
plt.plot(train_indices, co_avg_pred_forecast[0], '-', color='orange', label='Prediction' )
plt.plot(joined_df.iloc[train_size:]['CO AQI'], '-g', label='Validation' )
plt.plot(test_indices, co_avg_pred_forecast[1], '-', color='maroon', label='Testing')
plt.ylabel('CO AQI')
plt.title('CO AQI Average method')
plt.legend()
plt.show()

#%%
def naive_forecast(df, tr_size, ts_size):
    prediction = []
    for i in range(1, tr_size+1): # used as length rather than as an index
        prediction.append(df.iloc[i-1])  #  np.mean()
    forecast = [prediction[-1]] * ts_size

    if assert_equal(tr_size, len(prediction)):
        pass

    return prediction, forecast

co_naive_pred_forecast = naive_forecast(joined_df['CO AQI'], train_size, test_size)
indices = joined_df.index
# train_indices = indices[:train_size]
# test_indices = indices[train_size:]
plt.figure()
plt.plot(joined_df.iloc[:train_size]['CO AQI'], '-b', label='Training')
plt.plot(train_indices, co_naive_pred_forecast[0], '-', color='orange', label='Prediction' )
plt.plot(joined_df.iloc[train_size:]['CO AQI'], '-g', label='Validation' )
plt.plot(test_indices, co_naive_pred_forecast[1], '-', color='maroon', label='Testing')
plt.ylabel('CO AQI')
plt.title('CO AQI Naive method')
plt.legend()
plt.show()

#%%

def snaive_pred_forecast(X, last_tr_index, test_length, m):
    # 1-step prediction
    prediction = []
    for t in range((m-1), train_size):
        k = int((t - 1) / m)
        index = t + 1 - (m * (k + 1))
        prediction.append(X.iloc[index])

    # h-step prediction
    T = X.loc[:last_tr_index].shape[0] - 1
    for h in range(1, test_length + 1):
        k = int((h - 1) / m)
        index = T + h - (m * (k + 1))
        X.iloc[T + h] = X.iloc[index]
    return X

last_tr_index = indices[train_size]

co_snaive_pred_forecast = snaive_pred_forecast(joined_df['CO AQI'], last_tr_index, test_size, m=375)
plt.figure()
plt.plot(joined_df.iloc[:train_size]['CO AQI'], '-b', label='Training')
plt.plot(train_indices, co_snaive_pred_forecast[0], '-', color='orange', label='Prediction' )
plt.plot(joined_df.iloc[train_size:]['CO AQI'], '-g', label='Validation' )
plt.plot(test_indices, co_snaive_pred_forecast[1], '-', color='maroon', label='Testing')
plt.ylabel('CO AQI')
plt.title('CO AQI Naive method')
plt.legend()
plt.show()
print('debug...')



















