#%% Load the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ETS
from Utilities.WhitenessTest import WhitenessTest as WT
from Utilities.Correlation import Correlation as Corr
from statsmodels.tsa.seasonal import STL
import seaborn as sns
from numpy.testing import assert_equal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from Utilities.GPAC import gpac_table
from scipy.stats import chi2

#%%
def whiteness_test(x, name):
    wt = WT(x)
    print(f"ADF test for {name}:\n")
    wt.ADF_Cal()
    print(f"\nKPSS test for {name}:\n")
    wt.kpss_test()
    wt.Plot_Rolling_Mean_Var(name=name)

#%%
def plot_acf_pacf(x, lags, name, xlims=None):
    r_idx = 0
    if xlims:
        fig, axes = plt.subplots(len(xlims), 2, sharex=False, sharey=True, figsize=(15, 10))
        for xlim in xlims:
            plot_acf(x, lags=lags, ax=axes[r_idx, 0])
            axes[r_idx, 0].set_xlim(xlim)
            axes[r_idx, 0].set_title(f'ACF of {name}')

            plot_pacf(x, lags=lags, ax=axes[r_idx, 1])
            axes[r_idx, 1].set_xlim(xlim)
            axes[r_idx, 1].set_title(f'PACF of {name}')

            r_idx += 1
    else:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
        plot_acf(x, lags=lags, ax=axes[0])
        axes[0].set_title(f'ACF of {name}')
        plot_pacf(x, lags=lags, ax=axes[1])
        axes[1].set_title(f'PACF of {name}')
    plt.tight_layout()
    plt.show()

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

#%% Replace NaN's in AQI variables

# Replace NaN's in CO AQI
co_aqi = uspoll[['State', 'Date Local', 'CO AQI']]\
                    .groupby(['State', 'Date Local'])\
                    .agg('max').reset_index()

# Replace NaN's in SO2 AQI
so2_aqi = uspoll[['State', 'Date Local', 'SO2 AQI']]\
                .groupby(['State', 'Date Local'])\
                .agg('max').reset_index()

# Replace NaN's in O3 AQI
o3_aqi = uspoll[['State', 'Date Local', 'O3 AQI']]\
                .groupby(['State', 'Date Local'])\
                .agg('max').reset_index()

# Replace NaN's in NO2 AQI
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
# It is my understanding that the using one of the states would suffice the requirement
# since the requirement is to have at least 5000 instances.
chosen_state = states_equally_sized[0]
uspoll_state = uspoll_agg.loc[uspoll_agg.State == chosen_state]
print(uspoll_state.head())

#%%
"""
We do an left outer join between the filtered uspoll_state data frame (data corresponding to Arizona) 
and the 
data frame that contain only the dates between the start and last date found in uspoll_state.
Just in case there are dates that have missing values under any column. This would allow us
to identify the dates on which no information was registered. Perhaps we could use forecasting
techniques to replace the missing values.
"""
original_date_range = pd.date_range(start='2000-01-01', end='2016-03-31',
                                    freq='D', inclusive='both')
original_date_range = pd.DataFrame({'Date Local':original_date_range})

uspoll_state = pd.merge(left=original_date_range, right=uspoll_state, how='left',
                        on='Date Local', sort=False, copy=True)

#%% Extract the dates under which the data is missing
missing_dates = uspoll_state.loc[uspoll_state["CO AQI"].isnull(), "Date Local"]
print(f'List of dates for which values are missing:\n'
      f'{missing_dates}')

#%% Let's plot, and visualize the data until the start of the missing values
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

#%% Visualize the complete data after replacing the missing values using Seasonal Naive method
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
axes[0, 0].plot(uspoll_state['CO AQI'])
axes[0, 0].set_ylabel('CO AQI')
axes[0, 0].set_title('Carbon Monoxide Air Quality Index')
axes[0, 1].plot(uspoll_state['NO2 AQI'])
axes[0, 1].set_ylabel('NO2 AQI')
axes[0, 1].set_title('Nitrogen Dioxide Air Quality Index')
axes[1, 0].plot(uspoll_state['SO2 AQI'])
axes[1, 0].set_ylabel('SO2 AQI')
axes[1, 0].set_title('Sulfur Dioxide Air Quality Index')
axes[1, 1].plot(uspoll_state['O3 AQI'])
axes[1, 1].set_ylabel('O3 AQI')
axes[1, 1].set_title('Ground Level Ozone Air Quality Index')
fig.tight_layout()
plt.show()


# HWS removed and stored in backup.txt
#%%
"""
Weather Dataset is now included to aid in the process of regression.
"""

#%% Merge two dataframes
weather_df = pd.read_csv('arizona_weather.csv', header=0)
weather_df['Date Local'] = pd.to_datetime(weather_df['Date Local'])
joined_df = pd.merge(left=weather_df, right=uspoll_state.reset_index(), how='inner',
                     on='Date Local', sort=False)
joined_df = joined_df.set_index('Date Local')

#%% ADF-test of raw data
wt_raw_no = WT(x=joined_df['NO2 AQI'])
print("ADF of raw NO2 AQI data\n")
wt_raw_no.ADF_Cal()

wt_raw_co = WT(x=joined_df['CO AQI'])
print("\nADF of raw CO AQI data\n")
wt_raw_co.ADF_Cal()

wt_raw_so = WT(x=joined_df['SO2 AQI'])
print("\nADF of raw SO2 AQI data\n")
wt_raw_so.ADF_Cal()

wt_raw_o3 = WT(x=joined_df['O3 AQI'])
print("\nADF of raw O3 AQI data\n")
wt_raw_o3.ADF_Cal()


#%% Rolling mean and the variance of raw dataset

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
Step 7 - Print the strength of seasonality and trend
--------------------------------------------------------------------------"""

def print_strength_seas_tren(x, name):
    stl = STL(x, period=12).fit()
    residual = stl.resid
    trend = stl.trend
    seasonal = stl.seasonal
    f_t = np.max([0, 1-(np.var(residual)/np.var(residual + trend))]) # denominator = seasonally adjusted data
    f_s = np.max([0, 1-(np.var(residual)/np.var(residual + seasonal))]) # denominator = detrended data
    print(f"Strength of trend in {name} is: {f_t * 100}%")
    print(f"Strength of seasonality in {name} is {f_s * 100}%")

print_strength_seas_tren(joined_df['CO AQI'], 'CO AQI')
print("")
print_strength_seas_tren(joined_df['SO2 AQI'], 'SO2 AQI')
print("")
print_strength_seas_tren(joined_df['NO2 AQI'], 'NO2 AQI')
print("")
print_strength_seas_tren(joined_df['O3 AQI'], 'O3 AQI')


#%%


#%%
"""---------------------------------------------
Make the CO AQI data stationary
---------------------------------------------"""

def seasonal_differencing(y, seasonal_period):
    m = seasonal_period
    s_diff = []
    for t in range(m, len(y)):
        s_diff.append(y[t] - y[t-m])
    return s_diff

# Split the data into train, and test sets
# N = joined_df.shape[0]
# train_test_split = int(0.7 * N)
# test_len = N - train_test_split
# co = joined_df['CO AQI'][:train_test_split]
# co_test = joined_df['CO AQI'][train_test_split:]

N = joined_df.shape[0]
train_test_split = int(0.7 * N)
test_len = N - train_test_split
co = joined_df['CO AQI'][:train_test_split]
co_test = joined_df['CO AQI'][train_test_split:]

# Plot the data constrained to a narrow window so it is easier to visualize the
# seasonality.
plt.figure()
co.reset_index(drop=True).head(1000).plot()
plt.xlabel('Observation index')
plt.title('CO AQI of Arizona (Original Data) - limited window')
plt.ylabel('AQI')
plt.tight_layout()
plt.show()

# Strength of Seasonality and trend in the raw dataset
print_strength_seas_tren(co, name='CO AQI Original Data')

# ADF, and KPSS test on original data
whiteness_test(co, name='CO AQI Original Data')

"""
ACF of the dependent variable
"""
plot_acf_pacf(joined_df['CO AQI'], name='Carbon Monoxide AQI', lags=100)
# plot_acf_pacf(joined_df['NO2 AQI'], name='Nitrogen Dioxide AQI', lags=100)
# plot_acf_pacf(joined_df['SO2 AQI'], name='Sulfur Dioxide AQI', lags=100)
# plot_acf_pacf(joined_df['O3 AQI'], name='Ground Level Ozone AQI', lags=100)

"""
The Carbon Monoxide AQI data was very stubborn in not lending itself
to not being able to transform a stationary data despite several attempts
in making seasonal and non-seasonal differencing. Hence Log transformation is
preferred
"""
co_log = np.log(co)
co_test_log = np.log(co_test)

# Plot of Raw and log transformed data
fig, axes = plt.subplots(2, 1, figsize=(9, 7))
axes[0].plot(co.head(1000), label='Original training data')
axes[0].set_title('Original CO AQI')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('AQI')
axes[0].legend()
axes[1].plot(co_log.head(1000), label='Log transformed training data')
axes[1].set_title('Log transformed CO AQI')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Log AQI')
axes[1].legend()
fig.tight_layout()
plt.show()

# Strength of Seasonality and trend after log transformation
print_strength_seas_tren(co, name='CO AQI Original data')
print('')
print_strength_seas_tren(co_log, name='CO AQI after log transformation')

# ADF, and KPSS test
whiteness_test(co_log, name='CO AQI after log transformation')

# ACF, and PACF of Log transformed CO AQI
plot_acf_pacf(co_log, lags=400, name='Log transformed CO AQI',
              xlims=[[-5, 400], [-1, 50], [-1, 10], [360, 400]])


"""
There is still some level of seasonality present in the data. Going for a non-seasonal differencing.
"""
co_diff1 = co_log.diff()[1:]

# Plot of Raw and log transformed data
fig, axes = plt.subplots(2, 1, figsize=(9, 7))
axes[0].plot(co_log.head(1000), label='Log transformed training data')
axes[0].set_title('Log transformed CO AQI')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Log AQI')
axes[0].legend()
axes[1].plot(co_diff1[:1000], label='Differenced data')
axes[1].set_title('X -> Log -> Diff(1) - Stationary CO AQI')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('AQI')
axes[1].legend()
fig.tight_layout()
plt.show()

# Strength of Seasonality and trend after log transformation
print_strength_seas_tren(co, name='CO AQI Original data')
print('')
print_strength_seas_tren(co_log, name='CO AQI after log transformation')
print('')
print_strength_seas_tren(co_diff1, name='CO AQI after log transformation followed by a non-seasonal differencing')

# ADF, and KPSS test
whiteness_test(co_diff1, name='CO AQI after log transformation followed by a non-seasonal differencing')

# ACF, and PACF of Log transformed CO AQI
plot_acf_pacf(co_diff1, lags=400, name='CO AQI after log transformation followed by a non-seasonal differencing',
              xlims=[[-5, 400], [-1, 50], [-1, 10], [360, 400]])

# Only at this stage, GPAC allows the orders to be derived

#%% GPAC
"""--------------------------------
Generalized Partial AutoCorrelation
--------------------------------"""
corr = Corr()
lags = int(co.shape[0]/50)
acf_vals, _ = corr.acf(co_diff1.reset_index(drop=True), max_lag=lags, plot=False, return_acf=True)
gpac_vals = gpac_table(acf_vals, na=13, nb=13, plot=False)
plt.figure(figsize=(13, 10))
sns.heatmap(gpac_vals, annot=True)
plt.xticks(ticks=np.array(list(range(13))) + .5, labels=list(range(1, 14)))
plt.title('Generalized Partial Autocorrelation (GPAC) Table')
plt.xlabel('AR Order')
plt.ylabel('MA Order')
plt.tight_layout()
plt.show()

#%% ARIMA
"""
Main model - ARIMA
"""
na = 4
d = 1
nb = 3
arima_fit = sm.tsa.ARIMA(endog=co_log, order=(4,1,3), trend='n').fit()
print(arima_fit.summary())

# Residual Analysis
y_hat = arima_fit.predict()
residuals = co_log.values - y_hat.values

# Plot ACF of the residuals
plt.figure()
plot_acf(residuals, lags=50)
plt.title('ACF of the residuals based on the predicted by ARIMA(4,1,3)')
plt.xlabel('Lag')
plt.ylabel(r'p(lag)')
plt.show()

# Compute the Q value
arima_Q = co.shape[0] * (acf_vals[1:].T @ acf_vals[1:])
chi2_from_table = chi2.ppf(q=0.95, df=co.shape[0]-na-nb-1)

print(f"Chi-square critical value Q {arima_Q} is less than the value from the"
      f"table {chi2_from_table}. Since the Q value did not enter the critical region"
      f"in the distribution, we refuse to reject the null hypothesis which states"
      f"the data exhibits no statistical significance. In simple words, the residuals"
      f"are white.")

print(f"More accurate diagnosis - Ljung-Box test:")
print(sm.stats.acorr_ljungbox(x=residuals, lags=[50]))

#%%
"""
Plot that shows the fit between the original data and the predicted data
"""
co_log_mean = np.mean(co_log)
co_log_total = (co_log - co_log_mean).T @ (co_log - co_log_mean)
arima_res_ss = (co_log - y_hat).T @ (co_log - y_hat)
arima_r2 = 1 - (arima_res_ss/co_log_total)

y_hat_exp = np.exp(y_hat)

fig, axes = plt.subplots(2, 1, figsize=(15, 15))
axes[0].plot(co, label='Training data in original scale')
axes[0].plot(y_hat_exp, label='Predicted data exponentiated')
axes[0].set_title('Quality of model fitness to the data')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('CO AQI')
axes[0].legend()

axes[1].plot(co.head(1000), label='Training data in original scale')
axes[1].plot(y_hat_exp.head(1000), label='Predicted data exponentiated')
axes[1].set_title('Quality of model fitness to the data - Limited window')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('CO AQI')
axes[1].legend()
fig.tight_layout()
plt.show()

plt.figure()
plt.plot(co.head(1000), label='Training data in original scale')
plt.plot(co.head(1000).index.values[1:], y_hat_exp.head(1000)[1:], label='Predicted data exponentiated')
plt.title('Quality of model fitness to the data - Limited window')
plt.xlabel('Date')
plt.ylabel('CO AQI')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

#%%
"""
ARIMA(4,1,3) - MANUAL H-STEP PREDICTION - FORECAST FUNCTION WITH BACK TRANSFORMATION
"""

# joined_df = pd.read_csv('joined_df.csv', header=0, index_col='Date Local')
co_x = joined_df['CO AQI'][:train_test_split+50].reset_index(drop=True)
co_train_arima = np.log(co_x[:-50]).diff()
co_test_arima = np.log(co_x[-50:]).diff()
e = co_train_arima.diff()

co_train_arima = co_train_arima.values
co_test_arima = co_test_arima.values

test_set = joined_df['CO AQI'][-50:].reset_index(drop=True).values

def h_step_pred(h, y, yhat, e, T):

    a1 = -0.7758
    a2 = -0.3869
    a3 = 0.3138
    a4 = -0.1282

    b1 = 0.4002
    b2 = -0.2245
    b3 = -0.8534

    lag_1 = T+h-1
    lag_2 = T+h-2
    lag_3 = T+h-3
    lag_4 = T+h-4

    if lag_1 > T:
        lag1_val = yhat[lag_1]
        lag1_eval = 0
    else:
        lag1_val = y[lag_1]
        lag1_eval = e[lag_1]

    if lag_2 > T:
        lag2_val = yhat[lag_2]
        lag2_eval = 0
    else:
        lag2_val = y[lag_2]
        lag2_eval = e[lag_2]

    if lag_3 > T:
        lag3_val = yhat[lag_3]
        lag3_eval = 0
    else:
        lag3_val = y[lag_3]
        lag3_eval = e[lag_3]

    if lag_4 > T:
        lag4_val = yhat[lag_4]
    else:
        lag4_val = y[lag_4]

    y_val = a1 * lag1_val + a2 * lag2_val + a3 * lag3_val + a4 * lag4_val +\
            b1 * lag1_eval + b2 * lag2_eval + b3 * lag3_eval

    return y_val

y_h_step_pred = co_train_arima
for h in range(1, 52):
    tmp_val = h_step_pred(h=h, y=co_train_arima, yhat=y_h_step_pred,
                          e=e, T=co_train_arima.shape[0]-1)
    y_h_step_pred = np.r_[y_h_step_pred, tmp_val]

def back_transformation(y, p):
    reversed = []
    for i in range(len(y)-1):
        reversed.append(y[i+1] - p[i+1])
    return reversed

last_50_hstep = y_h_step_pred[-51:][:50]

p = [last_50_hstep[0] - co_log[-1]]
for i in range(1, 50):
    p.append(last_50_hstep[i] - last_50_hstep[i-1])

reversed_h_step = back_transformation(last_50_hstep, p)
h_step_complete_bt = np.exp(reversed_h_step)


plt.figure()
plt.plot(co.head(1000), label='Training data in original scale')
plt.plot(co.head(1000).index.values[1:], y_hat_exp.head(1000)[1:], label='Predicted data exponentiated')
plt.title('Quality of model fitness to the data - Limited window')
plt.xlabel('Date')
plt.ylabel('CO AQI')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(co_test_arima, label='test set')
plt.plot(last_50_hstep, label='h-step')
plt.legend()
plt.title('ARIMA(4,1,3) h-step prediction')
plt.grid(True)
plt.show()

#%%
"""
---------------------------------------------------------------------------------------
Holt-Winters Seasonal Method
----------------------------------------------------------------------------------------
As the professor mentioned in the class for my question on how to estimate 
the seasonality period is a very difficult question to answer, I thought it would
be reasonable to do a grid-search. Perhaps the parameter with the minimal mse
could be considered as the optimal seasonality period. I believe this is a more
systematic way to approach the estimate the parameter rather than to making assumption
that may not necessarily hold or rationale at all circumstances.
"""
# Iterate over different seasonality period and try fitting Holts Winter Seasonal method. Pick the
# seasonality period that yields the minimal MSE value.
#

# mse_list_co = [np.Inf, np.Inf]
#
# for i in range(2, 400):
#     ets_co = ETS(co, trend=None, damped_trend=False,
#                                       seasonal="mul", seasonal_periods=i).fit()
#     y_true = co_test.reset_index(drop=True).values.reshape([-1])
#     y_pred = ets_co.forecast(steps=co_test.shape[0])
#     diff = y_true - y_pred
#     mse = 1/len(diff) * (diff.T @ diff)
#     mse_list_co.append(mse)
#
# # we add an extra 2 because the range starts from 2
# seasonal_period = np.argmin(mse_list_co) + 2
seasonal_period = 375  # for convenient debugging...
print(f"The estimated optimal seasonality period would be {seasonal_period}")

# Fitting HW Seasonal based on estimated seasonality period
ets = ETS(endog=co, seasonal='mul', seasonal_periods=seasonal_period,
          trend=None).fit()

ets_yhat = ets.predict(start=0, end=co.shape[0]-1)

co_mean = np.mean(co)
co_tot_ss = (co - co_mean).T @ (co - co_mean)
holt_res_ss = (co - ets_yhat).T @ (co - ets_yhat)
holt_r2 = 1 - (holt_res_ss/co_tot_ss)

# Plotting y vs the yhat to illustrate the goodness of fit of the model to the data
plt.figure()
plt.plot(co.head(100), label='training data')
plt.plot(ets_yhat.head(100), label='predicted')
plt.title('Holts Winter Seasonal Method')
plt.xlabel('Date')
plt.ylabel('CO AQI')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

ets_forecast = ets.forecast(steps=co_test.shape[0])
plt.figure()
plt.plot(co, label='training data')
plt.plot(ets_yhat, label='predicted')
plt.plot(co_test, label='test data')
plt.plot(co_test.index.values, ets_forecast.reset_index(drop=True), label='forecast data')
plt.title('Holts Winter Seasonal Method')
plt.xlabel('Date')
plt.ylabel('CO AQI')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

tmp_diff = co_test.reset_index(drop=True).values - ets_forecast.reset_index(drop=True).values
ets_forecast_rmse = np.sqrt(np.mean(tmp_diff.T @ tmp_diff))

#%% Replace missing values in the exogenous variables using naive method

# def naive_iterpolate(df, vars):
#     df = df.set_index('Date Local')
#     indices = df.index
#     for var in vars:
#         missing_indices = np.where(df.loc[:, var].isna())[0]
#         # -1 here because .min() refers to the start of missing indices
#         # and we want values from the last entry of the non-missing dataset
#         # Assuming the missing_indices are consecutive after visual verification.
#         for m_index in missing_indices:
#             prev_time_index = indices[m_index - 1]
#             time_index = indices[m_index]
#             df.loc[time_index, var] = df.loc[prev_time_index, var]
#     return df

def snaive_interpolate(df, vars):
    df = df.set_index('Date Local')
    indices = df.index
    for var in vars:
        missing_indices = np.where(df.loc[:, var].isna())[0]
        # -1 here because .min() refers to the start of missing indices
        # and we want values from the last entry of the non-missing dataset
        # Assuming the missing_indices are consecutive after visual verification.
        for m_index in missing_indices:
            prev_time_index = indices[m_index - seasonal_period]
            time_index = indices[m_index]
            df.loc[time_index, var] = df.loc[prev_time_index, var]
    return df

#%%
"""
Impute one or very few missing values in feature columns
"""
joined_df = joined_df.drop(columns=['Year', 'Day_of_Year', 'Station_Number'])
data_types = joined_df.dtypes
float_vars = data_types.index.values[data_types == "float64"]
joined_df = joined_df.reset_index()
joined_df.loc[:, float_vars] = snaive_interpolate(joined_df, float_vars).reset_index()
joined_df = joined_df.set_index('Date Local')

#%%
"""----------------------------------------
Beginning of Regression Analysis
----------------------------------------"""

#%% Remove unncessary variables - At this point, I only wish to limit the analysis to CO AQI
# and leave out the other AQIs.
import re
data_types = joined_df.dtypes
# float_indices = np.where(data_types == "float64")[0]
# float_vars = joined_df.dtypes.reset_index().iloc[float_indices, 0]
discarded_aqi = ['SO2', 'NO2', 'O3', 'CO']
cnames = joined_df.columns
# p_formula -> pollutant formula
p_cnames = []
for p_formula in discarded_aqi:
    pattern = f"{p_formula}+[\s\w]*"
    p_cnames = np.unique(np.r_[p_cnames, re.findall(pattern, ", ".join(cnames))])
useful_vars = np.setdiff1d(float_vars, p_cnames)
useful_vars = np.r_[useful_vars, ['CO AQI']]

#%% Correlation plot between exogenous variables
plt.figure(figsize=(9, 8))
sns.heatmap(joined_df[useful_vars].corr(), linewidths=.5)
plt.title('Correlation between Exogenous variables')
plt.tight_layout()
plt.show()

#%%
co = joined_df['CO AQI']
X = joined_df[np.setdiff1d(useful_vars, ['CO AQI'])]
X_train = X[:round(0.7 * X.shape[0])]
X_test = X[round(0.7 * X.shape[0]):]
y = co
y_train = y[:round(0.7 * co.shape[0])]
y_test = y[round(0.7 * co.shape[0]):]

print("Printing sample of independent variables Linear Regression - not preprocessed yet")
print(X_train.head())

# Standardization
mu = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_z = (X_train - mu)/std
X_test_z = (X_test - mu)/std

# Eigen analysis
cov = (1/(X_train_z.shape[0]-1)) * (X_train_z.T @ X_train_z)
print(f"Covariance matrix:\n{cov}")
vals, vecs = np.linalg.eig(cov)

# Include intercept to the X feature set
X_train_z['intercept'] = np.ones([X_train_z.shape[0], 1])
X_test_z['intercept'] = np.ones([X_test_z.shape[0], 1])

print("Printing sample of independent variables Linear Regression - standardized")
print(X_train_z.head())

"""------------------------
OLS on standardized data
-----------------------"""
ols_fit2 = sm.regression.linear_model.OLS(endog=np.log(y_train), exog=X_train_z).fit()
print("OLS summary on standardized data")
print(ols_fit2.summary())

# Make prediction using the OLS fitted
ols2_train_pred = ols_fit2.predict(exog=X_train_z)
ols2_test_pred = ols_fit2.predict(exog=X_test_z)
diff2 = y_test - np.exp(ols2_test_pred)
ols2_mse = (1/diff2.shape[0]) * (diff2.T @ diff2)
ols2_tot_ss = (y_test - np.mean(y_test)).T @ (y_test - np.mean(y_test))
ols2_res_ss = diff2.T @ diff2
ols2_r2 = 1 - (ols2_res_ss/ols2_tot_ss)

spaced_xlabels = []
for i, index_val in zip(range(X_train.shape[0]), X_train.index.values):
    if i%375 == 0:
        spaced_xlabels.append(index_val)
    else:
        spaced_xlabels.append(None)

# Plot the quality of fit
plt.figure()
plt.plot(y_train, label='Training data')
plt.plot(np.exp(ols_fit2.predict(X_train_z)), label='Prediction on training set')
plt.plot(y_test, label='Testing data')
plt.plot(np.exp(ols_fit2.predict(X_test_z)), label='Prediction on testing set')
plt.legend()
plt.title('OLS Model on standardized data')
plt.xlabel('Date')
plt.ylabel('CO AQI')
plt.show()

# Print the variance/information present in each axis
indices_desc = np.argsort(vals)[::-1]
vals_normalized = vals/np.max(vals)
vals_normalized = vals_normalized[indices_desc]
print(f"Proportion of variance/information in each basis vector that span the vector space:\n"
      f"{vals_normalized}")

# Decorrelate the data
whiten_W = vecs @ np.diag((1/vals)**.5) @ vecs.T
X_train_w = X_train_z[np.setdiff1d(X_train_z.columns, ['intercept'])] @ whiten_W
X_test_w = X_test_z[np.setdiff1d(X_test_z.columns, ['intercept'])] @ whiten_W

plt.figure()
sns.heatmap(X_train_w.corr(), linewidths=0.05, linecolor='black')
plt.title(f"Correlation plot of data decorrelated PCA with sphering")
plt.show()

# Include intercept for whitened data
X_train_w['intercept'] = np.ones([X_train_w.shape[0], 1])
X_test_w['intercept'] = np.ones([X_test_w.shape[0], 1])

"""-------------------------------------------------------------
OLS on orthgonally projected data that is rotation neutralized
The following transformation whitens the data to faciliate/boost the
linear regression
-------------------------------------------------------------"""
ols_fit = sm.regression.linear_model.OLS(endog=np.log(y_train), exog=X_train_w).fit()
print("Printing summary of OLS fitted on decorrelated data")
print(ols_fit.summary())

# Plot the quality of fit
plt.figure()
plt.plot(y_train, label='Training data')
plt.plot(np.exp(ols_fit.predict(X_train_w)), label='Prediction on training set')
plt.plot(y_test, label='Testing data')
plt.plot(np.exp(ols_fit.predict(X_test_w)), label='Prediction on testing set')
plt.xlabel('Date')
plt.ylabel('CO AQI')
plt.legend()
plt.title('OLS Model on Decorrelated data')
plt.show()

# Making prediction using the model fitted on whitened data
ols1_train_pred = ols_fit.predict(exog=X_train_w)
ols1_test_pred = ols_fit.predict(exog=X_test_w)

# MSE
diff1 = y_test - np.exp(ols1_test_pred)
ols1_mse = (1/diff1.shape[0]) * (diff1.T @ diff1)

ols1_tot_ss = (y_test - np.mean(y_test)).T @ (y_test - np.mean(y_test))
ols1_res_ss = diff1.T @ diff1
ols1_r2 = 1 - (ols1_res_ss/ols1_tot_ss)

# PCA - dimensionality reduction
vals = vals[indices_desc]
vecs = vecs[:, indices_desc]
informative_cols = vals_normalized >= 0.05
retained_evecs = vecs[:, informative_cols]

# Orthogonal projection following including a intercept variable
X_train_l = X_train_z[np.setdiff1d(X_train_z.columns, ['intercept'])]
X_train_l = X_train_l @ retained_evecs
X_train_l['intercept'] = np.ones([X_train_l.shape[0], 1])

X_test_l = X_test_z[np.setdiff1d(X_test_z.columns, ['intercept'])]
X_test_l = X_test_l @ retained_evecs
X_test_l['intercept'] = np.ones([X_test_l.shape[0], 1])

"""-------------------------------------------------------------
OLS on data transformed onto low dimensional subspace
using orthogonal projection - PCA.
-------------------------------------------------------------"""
ols3_fit = sm.regression.linear_model.OLS(endog=np.log(y_train), exog=X_train_l).fit()
print("OLS Model on Latent space")
print(ols3_fit.summary())


# Making prediction using the model fitted on data that is reduced in dimension
ols3_train_pred = ols3_fit.predict(exog=X_train_l)
ols3_test_pred = ols3_fit.predict(exog=X_test_l)

diff3 = y_test - np.exp(ols3_test_pred)
ols3_mse = (1/diff3.shape[0]) * (diff3.T @ diff3)

ols3_tot_ss = (y_test - np.mean(y_test)).T @ (y_test - np.mean(y_test))
ols3_res_ss = diff3.T @ diff3
ols3_r2 = 1 - (ols1_res_ss/ols3_tot_ss)


# Plot the quality of fit
plt.figure()
plt.plot(y_train, label='Training data')
plt.plot(np.exp(ols3_fit.predict(X_train_l)), label='Prediction on training set')
plt.plot(y_test, label='Testing data')
plt.plot(np.exp(ols3_fit.predict(X_test_l)), label='Prediction on training set')
plt.legend()
plt.xlabel('Date')
plt.ylabel('CO AQI')
plt.title('OLS Model on PCA data')
plt.show()



print("Performance on test set")
print(f"RMSE of OLS - Whitened data is {ols1_mse**.5}")
print(f"RMSE of OLS - Standardized data is {ols2_mse**.5}")
print(f"RMSE of OLS - PCA is {ols3_mse**.5}")



#%%
from Utilities.Forecasts import *

#%%
co = joined_df['CO AQI'][:train_test_split]
train_size = co.shape[0]
test_size = co_test.shape[0]

#%%
co_avg_pred_forecast = avg_forecast(x=joined_df['CO AQI'], T=train_size,
                                    one=True, h=True, h_length=test_size,
                                    plot=True)
plt.show()

tmp_y = joined_df['CO AQI'].reset_index(drop=True)[train_size:train_size+test_size]
tmp_y_mean = np.mean(tmp_y)
co_avg_tot_ss = (tmp_y - tmp_y_mean).T @ (tmp_y - tmp_y_mean)
co_avg_res_ss = (tmp_y - co_avg_pred_forecast[1]).T @ (tmp_y - co_avg_pred_forecast[1])
avg_r2 = 1 - (co_avg_res_ss/co_avg_tot_ss)

indices = joined_df.index.values

train_indices = indices[:train_size]
test_indices = indices[train_size:]


#%%

co_naive_pred_forecast = naive_method(x=joined_df['CO AQI'], T=train_size,
                                    one=True, h=True, h_length=test_size)

tmp_y = joined_df['CO AQI'].reset_index(drop=True)[train_size:train_size+test_size]
tmp_y_mean = np.mean(tmp_y)
co_naive_tot_ss = (tmp_y - tmp_y_mean).T @ (tmp_y - tmp_y_mean)
co_naive_res_ss = (tmp_y - co_naive_pred_forecast[1]).T @ (tmp_y - co_naive_pred_forecast[1])
naive_r2 = 1 - (co_naive_res_ss/co_naive_tot_ss)

indices = joined_df.index
plt.figure()
plt.plot(joined_df.iloc[1:train_size]['CO AQI'].reset_index(drop=True), '-b', label='Training')
plt.plot(list(range(train_size, train_size+test_size)), joined_df.iloc[train_size:]['CO AQI'].reset_index(drop=True), '-g', label='Validation' )
plt.plot(list(range(train_size, train_size+test_size)), co_naive_pred_forecast[1], '-', color='orange', label='Testing')
plt.xlabel("Time Index")
plt.ylabel('CO AQI')
plt.title('CO AQI Naive method')
plt.legend()
plt.show()

#%%
co_drift_pred_forecast = drift_forecast(x=joined_df['CO AQI'], T=train_size,
                                        one=True, h=True, h_length=test_size,
                                        plot=True)

tmp_y = joined_df['CO AQI'].reset_index(drop=True)[train_size:train_size+test_size]
tmp_y_mean = np.mean(tmp_y)
co_drift_tot_ss = (tmp_y - tmp_y_mean).T @ (tmp_y - tmp_y_mean)
co_drift_res_ss = (tmp_y - co_drift_pred_forecast[1]).T @ (tmp_y - co_drift_pred_forecast[1])
drift_r2 = 1 - (co_drift_res_ss/co_drift_tot_ss)

#%%
co_ses_pred_forecast = ses(x=joined_df['CO AQI'], T=train_size, h_length=test_size)
co_ses_tot_ss = (tmp_y - tmp_y_mean).T @ (tmp_y - tmp_y_mean)
co_ses_res_ss = (tmp_y - co_ses_pred_forecast[1]).T @ (tmp_y - co_ses_pred_forecast[1])
ses_r2 = 1 - (co_ses_res_ss/co_ses_tot_ss)

plt.figure()
plt.plot(co.reset_index(drop=True), label='Training data')
plt.plot(co_ses_pred_forecast[0], label='Prediction')
plt.plot(range(train_size, train_size+test_size), co_test.reset_index(drop=True), label='Training data')
plt.plot(range(train_size, train_size+test_size), co_ses_pred_forecast[1], label='Training data')
plt.legend()
plt.xlabel('Time index')
plt.ylabel('CO AQI')
plt.title('CO AQI - Simple Exponential Smoothing Method')
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

cnames = X_train_z.columns
def feature_select_vif(df_train, y_train, target):
    df_train = pd.concat([pd.DataFrame({'bias_c': np.ones([df_train.shape[0]])}), df_train.reset_index(drop=True)],
                         axis=1)
    curr_value = 11
    filtered_cnames = np.setdiff1d(cnames, ['bias_c']+target)
    ignored_cols = []
    while curr_value > 10:
        ignore_col = None
        X_tr_subset = df_train[['bias_c'] + list(filtered_cnames)]
        vif_vals = pd.DataFrame([{'Variable':X_tr_subset.columns[i], 'VIF':VIF(X_tr_subset, i)} for i in range((X_tr_subset.shape[1])) if VIF(X_tr_subset, i) > 3])
        if X_tr_subset.shape[1] > 1:
            curr_value = vif_vals.VIF.iloc[1:].max()
            max_vif_idx = vif_vals.VIF.iloc[1:].argmax()
            ignore_col = vif_vals.Variable.iloc[1:].iloc[max_vif_idx]
            ols = sm.regression.linear_model.OLS(y_train.reshape([-1]), X_tr_subset).fit()
            aic = ols.aic
            bic = ols.bic
            adj_r2 = ols.rsquared_adj

        else:
            final_ols = sm.regression.linear_model.OLS(y_train.reshape([-1]), df_train[['bias_c'] + list(filtered_cnames)]).fit()
            return filtered_cnames, ignored_cols, final_ols

        if ignore_col:
            ##--------- Check for improvement
            tmp_filtered_cnames = list(filter(lambda x: x != ignore_col, filtered_cnames))
            tmp_X = df_train[['bias_c'] + list(tmp_filtered_cnames)]
            tmp_ols = sm.regression.linear_model.OLS(y_train.reshape([-1]), tmp_X).fit()
            new_adj_r2 = tmp_ols.rsquared_adj
            new_bic = tmp_ols.bic
            new_aic = tmp_ols.aic
            if new_adj_r2 < adj_r2 and np.abs(new_adj_r2 - adj_r2) > 2e-2:
                final_ols = sm.regression.linear_model.OLS(y_train.reshape([-1]), df_train[['bias_c'] + list(filtered_cnames)]).fit()
                return ignored_cols, filtered_cnames, final_ols
            ##--------
            ignored_cols.append(ignore_col)
            filtered_cnames = np.setdiff1d(cnames, ignore_col)
    final_ols = sm.regression.linear_model.OLS(y_train.reshape([-1]), df_train[['bias_c'] + list(filtered_cnames)]).fit()
    return filtered_cnames, ignored_cols, final_ols

tmp_train = X_train_z[np.setdiff1d(X_train_z.columns, ['intercept'])]
tmp_train['intercept'] = np.ones([tmp_train.shape[0], 1])
filtered_cnames_vif, ignored_cnames_vif, final_ols_vif = feature_select_vif(tmp_train, y_train.values, target=['CO AQI'])






# cols = np.setdiff1d(X_train_z.columns, ['intercept'])
# vif_vals = pd.DataFrame({"Feature": cols, "VIF":np.zeros([len(cols), 1])})
# vif_vals = vif_vals.set_index('Feature')
# for col_idx in range(len(cols)):
#     cname = cols[col_idx]
#     vif_vals.loc[cname, 'VIF'] = vif(X_train, col_idx)













