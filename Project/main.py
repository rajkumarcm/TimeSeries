#%% Load the libraries
from matplotlib import pylab
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os import chdir
from os.path import abspath, join
import statsmodels.api as sm
from statsmodels.tsa import holtwinters as ETS
from Utilities.WhitenessTest import WhitenessTest as WT
from Utilities.Correlation import Correlation as Corr
from statsmodels.tsa.seasonal import STL
import seaborn as sns
from numpy.testing import assert_equal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from Utilities.GPAC import gpac_table
from scipy.stats import chi2

#%% \/\/\/\/ NEED TO FIX THIS-----------------------------------------
def whiteness_test(x, name):
    wt = WT(x)
    print(f"ADF test for {name}:\n")
    wt.ADF_Cal()
    wt.Plot_Rolling_Mean_Var(name=name)

#%%
def plot_acf_pacf(x, lags, xlims=None):
    r_idx = 0
    if xlims:
        fig, axes = plt.subplots(len(xlims), 2, sharex=False, sharey=True, figsize=(15, 10))
        for xlim in xlims:
            plot_acf(x, lags=lags, ax=axes[r_idx, 0])
            plot_pacf(x, lags=lags, ax=axes[r_idx, 1])
            axes[r_idx, 0].set_xlim(xlim)
            axes[r_idx, 1].set_xlim(xlim)
            r_idx += 1
    else:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 5))
        plot_acf(x, lags=lags, ax=axes[0])
        plot_pacf(x, lags=lags, ax=axes[1])
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
"""
ACF of the dependent variables
"""
corr = Corr()
fig, axes = plt.subplots(2, 4, figsize=(20, 5))

plot_acf(joined_df['CO AQI'], ax=axes[0, 0])
plot_pacf(joined_df['CO AQI'], ax=axes[1, 0])
axes[0,0].set_title('CO AQI ACF')
axes[1,0].set_title('CO AQI PACF')

plot_acf(joined_df['NO2 AQI'], ax=axes[0, 1])
plot_pacf(joined_df['NO2 AQI'], ax=axes[1, 1])
axes[0,1].set_title('NO2 AQI ACF')
axes[1,1].set_title('NO2 AQI PACF')

plot_acf(joined_df['SO2 AQI'], ax=axes[0, 2])
plot_pacf(joined_df['SO2 AQI'], ax=axes[1, 2])
axes[0,2].set_title('SO2 AQI ACF')
axes[1,2].set_title('SO2 AQI PACF')

plot_acf(joined_df['O3 AQI'], ax=axes[0, 3])
plot_pacf(joined_df['O3 AQI'], ax=axes[1, 3])
axes[0,3].set_title('O3 AQI ACF')
axes[1,3].set_title('O3 AQI PACF')

fig.tight_layout()
plt.show()

#%%
"""
Having seen the ACF, and the PACF of the CO AQI, it appears the data was generated by an 
AutoRegressive process since the PACF exhibits cut-off pattern while the ACF exhibits tail-off.
"""
def seasonal_differencing(y, seasonal_period):
    m = seasonal_period
    s_diff = []
    for t in range(m, len(y)):
        s_diff.append(y[t] - y[t-m])
    return s_diff

co_diff1 = joined_df['CO AQI'].diff()[1:]
fig, axes = plt.subplots(2, 1)
plot_acf(co_diff1, lags=100, )
plot_pacf(co_diff1, lags=100)
plt.show()

"""
After applying first order differencing, the ACF, and the PACF now shows signs of an MA process. 
Perhaps this is a multiplicative model that generated the data.
"""
print_strength_seas_tren(co_diff1, name='CO AQI differenced')
wt = WT(co_diff1)
wt.Plot_Rolling_Mean_Var(name='CO AQI differenced')

co_diff2 = co_diff1.diff()[1:]
print_strength_seas_tren(co_diff2, name='CO AQI second-order differencing')
wt = WT(co_diff2)
wt.Plot_Rolling_Mean_Var(name='CO AQI second-order differencing')

fig, axes = plt.subplots(2, 1, sharex=True)
plot_acf(co_diff2, lags=100, ax=axes[0])
plot_pacf(co_diff2, lags=100, ax=axes[1])
plt.tight_layout()
plt.show()
whiteness_test(co_diff2, name='CO AQI second-order differenced')

acf_vals, _ = corr.acf(co_diff2.reset_index()['CO AQI'], max_lag=22, plot=False, return_acf=True)
gpac_vals = gpac_table(acf_vals, na=10, nb=10, plot=True)
plt.figure(figsize=(23, 20))
sns.heatmap(gpac_vals, annot=True)
plt.xticks(ticks=np.array(list(range(50))) + .5, labels=list(range(1, 51)))
plt.title('Generalized Partial Autocorrelation (GPAC) Table')
plt.xlabel('AR Order')
plt.ylabel('MA Order')
plt.tight_layout()
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



#%%

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

# from sklearn.decomposition import PCA
#
# pca = PCA()
# feature_cols = np.setdiff1d(float_vars, aqi_vars)
# pca.fit(joined_df[feature_cols], so2_stationary)
# # The following expression on the right side is expected to return values sorted in descending order
# n_components = len(pca.explained_variance_ratio_[pca.explained_variance_ratio_ >= 0.01])
#
# pca = PCA(n_components=n_components)
# X = pca.fit_transform(joined_df[feature_cols], so2_stationary)

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



#%% Q13 - ARIMA, and SARIMA

# split CO AQI into train, test
N = joined_df.shape[0]
train_test_split = int(0.7 * N)
co_train = joined_df.reset_index()['CO AQI'][:train_test_split]
co_test = joined_df.reset_index()['CO AQI'][train_test_split:]

#%% CO Log transformation
co_log = np.log(joined_df['CO AQI'])
plt.figure()
plt.plot(co_log)
plt.xlim([0, 1000])
plt.show()

#%%
print_strength_seas_tren(co_log, name='Original data')
print('')
print_strength_seas_tren(co_log, name='Log transformed data')

#%%
whiteness_test(pd.Series(co_log), 'CO AQI Log transformed')

#%% First order differencing following log transformation
co_diff1 = seasonal_differencing(co_log, seasonal_period=1)

#%%
plt.plot(co_diff1)
plt.xlim([0, 1000])
plt.show()

#%%
whiteness_test(pd.Series(co_diff1), 'CO diff1')

#%%
plot_acf_pacf(co_diff1, lags=400, xlims=[[-5, 400], [-1, 50], [-1, 11], [0, 100], [360, 400]])

#%%
acf_vals, _ = corr.acf(co_diff1, max_lag=28, plot=False, return_acf=True)
gpac_vals = gpac_table(acf_vals, na=13, nb=13, plot=False)
plt.figure(figsize=(13, 10))
sns.heatmap(gpac_vals, annot=True)
plt.xticks(ticks=np.array(list(range(13))) + .5, labels=list(range(1, 14)))
plt.title('Generalized Partial Autocorrelation (GPAC) Table')
plt.xlabel('AR Order')
plt.ylabel('MA Order')
plt.tight_layout()
plt.show()

#%%
na = 4
nb = 3
d = 1
arima_fit = sm.tsa.ARIMA(endog=co_log, order=(na,d,nb), trend='n').fit()
arima_fit.summary()

#%% Residual analysis
yhat = arima_fit.predict()
residual = co_log - yhat

lags = 30
plt.figure()
plot_acf(yhat, lags=lags)
plt.show()

Q = train_test_split * (acf_vals[1:].T @ acf_vals[1:])
max_Q = chi2.ppf(0.95, lags-na-nb)
print(f"Computed Q value: {Q} is less than max critical value from the table: {max_Q}. Hence the residuals are white.")

#%% Plot y vs yhat
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].plot()
plt.show()


#%%
def backwise_reg(df_train, y_train, target):
    grid = np.zeros([1, len(cnames)])
    df_train = df_train[np.setdiff1d(df_train.columns, target)]

    df_train = standardise(df_train)
    df_train = pd.concat([pd.DataFrame({'bias_c':np.ones([df_train.shape[0]])}), df_train.reset_index(drop=True)], axis=1)

    curr_value = 1
    filtered_cnames = np.setdiff1d(cnames, ['bias_c']+target)
    ignored_cnames = []
    while curr_value > 0.05 + 1e-3:
        X_tr_subset = df_train[['bias_c'] + list(filtered_cnames)]
        # X_ts_subset = df_test[['bias_c'] + list(np.setdiff1d(filtered_cnames, cname))]
        ols = OLS(y_train.reshape([-1]), X_tr_subset).fit()
        aic = ols.aic
        bic = ols.bic
        adj_r2 = ols.rsquared_adj

        max_pval_idx = ols.pvalues.iloc[1:].argmax()
        curr_value = ols.pvalues.iloc[1:][max_pval_idx]

        if curr_value <= 0.05:
            final_ols = OLS(y_train.reshape([-1]), df_train[['bias_c'] + list(filtered_cnames)]).fit()
            return ignored_cnames, filtered_cnames, final_ols

        ignore_col = ols.pvalues.iloc[1:].index[max_pval_idx]
        ##--------- Check for improvement
        tmp_filtered_cnames = list(filter(lambda x: x != ignore_col, filtered_cnames))
        tmp_X = df_train[['bias_c'] + list(tmp_filtered_cnames)]
        tmp_ols = OLS(y_train.reshape([-1]), tmp_X).fit()
        new_adj_r2 = tmp_ols.rsquared_adj
        new_bic = tmp_ols.bic
        new_aic = tmp_ols.aic
        if (new_adj_r2 < adj_r2 and np.abs(new_adj_r2 - adj_r2) > 2e-2) or new_bic > bic or new_aic > aic:
            final_ols = OLS(y_train.reshape([-1]), df_train[['bias_c'] + list(filtered_cnames)]).fit()
            return ignored_cnames, filtered_cnames, final_ols
        ##--------

        ignored_cnames.append(ignore_col)
        filtered_cnames = list(filter(lambda x: x != ignore_col, filtered_cnames))
    final_ols = OLS(y_train.reshape([-1]), df_train[['bias_c'] + list(filtered_cnames)]).fit()
    return ignored_cnames, filtered_cnames, final_ols















