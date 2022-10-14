#%%
import numpy as np
from numpy.testing import assert_equal
from matplotlib import pyplot as plt
import pandas as pd

#%%
# Question 2
x = [112, 118, 132, 129, 121, 135, 148, 136, 119, 104, 118, 115, 126, 141]
names = []
tr_mse_list = []
ts_mse_list = []
tr_res_var_list = []
ts_res_var_list = []
q_list = []
prediction_errors = {}

def avg_forecast(x, tr_size):
    prediction = [x[0]]
    for i in range(2, tr_size):
        prediction.append(np.mean(x[:i]))

    if assert_equal(tr_size-1, len(prediction)):
        pass

    plt.figure()
    plt.plot(list(range(tr_size+1)), x[:tr_size+1], '-b', label='Training data')
    plt.plot(list(range(tr_size, len(x))), x[tr_size:], '-', color='orange')
    plt.plot(list(range(tr_size, len(x))), [np.mean(x[:tr_size])]*(len(x)-tr_size), '-g', label='Test Forecast')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Average forecast method')
    plt.legend()
    plt.show()

    return prediction[1:], np.mean(x[:tr_size])

tr_size = 9
prediction, forecast = avg_forecast(x, tr_size)

#%%
# Question 3
def mse(y, y_pred):
    diff = np.subtract(y, y_pred)
    return 1/len(diff) * diff.T @ diff

# Deliberately set it to 2 so that all algorithms generalise to this starting point
tr_y = x[2:tr_size]
tr_pred = prediction
tr_res = np.subtract(tr_y, tr_pred)
tr_mse = 1/len(tr_res) * tr_res.T @ tr_res

ts_y = x[tr_size:]
ts_pred = [forecast] * (len(x) - tr_size)
# I manually computed mse so that I can use the residuals later for computing the variance
ts_res = np.subtract(ts_y, ts_pred)
ts_mse = 1/len(ts_res) * ts_res.T @ ts_res

# pd.DataFrame({'MSE':{'Prediction':tr_mse, 'Forecast':ts_mse}})

#%%
# Question 4
"""
The following snippet of code would produce the same result when you would use numpy
tmp_res_diff_ms = tr_res_diff-np.mean(tr_res_diff)
tmp_var = 1/len(tmp_res_diff_ms) * tmp_res_diff_ms.T @ tmp_res_diff_ms
"""
tr_res_diff_ms = tr_res - np.mean(tr_res)
tr_res_var = 1/len(tr_res_diff_ms) * tr_res_diff_ms.T @ tr_res_diff_ms
# tr_res_var = np.var(tr_res_diff)
ts_res_diff_ms = ts_res - np.mean(ts_res)
ts_res_var = 1/len(ts_res_diff_ms) * ts_res_diff_ms.T @ ts_res_diff_ms
# ts_res_var = np.var(ts_res_diff)
# pd.DataFrame({'Variance in residuals':{'Prediction':tr_res_var, 'Forecast':ts_res_var}})

#%%
# Question 5
def acf(x, max_lag):
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
    return acf_lags

h = 5

# Compute the ACF values
acf_values = acf(tr_res, max_lag=h)

# Ignore the first one
acf_values = acf_values[1:]

# Square and sum
Q = len(tr_res) * acf_values.T @ acf_values
# print(f'Q value: {Q}')

#%% Log metrics

names.append('Average')
tr_mse_list.append(tr_mse)
ts_mse_list.append(ts_mse)
tr_res_var_list.append(tr_res_var)
ts_res_var_list.append(ts_res_var)
q_list.append(Q)
prediction_errors['Average'] = tr_res

#%%
# Question 6
# Step 2
def naive_method(x, tr_size):
    prediction = []
    for i in range(1, tr_size):
        prediction.append(x[i-1])

    if assert_equal(tr_size-1, len(prediction)):
        pass

    plt.figure()
    plt.plot(list(range(tr_size+1)), x[:tr_size+1], '-b', label='Training data')
    plt.plot(list(range(tr_size, len(x))), x[tr_size:], '-', color='orange')
    plt.plot(list(range(tr_size, len(x))), [x[tr_size-1]]*(len(x)-tr_size), '-g', label='Test Forecast')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Naive forecast method')
    plt.legend()
    plt.show()

    return prediction[1:], x[tr_size-1]
# naive_method(x, tr_size)

#%%
# Step 3
prediction, naive_forecast = naive_method(x, tr_size)
tr_y = x[2:tr_size]
tr_pred = prediction
tr_res = np.subtract(tr_y, prediction)
tr_mse = 1 / len(tr_res) * tr_res.T @ tr_res

ts_y = x[tr_size:]
ts_pred = [naive_forecast] * (len(x)-tr_size)
# I manually computed mse so that I can use the residuals later for computing the variance
ts_res = np.subtract(ts_y, ts_pred)
ts_mse = 1 / len(ts_res) * ts_res.T @ ts_res
# pd.DataFrame({'MSE':{'Prediction':tr_mse, 'Forecast':ts_mse}})
#%%
# Step 4
tr_res_var = np.var(tr_res)
ts_res_var = np.var(ts_res)
# pd.DataFrame({'Variance in residuals':{'Prediction':tr_res_var, 'Forecast':ts_res_var}})

#%%
# Step 5
# Compute the ACF values
acf_values = acf(tr_res, max_lag=h)

# Ignore the first one
acf_values = acf_values[1:]

# Square and sum
Q = len(tr_res) * acf_values.T @ acf_values
# print(f'Q value: {Q}')

#%% Log metrics

names.append('Naive')
tr_mse_list.append(tr_mse)
ts_mse_list.append(ts_mse)
tr_res_var_list.append(tr_res_var)
ts_res_var_list.append(ts_res_var)
q_list.append(Q)
prediction_errors['Naive'] = tr_res

#%%
# Question 7 Step 2
def drift_forecast(x, tr_size):

    T = tr_size-1 # -1 to get the index
    ts_size = len(x) - tr_size
    straight_line_for_ts = x[T] - x[0]

    forecast = []
    for i in range(2, len(x)+1):
        if i <= tr_size:
            forecast.append(x[i-1] + ((x[i-1]-x[0])/(i-1)))
        else:
            forecast.append(x[T] + (i-tr_size) * (straight_line_for_ts/(T)))

    plt.figure()
    # plt.plot(list(range(1, tr_size+1)), x[:tr_size], '-b', label='Training data')
    # Only for aesthetic purpose
    plt.plot(list(range(1, tr_size + 2)), list(x[:tr_size])+[forecast[-ts_size]], '-b', label='Training data')
    # Prediction (not forecast) only starts from the second time-step
    plt.plot(list(range(tr_size+1, len(x)+1)), x[tr_size:], '-', color='orange', label='Test data')
    plt.plot(list(range(tr_size+1, len(x)+1)), forecast[-ts_size:], '-g', label='h-step prediction')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Drift forecast method')
    plt.legend()
    plt.show()
    return forecast
dr_forecast = drift_forecast(x, tr_size)

#%% Step 3 - MSE
tr_y = x[2:tr_size]
tr_pred = dr_forecast[:(tr_size-2)]
tr_res = np.subtract(tr_y, tr_pred)
tr_mse = 1 / len(tr_res) * tr_res.T @ tr_res
ts_y = x[tr_size:]
ts_pred = dr_forecast[-(len(x)-tr_size):]
# I manually computed mse so that I can use the residuals later for computing the variance
ts_res = np.subtract(ts_y, ts_pred)
ts_mse = 1 / len(ts_res) * ts_res.T @ ts_res
# pd.DataFrame({'MSE':{'Prediction':tr_mse, 'Forecast':ts_mse}})

#%%
# Step 4
tr_res_var = np.var(tr_res)
ts_res_var = np.var(ts_res)

#%% Step 5 - ACF and Q Value

acf_values = acf(tr_res, max_lag=5)

# Ignore the first one
acf_values = acf_values[1:]

# Square and sum
Q = len(tr_res) * acf_values.T @ acf_values
# print(f'Q value: {Q}')

#%% Log metrics

names.append('Drift')
tr_mse_list.append(tr_mse)
ts_mse_list.append(ts_mse)
tr_res_var_list.append(tr_res_var)
ts_res_var_list.append(ts_res_var)
q_list.append(Q)
prediction_errors['Drift'] = tr_res

#%%
# Question 8
def ses(x, alpha):
    a = alpha
    prediction = [x[0]] # prediction at time 1 is x[0]
    for i in range(1, tr_size): # 1 is the index of the second time-step
        prediction.append(a * x[i-1] + (1-a) * prediction[i-1])
    return prediction

alphas = [0, 0.25, 0.75, 0.99]
ts_size = len(x) - tr_size
# y values
tr_y = x[:tr_size]
ts_y = x[tr_size:]
ses_pred_list = {}
for alpha in alphas:
    ses_pred = ses(x, alpha)

    # Save for plotting together
    ses_pred_list[alpha] = ses_pred[-1]

     # Prediction - y_hat
    tr_pred = ses_pred[:tr_size]
    ts_pred = [ses_pred[-1]] * ts_size

    # Residuals
    tr_res = np.subtract(tr_y, tr_pred)
    ts_res = np.subtract(ts_y, ts_pred)

    # Step 3 - Compute the MSE on the train and test set
    ses_tr_mse = 1/len(tr_res) * tr_res.T @ tr_res
    ses_ts_mse = 1/ts_size * ts_res.T @ ts_res

    # Step 4 - Variance of prediction and forecast error
    ses_tr_var = np.var(tr_res)
    ses_ts_var = np.var(ts_res)

    # Step 5 - ACF and Q on training set
    ses_acf = acf(tr_res, max_lag=5)
    # Ignore the first one
    ses_acf = ses_acf[1:]
    # Square and sum
    Q = len(tr_res) * ses_acf.T @ ses_acf
    # print(f'Q value: {Q}')

    # Log the metrics
    names.append(f'SES a={alpha}')
    tr_mse_list.append(ses_tr_mse)
    ts_mse_list.append(ses_ts_mse)
    tr_res_var_list.append(ses_tr_var)
    ts_res_var_list.append(ses_ts_var)
    q_list.append(Q)
    prediction_errors[f'SES a={alpha}'] = tr_res

#Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
r_idx = 0
c_idx = 0
for alpha in alphas:
    axes[r_idx, c_idx].plot(list(range(1, tr_size+1)), tr_y, '-b', label='Training data')
    axes[r_idx, c_idx].plot(list(range(tr_size+1, len(x)+1)), x[tr_size:], '-', color='orange', label='Test data')
    axes[r_idx, c_idx].plot(list(range(tr_size+1, len(x)+1)), [ses_pred_list[alpha]]*ts_size, '-g', label='Predicted data')
    axes[r_idx, c_idx].set_xlabel('Sample')
    axes[r_idx, c_idx].set_ylabel('Value')
    axes[r_idx, c_idx].set_title(f'SES with alpha {alpha}')
    axes[r_idx, c_idx].grid(True)
    axes[r_idx, c_idx].legend()

    if c_idx == 1:
        c_idx = 0
        r_idx += 1
    else:
        c_idx += 1

plt.show()

#%% Question 10 - Display the results

metrics_df = pd.DataFrame({'Prediction MSE':tr_mse_list, 'Forecast MSE':ts_mse_list,
                          'Pred. Var':tr_res_var_list, 'Forec. Var':ts_res_var_list,
                           'Q':q_list}, index=names)
print(metrics_df)

#%% Question 11 - Plot ACF

from utils import acf as plot_acf

fig, axes = plt.subplots(4, 2, figsize=(13, 19))
r_idx = 0
c_idx = 0
for method_name in names:
    # prediction error of a method
    prediction_err_m = prediction_errors[method_name]
    plot_acf(prediction_err_m, max_lag=5, ax=axes[r_idx, c_idx])
    axes[r_idx, c_idx].set_title(f'ACF of {method_name} pred.error')
    if c_idx == 1:
        c_idx = 0
        r_idx += 1
    else:
        c_idx += 1
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.1,
                    hspace=0.3)
plt.show()

#%% Question 12 - Answer discloesd in the report.
















