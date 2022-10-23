#%% Import the libraries

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from utils import acf
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp
random_state = 123
np.random.seed(random_state)

#%% Q1
df = pd.read_csv('auto.clean.csv', header=0)
cnames = ['normalized-losses', 'wheel-base', 'length', 'width', 'height',
          'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio',
          'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
df_original = df
df = df[cnames]

#%%
df.head()

#%%
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state)
target = ['price']
y_train_original = df_train[target[0]]
y_test_original = df_test[target[0]]

#%% Q2
import seaborn as sns
plt.figure(figsize=(12, 11))
sns.heatmap(df_train.corr(), linewidths=0.2)
plt.title('Correlation plot')
plt.show()

#%% Q3 and Q4
from sklearn.preprocessing import StandardScaler

ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = df_train[np.setdiff1d(df_train.columns, target)].values
X_train = ss_X.fit_transform(X_train)

X_test = df_test[np.setdiff1d(df_test.columns, target)]
X_test = ss_X.transform(X_test)

y_train = ss_y.fit_transform(pd.DataFrame(df_train[target[0]]))
y_test = ss_y.transform(pd.DataFrame(df_test[target[0]]))

#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

cov = 1/X_train.shape[0] * (X_train.T @ X_train)
U, s, vh = np.linalg.svd(cov)
cond = np.max(np.real(s))/np.min(np.real(s))
print(f'Condition number: {cond}')

# The condition of over 1000 means severe degree of co-linearity

vif_vals = [{'Variable':df_train.columns[i], 'VIF':VIF(X_train, i)} for i in range((X_train.shape[1])) if VIF(X_train, i) > 5]
vif_vals = pd.DataFrame(vif_vals).sort_values(by='VIF', ascending=False)
if vif_vals.shape[0] > 0:
    print(f'Co-linearity exists, and will require removing {vif_vals.shape[0]} features')
    print(vif_vals)

# coll_cols = np.intersect1d(cnames, vif_vals.Variable)

#%% Q5. LSE Method
# Because the following results in a singular matrix. I now need to decorrelate the data

X_train = np.hstack([np.ones([X_train.shape[0], 1]), X_train])
X_test = np.hstack([np.ones([X_test.shape[0], 1]), X_test])

W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
lse_err = y_train - (X_train @ W)
lse_mse = 1/len(lse_err) * (lse_err.T @ lse_err)
print(f'\nLSE Coefficients: \n {list(W.reshape([-1]))}')
print(f'\nLSE MSE: {lse_mse[0][0]}')

#%% Q6. OLS
from statsmodels.regression.linear_model import OLS
ols = OLS(y_train, X_train).fit()
print(f'\nOLS Coefficients: \n {ols.params}')
ols_err = y_test - ols.predict(X_train)
ols_mse = 1/len(ols_err) * (ols_err.T @ ols_err)
print(f'\nOLS MSE: {ols_mse[0][0]}')

#%% Q7. Backward stepwise regression
# def r_sq(y_true, y_pred):
#     diff = y_true - y_pred
#     ss_res = diff.T @ diff
#     diff = y_true - np.mean(y_true)
#     ss_tot = diff.T @ diff
#     return 1 - (ss_res/ss_tot)
#
# def AIC(T, k, mse_loss):
#     return (T * np.log(mse_loss)) + (2 * (k + 2))
#
# def BIC(T, k, mse_loss):
#     return (T * np.log(mse_loss)) + ((k + 2) * np.log(T))
#
# def Adj_R_sq(T, k, r_sq):
#     return 1 - (((1 - r_sq) * T - 1)/(T - k - 1))
#
def mse(y_true, y_pred):
    diff = y_true - y_pred
    return (1/len(y_true)) * (diff.T @ diff)

def standardise(df):
    return (df - df.mean(axis=0))/df.std(axis=0)

def backwise_reg(df_train, y_train, target):
    grid = np.zeros([1, len(cnames)])
    df_train = df_train[np.setdiff1d(df_train.columns, target)]

    df_train = standardise(df_train)
    df_train = pd.concat([pd.DataFrame({'bias_c':np.ones([df_train.shape[0]])}), df_train.reset_index(drop=True)], axis=1)

    curr_value = 1
    filtered_cnames = np.setdiff1d(cnames, ['bias_c']+target)
    ignored_cnames = []
    while curr_value > 0.05 + 1e-3:
        prev_value = curr_value

        X_tr_subset = df_train[['bias_c'] + list(filtered_cnames)]
        # X_ts_subset = df_test[['bias_c'] + list(np.setdiff1d(filtered_cnames, cname))]
        ols = OLS(y_train.reshape([-1]), X_tr_subset).fit()

        max_pval_idx = ols.pvalues.iloc[1:].argmax()
        curr_value = ols.pvalues.iloc[1:][max_pval_idx]

        if curr_value <= 0.05:
            final_ols = OLS(y_train.reshape([-1]), df_train[['bias_c'] + list(filtered_cnames)]).fit()
            return ignored_cnames, filtered_cnames, final_ols

        ignore_col = ols.pvalues.iloc[1:].index[max_pval_idx]
        ignored_cnames.append(ignore_col)
        filtered_cnames = list(filter(lambda x: x != ignore_col, filtered_cnames))
    final_ols = OLS(y_train.reshape([-1]), df_train[['bias_c'] + list(filtered_cnames)]).fit()
    return ignored_cnames, filtered_cnames, final_ols

ignored_cnames_bw, filtered_cnames_bw, final_ols_bw = backwise_reg(df_train, y_train, target)

#%% Q8.

def feature_select_vif(df_train, target):
    df_train = df_train[np.setdiff1d(df_train.columns, target)]
    df_train = pd.concat([pd.DataFrame({'bias_c': np.ones([df_train.shape[0]])}), df_train.reset_index(drop=True)],
                         axis=1)
    curr_value = 6
    filtered_cnames = np.setdiff1d(cnames, ['bias_c']+target)
    ignored_cols = []
    while curr_value > 5:
        X_tr_subset = df_train[['bias_c'] + list(filtered_cnames)]
        vif_vals = pd.DataFrame([{'Variable':X_tr_subset.columns[i], 'VIF':VIF(X_tr_subset, i)} for i in range((X_tr_subset.shape[1])) if VIF(X_tr_subset, i) > 2])
        if vif_vals.shape[0] > 1:
            curr_value = vif_vals.VIF.iloc[1:].max()
            max_vif_idx = vif_vals.VIF.iloc[1:].argmax()
            ignore_col = vif_vals.Variable.iloc[1:].iloc[max_vif_idx]
        else:
            final_ols = OLS(y_train.reshape([-1]), df_train[['bias_c'] + list(filtered_cnames)]).fit()
            return filtered_cnames, ignored_cols, final_ols

        if curr_value > 5:
            filtered_cnames = list(filter(lambda x: x != ignore_col, filtered_cnames))
            ignored_cols.append(ignore_col)

    final_ols = OLS(y_train.reshape([-1]), df_train[['bias_c'] + list(filtered_cnames)]).fit()
    return filtered_cnames, ignored_cols, final_ols

filtered_cnames_vif, ignored_cnames_vif, final_ols_vif = feature_select_vif(df_train, target)

#%% Q9.
# Explanation will be disclosed in the report

#%% Q10.
best_model = final_ols_bw
print(best_model.summary())

#%% Q11.
X_tr_subset = df_train[list(filtered_cnames_bw)]
X_tr_subset = pd.concat([pd.DataFrame({'bias_c': np.ones([X_tr_subset.shape[0]])}), X_tr_subset.reset_index(drop=True)],
                         axis=1)
X_ts_subset = df_test[list(filtered_cnames_bw)]
X_ts_subset = pd.concat([pd.DataFrame({'bias_c': np.ones([X_ts_subset.shape[0]])}), X_ts_subset.reset_index(drop=True)],
                         axis=1)

tr_pred = best_model.predict(X_tr_subset)
tr_pred_error = np.subtract(y_train.reshape([-1]), tr_pred)
tr_mse_val = mse(y_train.reshape([-1]), tr_pred_error)

ts_pred = best_model.predict(X_ts_subset)
ts_pred_error = np.subtract(y_test.reshape([-1]), ts_pred)
ts_mse_val = mse(y_test.reshape([-1]), ts_pred_error)

plt.figure()
plt.plot(y_train, '-b', label='Training data')
plt.plot(tr_pred, '-m', label='Prediction')
# plt.plot(y_train, '-b', label='Training data')
plt.plot(range(len(y_train), len(y_train)+len(y_test)), y_test.reshape([-1])-1200, '-', color='orange', label='Test data')
# plt.plot(range(len(y_train), len(y_train)+len(y_test)), y_test-1200, '-', color='orange', label='Test data')
plt.plot(range(len(y_train), len(y_train)+len(y_test)), ts_pred-1200, '-g', label='Forecast')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('OLS Prediction')
plt.legend()
plt.grid()
plt.show()

#%% Q12.

acf(tr_pred_error, max_lag=20)

#%% Q13.
# This wasn't covered in the lecture other than just some slides showing the purpose.
# No mathematical coverage nor implementation details covered...
# Unreasonable question.
# T-test
r = np.zeros_like(best_model.params)
ttest = best_model.t_test(r_matrix=r)
print(ttest)

# F-test
f_value = np.var(best_model.params)
ftest = best_model.f_test(r)
print(ftest)

#%% Graph of the LSE model - Experiment...
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]
cov = (1/X_train.shape[0]) * (X_train.T @ X_train)
val, vec = np.linalg.eig(cov)
indices = np.argsort(val)[::-1]
val = val[indices]
vec = vec[:, indices]
var_threshold = 0.13
indices = np.where(val >= var_threshold)[0]
val = val[indices]
vec = vec[:, indices]

# Project the data onto lower dimensional subspace, where the features are de-correlated
X_train_dc = X_train @ vec
X_test_dc = X_test @ vec

X_train_dc = np.hstack([np.ones([X_train_dc.shape[0], 1]), X_train_dc])
X_test_dc = np.hstack([np.ones([X_test_dc.shape[0], 1]), X_test_dc])

W = np.linalg.inv(X_train_dc.T @ X_train_dc) @ X_train_dc.T @ y_train
lse_err = y_train - (X_train_dc @ W)
lse_mse = 1/len(lse_err) * (lse_err.T @ lse_err)
print(f'\nLSE Coefficients: \n {list(W.reshape([-1]))}')
print(f'\nLSE MSE: {lse_mse[0][0]}')

tr_pred = X_train_dc @ W
ts_pred = X_test_dc @ W
tr_pred = tr_pred - np.mean(tr_pred_error)
ts_pred = ts_pred - np.mean(ts_pred_error)
plt.figure()
# plt.plot(y_train, '-b', label='Training data')
plt.plot(tr_pred, '-m', label='Prediction')
# plt.plot(range(len(y_train), len(y_train)+len(y_test)), y_test-1200, '-', color='orange', label='Test data')
plt.plot(range(len(y_train), len(y_train)+len(y_test)), ts_pred-1200, '-g', label='Forecast')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('OLS Prediction')
plt.legend()
plt.grid()
plt.show()

















