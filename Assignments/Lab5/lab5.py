from LM import *

# The program would also work if you do not give any input as parameters.
# As per rubrics, it will ask the user to input the values.
lm_example1 = LM(T=10000, na=1, nb=0, ar_coeffs=[0.5], ma_coeffs=None)
lm_example1.fit()
lm_example2 = LM(T=10000, na=0, nb=1, ar_coeffs=None, ma_coeffs=[0.5])
lm_example2.fit()
lm_example3 = LM(T=10000, na=1, nb=1, ar_coeffs=[-0.5], ma_coeffs=[0.25])
lm_example3.fit()
lm_example4 = LM(T=10000, na=2, nb=0, ar_coeffs=[-0.5, -0.2], ma_coeffs=None)
lm_example4.fit()
lm_example5 = LM(T=10000, na=2, nb=1, ar_coeffs=[-0.5, -0.2], ma_coeffs=[-0.5])
lm_example5.fit()
lm_example6 = LM(T=10000, na=1, nb=2, ar_coeffs=[-0.5], ma_coeffs=[0.5, 0.4])
lm_example6.fit()
lm_example7 = LM(T=10000, na=0, nb=2, ar_coeffs=None, ma_coeffs=[0.5, -0.4])
lm_example7.fit()
# Example 8 requires pole cancellation simplification...
lm_example8 = LM(T=10000, na=2, nb=2, ar_coeffs=[-0.5,-0.2], ma_coeffs=[0.5, -0.4])
lm_example8.fit()

#%%
# Phase 2

# Note: I am using statsmodels version==0.13.5
# ARMA package from statsmodels is deprecated and is no longer available.

from statsmodels.tsa.arima.model import ARIMA
wn = np.random.normal(loc=0, scale=1, size=10000)

def stats_way(ar, ma, order, wn):
    _, y = dlsim((ma, ar, 1), wn)
    model = ARIMA(endog=y, order=order, trend=[1, 1, 0, 0]).fit()
    print_str = ""
    if order[0] > 0:
        ar_pred = model.arparams
        ar_pred = list(map(lambda x: np.round(x, 3), ar_pred))
        for i in range(len(ar_pred)):
            print_str += f"The AR coefficient a{i+1} is {ar_pred[i]}\n"

    if order[2] > 0:
        ma_pred = model.maparams
        ma_pred = list(map(lambda x: np.round(x, 3), ma_pred))
        for i in range(len(ma_pred)):
            print_str += f"The MA coefficient b{i+1} is {ma_pred[i]}\n"
    print(f"\n{print_str}")

    print(model.summary())

# system = coefficients of ma / coefficients of ar
#%% Example 1: ARMA(0,1): y(t) - 0.5y(t-1) = e(t)
order = (1, 0, 0)
ma = [1,0]
ar = [1,0.5]
stats_way(ar=ar, ma=ma, order=order, wn=wn)

#%% Example2: ARMA(0,1): y(t) = e(t) + 0.5e(t-1)
order = (0, 0, 1)
ma = [1, 0.5]
ar = [1, 0]
stats_way(ar=ar, ma=ma, order=order, wn=wn)

#%% Example3: ARMA(1,1): y(t) + 0.5y(t-1) = e(t) + 0.25e(t-1)
order = (1, 0, 1)
ma = [1, 0.25]
ar = [1, -0.5]
stats_way(ar=ar, ma=ma, order=order, wn=wn)

#%% Example4: ARMA(2,0): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t)
order = (2, 0, 0)
ma = [1, 0, 0]
ar = [1, -0.5, -0.2]
stats_way(ar=ar, ma=ma, order=order, wn=wn)

#%% Example5: ARMA(2,1): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) - 0.5e(t-1)
order = (2, 0, 1)
ma = [1, -0.5, 0]
ar = [1, -0.5, -0.2]
stats_way(ar=ar, ma=ma, order=order, wn=wn)

#%% Example6: ARMA(1,2): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1) - 0.4e(t-2)
order = (1, 0, 2)
ma = [1, 0.5, -0.4]
ar = [1, -0.5, 0]
stats_way(ar=ar, ma=ma, order=order, wn=wn)

# After zero pole cancellation:
order = (0, 0, 1)
ma = [1, -0.4]
ar = [1, 0]
stats_way(ar=ar, ma=ma, order=order, wn=wn)

#%% Example7: ARMA(0,2): y(t) = e(t) + 0.5e(t-1) - 0.4e(t-2)
order = (0, 0, 2)
ma = [1, 0.5, -0.4]
ar = [1, 0, 0]
stats_way(ar=ar, ma=ma, order=order, wn=wn)

#%% Example8: ARMA(2,2): y(t)+0.5y(t-1) +0.2y(t-2) = e(t)+ 0.5e(t-1) - 0.4e(t-2)
order = (2, 0, 2)
ma = [1, 0.5, -0.4]
ar = [1, -0.5, -0.2]
stats_way(ar=ar, ma=ma, order=order, wn=wn)

#%%









