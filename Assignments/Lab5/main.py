import numpy as np
# np.random.seed(123)
from scipy.signal import dlsim
from matplotlib import pyplot as plt

na = 2
nb = 0
num = [1, 0]
den = [1, 0.5, 0.25]
system = (num, den, 1)
T = 10000
mu = 0
std = 1
e = np.random.normal(loc=mu, scale=std, size=T)
_, y_true = dlsim(system, e)

# plt.figure()
# plt.plot(y_true[:, 0], 'b')
# plt.show()

epochs = 50
delta = 1e-7
ma_coeffs = [1, 0, 0]
ar_coeffs = [1, 0, 0]
n = na + nb
params = [0]*n

def get_sse(ma_coeffs, ar_coeffs):
    _, e = dlsim((ar_coeffs, ma_coeffs, 1), y_true) # reversed to get error i.e., the white noise
    return e, np.reshape(e.T @ e, [-1])[0]

def derivate1(error_original, theta_plus_delta, delta):
    ar_coeffs = [1]
    ma_coeffs = [1]
    if na > 0:
        ar_coeffs.extend(theta_plus_delta[:na])
    else:
        ar_coeffs.extend([0] * max_order)
    if nb > 0:
        ma_coeffs.extend(theta_plus_delta[-nb:])
    else:
        ma_coeffs.extend([0] * max_order)
    error_delta, _ = get_sse(ma_coeffs=ma_coeffs, ar_coeffs=ar_coeffs)
    return np.reshape(((error_original - error_delta) / delta), [-1])

def tune_hyperparams(sse_new, sse_old, new_theta, old_theta, delta_theta, A, mu, g):
    mu_max = 1e+45
    if sse_new < sse_old:
        if np.linalg.norm(delta_theta) < 1e-4:
            theta = new_theta
            std_error = sse_new/(T-n)
            cov = std_error * np.linalg.inv(A)
            return -1, theta, mu, sse_new, cov # signal to terminate the loop
        else:
            theta = new_theta
            mu = mu/10
            return 1, theta, mu, sse_new, np.eye(n, n) * mu

    return_code = -1
    while sse_new >= sse_old:
        mu *= 10
        if mu > mu_max:
            print('mu has exceeded maximum value and thus terminating the optimization process.')
            return_code = -1
            break
        else:
            sse_new, delta_theta, new_theta = second_step(old_theta, A, g, mu, cov=None)
            return_code = 1
    return return_code, new_theta, mu, sse_new, np.eye(n, n) * mu

def get_delta_theta(A, g, mu=None, cov=None):
    delta_theta = None
    if cov is None:
        delta_theta = np.linalg.inv(A + mu*np.eye(n, n))
        # return delta_theta @ g
    else:
        delta_theta = np.linalg.inv(A + cov)

    return delta_theta @ g

def second_step(theta, A, g, mu, cov, log_err=False):
    # Second order "approximation"
    # delta_theta = np.linalg.inv(A + (mu * np.eye(n, n))) @ g
    delta_theta = get_delta_theta(A=A, g=g, mu=mu, cov=cov)

    # Update theta now
    theta += delta_theta.reshape([-1])

    ar_coeffs = [1]
    ma_coeffs = [1]
    if na > 0:
        ar_coeffs.extend(theta[:na])
    else:
        ar_coeffs.extend([0]*max_order)
    if nb > 0:
        ma_coeffs.extend(theta[-nb:])
    else:
        ma_coeffs.extend([0]*max_order)

    # Log the Loss
    _, new_loss = get_sse(ma_coeffs=ma_coeffs, ar_coeffs=ar_coeffs)
    if log_err:
        print(f"Epoch: {epoch} SSE: {new_loss}")
    return new_loss, delta_theta, theta


mu = 1e-4
sse_old = None
sse_new = None
cov = None
max_order = np.max([na, nb])
for epoch in range(epochs):

    # 1. Construct X Matrix
    X = np.zeros([T, n])
    params_delta = params
    e, sse_old = get_sse(ma_coeffs=ma_coeffs, ar_coeffs=ar_coeffs)
    for i in range(len(params)):
        params_delta = params
        params_delta[i] += delta
        X[:, i] = derivate1(error_original=e, theta_plus_delta=params_delta,
                            delta=delta)

    # 2. Construct Hessian
    A = X.T @ X

    # 3. Construct Jacobian
    g = X.T @ e

    #---Step2---- Update parameters
    sse_new, delta_theta, new_theta = second_step(params, A, g, mu, cov, log_err=False)

    return_code, new_theta, mu, sse_new, cov = tune_hyperparams(sse_new, sse_old, new_theta, params, delta_theta, A, mu, g)
    print(f"Epoch: {epoch} SSE: {sse_new} theta: {new_theta}")

    params = new_theta
    ar_coeffs = [1]
    ma_coeffs = [1]
    if na > 0:
        ar_coeffs.extend(new_theta[:na])
    else:
        ar_coeffs.extend([0]*max_order)
    if nb > 0:
        ma_coeffs.extend(new_theta[-nb:])
    else:
        ma_coeffs.extend([0]*max_order)

    # if return_code == -1:
    #     break




















