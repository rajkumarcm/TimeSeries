import numpy as np
np.random.seed(123)
from scipy.signal import dlsim
from matplotlib import pyplot as plt

num = [1, 0, 0]
den = [1, 0.5, 0.25]
params_true = num[1:] + den[1:]
system = (num, den, 1)
T = 10000
mu = 1
std = 4
e = np.random.normal(loc=mu, scale=std, size=T)
_, y_true = dlsim(system, e)

# plt.figure()
# plt.plot(y_true[:, 0], 'b')
# plt.show()

epochs = 50
delta = 1e-6
ma_coeffs = [1, 0, 0]
ar_coeffs = [1, 0, 0]

def get_sse(ma_coeffs, ar_coeffs):
    _, e = dlsim(([1] + ar_coeffs, [1] + ma_coeffs, 1), y_true) # reversed to get error i.e., the white noise
    return e, np.reshape(e.T @ e, [-1])[0]

def derivate1(error_original, theta_plus_delta, delta):
    ar_coeffs = [1]
    ma_coeffs = [1]
    ar_coeffs.extend(theta_plus_delta[:(n//2)])
    ma_coeffs.extend(theta_plus_delta[(n//2):])
    error_delta, _ = get_sse(ma_coeffs=ma_coeffs, ar_coeffs=ar_coeffs)
    return np.reshape(((error_original - error_delta) / delta), [-1])

def tune_hyperparams(sse_new, sse_old, new_theta, old_theta, delta_theta, A, mu, g):
    mu_max = 1e+12
    if sse_new < sse_old:
        if np.linalg.norm(delta_theta) < 1e-3:
            theta = new_theta
            # std_error = loss(y_true, y_pred)/(T-n)
            # cov = std_error * np.linalg.inv(A)
            return -1, theta, mu, sse_new # signal to terminate the loop
        else:
            theta = new_theta
            mu = mu/10
            return 1, theta, mu, sse_new

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
    return return_code, new_theta, mu, sse_new

def get_delta_theta(A, g, mu=None, cov=None):
    delta_theta = None
    if cov is None:
        delta_theta = np.linalg.inv(A + mu*np.eye(n, n))
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
    ar_coeffs.extend(theta[:(n//2)])
    ma_coeffs.extend(theta[(n//2):])

    # Log the Loss
    _, new_loss = get_sse(ma_coeffs=ma_coeffs, ar_coeffs=ar_coeffs)
    if log_err:
        print(f"Epoch: {epoch} SSE: {new_loss}")
    return new_loss, delta_theta, theta

n = len(num) + len(den) - 2
mu = 1e-2
sse_old = None
sse_new = None
cov = None
for epoch in range(epochs):
    params = ar_coeffs[1:] + ma_coeffs[1:]

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

    return_code, new_theta, mu, sse_new = tune_hyperparams(sse_new, sse_old, new_theta, params, delta_theta, A, mu, g)
    print(f"Epoch: {epoch} SSE: {sse_new}")

    num_opt = [1]
    den_opt = [1]
    num_opt.extend(new_theta[:(n//2)])
    den_opt.extend(new_theta[(n//2):])
    ar_coeffs = num_opt
    ma_coeffs = den_opt

    # if return_code == -1:
    #     break




















