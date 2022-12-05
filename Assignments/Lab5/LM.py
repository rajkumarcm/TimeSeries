import numpy as np
np.random.seed(12345)
from scipy.signal import dlsim
from warnings import warn
from matplotlib import pyplot as plt

class LM:
    def __init__(self, T=None, na=None, nb=None, ar_coeffs=None, ma_coeffs=None):
        if (na is None and nb is None) or \
                (ar_coeffs is None and ma_coeffs) is None:
             T = int(input('Input the number of observations:'))
             na = int(input('Enter AR order:'))
             nb = int(input('Enter MA order:'))
             print('Note: For ARMA represented as y(t) + 0.5y(t-1) = e(t) you should enter a1 as -0.5')

             ar_coeffs = []
             for i in range(na):
                 ar_coeffs.append(float(input(f'Enter a{i+1} coefficient:')))

             ma_coeffs = []
             for i in range(nb):
                 ma_coeffs.append(float(input(f'Enter b{i+i} coefficient:')))

        wn = np.random.normal(loc=0, scale=1, size=T)
        max_order = np.max([na, nb])

        ar_coeffs = np.array(ar_coeffs)
        if na > 0:
            ar_coeffs = np.r_[1, -ar_coeffs]
            if na < max_order:
                ar_coeffs = np.r_[ar_coeffs, np.array([0] * (max_order - na))]
        else:
            ar_coeffs = np.r_[1, np.array([0] * max_order)]

        ma_coeffs = np.array(ma_coeffs)
        if nb > 0:
            ma_coeffs = np.r_[1, ma_coeffs]
            if nb < max_order:
                ma_coeffs = np.r_[ma_coeffs, np.array([0] * (max_order - nb))]
        else:
            ma_coeffs = np.r_[1, np.array([0] * max_order)]
        self.T = T
        self.na = na
        self.nb = nb
        self.n = na + nb
        self.max_order = max_order
        self.ar_coeffs = ar_coeffs
        self.ma_coeffs = ma_coeffs
        _, self.y_true = dlsim((ma_coeffs, ar_coeffs, 1), wn)

        # Hyper-parameters
        self.delta = 1e-7
        self.mu = 1e-2
        self.mu_max = 1e+33
        self.epochs = 10

    def construct_params(self, ar_coeffs, ma_coeffs):
        params = []
        if self.na > 0:
            params.extend(ar_coeffs[1:])
        if self.nb > 0:
            params.extend(ma_coeffs[1:])
        return params

    def extract_coeffs(self, params):
        ar_coeffs = [1]
        if self.na > 0:
            ar_coeffs.extend(params[:self.na])
        else:
            ar_coeffs.extend([0] * self.max_order)

        ma_coeffs = [1]
        if self.nb > 0:
            ma_coeffs.extend(params[-self.nb:])
        else:
            ma_coeffs.extend([0] * self.max_order)
        # print('debug checkpoint. Check ar and ma coefficients for correctness...')
        return ar_coeffs, ma_coeffs

    def get_sse(self, ma_coeffs, ar_coeffs):
        # reversed to get error i.e., the white noise
        _, e = dlsim((ar_coeffs, ma_coeffs, 1), self.y_true)
        return e, np.reshape(e.T @ e, [-1])[0]

    def derivate1(self, error_original, theta_plus_delta, delta):
        ar_coeffs, ma_coeffs = self.extract_coeffs(theta_plus_delta)
        error_delta, _ = self.get_sse(ma_coeffs=ma_coeffs, ar_coeffs=ar_coeffs)
        return np.reshape(((error_original - error_delta) / delta), [-1])

    def get_delta_theta(self, A, g, mu=None, cov=None):
        delta_theta = None
        if cov is None:
            delta_theta = np.linalg.inv(A + mu * np.eye(self.n, self.n))
        else:
            delta_theta = np.linalg.inv(A + cov)

        return delta_theta @ g

    def second_step(self, params, A, g, mu, cov):
        # Second order "approximation"
        # delta_theta = np.linalg.inv(A + (mu * np.eye(n, n))) @ g
        delta_theta = self.get_delta_theta(A=A, g=g, mu=mu, cov=cov)

        # Update theta now
        theta = np.copy(params)
        theta += delta_theta.reshape([-1])

        ar_coeffs = [1]
        ma_coeffs = [1]
        if self.na > 0:
            ar_coeffs.extend(theta[:self.na])
        else:
            ar_coeffs.extend([0] * self.max_order)
        if self.nb > 0:
            ma_coeffs.extend(theta[-self.nb:])
        else:
            ma_coeffs.extend([0] * self.max_order)

        # Log the Loss
        _, new_loss = self.get_sse(ma_coeffs=ma_coeffs, ar_coeffs=ar_coeffs)

        return new_loss, delta_theta, theta

    def tune_hyperparams(self, sse_new, sse_old, new_theta, old_theta, delta_theta, A, mu, g):
        if sse_new < sse_old:
            if np.linalg.norm(delta_theta) < 1e-4:
                theta = new_theta
                return -1, theta, mu, sse_new  # signal to terminate the loop
            else:
                theta = new_theta
                mu = mu / 10
                return 1, theta, mu, sse_new

        return_code = -1
        while sse_new >= sse_old:
            mu *= 10
            if mu > self.mu_max:
                warn('mu has exceeded maximum value and thus terminating the optimization process.')
                return_code = -1
                break
            else:
                sse_new, delta_theta, new_theta = self.second_step(old_theta, A, g, mu, cov=None)
                return_code = 1

        return return_code, new_theta, mu, sse_new

    def confidence_interval(self):
        conf_int = []
        for i in range(self.n):
            b1 = self.theta_pred[i] - 2 * np.sqrt(self.cov[i, i])
            b2 = self.theta_pred[i] + 2 * np.sqrt(self.cov[i, i])
            # lower bound and upper bound
            lb = np.min([b1, b2])
            ub = np.max([b1, b2])
            conf_int.append([lb, ub])
        return conf_int

    def zero_pole_c(self):
        if self.na > 0:
            print(f"Zeros: {np.roots(np.r_[1, self.theta_pred[:self.na]])}")
        if self.nb > 0:
            print(f"Poles: {np.roots(np.r_[1, self.theta_pred[-self.nb:]])}")
        pass

    def plot_opt_progress(self, losses):
        plt.figure()
        plt.plot(range(1, len(losses)+1), losses, '-b', label='Loss')
        plt.title('Optimization progress')
        plt.xlabel('Epoch/Iteration')
        plt.ylabel('SSE')
        plt.tight_layout()
        plt.show()

    def fit(self):
        sse_list = []
        cov = None
        std_error = None
        ma_coeffs = [1] + [0] * self.max_order
        ar_coeffs = [1] + [0] * self.max_order
        params = np.array([0] * self.n, dtype=np.float32)
        for epoch in range(self.epochs):

            # 1. Construct X Matrix
            X = np.zeros([self.T, self.n])
            e, sse_old = self.get_sse(ma_coeffs=ma_coeffs, ar_coeffs=ar_coeffs)
            for i in range(len(params)):
                params_delta = np.copy(params)
                params_delta[i] += self.delta
                X[:, i] = self.derivate1(error_original=e, theta_plus_delta=params_delta,
                                         delta=self.delta)

            # 2. Construct Hessian
            A = X.T @ X

            # 3. Construct Jacobian
            g = X.T @ e

            # ---Step2---- Update parameters
            sse_new, delta_theta, new_theta = self.second_step(params, A, g, self.mu, cov)

            return_code, new_theta, self.mu, sse_new = self.tune_hyperparams(sse_new, sse_old, new_theta,
                                                                             params, delta_theta, A,
                                                                             self.mu, g)

            # Log the error information
            sse_list.append(sse_new)
            print(f"Epoch: {epoch} SSE: {sse_new} MSE: {sse_new/self.T} theta: {new_theta}")

            ar_coeffs, ma_coeffs = self.extract_coeffs(new_theta)
            params = np.copy(new_theta)

            if return_code == -1:
                self.std_error = sse_new / (self.T - self.n)
                self.cov = self.std_error * np.linalg.inv(A)
                self.theta_pred = params
                self.conf_int = self.confidence_interval()
                print_str = ""
                for i in range(self.na):
                    print_str += f"a{i+1}: {params[i]} conf_int: {self.conf_int[i]}\n"

                for i in range(self.nb):
                    print_str += f"b{i+1}: {params[self.na + i]} conf_int: {self.conf_int[self.na + i]}\n"

                print(f"Estimated ARMA Coefficients:\n{print_str}")
                print(f"Covariance of the estimated coefficients:\n{self.cov}")
                # print("debug checkpoint... Check for correctness of coefficients extraction.")
                self.zero_pole_c()
                self.plot_opt_progress(sse_list)
                break


if __name__ == "__main__":
    # The program would also work if you do not give any input as parameters.
    # As per rubrics, it will ask the user to input the values.
    # lm_example1 = LM(T=10000, na=1, nb=0, ar_coeffs=[0.5], ma_coeffs=None)
    # lm_example2 = LM(T=10000, na=0, nb=1, ar_coeffs=None, ma_coeffs=[0.5])
    # lm_example3 = LM(T=10000, na=1, nb=1, ar_coeffs=[-0.5], ma_coeffs=[0.25])
    # lm_example4 = LM(T=10000, na=2, nb=0, ar_coeffs=[-0.5, -0.2], ma_coeffs=None)
    # lm_example5 = LM(T=10000, na=2, nb=1, ar_coeffs=[-0.5, -0.2], ma_coeffs=[-0.5])
    # lm_example6 = LM(T=10000, na=1, nb=2, ar_coeffs=[-0.5], ma_coeffs=[0.5, 0.4])
    lm_example7 = LM(T=10000, na=0, nb=2, ar_coeffs=None, ma_coeffs=[0.5, -0.4])
    # Example 8 requires pole cancellation simplification...
    # lm_example8 = LM(T=10000, na=2, nb=2, ar_coeffs=[-0.5,-0.2], ma_coeffs=[0.5, -0.4])
    lm_example7.fit()













