import math
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

import warnings
warnings.filterwarnings("ignore")

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

""" Solution """

class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """
        # constants
        self.v_max = SAFETY_THRESHOLD

        # data holders
        self.x_sample = np.array([]).reshape(-1, DOMAIN.shape[0])
        self.f_sample = np.array([]).reshape(-1, DOMAIN.shape[0])
        self.v_sample = np.array([]).reshape(-1, DOMAIN.shape[0])
        self.logv_sample = np.array([]).reshape(-1, DOMAIN.shape[0])
        self.gv_sample = np.array([]).reshape(-1, DOMAIN.shape[0])

        # incorporate prior beliefs about f() and v()
        self.f_sigma = 0.15
        self.f_variance = 0.5
        self.f_lengthscale = 10
        self.f_kernel = 0.125* Matern(length_scale=self.f_lengthscale, length_scale_bounds=(1e-5, 1e5), nu=2.5) + WhiteKernel(0.5**2)
        self.f_gpr = GaussianProcessRegressor(
            kernel=self.f_kernel,
            alpha=self.f_sigma
        )
        self.v_sigma = 0.0001
        self.v_variance = math.sqrt(2)
        self.v_lengthscale = 0.5
        self.v_const = 4
        self.v_kernel = ConstantKernel(self.v_const) + math.sqrt(2) * Matern(length_scale=self.v_lengthscale, length_scale_bounds=(1e-5, 1e5), nu=2.5) + WhiteKernel(self.v_sigma**2)
        self.v_gpr = GaussianProcessRegressor(
            kernel=self.v_kernel,
            alpha=self.v_sigma
        )


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x DOMAIN.shape[0] array containing the next point to evaluate
        """

        # In implementing this function, you may use optimize_acquisition_function() defined below.
        if self.x_sample.size == 0:
            # if no point has been sampled yet, we can't optimize the acquisition function yet
            # we instead sample a random starting point in the DOMAIN
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * np.random.rand(DOMAIN.shape[0])
            next_x = np.array([x0]).reshape(-1, DOMAIN.shape[0])
        else:
            if len(self.f_sample) == 12 and np.all(self.f_sample < 0.4):
                x0 = (self.x_sample[0] + (DOMAIN[:, 1] - DOMAIN[:, 0])/2) % DOMAIN[:, 1]
                #x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * np.random.rand(DOMAIN.shape[0])
                next_x = np.array([x0]).reshape(-1, DOMAIN.shape[0])
            else:
                next_x = self.optimize_acquisition_function()

        assert next_x.shape == (1, DOMAIN.shape[0])
        return next_x

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """
        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):########################
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(
                func=objective, x0=x0, bounds=DOMAIN, approx_grad=True
            )

            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.
        Constrained acquisition function as proposed by https://arxiv.org/abs/1403.5607

        Parameters
        ----------
        x: np.ndarray
            x in DOMAIN of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """
        return float(self.expected_improvement(x) * self.constraint_function(x))

    def expected_improvement(self, x):
        """
        Compute expected improvement at points x based on samples x_samples
        and y_samples using Gaussian process surrogate

        Args:
            x: Points at which EI should be computed
        """
        mu, sigma = self.f_gpr.predict([x], return_std=True)
        mu_sample = self.f_gpr.predict(self.x_sample)
        
        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def constraint_function(self, x):
        """
        Model constraint condition v(theta) < v_max as a real-valued latent constraint function
        g_k(x) with g_k(theta) =  v_max - v(theta) > 0 and then infer PR(g_k > 0) from its posterior

        Following: https://arxiv.org/abs/1403.5607
        """

        # predict distribution of speed v
        mu, sigma = self.v_gpr.predict([x], return_std=True)

        # Gaussian CDF with params from GPR prediction
        if sigma != 0:
            pr = norm.cdf(self.v_max, loc=mu, scale=sigma)
        else:
            pr = 0.98 * (self.v_max - mu) if mu <= self.v_max else 0.02 * (mu - self.v_max)

        return pr

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float--> np.ndarray
            structural features
        f: float --> np.ndarray
            logP obj func
        v: float --> np.ndarray
            SA constraint func
        """

        # stack the newly obtained data point onto the existing data points
        self.x_sample    = np.vstack((self.x_sample, x))
        self.f_sample    = np.vstack((self.f_sample, f))
        self.v_sample    = np.vstack((self.v_sample, v))

        # add new datapoint to GPs and retrain
        self.f_gpr.fit(self.x_sample, self.f_sample)
        self.v_gpr.fit(self.x_sample, self.v_sample)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x DOMAIN.shape[0] array containing the optimal solution of the problem
        """

        # select the highest accuracy sample from all valid samples (i.e. samples above the speed threshold)
        valid_samples = self.f_sample
        valid_samples[self.v_sample > self.v_max] = -1e6 # heuristically high number
        best_index = np.argmax(valid_samples)     # get the index of highest accuracy
        x_opt = self.x_sample[best_index]         # get the corresponding x value

        return x_opt


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_DOMAIN(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_DOMAIN = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_DOMAIN)
    x_valid = x_DOMAIN[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)
    # Validate solution
    solution = agent.get_solution()
    assert check_in_DOMAIN(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
