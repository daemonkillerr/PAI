"""Solution."""

import math
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

import warnings
warnings.filterwarnings("ignore")

# global variables
np.random.seed(14504)
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

""" Solution """

class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """
        self.v_max = SAFETY_THRESHOLD

        self.x_sample = np.empty((0, DOMAIN.shape[0]))
        self.f_sample = np.empty((0, DOMAIN.shape[0]))
        self.v_sample = np.empty((0, DOMAIN.shape[0]))
        #self.logv_sample = np.empty((0, DOMAIN.shape[0]))
        self.gv_sample = np.empty((0, DOMAIN.shape[0]))

        self.f_kernel = 0.125* Matern(length_scale=10, length_scale_bounds=(1e-5, 1e5), nu=2.5) + WhiteKernel(0.5**2)
        self.f_gpr = GaussianProcessRegressor(
            kernel=self.f_kernel,
            alpha=0.15
        )

        self.v_kernel = ConstantKernel(4) + math.sqrt(2) * Matern(length_scale=0.5, length_scale_bounds=(1e-5, 1e5), nu=2.5) + WhiteKernel(0.0001**2)
        self.v_gpr = GaussianProcessRegressor(
            kernel=self.v_kernel,
            alpha=0.0001
        )


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        if self.x_sample.size == 0:
            init_recommendation = np.random.uniform(low=DOMAIN[:, 0], high=DOMAIN[:,1])
            next_recommend = init_recommendation.reshape(-1, DOMAIN.shape[0])
        else:
            if len(self.f_sample) == 12 and np.all(self.f_sample < 0.4):
                DOMAIN_midpoint = (DOMAIN[:, 1] - DOMAIN[:, 0])/2
                init_recommendation = (self.x_sample[0] + DOMAIN_midpoint) % DOMAIN[:, 1]
                next_recommend = init_recommendation.reshape(-1, DOMAIN.shape[0])
            else:
                next_recommend = self.optimize_acquisition_function()

        return np.atleast_2d(next_recommend)

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
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        
        # Expected improvement
        predicted_mean, predicted_std = self.f_gpr.predict(x, return_std=True)

        mean_sample = self.f_gpr.predict(self.x_sample)
        predicted_std = predicted_std.reshape(-1, 1)
        mean_sample_max = np.max(mean_sample)
        with np.errstate(divide='warn'):
            improvement = predicted_mean - mean_sample_max
            Z = improvement / predicted_std
            cdf_Z = norm.cdf(Z)
            pdf_Z = norm.pdf(Z)
            expected_improvement = improvement * cdf_Z + predicted_std * pdf_Z
            idx = predicted_std==0.0
            expected_improvement[idx]=0.0

        # Constraint function
        predicted_values = self.v_gpr.predict(x, return_std=True)
        mu, sigma = predicted_values

        if sigma != 0:
            pr = norm.cdf(self.v_max, loc=mu, scale=sigma)
        else:
            if mu <= self.v_max:
                pr = 0.98 * (self.v_max - mu)
            else:
                pr = 0.02 * (mu-self.v_max)

        return float(expected_improvement * pr)

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        self.x_sample, self.f_sample, self.v_sample = np.vstack((self.x_sample, x)), np.vstack((self.f_sample, f)), np.vstack((self.v_sample, v))
        self.f_gpr.fit(self.x_sample, self.f_sample)
        self.v_gpr.fit(self.x_sample, self.v_sample)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        valid_samples = np.where(self.v_sample > self.v_max, -1e9, self.f_sample)
        best_index = np.argmax(valid_samples)
        x_opt = self.x_sample[best_index]       
        return x_opt
    
    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


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
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
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



'''
# Original
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        pass

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        raise NotImplementedError

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
        for _ in range(20):
            init_recommendation = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, init_recommendation=init_recommendation, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        raise NotImplementedError

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        raise NotImplementedError

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
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
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
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
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
'''