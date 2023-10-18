import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.kernel_approximation import Nystroem
from sklearn.cluster import KMeans
from sklearn import svm

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        # Optimal kernel as per CV
        self.kernel = Matern(1.0, (1e-5, 1e4), nu=2.5) + WhiteKernel(1.0, (1e-5, 1e2))
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            copy_X_train=False,
            random_state=22,
        )
        # self.nystroem_feature_map = Nystroem(
        #    gamma=0.2, random_state=1, n_components=1000, n_jobs=-1
        # )
        # self.lc = svm.LinearSVR()
        # TODO: Add custom initialization for your model here if necessary

    def predict_cost_align(
        self, gp_mean: np.ndarray, gp_std: np.ndarray, AREA_idxs: np.ndarray
    ) -> np.ndarray:
        """
        Custom prediction assignment based on asymmetric cost.
        If threshold value within one std of pred mean, then pred = threshold.
        Else attempt to underpredict by subtracting a fraction of the std.

        :returns: predictions based on above rule
        """
        thresh_mask = AREA_idxs

        pred = np.zeros(len(gp_mean))
        pred[thresh_mask] = gp_mean[thresh_mask]
        pred[~thresh_mask] = gp_mean[~thresh_mask] + 0.75 * gp_std[~thresh_mask]
        print("Prediction aligned for asymmetric cost")

        return pred

    def make_predictions(
        self, test_x_2D: np.ndarray, test_x_AREA: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_x_2D: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_x_AREA: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each city_area here
        gp_mean, gp_std = self.gp.predict(test_x_2D, return_std=True)

        pred = self.predict_cost_align(gp_mean, gp_std, test_x_AREA.astype(bool))
        # print(gp_std)

        return pred, gp_mean, gp_std

        # return self.lc.predict(self.nystroem_feature_map.transform(test_x_2D)),[],[]

    def fitting_model(self, train_y: np.ndarray, train_x_2D: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x_2D: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        train_x_2D, train_y = samp_reduce(train_x_2D, train_y)
        print("Sample size reduced: %s / %s" % (train_x_2D.shape, train_y.shape))

        self.gp.fit(train_x_2D, train_y)
        print("Kernel:", self.gp.kernel_)
        # self.lc.fit(self.nystroem_feature_map.fit_transform(train_x_2D), train_y)


def samp_reduce(train_x, train_y, k=5000):
    """
    Reduces the sample size via k-means clustering
    to deal with GP inference complexity.

    :k: desired sample size
    """

    cluster = KMeans(n_clusters=k)
    cluster.fit(train_x, train_y)

    train_y_new = np.empty((k,))
    for i in range(0, k):
        idx = cluster.labels_ == i
        train_y_new[i] = train_y[idx].mean()
    train_x_new = cluster.cluster_centers_
    return train_x_new, train_y_new


# You don't have to change this function
def cost_function(
    ground_truth: np.ndarray, predictions: np.ndarray, AREA_idxs: np.ndarray
) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert (
        ground_truth.ndim == 1
        and predictions.ndim == 1
        and ground_truth.shape == predictions.shape
    )

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(AREA_idx) for AREA_idx in AREA_idxs]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_in_circle(coor, circle_coor):
    """
    Checks if a coordinate is inside a circle.
    :param coor: 2D coordinate
    :param circle_coor: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coor[0] - circle_coor[0]) ** 2 + (
        coor[1] - circle_coor[1]
    ) ** 2 < circle_coor[2] ** 2


# You don't have to change this function
def determine_city_area_idx(visualization_xs_2D):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param visualization_xs_2D: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array(
        [
            [0.5488135, 0.71518937, 0.17167342],
            [0.79915856, 0.46147936, 0.1567626],
            [0.26455561, 0.77423369, 0.10298338],
            [0.6976312, 0.06022547, 0.04015634],
            [0.31542835, 0.36371077, 0.17985623],
            [0.15896958, 0.11037514, 0.07244247],
            [0.82099323, 0.09710128, 0.08136552],
            [0.41426299, 0.0641475, 0.04442035],
            [0.09394051, 0.5759465, 0.08729856],
            [0.84640867, 0.69947928, 0.04568374],
            [0.23789282, 0.934214, 0.04039037],
            [0.82076712, 0.90884372, 0.07434012],
            [0.09961493, 0.94530153, 0.04755969],
            [0.88172021, 0.2724369, 0.04483477],
            [0.9425836, 0.6339977, 0.04979664],
        ]
    )

    visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0],))

    for i, coor in enumerate(visualization_xs_2D):
        visualization_xs_AREA[i] = any([is_in_circle(coor, circ) for circ in circles])

    return visualization_xs_AREA


# You don't have to change this function
def perform_extended_evaluation(model: Model, output_dir: str = "/results"):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print("Performing extended evaluation")

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS)
        / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS)
        / EVALUATION_GRID_POINTS,
    )
    visualization_xs_2D = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(
        visualization_xs_2D, visualization_xs_AREA
    )
    predictions = np.reshape(
        predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    )
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title("Extended visualization of task 1")
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, "extended_evaluation.pdf")
    fig.savefig(figure_path)
    print(f"Saved extended evaluation to {figure_path}")

    plt.show()


def extract_city_area_information(
    train_x: np.ndarray, test_x: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_x_2D = np.zeros((train_x.shape[0], 2), dtype=float)
    train_x_AREA = np.zeros((train_x.shape[0],), dtype=bool)
    test_x_2D = np.zeros((test_x.shape[0], 2), dtype=float)
    test_x_AREA = np.zeros((test_x.shape[0],), dtype=bool)

    # TODO: Extract the city_area information from the training and test features
    train_x_2D = train_x[:, :2]
    train_x_AREA = train_x[:, 2]
    test_x_2D = test_x[:, :2]
    test_x_AREA = test_x[:, 2]

    assert (
        train_x_2D.shape[0] == train_x_AREA.shape[0]
        and test_x_2D.shape[0] == test_x_AREA.shape[0]
    )
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA


# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt("train_x.csv", delimiter=",", skiprows=1)
    train_y = np.loadtxt("train_y.csv", delimiter=",", skiprows=1)
    test_x = np.loadtxt("test_x.csv", delimiter=",", skiprows=1)

    # Extract the city_area information
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(
        train_x, test_x
    )
    # Fit the model
    print("Fitting model")
    model = Model()
    model.fitting_model(train_y, train_x_2D)

    # Predict on the test features
    print("Predicting on test features")
    predictions = model.make_predictions(test_x_2D, test_x_AREA)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir=".")


if __name__ == "__main__":
    main()
