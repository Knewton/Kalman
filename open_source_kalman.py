"""
Open-source code by @andersenchen and @sbchou at Knewton.

August 2013

Contains class KalmanFilter, an implementation of the hybrid Kalman
filter algorithm), and a Simulator class for testing.
"""

import itertools
import math
import random
import numpy as np

class KalmanFilter:
    """
    A class for statistical inference of timeseries data using 
    the Kalman Filter algorithm.

    The system we solve is a set of latent scalars X = {x_1, x_2,...,x_k} given
    scalar observations Z = {z_1, z_2,...,z_k} occuring at times {t_1, t_2,...,t_k},
    where our latent state is evolved as a Gaussian random walk:

    x_k = x_{k_1} + sqrt(t_k - t_{k-1}) * w_k
    with w_k ~ N(0, q)
    and t_k - t_{k-1} being the time difference between our current and previous
    observations.

    And an observation of our latent state is made according to:
    z_k = x_k + v_k
    with v_k ~ N(0, r)
    """
    def __init__(self, x0=0, var0=0.001, q=0.01, r=0.1):
        """
        Initialize starting values. (x0, var0) represent starting mean and 
        variance estimates of the filter. q is the variance of the latent process,
        and r is the variance of the observation noise.

        Parameters:
            x0: Scalar that represents the starting mean estimate of the filter
            var0: Positive scalar that represents the starting variance estimate of the filter
            q: Positive scalar representing the latent process variance
            r: Positive scalar representing the observation noise variance
        """
        assert var0 > 0, "starting variance should be a positive scalar"
        assert q > 0, "latent state variance should be a positive scalar"
        assert r > 0, "observation noise variance should be positive scalar"

        self.x_curr = x0
        self.var_curr = var0
        self.q = q
        self.r = r
        self.initialize_estimates()

    def initialize_estimates(self):
        """
        Start with empty list of means and variances
        """
        self.means = []
        self.variances = []

    def update(self, observation, timeDifference):
        """
        Update our posterior to account for a new (observation, timeDifference))
        """
        assert timeDifference > 0, "timeDifference should be a positive scalar"
        x_predict = self.x_curr

        #prediction step
        var_predict = self.var_curr + self.q * math.sqrt(timeDifference)

        kalman_gain = var_predict / (var_predict + self.r)

        self.x_curr = x_predict + kalman_gain * (observation - x_predict)
        self.var_curr = kalman_gain * self.r

        self.means.append(self.x_curr)
        self.variances.append(self.var_curr)

        return self.x_curr, self.var_curr

    def filter_observations(self, observations, timeDifferences):
        """
        Run the filter on multiple observations.
        The length of observations should equal the length of timeDifferences.
        """
        assert len(observations) == len(timeDifferences)

        for obs, t in itertools.izip(observations, timeDifferences):
            self.update(obs, t)

        return self.means, self.variances

class Simulator:
    """
    A class for generating simulation data and running it through 
    the Kalman filter.
    """
    def __init__(self):
        pass

    def test_constant(self, var_z, n=1000):
        """
        We have a constant latent state at zero having observation noise with
        variance var_z. Observation times are assumed to be equally spaced.
        We simulate data according to this model and do parameter recovery.

        Parameters:
            var_z:
            Positive scalar representing the variance of the observation noise
        """
        assert var_z > 0, "var_z should be a positive scalar"

        observations = [random.gauss(0, var_z) for i in xrange(n)]
        timeDifferences = [1] * n

        kalman = KalmanFilter()
        kalman.filter_observations(observations, timeDifferences)
        self.visualize(kalman, [0] * n, observations, timeDifferences)

    def generate_noisy_gaussian_walk(self, var_x, var_z, max_time, n=1000):
        """
        Generate simulated data with n continuous random gaussian observations
        and uniform random time differences, where var_x is the
        variance of the latent process governing x_k, and
        var_z is the variance of the observation noise, and n is the number of
        observations made. We draw time differences ~ Unif(0, max_time)
        where max_time is the maximum time difference.

        Parameters:
            var_x:
            Positive scalar representing the variance of the latentprocess
            var_z:
            Positive scalar representing the variance of the observation noise
            max_time:
            Positive scalar representing the maximum possible time difference
            between two observations
            n:
            Positive integer representing the number of observations to be
            generated
        """

        assert var_x > 0, "var_x should be a positive scalar"
        assert var_z > 0, "var_z should be a positive scalar"
        assert max_time > 0, "max_time should be a positive scalar"
        assert n > 0 and int(n) == n, "n should be a positive integer"

        observations = []
        truth = []
        timeDifferences = [random.random() * max_time for i in xrange(n)]

        latent = 0
        for i in xrange(n):
            latent += random.gauss(0, var_x) * math.sqrt(timeDifferences[i])
            truth.append(latent)
            observations.append(latent + random.gauss(0, var_z))

        return truth, observations, timeDifferences

    def test_noisy_gaussian_walk(self, test_var_latent=0.01,
        test_var_observed=0.1, test_max_time_diff=5.):
        """
        Test recovery of latent parameters generated by noisy gaussian walk 
        (continuous random time differences) using the Kalman Filter.
        Parameters:
            test_var_latent:
            Positive scalar representing the variance of thelatent process
            we generate
            test_var_observed: Positive scalar representing the variance of the
            observation noise we generate
            test_max_time_diff:

        """
        assert test_var_latent > 0, "test_var_latent should be a positive scalar"
        assert test_var_observed > 0, "test_var_observed should be a positive scalar"
        assert test_max_time_diff > 0, "test_max_time_diff should be a positive scalar"

        truth, observations, timeDifferences = \
            self.generate_noisy_gaussian_walk(test_var_latent,
                test_var_observed, test_max_time_diff)
        kalman = KalmanFilter()
        kalman.filter_observations(observations, timeDifferences)
        self.visualize(kalman, truth, observations, timeDifferences)

    def visualize(self, kalman, truth, observations, timeDifferences):
        """
        Plot recovered and original latent variables along with observations
        using matplotlib
        """
        from matplotlib import pyplot as plt
        plt.xlabel("Time")
        plt.ylabel("Latent state")
        plt.title("Latent state vs. time")
        p1, = plt.plot(np.cumsum(timeDifferences), truth)
        p2, = plt.plot(np.cumsum(timeDifferences), kalman.means)
        xLength = len(kalman.means)
        p3 = plt.scatter(np.cumsum(timeDifferences), observations, alpha = 0.2)
        bottom = [kalman.means[i] - 2 * math.sqrt(kalman.variances[i]) for 
            i in xrange(xLength)]
        top = [kalman.means[i] + 2 * math.sqrt(kalman.variances[i]) for
            i in xrange(xLength)]
        plt.fill_between(np.cumsum(timeDifferences),
            bottom, top, color="green", alpha=0.2)
        p = plt.Rectangle((0,0), 1, 1, fc="g", alpha=0.2)
        plt.legend((p1, p2, p3, p), ("Truth", "Posterior means", "Observations",
            "95 Percent Confidence Interval"))
        plt.show()


if __name__ == '__main__':
    s = Simulator()
    s.test_noisy_gaussian_walk()
    