# --------------------------------------------------------
"""
This script includes the classes to generate weights for the independent variables

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# --------------------------------------------------------

import numpy as np
from scipy.stats import laplace, cauchy, norm

# ------------------------------------------------------------------

class NormWeight:
    """ Class to generate weights from the normal distribution """
    def __init__(self, ind_var: int = 3):
        """
        Args:
            ind_var (int, optional): number of independent variables. Defaults to 3.
        """
        self.ind_var = ind_var

    def gen_weight(self):
        """
        Generate weights for the independent variables
        Returns:
            w (np.array): weights for the independent variables [Variables]
        """
        w = norm.rvs(size=self.ind_var)
        w /= np.sqrt(np.sum(np.abs(w) ** 2))
        return w


class DisturbedNormWeight:
    """ Class to generate weights from the normal distribution and disturbed based on anomalous events """
    def __init__(self, ind_var: int = 3, kw: float = 0.5):
        """
        Args:
            ind_var (int, optional): number of independent variables. Defaults to 3
            kw (float, optional): weight of disturbance. Defaults to 0.5
        """
        self.ind_var = ind_var
        self.kw = kw

    def gen_weight(self, events):
        """
        Generate weights for the independent variables.
        Args:
            events (np.ndarray): events [Variables, Ntime, Nlat, Nlon]
        Returns:
            w_events (np.ndarray): disturbed weights for the independent variables [Variables, Ntime, Nlat, Nlon]
        """

        w = norm.rvs(size=self.ind_var)
        wd = norm.rvs(size=self.ind_var)
        w /= np.sqrt(np.sum(np.abs(w) ** 2))

        # Make sure disturbance is orthogonal to original w
        wd_ort = wd - np.dot(wd, w) * w
        wd_ort /= np.sqrt(np.sum(np.abs(wd_ort) ** 2))
        wd = wd_ort

        w_events = np.zeros(events.shape, dtype=float)

        for v in range(self.ind_var):
            w_events[v, events[v] == 0] = w[v]
            w_events[v, events[v] == 1] = self.kw * wd[v] + (1.0 - self.kw) * w[v]

        return w_events



class CauchyWeight:
    """ Class to generate weights from the Cauchy distribution """
    def __init__(self, ind_var: int = 3):
        """
        Args:
            ind_var (int, optional): number of independent variables. Defaults to 3
        """
        self.ind_var = ind_var

    def gen_weight(self):
        """
        Generate weights for the independent variables
        Returns:
            w (np.array): weights for the independent variables [Variables]
        """
        w = cauchy.rvs(size=self.ind_var)
        w /= np.sqrt(np.sum(np.abs(w) ** 2))
        return w


class DisturbedCauchyWeight:
    """ Class to generate weights from the Cauchy distribution and disturbed based on anomalous events """

    def __init__(self, ind_var: int = 3, kw: float = 0.5):
        """
        Args:
            ind_var (int, optional): number of independent variables. Defaults to 3
            kw (float, optional): weight of disturbance. Defaults to 0.5
        """
        self.ind_var = ind_var
        self.kw = kw

    def gen_weight(self, events):
        """
        Generate weights for the independent variables.
        Args:
            events (np.ndarray): events [Variables, Ntime, Nlat, Nlon]
        Returns:
            w_events (np.ndarray): disturbed weights for the independent variables [Variables, Ntime, Nlat, Nlon]
        """
        w = cauchy.rvs(size=self.ind_var)
        wd = cauchy.rvs(size=self.ind_var)
        w /= np.sqrt(np.sum(np.abs(w) ** 2))

        # Make sure disturbance is orthogonal to original w
        wd_ort = wd - np.dot(wd, w) * w
        wd_ort /= np.sqrt(np.sum(np.abs(wd_ort) ** 2))
        wd = wd_ort

        w_events = np.zeros(events.shape, dtype=float)

        for v in range(self.ind_var):
            w_events[v, events[v] == 0] = w[v]
            w_events[v, events[v] == 1] = self.kw * wd[v] + (1.0 - self.kw) * w[v]

        return w_events


class LaplacWeight:
    """ Class to generate weights from the Laplacian distribution """
    def __init__(self, ind_var: int = 3):
        """
        Args:
            ind_var (int, optional): number of independent variables. Defaults to 3
        """
        self.ind_var = ind_var

    def gen_weight(self):
        """
        Generate weights for the independent variables
        Returns:
            w (np.array): weights for the independent variables [Variables]
        """
        w = laplace.rvs(size=self.ind_var)
        w /= np.sqrt(np.sum(np.abs(w) ** 2))
        return w


class DisturbedLaplaceWeight:
    """ Class to generate weights from the Laplacian distribution and disturbed based on anomalous events """

    def __init__(self, ind_var: int = 3, kw: float = 0.5):
        """
        Args:
            ind_var (int, optional): number of independent variables. Defaults to 3
            kw (float, optional): weight of disturbance. Defaults to 0.5
        """
        self.ind_var = ind_var
        self.kw = kw

    def gen_weight(self, events):
        """
        Generate weights for the independent variables.
        Args:
            events (np.ndarray): events [Variables, Ntime, Nlat, Nlon]
        Returns:
            w_events (np.ndarray): disturbed weights for the independent variables [Variables, Ntime, Nlat, Nlon]
        """
        w = laplace.rvs(size=self.ind_var)
        wd = laplace.rvs(size=self.ind_var)
        w /= np.sqrt(np.sum(np.abs(w) ** 2))

        # Make sure disturbance is orthogonal to original w
        wd_ort = wd - np.dot(wd, w) * w
        wd_ort /= np.sqrt(np.sum(np.abs(wd_ort) ** 2))
        wd = wd_ort

        w_events = np.zeros(events.shape, dtype=float)

        for v in range(self.ind_var):
            w_events[v, events[v] == 0] = w[v]
            w_events[v, events[v] == 1] = self.kw * wd[v] + (1.0 - self.kw) * w[v]

        return w_events






if __name__ == '__main__':

    print('test weights...')
    print('test normal weights...')
    W = NormWeight()
    print(W.gen_weight())

    W = DisturbedNormWeight()
    e = np.random.randint(0, 2, (3, 52*4, 100, 100), dtype=np.uint8)
    print(W.gen_weight(e).shape)

    print('test cauchy weights...')
    W = CauchyWeight()
    print(W.gen_weight())

    W = DisturbedCauchyWeight()
    print(W.gen_weight(e).shape)

    print('test laplace weights...')

    W = LaplacWeight()
    print(W.gen_weight())

    W = DisturbedLaplaceWeight()
    print(W.gen_weight(e).shape)

