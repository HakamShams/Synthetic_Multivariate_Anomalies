# --------------------------------------------------------
"""
This script includes the classes to generate coupling types

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""

# --------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------

class LinearCoupling:
    """ Class to generate dependent variables based on a linear coupling between the independent variables """

    def combine_variables(self, datacube: np.array, event: np.array, weight: np.array, noise: np.array):
        """
        Generate dependent signals

        Args:
            datacube (np.array): the input independent variables [Variables, Ntime, Nlat, Nlon]
            event (np.array): the input anomalous event [Variables, Ntime, Nlat, Nlon]
            weight (np.array): the input weight [Variables] or [Variables, Ntime, Nlat, Nlon]
            noise (np.array): the input noise [Variables, Ntime, Nlat, Nlon]
        Returns:
            datacube_combined (np.array): the dependent signals [Variables, Ntime, Nlat, Nlon]
            event_combined (np.array): the anomalous events for the dependent signals [Variables, Ntime, Nlat, Nlon]
        """
        datacube_combined = np.zeros(noise.shape, dtype=float)
        event_combined = np.zeros(noise.shape, dtype=float)

        for v in range(len(datacube)):
            datacube_combined += datacube[v] * weight[v]
            #event_combined += event[v] * np.abs(weight[v])

        datacube_combined += noise

        # generate anomalous events if half of the independent variables have anomalous event
        #event_combined[event_combined >= 0.5] = 1
        #event_combined[event_combined < 0.5] = 0
        # generate anomalous events if there is at least one anomalous event  in the independent variables
        event_combined[np.sum(event, axis=0) > 0] = 1

        return datacube_combined, event_combined


class QuadraticCoupling:
    """ Class to generate dependent variables based on a quadratic coupling between the independent variables """

    def combine_variables(self, datacube: np.array, event: np.array, weight: np.array, noise: np.array):
        """
        Generate dependent signals

        Args:
            datacube (np.array): the input independent variables [Variables, Ntime, Nlat, Nlon]
            event (np.array): the input anomalous event [Variables, Ntime, Nlat, Nlon]
            weight (np.array): the input weight [Variables] or [Variables, Ntime, Nlat, Nlon]
            noise (np.array): the input noise [Variables, Ntime, Nlat, Nlon]
        Returns:
            datacube_combined (np.array): the dependent signals [Variables, Ntime, Nlat, Nlon]
            event_combined (np.array): the anomalous events for the dependent signals [Variables, Ntime, Nlat, Nlon]
        """
        datacube_combined = np.zeros(noise.shape, dtype=float)
        event_combined = np.zeros(noise.shape, dtype=float)

        for v in range(len(datacube)):
            datacube_combined += (datacube[v] ** 2 - 1) * (2**-0.5) * weight[v]
            #event_combined += event[v] * np.abs(weight[v])

        datacube_combined += noise

        # generate anomalous events if half of the independent variables have anomalous event
        #event_combined[event_combined >= 0.5] = 1
        #event_combined[event_combined < 0.5] = 0
        # generate anomalous events if there is at least one anomalous event  in the independent variables
        event_combined[np.sum(event, axis=0) > 0] = 1

        return datacube_combined, event_combined


class ExtremeCoupling:
    """ Generate the coupling matrix between extremes and anomalous events """
    def __init__(self, var: int = 6, var_drop: list = None, interval: int = 10):
        """
        Args:
            var (int, optional): the number of variables to generate. Defaults to 6
            var_drop (list, optional): the list of variables to exclude. Defaults to None
            interval (int, optional): the time interval. Defaults to 10
        """
        self.var = var
        self.var_drop = var_drop
        self.interval = interval

    def couple_event(self):
        """
        Generate the coupling matrix between extremes and anomalous events
        Returns:
            events (np.ndarray): the coupling matrix [Variables, Interval]
        """

        events = np.zeros((self.var, self.interval), dtype=np.uint8)

        for v in range(self.var):

            if v + 1 in self.var_drop:
                continue

            p_e = np.random.randint(0, self.interval)
            s_e = np.random.randint(5, self.interval + 1) // 2

            events[v, max(0, p_e - s_e): min(self.interval, p_e + s_e)] = 1

        return events


if __name__ == '__main__':

    print('test coupling...')
    datacube = np.random.randn(3, 52*4, 100, 100)
    noise = np.random.randn(52*4, 100, 100) * 0.1
    weight = np.random.randn(3, 52*4, 100, 100)
    event = np.random.randint(0, 2, (3, 52*4, 100, 100))

    print('test extreme coupling...')
    E_var = ExtremeCoupling(var_drop=[4, 5]).couple_event()
    print(E_var)
    plt.imshow(E_var)
    plt.show()

    print('test linear coupling...')
    L = LinearCoupling()
    d, e = L.combine_variables(datacube, event, weight, noise)
    print(np.unique(d))
    print(np.unique(e))

    print('test quadratic coupling...')
    L = QuadraticCoupling()
    d, e = L.combine_variables(datacube, event, weight, noise)
    print(np.unique(d))
    print(np.unique(e))

