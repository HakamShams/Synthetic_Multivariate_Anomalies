# --------------------------------------------------------
"""
This script includes the classes to generate different events types

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# --------------------------------------------------------
import numpy as np
from scipy import ndimage

# ------------------------------------------------------------------

class CubeEvent:
    """ Class to generate a cube event """
    def __init__(self, n: int = 10, sx: int = 17, sy: int = 17, sz: int = 17):
        """
        Args:
            n (int, optional): number of events. Defaults to 10.
            sx (int, optional): maximum extension of the event in the x direction. Defaults to 17.
            sy (int, optional): maximum extension of the event in the y direction. Defaults to 17.
            sz (int, optional): maximum extension of the event in the z direction. Defaults to 17.
        """
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.n = n

    def gen_event(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate base signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            events (np.ndarray): events [Ntime, Nlat, Nlon]
        """
        events = np.zeros((Ntime, Nlat, Nlon), dtype=np.uint8)

        for _ in range(self.n):
            px_e = np.random.randint(0, Nlon)
            py_e = np.random.randint(0, Nlat)
            pz_e = np.random.randint(0, Ntime)

            sx_e = np.random.randint(3, self.sx + 1) // 2
            sy_e = np.random.randint(3, self.sy + 1) // 2
            sz_e = np.random.randint(3, self.sz + 1) // 2

            events[
            max(0, pz_e - sz_e): min(Ntime, pz_e + sz_e),
            max(0, py_e - sy_e): min(Nlat, py_e + sy_e),
            max(0, px_e - sx_e): min(Nlon, px_e + sx_e)
            ] = 1

        return events


class LocalEvent:
    """ Class to generate a local event """
    def __init__(self, n: int = 10, sz: int = 17):
        """
        Args:
            n (int, optional): number of events. Defaults to 10.
            sz (int, optional): maximum extension of the event in the z direction. Defaults to 17.
        """
        self.sz = sz
        self.n = n

    def gen_event(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate base signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            events (np.ndarray): events [Ntime, Nlat, Nlon]
        """
        events = np.zeros((Ntime, Nlat, Nlon), dtype=np.uint8)

        for _ in range(self.n):
            px_e = np.random.randint(0, Nlon)
            py_e = np.random.randint(0, Nlat)
            pz_e = np.random.randint(0, Ntime)

            sz_e = np.random.randint(3, self.sz + 1) // 2

            events[max(0, pz_e - sz_e): min(Ntime, pz_e + sz_e), py_e, px_e] = 1

        return events


class EmptyEvent:
    """ Class to generate an empty event """
    def gen_event(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate base signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            events (np.ndarray): events [Ntime, Nlat, Nlon]
        """
        return np.zeros((Ntime, Nlat, Nlon), dtype=np.uint8)


class GaussianEvent:
    """ Class to generate a gaussian event """
    def __init__(self, n: int = 10, sx: int = 23, sy: int = 23, sz: int = 23):
        """
        Args:
            n (int, optional): number of events. Defaults to 10.
            sx (int, optional): maximum extension of the event in the x direction. Defaults to 23.
            sy (int, optional): maximum extension of the event in the y direction. Defaults to 23.
            sz (int, optional): maximum extension of the event in the z direction. Defaults to 23.
        """

        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.n = n

    def gen_event(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate base signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            events (np.ndarray): events [Ntime, Nlat, Nlon]
        """
        events = np.zeros((Ntime, Nlat, Nlon), dtype=np.uint8)

        for _ in range(self.n):

            px_e = np.random.randint(0, Nlon)
            py_e = np.random.randint(0, Nlat)
            pz_e = np.random.randint(0, Ntime)

            sx_e = np.random.randint(7, self.sx + 1) // 2
            sy_e = np.random.randint(7, self.sy + 1) // 2
            sz_e = np.random.randint(7, self.sz + 1) // 2

            x_array = np.arange(Nlon)
            x_array = np.repeat(x_array[np.newaxis, :], Nlat, axis=0)
            x_array = np.repeat(x_array[np.newaxis, :], Ntime, axis=0)

            y_array = np.arange(Nlat)
            y_array = np.repeat(y_array[:, np.newaxis], Nlon, axis=1)
            y_array = np.repeat(y_array[np.newaxis, :], Ntime, axis=0)

            z_array = np.arange(Ntime)
            z_array = np.repeat(z_array[:, np.newaxis], Nlat, axis=1)
            z_array = np.repeat(z_array[:, :, np.newaxis], Nlon, axis=2)

            events_i = np.exp(-0.5 * (((x_array - px_e) / sx_e) ** 2 +
                                      ((y_array - py_e) / sy_e) ** 2 +
                                      ((z_array - pz_e) / sz_e) ** 2))

            #events_i = np.zeros((Ntime, Nlat, Nlon), dtype=float)
            #for k in range(Ntime):
            #    for j in range(Nlat):
            #        for i in range(Nlon):
            #            events_i[k, j, i] = np.exp(-0.5 * (
            #                    (i - px_e) ** 2 / sx_e ** 2
            #                    + (j - py_e) ** 2 / sy_e ** 2
            #                    + (k - pz_e) ** 2 / sz_e ** 2
            #            ))

            det = (sx_e ** 2) * (sy_e ** 2) * (sz_e ** 2)

            events_i = events_i * det ** (-0.5) * (2 * np.pi) ** (-3 / 2) * 100

            events_i = (events_i - np.min(events_i)) / (np.max(events_i) - np.min(events_i))

            thr = (100 - np.max([sx_e, sy_e, sz_e])) / 100
            events_i[events_i >= thr] = 1
            events_i[events_i < thr] = 0

            events[events_i == 1] = 1

        return events.astype(np.uint8)


class OnsetEvent:
    """ Class to generate an onset event
        This generates an event of spatial size sx, sy at px_e, py_e that starts after the time os
        and lasts until the end of the time series
    """

    def __init__(self, n: int = 10, sx: int = 17, sy: int = 17, os: float = 0.9):
        """
        Args:
            n (int, optional): number of events. Defaults to 10.
            sx (int, optional): maximum extension of the event in the x direction. Defaults to 17.
            sy (int, optional): maximum extension of the event in the y direction. Defaults to 17.
            os (int, optional): time step at which the event can start. os is given in percent. Defaults to 0.9.
        """
        self.sx = sx
        self.sy = sy
        self.os = os
        self.n = n

    def gen_event(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate base signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            events (np.ndarray): events [Ntime, Nlat, Nlon]
        """
        events = np.zeros((Ntime, Nlat, Nlon), dtype=np.uint8)

        for _ in range(self.n):

            px_e = np.random.randint(0, Nlon)
            py_e = np.random.randint(0, Nlat)
            pz_e = np.random.randint(self.os * Ntime, Ntime)

            sx_e = np.random.randint(3, self.sx + 1) // 2
            sy_e = np.random.randint(3, self.sy + 1) // 2

            events[
            pz_e:,
            max(0, py_e - sy_e): min(Nlat, py_e + sy_e),
            max(0, px_e - sx_e): min(Nlon, px_e + sx_e)
            ] = 1

        return events


class RandomWalkEvent:
    """
    Class to generate a random walk event
    This generates an event by doing a random walk in the 3D volume starting at a random location px_e, py_e and pz_e.
    """

    def __init__(self, n: int = 10, s: int = 125):
        """
        Args:
            n (int, optional): number of events. Defaults to 10.
            s (int, optional): number of random steps. Defaults to 125.
        """
        self.s = s
        self.n = n

    def gen_event(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate base signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            events (np.ndarray): events [Ntime, Nlat, Nlon]
        """
        events = np.zeros((Ntime, Nlat, Nlon), dtype=np.uint8)

        for _ in range(self.n):

            px_e = np.random.randint(0, Nlon)
            py_e = np.random.randint(0, Nlat)
            pz_e = np.random.randint(0, Ntime)

            n_step = 0
            n_step_limit = 0

            events[pz_e, py_e, px_e] = 1

            while n_step < self.s and n_step_limit < self.s*10:

                px_e_rand = np.random.randint(-1, 2)
                px_e_neu = px_e + px_e_rand
                py_e_rand = np.random.randint(-1, 2)
                py_e_neu = py_e + py_e_rand
                pz_e_rand = np.random.randint(-1, 2)
                pz_e_neu = pz_e + pz_e_rand

                n_step_limit += 1

                if px_e_neu >= 0 and px_e_neu < Nlon:
                    if py_e_neu >= 0 and py_e_neu < Nlat:
                        if pz_e_neu >= 0 and pz_e_neu < Ntime:
                            if events[pz_e_neu, py_e_neu, px_e_neu] != 1:
                                events[pz_e_neu, py_e_neu, px_e_neu] = 1
                                px_e = px_e_neu
                                py_e = py_e_neu
                                pz_e = pz_e_neu
                                n_step += 1

        return events


class ExtremeEvent:
    """ Class to generate extreme events based on the number of anomalies before the event """
    def __init__(self, a_thr: int = 20, t_interval: int = 10):
        """
        Args:
            a_thr (int, optional): number of anomalies before the extreme event. Defaults to 20.
            t_interval (int, optional): time interval between anomalies and extremes. Defaults to 10.
        """
        self.a_thr = a_thr
        self.t_interval = t_interval  #including the time t

    def gen_event(self, anomalous_events):
        """
        Generate extreme events
        Args:
            anomalous_events (np.ndarray): anomalous events [Variable, Ntime, Nlat, Nlon]
        Returns:
            extreme_events (np.ndarray): extreme events [Ntime, Nlat, Nlon]
        """
        V, T, H, W = anomalous_events.shape
        extreme_events = np.zeros((T, H, W), np.uint8)

        for t in range(T):
            if t < self.t_interval - 1:
                continue

            n_anomalous = np.sum(anomalous_events[:, t + 1 - self.t_interval: t + 1, ...], axis=(0, 1))
            extreme_events[t, n_anomalous > self.a_thr] = 1

        return extreme_events



class ExtremeClass:
    """ Class to generate extreme event classes """
    def __init__(self, n_added_classes: int = 2):
        """
        Args:
            n_added_classes (int, optional): number of added classes. Defaults to 2.
        """
        self.n_added_classes = n_added_classes

    def gen_event(self, extreme_events):
        """
        Generate extreme events
        Args:
            extreme_events (np.ndarray): extreme events with only one class representing the most extreme class [Ntime, Nlat, Nlon]
        Returns:
            extreme_events_all (np.ndarray): extreme events with added classes [Ntime, Nlat, Nlon]
        """

        extreme_events_all = extreme_events.copy()

        for c in range(self.n_added_classes):
            extreme_events_c = ndimage.generic_filter(extreme_events, np.max, size=(3, 1, 1), mode='constant')
            extreme_events_all[extreme_events_c - extreme_events == 1] = c + 2
            extreme_events[extreme_events_c - extreme_events > 0] = 1

        return extreme_events_all



if __name__ == '__main__':

    print('test events...')

    Ntime = 52 * 46 // 4
    Nlat = 100
    Nlon = 100

    print('CubeEvent...')
    E = CubeEvent(n=200, sx=25, sy=25, sz=25).gen_event(Ntime, Nlat, Nlon)
    print('number of anomalous pixels: ', np.sum(E))
    E = ExtremeClass(n_added_classes=2).gen_event(E)

    print('ExtremeEvent...')
    D = ExtremeEvent().gen_event(np.concatenate((E[None, ...], E[None, ...]), axis=0))
    print('number of extreme pixels: ', np.sum(D))

    print('LocalEvent...')
    E = LocalEvent().gen_event(Ntime, Nlat, Nlon)
    print('number of anomalous pixels: ', np.sum(E))

    print('EmptyEvent...')
    E = EmptyEvent().gen_event(Ntime, Nlat, Nlon)
    print('number of anomalous pixels: ', np.sum(E))

    print('GaussianEvent...')
    E = GaussianEvent().gen_event(Ntime, Nlat, Nlon)
    print('number of anomalous pixels: ', np.sum(E))

    print('OnsetEvent...')
    E = OnsetEvent().gen_event(Ntime, Nlat, Nlon)
    print('number of anomalous pixels: ', np.sum(E))

    print('RandomWalkEvent...')
    E = RandomWalkEvent().gen_event(Ntime, Nlat, Nlon)
    print('number of anomalous pixels: ', np.sum(E))

