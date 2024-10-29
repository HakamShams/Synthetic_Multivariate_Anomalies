# --------------------------------------------------------
"""
This script includes the classes to generate different noise signal types

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# --------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn

# ------------------------------------------------------------------


class WhiteNoise:
    """ Class to generate white noise signal """
    def __init__(self, meu: float = 0., sigma: float = 1.):
        """
        Args:
            meu (float, optional): mean noise value. Defaults to 0.
            sigma (float, optional): standard deviation noise value. Defaults to 1.
        """
        self.meu = meu
        self.sigma = sigma

    def gen_noise(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate noise signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            noise (np.ndarray): noise signal [Ntime, Nlat, Nlon]
        """
        noise = np.random.default_rng().normal(self.meu, self.sigma, (Ntime, Nlat, Nlon))
        return noise


class CauchyNoise:
    """ Class to generate cauchy noise signal """
    def __init__(self, meu: float = 0., sigma: float = 1.):
        """
        Args:
            meu (float, optional): mean noise value. Defaults to 0.
            sigma (float, optional): standard deviation noise value. Defaults to 1.
        """
        self.meu = meu
        self.sigma = sigma

    def gen_noise(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate noise signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            noise (np.ndarray): noise signal [Ntime, Nlat, Nlon]
        """
        noise = np.random.default_rng().standard_cauchy((Ntime, Nlat, Nlon)) + self.meu
        if self.sigma > 0:
            noise *= (self.sigma / np.std(noise))
        return noise


class LaplaceNoise:
    """ Class to generate laplace noise signal """
    def __init__(self, meu: float = 0., sigma: float = 1., lambda_laplace: float = 1.):
        """
        Args:
            meu (float, optional): mean noise value. Defaults to 0.
            sigma (float, optional): standard deviation noise value. Defaults to 1.
            lambda_laplace (float, optional): the exponential decay Lambda. Defaults to 1.
        """
        self.meu = meu
        self.sigma = sigma
        self.lambda_laplace = lambda_laplace

    def gen_noise(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate noise signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            noise (np.ndarray): noise signal [Ntime, Nlat, Nlon]
        """
        noise = np.random.default_rng().laplace(self.meu, self.lambda_laplace, (Ntime, Nlat, Nlon))
        if self.sigma > 0:
            noise *= (self.sigma / np.std(noise))
        return noise

class RedNoise:
    """ Class to generate red noise signal """
    def __init__(self, meu: float = 0., sigma: float = 0.2, btime: float = 1., blat: float = 1., blon: float = 1.):

        """
        Args:
            meu (float, optional): mean noise value. Defaults to 0.
            sigma (float, optional): standard deviation noise value. Defaults to 0.2.
            btime (float, optional): scaling factor in the time direction. Defaults to 1.
            blat (float, optional): scaling factor in the latitude direction. Defaults to 1.
            blon (float, optional): scaling factor in the longitude direction. Defaults to 1.
        """
        self.meu = meu
        self.sigma = sigma

        self.btime = btime
        self.blat = blat
        self.blon = blon

    def gen_noise(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate noise signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            noise (np.ndarray): noise signal [Ntime, Nlat, Nlon]
        """
        noise = np.random.normal(self.meu, self.sigma, (Ntime, Nlat, Nlon))
        noise = fftn(noise)

        """
        for i in range(1, Nlon+1):
            flon = Nlon - i + 1 if i > (Nlon + 1) / 2 else i - 1
            flon = np.max([flon, 1.0])
            for j in range(1, Nlat + 1):
                flat = Nlat - j + 1 if j > (Nlat + 1) / 2 else j - 1
                flat = np.max([flat, 1.0])
                for k in range(1, Ntime + 1):
                    ftime = Ntime - k + 1 if k > (Ntime + 1) / 2 else k - 1
                    ftime = np.max([ftime, 1.0])
                    noise[k-1, j-1, i-1] /= (ftime ** self.btime) * (flon ** self.blon) * (flat ** self.blat)
        """

        # this is more efficient than for loops
        flon_array = np.arange(1, Nlon + 1)
        flon_array = np.repeat(flon_array[np.newaxis, :], Nlat, axis=0)
        flon_array = np.repeat(flon_array[np.newaxis, :], Ntime, axis=0)
        idx_flon = (flon_array > ((Nlon + 1) / 2))
        flon_array[idx_flon] = Nlon - flon_array[idx_flon] + 1
        flon_array[~idx_flon] = flon_array[~idx_flon] - 1
        flon_array = np.max(np.concatenate((flon_array[None, ...], np.ones((1, Ntime, Nlat, Nlon))), axis=0), 0)

        flat_array = np.arange(1, Nlat + 1)
        flat_array = np.repeat(flat_array[:, np.newaxis], Nlon, axis=1)
        flat_array = np.repeat(flat_array[np.newaxis, :], Ntime, axis=0)
        idx_flat = (flat_array > ((Nlat + 1) / 2))
        flat_array[idx_flat] = Nlat - flat_array[idx_flat] + 1
        flat_array[~idx_flat] = flat_array[~idx_flat] - 1
        flat_array = np.max(np.concatenate((flat_array[None, ...], np.ones((1, Ntime, Nlat, Nlon))), axis=0), 0)

        ftime_array = np.arange(1, Ntime + 1)
        ftime_array = np.repeat(ftime_array[:, np.newaxis], Nlat, axis=1)
        ftime_array = np.repeat(ftime_array[:, :, np.newaxis], Nlon, axis=2)
        idx_ftime = (ftime_array > ((Ntime + 1) / 2))
        ftime_array[idx_ftime] = Ntime - ftime_array[idx_ftime] + 1
        ftime_array[~idx_ftime] = ftime_array[~idx_ftime] - 1
        ftime_array = np.max(np.concatenate((ftime_array[None, ...], np.ones((1, Ntime, Nlat, Nlon))), axis=0), 0)

        noise /= (ftime_array ** self.btime) * (flat_array ** self.blat) * (flon_array ** self.blon)

        noise = ifftn(noise)
        noise = noise.real
        noise *= self.sigma / np.std(noise)

        return noise


if __name__ == '__main__':

    print('test noises...')

    Ntime = 52*40
    Nlat = 200
    Nlon = 200

    print('WhiteNoise')
    N = WhiteNoise(0, 1.).gen_noise(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), N[:, Nlat//2, Nlon//2])
    plt.show()
    plt.imshow(N[Ntime//2])
    plt.show()

    print('CauchyNoise')
    N = CauchyNoise(0, 1.).gen_noise(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), N[:, Nlat//2, Nlon//2])
    plt.show()
    plt.imshow(N[Ntime//2])
    plt.show()

    print('LaplaceNoise')
    N = LaplaceNoise(0, 1., 1.).gen_noise(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), N[:, Nlat//2, Nlon//2])
    plt.show()
    plt.imshow(N[Ntime//2])
    plt.show()

    print('RedNoise')
    N = RedNoise().gen_noise(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), N[:, Nlat // 2, Nlon // 2])
    plt.show()
    plt.imshow(N[Ntime//2])
    plt.show()

