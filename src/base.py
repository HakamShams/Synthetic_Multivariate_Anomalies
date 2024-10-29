# --------------------------------------------------------
"""
This script includes the classes to generate different base signal types

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# --------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

np.set_printoptions(suppress=True)
xr.set_options(display_max_rows=40)
xr.set_options(display_width=1000)

# ------------------------------------------------------------------


class ConstantBase:
    """ Class to generate constant base signal """
    def __init__(self, const: float = 0., latGrad: bool = True):
        """
        Args:
            const (float, optional): constant value for the signal. Defaults to 0.
            latGrad (bool, optional): whether to add gradient for the base along latitude. Defaults to True.
        """
        self.const = const
        self.latGrad = latGrad

    def gen_base(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate base signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            base (np.ndarray): base signal [Ntime, Nlat, Nlon]
        """
        base = np.full((Ntime, Nlat, Nlon), self.const, dtype=np.float32)

        if self.latGrad:
            for lat in reversed(range(Nlat)):
                base[:, lat, :] = base[:, lat, :] + lat / float(Nlat)
            base = base - 0.5

        return base


class SineBase:
    """ Class to generate sine base signal """
    def __init__(self, shift: float = 0., amp: float = 1, nOsc: int = 4, latGrad: bool = True):
        """
        Args:
            shift (float, optional): shift value for the sine signal. Defaults to 0.
            amp (float, optional): amplitude value for the sine signal. Defaults to 1.
            nOsc (int, optional): number of Oscillators. Defaults to 4.
            latGrad (bool, optional): whether to add gradient for the sine signal. Defaults to True.
        """
        self.shift = shift
        self.amp = amp
        self.nOsc = nOsc
        self.latGrad = latGrad

    def gen_base(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate base signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            base (np.ndarray): base signal [Ntime, Nlat, Nlon]
        """

        base = np.zeros((Ntime, Nlat, Nlon), dtype=np.float32)

        for t in range(Ntime):
            base[t, :, :] = self.amp * np.sin(t * self.nOsc / (Ntime - 1) * 2 * np.pi) + self.shift

        if self.latGrad:
            for lat in reversed(range(Nlat)):
                base[:, lat, :] = base[:, lat, :] + lat / float(Nlat)

            base = base - 0.5

        return base


class CosineBase:
    """ Class to generate cosine base signal """
    def __init__(self, shift: float = 0., amp: float = 1, nOsc: int = 4, latGrad: bool = True):
        """
        Args:
            shift (float, optional): shift value for the cosine signal. Defaults to 0.
            amp (float, optional): amplitude value for the cosine signal. Defaults to 1.
            nOsc (int, optional): number of Oscillators. Defaults to 4.
            latGrad (bool, optional): whether to add gradient for the cosine signal. Defaults to True.
        """
        self.shift = shift
        self.amp = amp
        self.nOsc = nOsc
        self.latGrad = latGrad

    def gen_base(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate base signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            base (np.ndarray): base signal [Ntime, Nlat, Nlon]
        """
        base = np.zeros((Ntime, Nlat, Nlon), dtype=np.float32)

        for t in range(Ntime):
            base[t, :, :] = self.amp * np.cos(t * self.nOsc / (Ntime - 1) * 2 * np.pi) + self.shift

        if self.latGrad:
            for lat in reversed(range(Nlat)):
                base[:, lat, :] = base[:, lat, :] + lat / float(Nlat)

            base = base - 0.5

        return base



class TrendBase:
    """ Class to generate trend base signal """
    def __init__(self, ttime: int = 2, tlat: int = 2, tlon: int = 2, latGrad: bool = True):
        """
        Args:
            ttime (int, optional): trend in the time direction. Defaults to 2.
            tlat (int, optional): trend in the latitude direction. Defaults to 2.
            tlon (int, optional): trend in the longitude direction. Defaults to 2.
            latGrad (bool, optional): whether to add gradient for the trend signal. Defaults to True.
        """
        self.ttime = ttime
        self.tlat = tlat
        self.tlon = tlon
        self.latGrad = latGrad

    def gen_base(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate base signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            base (np.ndarray): base signal [Ntime, Nlat, Nlon]
        """

        base = np.zeros((Ntime, Nlat, Nlon), dtype=np.float32)

        for k in range(Ntime):
            for i in range(Nlon):
                for j in range(Nlat):
                    base[k, j, i] = i / (Nlon - 1) * 2 * self.tlon - self.tlon \
                                    + j / (Nlat - 1) * 2 * self.tlat - self.tlat \
                                    + k / (Ntime - 1) * 2 * self.ttime - self.ttime

        if self.latGrad:
            for lat in reversed(range(Nlat)):
                base[:, lat, :] = base[:, lat, :] + lat / float(Nlat)

            base = base - 0.5

        return base



class NOAABase:
    """ Class to generate NOAA Remote Sensing base signal """
    def __init__(self, var: str = 'SMN', climatology: str = 'mean', nan_fill: float = 0.,
                 lat_min: int = 80, lat_max: int = 280,
                 lon_min: int = 80, lon_max: int = 280,
                 root: str = r'../data_base/NOAA.nc'):
        """
        Args:
            var (str, optional): variable name. Defaults to 'SMN'.
            climatology (str, optional): climatology to be used i.e., mean or median. Defaults to 'mean'.
            nan_fill (float, optional): value to fill in invalid pixels. Defaults to 0.
            lat_min (int, optional): minimum grid point in the latitude direction. Defaults to 80.
            lat_max (int, optional): maximum grid point in the latitude direction. Defaults to 280.
            lon_min (int, optional): minimum grid point in the longitude direction. Defaults to 80.
            lon_max (int, optional): maximum grid point in the longitude direction. Defaults to 280.
            root (str, optional): path to data directory. Defaults to '../data_base/NOAA.nc'.
        """

        self.var = var
        self.climatology = climatology
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max

        self.nan_fill = nan_fill
        self.root = root

    def gen_invalid_mask(self):
        """
        Generate mask of invalid pixels i.e., over water
        Returns:
            invalid_mask (np.ndarray): mask of invalid pixels [Nlat, Nlon]
        """

        noaa_dataset = xr.open_dataset(self.root)['SMN']
        noaa_dataset = noaa_dataset.sel(statistic='mean').isel(rlat=slice(self.lat_min, self.lat_max),
                                                               rlon=slice(self.lon_min, self.lon_max)).values
        noaa_dataset = np.flip(noaa_dataset, axis=1)

        invalid_mask = np.sum(np.isnan(noaa_dataset), axis=0) / 52

        return invalid_mask.astype(np.uint8)

    def gen_base(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate base signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            base (np.ndarray): base signal [Ntime, Nlat, Nlon]
        """

        base = np.zeros((Ntime, Nlat, Nlon), dtype=np.float32)

        noaa_dataset = xr.open_dataset(self.root)[self.var]
        noaa_dataset = noaa_dataset.sel(statistic=self.climatology).isel(rlat=slice(self.lat_min, self.lat_max),
                                                                         rlon=slice(self.lon_min, self.lon_max)).values

        noaa_dataset = np.flip(noaa_dataset, axis=1)
        noaa_dataset[np.isnan(noaa_dataset)] = self.nan_fill

        for t in range(Ntime):
            base[t] = noaa_dataset[t - 52 * (t // 52) if t // 52 != 0 else t]

        # scale NDVI/SMN by 10
        if self.var == 'SMN':
            base *= 10

        return base



class CERRABase:
    """ Class to generate CERRA Reanalysis base signal """
    def __init__(self, var: str = 'r2', climatology: str = 'mean', nan_fill: float = 0.,
                 lat_min: int = 400, lat_max: int = 600,
                 lon_min: int = 450, lon_max: int = 650,
                 root: str = r'../data_base/CERRA_climatology_pixels_train.nc'):

        """
        Args:
            var (str, optional): variable name. Defaults to 'r2'. variables from CERRA could be:
            (al, hcc, lcc, liqvsm, mcc, msl, r2, si10, skt, sot, sp, sr, t2m, tcc, tciwv, tp, vsw, wdir10)
            climatology (str, optional): climatology to be used i.e., mean or median. Defaults to 'mean'.
            nan_fill (float, optional): value to fill in invalid pixels. Defaults to 0.
            lat_min (int, optional): minimum grid point in the latitude direction. Defaults to 80.
            lat_max (int, optional): maximum grid point in the latitude direction. Defaults to 280.
            lon_min (int, optional): minimum grid point in the longitude direction. Defaults to 80.
            lon_max (int, optional): maximum grid point in the longitude direction. Defaults to 280.
            root (str, optional): path to data directory. Defaults to '../data_base/CERRA_climatology_pixels_train.nc'.
        """

        self.var = var
        self.climatology = climatology
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max

        self.nan_fill = nan_fill
        self.root = root

    def gen_invalid_mask(self):
        """
        Generate mask of invalid pixels i.e., over water

        Returns:
            invalid_mask (np.ndarray): mask of invalid pixels [Nlat, Nlon]
        """
        cerra_dataset = xr.open_dataset(self.root)['vsw']
        cerra_dataset = cerra_dataset.sel(statistic='mean', climatology='mean').isel(y=slice(self.lat_min, self.lat_max),
                                                                                     x=slice(self.lon_min, self.lon_max)).values
        cerra_dataset = np.flip(cerra_dataset, axis=1)

        invalid_mask = np.sum(np.isnan(cerra_dataset), axis=0) / 52

        return invalid_mask.astype(np.uint8)

    def gen_base(self, Ntime: int, Nlat: int, Nlon: int):
        """
        Generate base signal
        Args:
            Ntime (int): number of time steps
            Nlat (int): number of grid points in the latitude direction
            Nlon (int): number grid points in the longitude direction
        Returns:
            base (np.ndarray): base signal [Ntime, Nlat, Nlon]
        """

        base = np.zeros((Ntime, Nlat, Nlon), dtype=np.float32)

        cerra_dataset = xr.open_dataset(self.root)[self.var]

        cerra_dataset = cerra_dataset.sel(statistic='mean', climatology=self.climatology)\
            .isel(y=slice(self.lat_min, self.lat_max),
                  x=slice(self.lon_min, self.lon_max)).values

        cerra_dataset = np.flip(cerra_dataset, axis=1)
        cerra_dataset[np.isnan(cerra_dataset)] = self.nan_fill

        for t in range(Ntime):
            base[t] = cerra_dataset[t - 52 * (t // 52) if t // 52 != 0 else t]

        return base


if __name__ == '__main__':

    print('test bases...')

    Ntime = 52*4
    Nlat = 200
    Nlon = 200

    print('CERRABase')
    B = CERRABase().gen_base(Ntime, Nlat, Nlon)
    M = CERRABase().gen_invalid_mask()
    plt.imshow(M)
    plt.show()
    plt.plot(np.arange(Ntime), B[:, Nlat // 2, Nlon // 2])
    plt.show()

    print('NOAABase')
    B = NOAABase().gen_base(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), B[:, Nlat // 2, Nlon // 2])
    plt.show()
    plt.imshow(B[Ntime//2, :, :])
    plt.show()

    print('ConstantBase')
    B = ConstantBase(0, False).gen_base(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), B[:, Nlat//2, Nlon//2])
    plt.show()
    plt.imshow(B[Ntime//2, :, :])
    plt.show()

    print('ConstantBase with latitude gradient')
    B = ConstantBase(0, True).gen_base(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), B[:, Nlat//4, Nlon//2])
    plt.show()
    plt.imshow(B[Ntime//2, :, :])
    plt.show()

    print('SineBase')
    B = SineBase(0, 1, 4, False).gen_base(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), B[:, Nlat // 2, Nlon // 2])
    plt.show()
    plt.imshow(B[Ntime//2, :, :])
    plt.show()

    print('SineBase with latitude gradient')
    B = SineBase(0, 1, 4, True).gen_base(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), B[:, Nlat // 4, Nlon // 2])
    plt.show()
    plt.imshow(B[Ntime//2, :, :])
    plt.show()

    print('CosineBase')
    B = CosineBase(0, 1, 4, False).gen_base(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), B[:, Nlat // 2, Nlon // 2])
    plt.show()
    plt.imshow(B[Ntime//2, :, :])
    plt.show()

    print('CosineBase with latitude gradient')
    B = CosineBase(0, 1, 4, True).gen_base(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), B[:, Nlat // 4, Nlon // 2])
    plt.show()
    plt.imshow(B[Ntime//2, :, :])
    plt.show()

    print('TrendBase')
    B = TrendBase(2, 2, 2, False).gen_base(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), B[:, Nlat // 2, Nlon // 2])
    plt.show()
    plt.imshow(B[Ntime//2, :, :])
    plt.show()

    print('TrendBase with latitude gradient')
    B = TrendBase(2, 2, 2, True).gen_base(Ntime, Nlat, Nlon)
    plt.plot(np.arange(Ntime), B[:, Nlat // 4, Nlon // 2])
    plt.show()
    plt.imshow(B[Ntime//2, :, :])
    plt.show()

