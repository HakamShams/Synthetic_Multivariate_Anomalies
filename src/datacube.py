# --------------------------------------------------------
"""
Synthetic Multivariate Anomalies
This script includes the main class and functions to generate and save the datacube

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# --------------------------------------------------------

import numpy as np
import xarray as xr
import os
import datetime
from scipy import ndimage
import random
import json

np.set_printoptions(suppress=True)
xr.set_options(display_max_rows=40)
xr.set_options(display_width=1000)

# ------------------------------------------------------------------

class DataCube:
    """ Class to generate datacube including multivariate anomalies and extreme events """
    def __init__(self, config):
        """
        Args:
            config (module): configuration file
        """

        # get config file
        self.exp_dir = config.exp_dir

        self.Ntime = config.n_time
        self.Nlat = config.n_lat
        self.Nlon = config.n_lon

        self.ind_var = config.ind_var
        self.dep_var = config.dep_var

        self.base_ind_var = config.base_ind_var if type(config.base_ind_var) is list \
            else [config.base_ind_var] * config.ind_var
        self.noise_ind_var = config.noise_ind_var if type(config.noise_ind_var) is list \
            else [config.noise_ind_var] * config.ind_var
        self.event_ind_var = config.event_ind_var if type(config.event_ind_var) is list \
            else [config.event_ind_var] * config.ind_var

        self.kb = config.kb if type(config.kb) is list \
            else [config.kb] * (config.ind_var + config.dep_var)
        self.kn = config.kn if type(config.kn) is list \
            else [config.kn] * (config.ind_var + config.dep_var)
        self.ks = config.ks if type(config.ks) is list \
            else [config.ks] * (config.ind_var + config.dep_var)

        self.is_conv = config.is_conv
        self.kernel_size = config.kernel_size

        if config.dep_var > 0:

            self.noise_dep_var = config.noise_dep_var if type(config.noise_dep_var) is list \
                else [config.noise_dep_var] * config.dep_var
            self.weight_dep_var = config.weight_dep_var if type(config.weight_dep_var) is list \
                else [config.weight_dep_var] * config.dep_var
            self.coupling_dep_var = config.coupling_dep_var if type(config.coupling_dep_var) is list \
                else [config.coupling_dep_var] * config.dep_var

        self.extreme_events = config.extreme_events
        self.pos_extreme_interval = config.pos_extreme_interval
        self.extreme_classes = config.extreme_classes
        self.extreme_coupling = config.extreme_coupling
        self.sign_extreme_coupling = config.sign_extreme_coupling
        self.training_end_time = config.training_end_time

        self.is_mask_invalid = config.is_mask_invalid

        # fix random seed
        self.seed = config.seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def gen_datastream(self, base, noise, event,
                       Ntime: int = 208, Nlat: int = 100, Nlon: int = 100,
                       kb: float = 1., kn: float = 1., ks: float = 1., is_conv: bool = True, kernel_size: int = 3):
        """
        Function that generates a single datacube representing an independent variable
        Args:
            base (class): class to generate the base signal
            noise (class): class to generate the noise signal
            event (class or list of classes): the class to generate events
            Ntime (int, optional): number of time steps. Defaults to 208
            Nlat (int, optional): number of grid points in the latitude direction. Defaults to 100
            Nlon (int, optional): number grid points in the longitude direction. Defaults to 100
            kb (float, optional): strength of the base signal modulation by events. Defaults to 1.
            kn (float, optional): strength of the noise signal modulation by events. Defaults to 1.
            ks (float, optional): strength of the mean shift event. Defaults to 1.
            is_conv (bool, optional): whether to use convolution on the generated signal. Defaults to True
            kernel_size (int, optional): the kernel size for convolution. Defaults to 3
        Returns:
            B (ndarray): the generated signal [Ntime, Nlat, Nlon]
            E (np.ndarray): the generated events [Ntime, Nlat, Nlon]
        """

        B = base.gen_base(Ntime, Nlat, Nlon)
        #N = noise.gen_noise(Ntime, Nlat, Nlon)

        if type(event) is list:
            E = np.zeros((Ntime, Nlat, Nlon), dtype=np.uint8)
            for event_i in event:
                E_i = event_i.gen_event(Ntime, Nlat, Nlon)
                E[E_i == 1] = 1
        else:
            E = event.gen_event(Ntime, Nlat, Nlon)

        if is_conv:
            B = ndimage.generic_filter(B, np.mean, size=kernel_size, mode='nearest')

        if self.is_mask_invalid:
            M = base.gen_invalid_mask()
            E[:, M == 1] = 0

        #Y = B * 2 ** (kb * E) + N * 2 ** (kn * E) + np.std(N) * ks * E

        return B, E

    def gen_extreme_events(self, event, Ntime: int = 208, Nlat: int = 100, Nlon: int = 100):
        """
        Function that generates extreme events
        Args:
            event (class or list of classes): the class to generate events
            Ntime (int, optional): number of time steps. Defaults to 208
            Nlat (int, optional): number of grid points in the latitude direction. Defaults to 100
            Nlon (int, optional): number grid points in the longitude direction. Defaults to 100
        Returns:
            E (np.ndarray): the generated events [Ntime, Nlat, Nlon]
        """
        if type(event) is list:
            E = np.zeros((Ntime, Nlat, Nlon), dtype=np.uint8)
            for event_i in event:
                E_i = event_i.gen_event(Ntime, Nlat, Nlon)
                E[E_i == 1] = 1
        else:
            E = event.gen_event(Ntime, Nlat, Nlon)

        return E

    def propagate_extreme_events(self, E: np.array, E_extreme: np.array, ECM: np.array, interval: int = 10,
                                 pos_extreme_interval: int = 6, Ntime: int = 208):
        """
        Function that generates anomalies based on extreme events and coupling matrix
        Args:
            E (np.ndarray): placeholder for events [Ntime, Nlat, Nlon]
            E_extreme (np.ndarray): extreme events [Ntime, Nlat, Nlon]
            ECM (np.ndarray): coupling matrix [Variables, Interval]
            interval (int, optional): the time interval. Defaults to 10.
            pos_extreme_interval (int, optional): the position of the extreme within the time interval. Defaults to 6.
            Ntime (int, optional): number of time steps. Defaults to 208
        Returns:
            E (np.ndarray): the generated anomalous events [Ntime, Nlat, Nlon]
        """
        #N = noise.gen_noise(Ntime, Nlat, Nlon)

        for t in range(Ntime):
            idx = (E_extreme[t] == 1)
            n_extreme = np.sum(idx)

            if n_extreme > 0:
                if t < pos_extreme_interval - 1:
                    E[0: t + 1, idx] += np.repeat(ECM[pos_extreme_interval - t - 1:pos_extreme_interval, np.newaxis], n_extreme, axis=1)
                else:
                    E[t + 1 - pos_extreme_interval: t + 1, idx] += np.repeat(ECM[:pos_extreme_interval, np.newaxis], n_extreme, axis=1)

                if t + 1 + interval - pos_extreme_interval > Ntime:
                    E[t+1:, idx] += np.repeat(ECM[pos_extreme_interval: pos_extreme_interval + Ntime - t - 1, np.newaxis], n_extreme, axis=1)
                else:
                    E[t + 1: t + interval - pos_extreme_interval + 1, idx] += np.repeat(ECM[pos_extreme_interval:, np.newaxis], n_extreme, axis=1)

        E[E > 1] = 1

        #Y = Y * 2 ** (kb * E) + N * 2 ** (kn * E) + np.std(N) * ks * E

        return E

    def gen_datacube(self):
        """
        Generate the datacube
        Returns:
            datacube (np.ndarray): the generated datacube [Variables, Ntime, Nlat, Nlon]
            datacube_events (np.ndarray): the generated anomalous events [Variables, Ntime, Nlat, Nlon]
            datacube_events_extreme (np.ndarray): the generated anomalous events that are coupled with the extremes [Variables, Ntime, Nlat, Nlon]
            datacube_extreme_events (np.ndarray): the generated extreme events [Ntime, Nlat, Nlon]
            extreme_coupling_matrix (np.ndarray): the generated coupling matrix [Variables, Interval]
        """

        # prepare placeholders for the datacube and anomalous events
        datacube = np.zeros((self.ind_var + self.dep_var, self.Ntime, self.Nlat, self.Nlon), dtype=float)
        datacube_events = np.zeros((self.ind_var + self.dep_var, self.Ntime, self.Nlat, self.Nlon), dtype=np.uint8)

        # generate extreme events
        print('generating extreme events')
        datacube_extreme_events = self.gen_extreme_events(self.extreme_events, self.Ntime, self.Nlat, self.Nlon)
        print('generating extreme event classes')
        datacube_extreme_events = self.extreme_classes.gen_event(datacube_extreme_events)

        if self.is_mask_invalid:
            M = self.base_ind_var[0].gen_invalid_mask()
            datacube_extreme_events[:, M == 1] = 0

        # generate extreme coupling matrix
        extreme_coupling_matrix = self.extreme_coupling.couple_event()

        # generate the independent variables of the datacube
        for v in range(self.ind_var):
            print('variable: ', v)
            datacube[v, ...], datacube_events[v, ...] = self.gen_datastream(self.base_ind_var[v],
                                                                            self.noise_ind_var[v],
                                                                            self.event_ind_var[v],
                                                                            self.Ntime,
                                                                            self.Nlat,
                                                                            self.Nlon,
                                                                            self.kb[v],
                                                                            self.kn[v],
                                                                            self.ks[v],
                                                                            self.is_conv,
                                                                            self.kernel_size)

        # generate the dependent variables of the datacube
        if self.dep_var > 0:
            for v in range(self.dep_var):
                print('dep variable: ', v)
                if type(self.weight_dep_var[v]).__name__ in ['DisturbedLaplaceWeight',
                                                             'DisturbedCauchyWeight',
                                                             'DisturbedNormWeight']:

                    weight = self.weight_dep_var[v].gen_weight(datacube_events[:self.ind_var])
                else:
                    weight = self.weight_dep_var[v].gen_weight()

                #noise = self.noise_dep_var[v].gen_noise(self.Ntime, self.Nlat, self.Nlon)
                noise = np.zeros((self.Ntime, self.Nlat, self.Nlon))
                datacube[v+self.ind_var], datacube_events[v+self.ind_var] = \
                    self.coupling_dep_var[v].combine_variables(datacube[:self.ind_var],
                                                               datacube_events[:self.ind_var],
                                                               weight,
                                                               noise)

        if self.is_mask_invalid:
            datacube_events[:, :, M == 1] = 0

        # generate anomaly based on the extreme coupling matrix
        datacube_events_extreme = np.zeros_like(datacube_events)

        for v in range(self.ind_var + self.dep_var):
            print('propagate extreme to variable: ', v)

            if v >= self.ind_var:
                noise_var = self.noise_dep_var[v-self.ind_var]
            else:
                noise_var = self.noise_ind_var[v]

            N = noise_var.gen_noise(self.Ntime, self.Nlat, self.Nlon)

            # skip uncorrelated variable with extreme
            if v + 1 not in self.extreme_coupling.var_drop:

                datacube_events_extreme[v, ...] = self.propagate_extreme_events(datacube_events_extreme[v],
                                                                                datacube_extreme_events,
                                                                                extreme_coupling_matrix[v],
                                                                                self.extreme_coupling.interval,
                                                                                self.pos_extreme_interval,
                                                                                self.Ntime
                                                                                )

            E_union = datacube_events_extreme[v, ...] + datacube_events[v, ...]
            E_union[E_union > 1] = 1

            theta = datacube[v, ...] * (-1 + 2 ** (self.kb[v] * E_union)) \
                    + N * 2 ** (self.kn[v] * E_union) \
                    + np.std(N) * self.ks[v] * E_union

            sign_theta = np.ones_like(theta)
            sign_theta[theta < 0] = -1

            lambda_theta = sign_theta / self.sign_extreme_coupling[v]
            lambda_theta[datacube_events_extreme[v, ...] != 1] = 1
            datacube[v, ...] = datacube[v, ...] + lambda_theta * theta

        return datacube, datacube_events, datacube_events_extreme, datacube_extreme_events, extreme_coupling_matrix

    def save_datacube(self, datacube: np.array, datacube_events: np.array, datacube_events_extreme: np.array,
                      datacube_extremes: np.array, extreme_coupling: np.array):
        """
        Function to save the generated datacube. Results will be saved into a NetCDF file.
        Args:
            datacube (np.ndarray): the generated datacube [Variables, Ntime, Nlat, Nlon]
            datacube_events (np.ndarray): the generated anomalous events [Variables, Ntime, Nlat, Nlon]
            datacube_events_extreme (np.ndarray): the generated anomalous events that are coupled with the extremes [Variables, Ntime, Nlat, Nlon]
            datacube_extremes (np.ndarray): the generated extreme events [Ntime, Nlat, Nlon]
            extreme_coupling (np.ndarray): the generated coupling matrix [Variables, Interval]
        """

        # prepare variables
        x = np.linspace(0, self.Nlon/10, self.Nlon)
        y = np.linspace(0, self.Nlat/10, self.Nlat)
        longitude, latitude = np.meshgrid(x, y)

        dataset = xr.Dataset()
        dataset = dataset.assign_coords(
                                        time=np.array([i + 1 for i in range(self.Ntime)], dtype=np.int32),
                                        latitude=(["y", "x"], np.ascontiguousarray(latitude).astype(np.float32)),
                                        longitude=(["y", "x"], np.ascontiguousarray(longitude).astype(np.float32))
                                        )

        var_names = ['var_0' + str(i+1) if i < 9 else 'var' + str(i+1) for i in range(self.ind_var + self.dep_var)]

        for v in range(len(var_names)):
            dataset[var_names[v]] = (["time", "y", "x"], datacube[v].astype(np.float32))

        dataset['anomaly'] = (["var", "time", "y", "x"], datacube_events.astype(np.uint8))
        dataset['anomaly_extreme'] = (["var", "time", "y", "x"], datacube_events_extreme.astype(np.uint8))
        dataset['extreme'] = (["time", "y", "x"], datacube_extremes.astype(np.uint8))
        dataset['anomaly'] = dataset['anomaly'].assign_coords(var=var_names)

        # prepare attributes
        dict_base_ind_var, dict_noise_ind_var, dict_event_ind_var = {}, {}, {}
        dict_kb_ind_var, dict_kn_ind_var, dict_ks_ind_var = {}, {}, {}
        dict_sign_ind_var = {}

        for v in range(self.ind_var):
            dict_base_ind_var[var_names[v]] = {type(self.base_ind_var[v]).__name__: self.base_ind_var[v].__dict__}
            dict_noise_ind_var[var_names[v]] = {type(self.noise_ind_var[v]).__name__: self.noise_ind_var[v].__dict__}

            if type(self.event_ind_var[v]) is list:
                dict_var_t = {}
                for k in range(len(self.event_ind_var[v])):
                    dict_var_t[type(self.event_ind_var[v][k]).__name__] = self.event_ind_var[v][k].__dict__
                dict_event_ind_var[var_names[v]] = dict_var_t
            else:
                dict_event_ind_var[var_names[v]] = {type(self.event_ind_var[v]).__name__: self.event_ind_var[v].__dict__}

            dict_kb_ind_var[var_names[v]] = {'kb': self.kb[v]}
            dict_kn_ind_var[var_names[v]] = {'kn': self.kn[v]}
            dict_ks_ind_var[var_names[v]] = {'ks': self.ks[v]}

            dict_sign_ind_var[var_names[v]] = {'anomaly sign': self.sign_extreme_coupling[v]}

        if self.dep_var > 0:
            dict_noise_dep_var, dict_weight_dep_var, dict_coupling_dep_var = {}, {}, {}
            dict_kb_dep_var, dict_kn_dep_var, dict_ks_dep_var = {}, {}, {}
            dict_sign_dep_var = {}

            for v in range(self.dep_var):
                dict_noise_dep_var[var_names[v+self.ind_var]] = {type(self.noise_dep_var[v]).__name__: self.noise_dep_var[v].__dict__}
                dict_weight_dep_var[var_names[v+self.ind_var]] = {type(self.weight_dep_var[v]).__name__: self.weight_dep_var[v].__dict__}
                dict_coupling_dep_var[var_names[v+self.ind_var]] = {type(self.coupling_dep_var[v]).__name__: self.coupling_dep_var[v].__dict__}

                dict_kb_dep_var[var_names[v+self.ind_var]] = {'kb': self.kb[v+self.ind_var]}
                dict_kn_dep_var[var_names[v+self.ind_var]] = {'kn': self.kn[v+self.ind_var]}
                dict_ks_dep_var[var_names[v+self.ind_var]] = {'ks': self.ks[v+self.ind_var]}

                dict_sign_dep_var[var_names[v+self.ind_var]] = {'anomaly sign': self.sign_extreme_coupling[v+self.ind_var]}

        if type(self.extreme_events) is list:
            dict_extreme_events = {}
            for k in range(len(self.extreme_events)):
                dict_extreme_events[type(self.extreme_events[k]).__name__] = self.extreme_events[k].__dict__
        else:
            dict_extreme_events = {type(self.extreme_events).__name__: self.extreme_events.__dict__}

        dataset = dataset.assign_attrs(
            creation_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data="artificial datacube with anomalous and extreme events",
            random_seed=self.seed,
            time_extent=self.Ntime,
            longitudinal_extent=self.Nlon,
            latitudinal_extent=self.Nlat,
            n_independent_variables=self.ind_var,
            independent_variables=var_names[:self.ind_var],
            base_independent_variables=str(dict_base_ind_var),
            noise_independent_variables=str(dict_noise_ind_var),
            event_independent_variables=str(dict_event_ind_var),
            kb_independent_variables=str(dict_kb_ind_var),
            kn_independent_variables=str(dict_kn_ind_var),
            ks_independent_variables=str(dict_ks_ind_var),
            coupling_sign_independent_variables=str(dict_sign_ind_var),
            is_conv_independent_variables="True" if self.is_conv else "False",
            kernel_size_independent_variables=self.kernel_size,
        )

        if self.dep_var > 0:
            dataset = dataset.assign_attrs(
                n_dependent_variables=self.dep_var,
                dependent_variables=var_names[self.ind_var:],
                noise_dependent_variables=str(dict_noise_dep_var),
                weight_dependent_variables=str(dict_weight_dep_var),
                coupling_dependent_variables=str(dict_coupling_dep_var),
                kb_dependent_variables=str(dict_kb_dep_var),
                kn_dependent_variables=str(dict_kn_dep_var),
                ks_dependent_variables=str(dict_ks_dep_var),
                coupling_sign_dependent=str(dict_sign_dep_var),
            )

        dataset = dataset.assign_attrs(
            is_mask_invalid="True" if self.is_mask_invalid else "False",
            extreme_events=str(dict_extreme_events),
            extreme_coupling=str({type(self.extreme_coupling).__name__: self.extreme_coupling.__dict__}),
            extreme_timestep=self.pos_extreme_interval,
            extreme_added_classes=self.extreme_classes.n_added_classes,
        )

        exp_name = os.path.basename(os.path.normpath(self.exp_dir))
        dir_out = os.path.join(self.exp_dir, 'datacube_{}.nc'.format(exp_name))
        dataset.to_netcdf(dir_out)
        np.save(os.path.join(self.exp_dir, 'coupling_matrix_{}.npy'.format(exp_name)), extreme_coupling)

    def comp_weekly_climatology(self, datacube: np.array, time_step: int = -1):
        """
        Function to compute weekly climatology of the training set in the datacube. Results will be saved into a NetCDF file.
        Args:
            datacube (np.ndarray): the input datacube [Variables, Ntime, Nlat, Nlon]
            time_step (int, optional): the time step at which the training set ends. Defaults to -1.
        """
        x = np.linspace(0, 1, self.Nlon)
        y = np.linspace(0, 1, self.Nlat)
        longitude, latitude = np.meshgrid(x, y)

        dataset = xr.Dataset()
        dataset = dataset.assign_coords(climatology=np.array(["min", "max", "mean", "median", "std"], dtype=object),
                                        time=np.array([i + 1 for i in range(52)], dtype=np.int32),
                                        latitude=(["y", "x"], np.ascontiguousarray(latitude).astype(np.float32)),
                                        longitude=(["y", "x"], np.ascontiguousarray(longitude).astype(np.float32)),
                                        )

        var_names = ['var_0' + str(i+1) if i < 9 else 'var' + str(i+1) for i in range(self.ind_var + self.dep_var)]

        datacube = datacube[:, :time_step, ...]

        for v, variable in enumerate(var_names):

            datacube_v = np.zeros((5, 52, self.Nlat, self.Nlon))

            for week in range(52):

                datacube_v_w = datacube[v, week::52, ...]
                datacube_v[0, week, ...] = np.min(datacube_v_w, axis=0)
                datacube_v[1, week, ...] = np.max(datacube_v_w, axis=0)
                datacube_v[2, week, ...] = np.mean(datacube_v_w, axis=0)
                datacube_v[3, week, ...] = np.median(datacube_v_w, axis=0)
                datacube_v[4, week, ...] = np.std(datacube_v_w, axis=0)

            dataset[variable] = (["climatology", "time", "y", "x"], datacube_v.astype(np.float32))

            del datacube_v_w, datacube_v

        dataset = dataset.assign_attrs(
            creation_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data="climatology for artificial datacube",
            data_source="climatology_{}.nc".format(os.path.basename(os.path.normpath(self.exp_dir))),
            random_seed=self.seed,
            time_extent=self.Ntime,
            longitudinal_extent=self.Nlon,
            latitudinal_extent=self.Nlat,
            n_independent_variables=self.ind_var,
            independent_variables=var_names[:self.ind_var],
        )

        if self.dep_var > 0:
            dataset = dataset.assign_attrs(
                n_dependent_variables=self.dep_var,
                dependent_variables=var_names[self.ind_var:],
            )

        exp_name = os.path.basename(os.path.normpath(self.exp_dir))
        dir_out = os.path.join(self.exp_dir, 'climatology_{}.nc'.format(exp_name))
        dataset.to_netcdf(dir_out)


    def comp_statistic(self, datacube: np.array, time_step: int = -1):
        """
        Function to compute statistics of the training set in the datacube. Results will be saved into a json file.
        Args:
            datacube (np.ndarray): the input datacube [Variables, Ntime, Nlat, Nlon]
            time_step (int, optional): the time step at which the training set ends. Defaults to -1.
        """

        var_names = ['var_0' + str(i+1) if i < 9 else 'var' + str(i+1) for i in range(self.ind_var + self.dep_var)]

        datacube = datacube[:, :time_step, ...]

        dict_v = {'min': {v: 0.0 for v in var_names},
                  'max': {v: 0.0 for v in var_names},
                  'mean': {v: 0.0 for v in var_names},
                  'median': {v: 0.0 for v in var_names},
                  'std': {v: 0.0 for v in var_names},
                  }

        for v, variable in enumerate(var_names):

            dict_v['min'][variable] = str(np.min(datacube[v]))
            dict_v['max'][variable] = str(np.max(datacube[v]))
            dict_v['mean'][variable] = str(np.mean(datacube[v]))
            dict_v['median'][variable] = str(np.median(datacube[v]))
            dict_v['std'][variable] = str(np.std(datacube[v]))

        exp_name = os.path.basename(os.path.normpath(self.exp_dir))

        with open(os.path.join(self.exp_dir, 'statistic_{}.json'.format(exp_name)), 'w') as j_file:
            json.dump(dict_v, j_file)
