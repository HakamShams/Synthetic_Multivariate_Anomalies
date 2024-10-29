from src.base import ConstantBase, SineBase, CosineBase, TrendBase, NOAABase, CERRABase
from src.noise import WhiteNoise, CauchyNoise, LaplaceNoise, RedNoise
from src.event import CubeEvent, LocalEvent, EmptyEvent, GaussianEvent, OnsetEvent, RandomWalkEvent, ExtremeEvent, ExtremeClass
from src.weight import NormWeight, DisturbedNormWeight, CauchyWeight, DisturbedCauchyWeight, LaplacWeight, \
    DisturbedLaplaceWeight
from src.coupling import LinearCoupling, QuadraticCoupling, ExtremeCoupling

# ------------------------------------------------------------------
# define the parameters
# directory to save experiment
exp_dir = r'./log/exp_332'
# random seed
seed = 0

# Cube dimensions
# time extension -z axis
n_time = 52 * 46
# latitude extension -y axis
n_lat = 200
# longitude extension -x axis
n_lon = 200

# at which time step the training set ends. This is used to compute statistics of training data
training_end_time = 52*34

# number of independent variables
ind_var = 6
# number of dependent variables
dep_var = 0
# base type for the independent variables

base_ind_var = [CERRABase(var='al', climatology='mean', nan_fill=0.,
                          lat_min=400, lat_max=600,
                          lon_min=450, lon_max=650,
                          root=r'./data_base/CERRA_climatology_pixels_train.nc'),
                CERRABase(var='t2m', climatology='mean', nan_fill=260.,
                                          lat_min=400, lat_max=600,
                                          lon_min=450, lon_max=650,
                                          root=r'./data_base/CERRA_climatology_pixels_train.nc'),
                CERRABase(var='tcc', climatology='mean', nan_fill=50.,
                                          lat_min=400, lat_max=600,
                                          lon_min=450, lon_max=650,
                                          root=r'./data_base/CERRA_climatology_pixels_train.nc'),
                CERRABase(var='tp', climatology='mean', nan_fill=0.,
                                          lat_min=400, lat_max=600,
                                          lon_min=450, lon_max=650,
                                          root=r'./data_base/CERRA_climatology_pixels_train.nc'),
                CERRABase(var='r2', climatology='mean', nan_fill=50.,
                          lat_min=400, lat_max=600,
                          lon_min=450, lon_max=650,
                          root=r'./data_base/CERRA_climatology_pixels_train.nc'),
                CERRABase(var='vsw', climatology='mean', nan_fill=0.,
                          lat_min=400, lat_max=600,
                          lon_min=450, lon_max=650,
                          root=r'./data_base/CERRA_climatology_pixels_train.nc'),
                ]


# noise type for the independent variables
noise_ind_var = [WhiteNoise(meu=0., sigma=.01),
                 RedNoise(meu=0, sigma=.9, btime=1, blat=1, blon=1),
                 LaplaceNoise(meu=0., sigma=.7),
                 WhiteNoise(meu=0., sigma=.04),
                 WhiteNoise(meu=0., sigma=.7),
                 WhiteNoise(meu=0., sigma=.017)
                 ]

# event type for the independent variables
event_ind_var = [[CubeEvent(n=320, sx=35, sy=35, sz=25),
                  RandomWalkEvent(n=3000, s=125),
                  LocalEvent(n=4000, sz=17),
                  GaussianEvent(n=300, sx=35, sy=35, sz=25)],
                 [OnsetEvent(n=18, sx=17, sy=17, os=0.98),
                  RandomWalkEvent(n=1800, s=125),
                  LocalEvent(n=160, sz=17),
                  GaussianEvent(n=350, sx=35, sy=35, sz=25)],
                 [CubeEvent(n=300, sx=35, sy=35, sz=25),
                  RandomWalkEvent(n=2000, s=125),
                  LocalEvent(n=2800, sz=17),
                  GaussianEvent(n=290, sx=35, sy=35, sz=25)],
                 [CubeEvent(n=320, sx=35, sy=35, sz=25),
                  RandomWalkEvent(n=3000, s=125),
                  LocalEvent(n=4000, sz=17),
                  GaussianEvent(n=300, sx=35, sy=35, sz=25)],
                 [OnsetEvent(n=18, sx=17, sy=17, os=0.98),
                  RandomWalkEvent(n=1800, s=125),
                  LocalEvent(n=160, sz=17),
                  GaussianEvent(n=350, sx=35, sy=35, sz=25)],
                 [CubeEvent(n=300, sx=35, sy=35, sz=25),
                  RandomWalkEvent(n=2000, s=125),
                  LocalEvent(n=2800, sz=17),
                  GaussianEvent(n=290, sx=35, sy=35, sz=25)]
                 ]

# disturbance parameters
kb = [0.3, 0.01, 0.03, 0.07, 0.06, 0.1]
kn = [0.2, 0.01, 0.08, 0.20, 0.06, 0.1]
ks = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# whether to smooth the independent variables with conv or not
is_conv = False
# kernel size for convolution
kernel_size = 3

# whether to not generate anomalous on invalid pixels i.e., on sea
is_mask_invalid = True

# noise type for the dependent variables
noise_dep_var = WhiteNoise(meu=0., sigma=.065)

# weight type for the dependent variables
weight_dep_var = [NormWeight(ind_var=ind_var),
                  LaplacWeight(ind_var=ind_var),
                  LaplacWeight(ind_var=ind_var)
                  ]

# coupling type for the dependent variables
coupling_dep_var = [QuadraticCoupling(), LinearCoupling(), LinearCoupling()]

# parameters to define extreme events
# interval in time and threshold for anomalous events
extreme_events = [CubeEvent(n=200, sx=35, sy=35, sz=25),
                  RandomWalkEvent(n=1100, s=125),
                  LocalEvent(n=2600, sz=17),
                  GaussianEvent(n=340, sx=35, sy=35, sz=25),
                  OnsetEvent(n=25, sx=17, sy=17, os=0.98),
                  ]


# define coupling between extremes and variables
extreme_coupling = ExtremeCoupling(var=ind_var + dep_var, var_drop=[1], interval=14)
# position of the extreme in the coupling matrix
pos_extreme_interval = 10  # i.e., if the interval is 14 the extreme will happen at time step 10
# add two more classes for extremes. These classes will be added before and after the extremes
extreme_classes = ExtremeClass(n_added_classes=2)
# signs for coupling between variables and extreme events
sign_extreme_coupling = [1, 1, -1, -1, -1, -1]
