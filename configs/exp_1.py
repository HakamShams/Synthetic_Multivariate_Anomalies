from src.base import ConstantBase, SineBase, CosineBase, TrendBase, NOAABase, CERRABase
from src.noise import WhiteNoise, CauchyNoise, LaplaceNoise, RedNoise
from src.event import CubeEvent, LocalEvent, EmptyEvent, GaussianEvent, OnsetEvent, RandomWalkEvent, ExtremeEvent, ExtremeClass
from src.weight import NormWeight, DisturbedNormWeight, CauchyWeight, DisturbedCauchyWeight, LaplacWeight, \
    DisturbedLaplaceWeight
from src.coupling import LinearCoupling, QuadraticCoupling, ExtremeCoupling

# ------------------------------------------------------------------
# define the parameters
# directory to save experiment
exp_dir = r'./log/exp_1'
# random seed
seed = 44

# Cube dimensions
# time extension -z axis
n_time = 52 * 46
# latitude extension -y axis
n_lat = 200
# longitude extension -x axis
n_lon = 200

# total number of time steps
training_end_time = 52 * 46

# number of independent variables
ind_var = 3
# number of dependent variables
dep_var = 3
# base type for the independent variables
base_ind_var = [SineBase(shift=0., amp=3., nOsc=46, latGrad=True),
                CosineBase(shift=0., amp=3., nOsc=46, latGrad=True),
                ConstantBase(const=.0, latGrad=True)
                ]

# TrendBase(ttime=2, tlat=0, tlon=1)]

# noise type for the independent variables
noise_ind_var = [
    RedNoise(sigma=.2, btime=1.0, blat=1.0, blon=1.0),
    #WhiteNoise(meu=0., sigma=.05),
    LaplaceNoise(meu=0., sigma=.08, lambda_laplace=1.),
    WhiteNoise(meu=0., sigma=.07),
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
                  GaussianEvent(n=290, sx=35, sy=35, sz=25)]
                 ]

# disturbance parameters
kb = [.35, .35, .9, .35, .35, .35]
kn = [.35, .35, .9, .35, .35, .35]
ks = [.35, .35, .9, .35, .35, .35]

# whether to smooth the independent variables with conv or not
is_conv = False
# kernel size for convolution
kernel_size = 3

# whether to not generate anomalous on invalid pixels i.e., on sea
# only works for CERRA and NOAA base types
is_mask_invalid = False

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
extreme_events = [CubeEvent(n=200, sx=35, sy=35, sz=25),
                  RandomWalkEvent(n=1100, s=125),
                  LocalEvent(n=2600, sz=17),
                  GaussianEvent(n=340, sx=35, sy=35, sz=25),
                  OnsetEvent(n=25, sx=17, sy=17, os=0.98),
                  ]

# define coupling between extremes and variables
extreme_coupling = ExtremeCoupling(var=ind_var + dep_var, var_drop=[1, 5], interval=14)

# position of the extreme in the coupling matrix
pos_extreme_interval = 10  # i.e., if the interval is 14 the extreme will happen at time step 10
# add two more classes for extremes. These classes will be added before and after the extremes
extreme_classes = ExtremeClass(n_added_classes=2)
# signs for coupling between variables and extreme events
sign_extreme_coupling = [1, 1, -1, -1, -1, -1]