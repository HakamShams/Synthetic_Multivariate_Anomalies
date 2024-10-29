# --------------------------------------------------------
"""
Synthetic Multivariate Anomalies
Generate synthetic datacube with multivariate anomalies and extreme events

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# --------------------------------------------------------

import argparse
import importlib
import shutil
import numpy as np
import os
from src import utils
from src.datacube import DataCube

# ------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser('DataCube_Generator')
    parser.add_argument('--config_file', type=str, default='exp_1',
                        help='configuration file (default: \'exp_1\')')
    return parser.parse_args()

# ------------------------------------------------------------------

def gen_datacube(args):

    # get config file
    config_file = args.config_file
    config = importlib.import_module('configs' + '.' + config_file)

    # get logger
    logger = utils.get_logger(config.exp_dir)

    utils.log_string(logger, "\ngenerating datacube ...")

    shutil.copy(os.path.join('configs', config_file) + '.py', os.path.join(config.exp_dir, 'config.py'))

    # define the configuration for the datacube
    datacube_cls = DataCube(config)

    # generate the data
    d_data, d_events, d_events_extremes, d_extremes, d_coupling = datacube_cls.gen_datacube()

    # save the datacube
    utils.log_string(logger, "saving datacube into {}...".format(config.exp_dir))
    datacube_cls.save_datacube(d_data, d_events, d_events_extremes, d_extremes, d_coupling)

    # compute climatology
    utils.log_string(logger, "computing climatology and save it into {}...".format(config.exp_dir))
    datacube_cls.comp_weekly_climatology(d_data, time_step=config.training_end_time)
    # compute statistic
    utils.log_string(logger, "computing statistic and save it into {}...".format(config.exp_dir))
    datacube_cls.comp_statistic(d_data, time_step=config.training_end_time)

    # add additional information to the log file
    d_events_union = d_events + d_events_extremes
    d_events_union[d_events_union > 1] = 1

    utils.log_string(logger, "number of all anomalous events {}".format(np.sum(d_events_union)))
    utils.log_string(logger, "number of anomalous extreme events {}".format(np.sum(d_events_extremes)))
    utils.log_string(logger, "number of extreme events {}".format(np.sum(d_extremes)))

    utils.log_string(logger, "percentage of all anomalous events {}".format(100 * np.sum(d_events_union)/d_events_union.size))
    utils.log_string(logger, "percentage of anomalous extreme events {}".format(100 * np.sum(d_events_extremes)/d_events_extremes.size))
    utils.log_string(logger, "percentage of extreme events {}".format(100 * np.sum(d_extremes == 1)/d_extremes.size))

    utils.log_string(logger, config_file + " Done!")


if __name__ == '__main__':

    args = parse_args()
    gen_datacube(args)

