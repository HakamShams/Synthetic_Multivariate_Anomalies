# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

import numpy as np
import random
import os
import datetime
import logging

np.set_printoptions(suppress=True)

# ------------------------------------------------------------------

def log_string(logger, str):
    logger.info(str)
    print(str)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)


def get_logger(exp_dir):
    # Set Logger and create Directories

    if exp_dir is None or len(exp_dir) == 0:
        exp_dir = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    make_dir(exp_dir)

    logger = logging.getLogger("DataCube_Generator")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log_file.txt' % exp_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def min_max_scale(array, min_array, max_array, min_new=-1., max_new=1.):
    array = ((max_new - min_new) * (array - min_array) / (max_array - min_array)) + min_new
    return array


