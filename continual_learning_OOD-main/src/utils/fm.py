import torch
import pickle
import os
import logging
import numpy as np
import random

##################################################################################
# FILE MANAGEMENT 
##################################################################################

get_device = lambda id=0: f"cuda:{id}" if torch.cuda.is_available() else 'cpu'


# TODO: Add documentation
def my_load(path, format='rb'):
    with open(path, format) as f:
        object = pickle.load(f)
    return object


# TODO: Add documentation
def my_save(object, path, format='wb'):
    with open(path, format) as f:
        pickle.dump(object, f)


# Same as os.path.join, but it replaces all '\\' with '/' when running on Windows
def join(*args):
    path = os.path.join(*args).replace('\\', '/')
    return path


# Procedure for setting all the seeds
def set_all_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# Procedure to set a <fname>.log file in the directory specified by <root>
def init_logger(root, fname='progress', common_logger_path=None):
    logger = logging.getLogger(fname)
    logger.setLevel(logging.DEBUG)

    logger.handlers = []
    # set the file on which to write the logging messages
    fh = logging.FileHandler(os.path.join(root, f'{fname}.log'))
    # formatter_file = logging.Formatter('%(asctime)s - %(message)s')
    # example of output format:
    # [02-12 12:48:17] /home/dangioni/UncertaintyQuantificationUnderAttack/main_attack.py:174} DEBUG - <message>
    formatter = logging.Formatter('[%(asctime)s] %(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                                  '%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if common_logger_path is not None:
        # date = datetime.now().strftime("day-%d-%m-%Y_hr-%H-%M-%S")
        # common_fh = logging.FileHandler(os.path.join(common_root, f'{date}_progress.log'))
        common_fh = logging.FileHandler(common_logger_path)
        common_formatter = logging.Formatter('[%(asctime)s] %(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                                             '%m-%d %H:%M:%S')
        common_fh.setFormatter(common_formatter)
        logger.addHandler(common_fh)

    # Don't remember exactly, something like flushing the messages in real time in the logging file
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(fh)
    logger.addHandler(streamhandler)
    return logger


# If script parameters are collected as a dictionary, write them in a readable text format
def write_kwargs(kwargs, dirname, fname):
    s = ''
    for k, v in kwargs.items():
        s += f"{k}: {v}\n"
    with open(join(dirname, f"{fname}.txt"), 'w') as f:
        f.write(s)
    return s