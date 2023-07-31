

import numpy as np
import torch
import logging
import configparser

from training.train_model import training_loop
from training.inference import inference_loop
from training.testing import testing_loop
from utils import Messages, set_logging, get_device
from networks.network import MainNet


# Load config values
config = configparser.ConfigParser()
config.read('configs/training.cfg')

#--------------------
#   Config Commons
#--------------------
SEED = config.getint('Commons', 'seed')
TRAIN_MODEL = config.getboolean('Commons', 'train_model')
TEST_MODEL = config.getboolean('Commons', 'test_model')
INFERENCE_MODEL = config.getboolean('Commons', 'inference_model')

# Set Logging Level
set_logging(set_level=config.get('Commons', 'logging_level'))
# Class for displaying messages in terminal
m = Messages()

#---------------
#   Set Seeds
#---------------
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = get_device()
logging.critical(m.device(device))

#-----------------------------------
#           Model/Network
#-----------------------------------

# Model initialization
net = MainNet(device)

################
#   Training   #
################
if TRAIN_MODEL:
    training_loop(net)

###############
#   Testing   #
###############
if TEST_MODEL:
    test_metrics = testing_loop(net, device)
    print('PSNR:', test_metrics['PSNR'])
    print('SSIM:', test_metrics['SSIM'])
    print('FID:', test_metrics['FID'])

#################
#   Inference   #
#################
if INFERENCE_MODEL:
    inference_loop(net, device)
