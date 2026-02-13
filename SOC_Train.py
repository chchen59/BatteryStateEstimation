import numpy as np
import pandas as pd
import scipy.io
import math
import os
import ntpath
import sys
import logging
import time
import sys

from importlib import reload
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
#from keras.utils import np_utils
from keras.layers import LSTM, Embedding, RepeatVector, TimeDistributed, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback

data_path = "./"
sys.path.append(data_path)

from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols, CapacityCols
from data_processing.model_data_handler import ModelDataHandler

# Config logging
reload(logging)
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')

# Load the cycle and capacity data to memory based on the specified chunk size
dataset = UniboPowertoolsData(
    test_types=['S'],
    chunk_size=1000000,
    lines=[37, 40],
    charge_line=37,
    discharge_line=40,
    base_path=data_path
)

# Prepare the training and testing data for model data handler to load the model input and output data.
train_data_test_names = [
    '000-DM-3.0-4019-S', 
    #'001-DM-3.0-4019-S', 
    #'002-DM-3.0-4019-S', 
    #'006-EE-2.85-0820-S', 
    #'007-EE-2.85-0820-S', 
    #'018-DP-2.00-1320-S', 
    #'019-DP-2.00-1320-S',
    #'036-DP-2.00-1720-S', 
    #'037-DP-2.00-1720-S', 
    #'038-DP-2.00-2420-S', 
    #'040-DM-4.00-2320-S',
    #'042-EE-2.85-0820-S', 
    #'045-BE-2.75-2019-S'
]

test_data_test_names = [
    '003-DM-3.0-4019-S',
    #'008-EE-2.85-0820-S',
    #'039-DP-2.00-2420-S', 
    #'041-DM-4.00-2320-S',    
]

dataset.prepare_data(train_data_test_names, test_data_test_names)

# Model data handler will be used to get the model input and output data for further training purpose.
mdh = ModelDataHandler(dataset, [
    CycleCols.VOLTAGE,
    CycleCols.CURRENT,
    CycleCols.TEMPERATURE,
], [CapacityCols.SOH, CapacityCols.AVERAGE_TENSION])

train_x, train_y, test_x, test_y = mdh.get_discharge_whole_cycle(soh = False, output_capacity = False)

train_y = mdh.keep_only_capacity(train_y, is_multiple_output = True)
test_y = mdh.keep_only_capacity(test_y, is_multiple_output = True)

print(train_x.shape)