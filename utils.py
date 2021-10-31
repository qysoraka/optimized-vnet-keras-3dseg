
import os
import glob
import keras
import nibabel
import time
from pathlib import Path
import numpy as np
from itertools import cycle
from numpy.random import random
from scipy.ndimage import interpolation
from collections import defaultdict
from scipy.ndimage import zoom
from keras import backend as K
from keras.callbacks import Callback


class ModelAndWeightsCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.