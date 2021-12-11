
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
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, jsonpath, monitor='val_loss', verbose=0,
                 save_best_only=False, 
                 mode='auto', period=1):
        super(ModelAndWeightsCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.jsonpath = jsonpath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            jsonpath = self.jsonpath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        self.model.save_weights(filepath, overwrite=True)
                        with open(jsonpath, 'w') as f:
                            f.write(self.model.to_json())
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                self.model.save_weights(filepath, overwrite=True)
                with open(jsonpath, 'w') as f:
                    f.write(self.model.to_json())


def add_midlines(data):
    assert isinstance(data, np.ndarray), "[ERROR] input image is not a np.array: {}".format(type(data))
    arr = data.copy()
    x_mid, y_mid, z_mid = np.median(np.array(([0]*3, data.shape)), axis=0).astype(int)
    max_val = np.max(arr)
    arr[x_mid-1:x_mid+1, :, :] = (max_val*0.2) * np.ones_like(arr[x_mid-1:x_mid+1, :, :])
    arr[:, y_mid-1:y_mid+1, :] = (max_val*0.5) * np.ones_like(arr[:, y_mid-1:y_mid+1, :])
    arr[:, :, z_mid-1:z_mid+1] = max_val * np.ones_like(arr[:, :, z_mid-1:z_mid+1])
    return arr


def dice_coefficient(y_true, y_pred, squared=True, smooth=1e-8):
    y_true_flat, y_pred_flat = K.flatten(y_true), K.flatten(y_pred)
    dice_nom = 2 * K.sum(y_true_flat * y_pred_flat)