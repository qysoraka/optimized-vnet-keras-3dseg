
if __name__ == '__main__':
    import glob
    import os
    import numpy as np
    import argparse
    import time, re
    import tensorflow as tf
    from keras.optimizers import Adam, SGD
    from utils import DataGenerator, dice_loss, dice_coefficient, ModelAndWeightsCheckpoint
    from vnet3d import VNet
    from keras.callbacks import LearningRateScheduler, Callback, TensorBoard, EarlyStopping
    
    parser = argparse.ArgumentParser(description="Script to run UNet3D")
    parser.add_argument('--core_tag', '-ct', required=True)
    parser.add_argument('--nii_dir', '-I', required=True)