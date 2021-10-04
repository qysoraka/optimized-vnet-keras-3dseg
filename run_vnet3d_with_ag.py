
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
    parser.add_argument('--batch_size', '-bs', required=True, type=int)
    parser.add_argument('--image_size', '-is', required=True, type=int)
    parser.add_argument('--learning_rate', '-lr', required=True, type=float)
    parser.add_argument('--group_size', '-gs', required=True, type=int)
    parser.add_argument('--f_root', '-fr', required=True, type=int)
    parser.add_argument('--n_validation', required=True, type=int)
    parser.add_argument('--n_test', required=True, type=int)
    parser.add_argument('--optimizer', '-op', required=True, default='adam')
    parser.add_argument('--print_summary_only', action='store_true')
    parser.set_defaults(print_summary_only=False)
