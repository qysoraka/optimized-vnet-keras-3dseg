
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

    args = parser.parse_args()
    if args.optimizer == 'adam':
        args.learning_rate /= 20 # reduce lr for adam
    elif args.optimizer == 'sgd':
        pass
    else:
        raise Exception('[ERROR] optimizer = {}'.format(args.optimizer))

    # Cloud settings
    home_dir = os.path.expanduser("~")
    hostname = os.uname()[1]
    cloud_dir = '{}/gdrive/cloud/{}'.format(home_dir, hostname)
    try:
        os.system('mkdir -p ' + cloud_dir)
    except:
        pass

    # Get data
    # [IDs] Get sample IDs from src_dir
    src_dir = args.nii_dir #'data/data_with_augmentation/'
    assert os.path.exists(src_dir), "[ERROR] {} does not exist".format(src_dir)
    fpaths = glob.glob(src_dir + '/*.nii.gz')
    sids = sorted(set([os.path.split(x)[-1].rsplit('_', 1)[0] for x in fpaths]))
    
    seed = 0
    np.random.seed(seed)
    shuffle = True
    if shuffle:
        np.random.shuffle(sids)
    
    # Modules
    def lr_schedule_wrapper(learning_rate):
        learning_rate = learning_rate
        def lr_schedule(epoch):
            #learning_rate = 1e-4
            if epoch > 10:
                learning_rate /= 2
            if epoch > 20:
                learning_rate /= 2
            if epoch > 50:
                learning_rate /= 2
            tf.summary.scalar('learning_rate', learning_rate)
            #tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            return learning_rate
        return lr_schedule
    
    # Set params and callbacks
    n_val, n_test = args.n_validation, args.n_test
    n_train = len(sids) - n_val - n_test
    if n_train < 0:
        raise Exception("n_train({}) < n_validation({})+n_test({})".format(n_train, n_val, n_test))
    elif n_train < n_val + n_test:
        raise Exception("n_train({}) <  n_validation({})+n_test({})".format(n_train, n_val, n_test))

    train_ids = sids[:n_train]
    valid_ids = sids[n_train : n_train+n_val]
    test_ids = sids[n_train+n_val : n_train+n_val+n_test]
    print("IDs", len(sids), len(train_ids), len(valid_ids), len(test_ids), n_train)
    epochs = 100
    h5_dir = os.path.join(cloud_dir, 'models')
    if not os.path.exists(h5_dir):
        os.system('mkdir {}'.format(h5_dir))
    prefix = os.path.join(h5_dir, args.core_tag + 