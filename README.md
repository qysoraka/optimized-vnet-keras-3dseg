# 3D Segmentor: V-Net Implementation with Keras

To setup the environment, install necessary packages using this command:
```pip3 install -r requirements.txt```

Key features:
- Implementation of modified V-Net [https://arxiv.org/abs/1606.04797]
- Data augmentation features such as translate, zoom, shear, flip, and rotate
- Supports group normalization (default group size 8)

Help:
```bash
usage: run_vnet3d.py [-h] --core_tag CORE_TAG --nii_dir NII_DIR --batch_size
                     BATCH_SIZE --image_size IMAGE_SIZE --learning_rate
                     LEARNING_RATE --group_size GROUP_SIZE --f_root F_ROOT
                     --n_validation N_VALIDATION --n_test N_TEST --optimizer
                     OPTIMIZER [--print_summary_only]

Script to run VNet3D

optional arguments:
  -h, --help            show this help message and exit
  --core_tag CORE_TAG, -ct CORE_TAG
  --nii_dir NII_DIR, -I NII_DIR
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
  --image_size IMAGE_SIZE, -is IMAGE_SIZE
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --group_size GROUP_SIZE, -gs GROUP_SIZE
  --f_root F_ROOT, -fr F_ROOT
  --n_validation N_VALIDATION
  --n_test N_TEST
  --optimizer OPTIMIZER, -op OPTIMIZER
  --print_summary_only
```
Only .nii.gz files should be stored in the `--nii_dir`. Each sample in the `--nii_dir` should have the following 5 suffices:
- `_flair.nii.gz` 
- `_t1.nii.gz`
- `_t1ce.nii.gz`
- `_t2.nii.gz`
- `_seg.nii.gz` (label)
The label file with `_seg.nii.gz` should have only 0, 1, 2, 4 as its value, corresponding to the BraTS 2018 brain tumor segmentation data [https://www.med.upenn.edu/sbia/brats2018/data.html].
