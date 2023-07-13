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
                