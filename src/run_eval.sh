#!/usr/bin/env bash
python run.py \
      --mode='eval' \
      --test_data_url='/home/nowburn/python_projects/python/Garbage_Classify/datasets/origin_data/my_data' \
      --num_classes=40 \
      --eval_pb_path='/home/nowburn/disk/data/Garbage_Classify/models/nas-label_smoothing-tta-11plus1/model' \
#      --eval_weights_path='/home/nowburn/disk/data/Garbage_Classify/new/nas-new_all-origin_aug-7/weights_005_0.8894.h5' \
