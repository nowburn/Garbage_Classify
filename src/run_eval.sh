#!/usr/bin/env bash
python run.py \
      --mode='eval' \
      --eval_pb_path='/home/nowburn/disk/data/Garbage_Classify/model-finetune-all-11/model' \
      --test_data_url='/home/nowburn/python_projects/python/Garbage_Classify/datasets/garbage_classify/train_data' \
      --num_classes=40
#      --eval_weights_path='/home/nowburn/disk/data/Garbage_Classify/model-finetune/weights_008_0.9798.h5' \
