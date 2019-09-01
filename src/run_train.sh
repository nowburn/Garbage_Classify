#!/usr/bin/env bash
python run.py \
      --data_url='../datasets/train_data,../datasets/val_data' \
      --train_url='../model_snapshots' \
      --num_classes=40 \
      --deploy_script_path='./deploy_scripts' \
      --max_epochs=5
      #--test_data_url='../datasets/test_data'