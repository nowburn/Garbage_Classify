#!/usr/bin/env bash
python run.py \
      --data_url='../datasets/origin_data/train' \
      --test_data_url='../datasets/origin_data/test' \
      --train_url='../model_snapshots' \
      --deploy_script_path='./deploy_scripts' \
      --num_classes=40 \
      --max_epochs=40 \
