#!/bin/bash

python train.py --cfg=basic_config --gpu 0,1,2,3
python train.py --cfg=detection_config --gpu 0,1,2,3
python train.py --cfg=global_config --gpu 0,1,2,3