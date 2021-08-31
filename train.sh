#!/bin/bash

python3 train.py --cfg=basic_config --gpu 0
python3 train.py --cfg=detection_config --gpu 0
python3 train.py --cfg=global_config --gpu 0
