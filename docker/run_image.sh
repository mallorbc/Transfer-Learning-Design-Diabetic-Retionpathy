#!/bin/bash
dir_to_mount=$(pwd)
docker run --gpus all -d -p 5901:5901 -v $dir_to_mount:/shared_drive diabetic-kaggle
