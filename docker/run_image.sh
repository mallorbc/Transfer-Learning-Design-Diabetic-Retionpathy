#!/bin/bash
cd ..
dir_to_mount=$(pwd)
docker run --gpus all -d -p 7901:5901 -p 8901:6901 -v $dir_to_mount:/shared_drive diabetic-kaggle
