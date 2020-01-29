#!/bin/bash
cd ..
cd ..
dir_to_mount=$(pwd)
docker run -d -p 17901:5901 -p 18901:6901 -v $dir_to_mount:/shared_drive --restart always diabetic-kaggle-no-gpu
