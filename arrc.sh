#!/bin/bash

# set the partition where the job will run
#SBATCH --partition=gpu
# set the number of nodes
#SBATCH --nodes=1
# set the nodelist
#SBATCH --nodelist gpu-compute-1
# set max wallclock time
#SBATCH --time=1:00:00
# set name of job
#SBATCH --job-name=trainingmodel
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=lynchc2@mail.uc.edu
# run the application

module load anaconda/3.0
conda init bash
cd /home/lynchc2/project/
source activate /home/lynchc2/.conda/envs/diabetic
cd code
python main.py -n reg_crop_512_blur_00_5_3 -d /home/lynchc2/connor/reg_crop_512_blur_00_5/images/ -model 1 -si 0.25 -ti 0.25 -test_csv /home/lynchc2/project/code/reg_crop_512/csv_files/test.csv -train_csv /home/lynchc2/project/code/reg_crop_512/csv_files/train.csv -m 6 -csv /home/lynchc2/data/trainLabels_cropped.csv -b 36